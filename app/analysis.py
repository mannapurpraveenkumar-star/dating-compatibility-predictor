"""Analysis pipeline: features for each person, a rating for every pair, the likely couples.

1. Read each detected person's features (app.attributes): age, apparent gender,
   attractiveness rating, outfit colour.
2. Enumerate every unique pair once: n(n-1)/2, across all uploaded photos.
3. Rate each pair on every factor and score it with the chosen weights (app.matching).
4. Choose the best set of pairs (app.matching.predict).

All per-person ratings and per-pair ratings are returned, so the website can
show the ratings themselves, not only the matches.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timezone

import cv2
import numpy as np

from app import __version__, config
from app.attributes import describe_person
from app.identity import same_people
from app.matching import MIN_MATCH_SCORE, factor_catalogue, is_eligible, parse_weights, predict, score_pair
from app.vision import DecodedImage, Detection, Face, colour_hex, face_region, grey_stats, label_across_photos

RESEARCH_QUESTION = "Can a Machine Learning Model predict whether two people are meant to be together?"

METHOD = (
    "Each person's estimated age, apparent gender, attractiveness rating and outfit colour are read from "
    "the image. Every possible man-woman pair is rated on each factor (how similar the two people are, "
    "and optionally how close they stand), the ratings are combined using adjustable weights, and the "
    "best set of pairs is the prediction. Faces can come from one photo or several; the same person "
    "appearing in more than one photo is counted once."
)

LIMITATIONS = [
    "This is an experimental prediction. It has not yet been tested against real couples, "
    "so its accuracy is unknown.",
    "Friends, family and colleagues can look alike or pose close together, so a high score does not prove "
    "two people are dating.",
    "Estimated age is often several years out, and height in the photo is not real height.",
    "The attractiveness rating comes from a model trained on 5,500 faces (73% East Asian, 27% white) "
    "rated by 60 volunteers. It reflects their taste, not a fact, and is unreliable for other groups.",
    "Skin colour is never used. People who may be under 18, or whose faces are too small to check, are never matched.",
    "Pairs are limited to one man and one woman by apparent gender, so same-sex couples are not predicted, "
    "and gender read from a face can be wrong (the rule only applies when the model is confident).",
    "With several photos, faces are compared with each other to spot the same person twice "
    "(SFace face matching). This only works within one upload: no names, no database, nothing stored. "
    "Look-alikes such as twins can be merged by mistake, and a very different angle can be missed.",
]

# Numeric features per pair, in a fixed order, shared with research.py.
PAIR_FEATURE_NAMES = [
    "centre_distance_face_widths",
    "size_ratio",
    "vertical_offset_face_heights",
    "adjacent_in_row",
    "box_overlap_iou",
]


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------
def pair_count(n: int) -> int:
    """Number of unique unordered pairs among n people: n(n-1)/2."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (n - 1) // 2


def unique_pairs(items: list) -> list[tuple]:
    """Every unordered pair exactly once, in a stable order (A-B, A-C, ..., B-C, ...)."""
    return list(itertools.combinations(items, 2))


def iou(a: Face, b: Face) -> float:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def pair_features(a: Face, b: Face) -> dict[str, float]:
    """Measurable geometry of two detected faces within the frame."""
    mean_w = (a.w + b.w) / 2
    mean_h = (a.h + b.h) / 2
    dist = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
    return {
        "centre_distance_px": round(dist, 1),
        "centre_distance_face_widths": round(dist / mean_w, 2),
        "size_ratio": round(min(a.area, b.area) / max(a.area, b.area), 3),
        "vertical_offset_face_heights": round(abs(a.cy - b.cy) / mean_h, 2),
        "adjacent_in_row": int(abs(a.index - b.index) == 1),
        "box_overlap_iou": round(iou(a, b), 3),
    }


def feature_vector(features: dict[str, float]) -> list[float]:
    return [float(features[k]) for k in PAIR_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Quality flags (used to warn about unreliable photos)
# ---------------------------------------------------------------------------
def quality_flags(stats: dict[str, float], *, face: Face | None = None) -> list[str]:
    flags: list[str] = []
    if stats["brightness"] < config.DARK_MEAN:
        flags.append("Dark")
    elif stats["brightness"] > config.BRIGHT_MEAN:
        flags.append("Over-exposed")
    if stats["contrast"] < config.LOW_CONTRAST_STD:
        flags.append("Low contrast")
    if stats["sharpness"] < config.BLUR_LAPLACIAN_VAR:
        flags.append("Possibly blurred")
    if face is not None:
        if min(face.w, face.h) < config.MIN_FACE_SIDE_FOR_QUALITY:
            flags.append("Small face")
        if face.score < config.LOW_CONFIDENCE:
            flags.append("Low detection confidence")
    return flags


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def apply_age_answers(age: dict, answer: str | None) -> dict:
    """The user's own answer about who is an adult always wins over the models.

    The models can't reliably tell a child from an adult on small faces, but a person
    looking at the photo can. A child marked "child" is never matched; a person
    marked "adult" is matched using the model's estimated age for comparison.
    """
    if answer == "adult":
        return {**age, "adult": True, "reason": None, "confirmed_by_user": True}
    if answer == "child":
        return {**age, "adult": False, "reason": "marked as under 18 by you", "confirmed_by_user": True}
    return age


def analyse(decoded: DecodedImage, detection: Detection, age_answers: dict[str, str] | None = None,
            weights: dict | None = None) -> dict:
    """Analyse one photo. See analyse_photos for the details."""
    return analyse_photos([(decoded, detection)], age_answers, weights)


def analyse_photos(photos: list[tuple[DecodedImage, Detection]],
                   age_answers: dict[str, str] | None = None,
                   weights: dict | None = None) -> dict:
    """Build the structured, JSON-serialisable analysis result for one or more photos.

    * Faces from all photos form ONE group, labelled Person A, B, ... photo by photo.
    * Pairs from the same photo are rated on every factor; pairs from different
      photos have no shared geometry, so proximity factors are skipped for them.
    * ``age_answers`` maps a label to "adult" or "child", as confirmed by the user.
    * ``weights`` maps factor keys to 0-100 (missing keys use the defaults).
    """
    age_answers = age_answers or {}
    weights = parse_weights(weights)
    faces = label_across_photos([det.faces for _, det in photos])
    n = len(faces)

    images = []
    for p, (decoded, det) in enumerate(photos):
        img = decoded.image
        h, w = img.shape[:2]
        s = min(1.0, 1024 / max(h, w))
        small = img if s == 1.0 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        images.append({"photo": p + 1, "width": decoded.width, "height": decoded.height,
                       "format": decoded.fmt.upper(), "quality_flags": quality_flags(grey_stats(small)),
                       "face_count": len(det.faces)})

    face_rows, people = [], {}
    for f in faces:
        img = photos[f.photo][0].image
        person = describe_person(img, f)
        person["age"] = apply_age_answers(person["age"], age_answers.get(f.label))
        people[f.label] = person
        face_rows.append({
            "index": f.index,
            "label": f.label,
            "letter": f.letter,
            "photo": f.photo + 1,
            "colour": colour_hex(f.index),
            "box": {"x": f.x, "y": f.y, "w": f.w, "h": f.h},
            "confidence": round(f.score, 3),
            "quality_flags": quality_flags(grey_stats(face_region(img, f)), face=f),
            "features": person,
        })

    # The same person in two photos is one person: keep the largest face, mark the rest.
    same = same_people([d.image for d, _ in photos], faces)
    for label, keep in same["duplicate_of"].items():
        people[label]["duplicate_of"] = keep
        people[label]["same_person_similarity"] = same["similarity"].get(label)
    for group in same["groups"]:
        people[group[0]]["also_in"] = [{"label": x, "photo": next(f.photo for f in faces if f.label == x) + 1}
                                       for x in group[1:]]

    # "Side by side" is judged among the adults of the same photo only, so a child
    # standing between two parents does not count against them.
    adult_order = {p: [f.label for f in faces if f.photo == p and is_eligible(people[f.label])]
                   for p in range(len(photos))}
    scored = []
    for a, b in unique_pairs(faces):
        if a.photo == b.photo:
            feat = pair_features(a, b)
            order = adult_order[a.photo]
            adjacent = (a.label in order and b.label in order
                        and abs(order.index(a.label) - order.index(b.label)) == 1)
            geometry = {**feat, "adjacent_in_row": int(adjacent)}
        else:
            feat, geometry = None, None
        row = score_pair(a, b, people[a.label], people[b.label], geometry, weights)
        row["geometry"] = feat
        scored.append(row)
    assert len(scored) == pair_count(n)

    prediction = predict(scored, people, n)

    return {
        "app_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "research_question": RESEARCH_QUESTION,
        "method": METHOD,
        "photo_count": len(photos),
        "image": images[0] if images else None,
        "images": images,
        "detector": {
            "name": "YuNet (OpenCV FaceDetectorYN)",
            "opencv_version": cv2.__version__,
            "total_detected": sum(det.total_detected for _, det in photos),
            "capped_at": config.MAX_FACES if any(det.capped for _, det in photos) else None,
        },
        "face_count": n,
        "people_count": n - len(same["duplicate_of"]),
        "same_person": {"checked": same.get("checked", False), "groups": same["groups"],
                        "threshold": config.SAME_PERSON_COSINE},
        "pair_count": pair_count(n),
        "weights": weights,
        "min_match_score": MIN_MATCH_SCORE,
        "factors": factor_catalogue(),
        "prediction": prediction,
        "age_answers_used": {k: v for k, v in age_answers.items() if k in people},
        "faces": face_rows,
        "pair_scores": sorted(scored, key=lambda p: (-p["match_score"], p["a"], p["b"])),
        "limitations": LIMITATIONS,
    }
