"""Analysis pipeline: features for each person, a score for every pair, one predicted pair.

1. Read each detected person's features (app.attributes) in the background.
2. Enumerate every unique pair once: n(n-1)/2.
3. Score each pair for similarity and togetherness (app.matching).
4. Return the single best pair as the prediction.

The per-person features and per-pair scores stay in the JSON result for
research and debugging, but the website shows only the prediction.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timezone

import cv2
import numpy as np

from app import __version__, config
from app.attributes import describe_person
from app.matching import is_eligible, predict, score_pair
from app.vision import DecodedImage, Detection, Face, colour_hex, face_region, grey_stats

RESEARCH_QUESTION = "Can a Machine Learning Model predict whether two people are meant to be together?"

METHOD = (
    "Each person's estimated age, apparent gender, height in the photo and outfit colour are read from "
    "the image. Every possible man-woman pair is scored on how similar the two people are and how they "
    "are posed (close together, side by side, leaning in). The best set of pairs is the prediction."
)

LIMITATIONS = [
    "This is an experimental prediction from one photo. It has not yet been tested against real couples, "
    "so its accuracy is unknown.",
    "Friends, family and colleagues often pose close together, so a high score does not prove two people are dating.",
    "Estimated age is often several years out, and height in the photo is not real height.",
    "Skin colour is never used. People who may be under 18, or whose faces are too small to check, are never matched.",
    "Pairs are limited to one man and one woman by apparent gender, so same-sex couples are not predicted, "
    "and gender read from a face can be wrong (the rule only applies when the model is confident).",
    "Nobody is identified: there is no face recognition and no names.",
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


def analyse(decoded: DecodedImage, detection: Detection, age_answers: dict[str, str] | None = None) -> dict:
    """Build the structured, JSON-serialisable analysis result.

    ``age_answers`` maps a person's label to "adult" or "child", as confirmed by the user.
    """
    age_answers = age_answers or {}
    img = decoded.image
    faces = detection.faces
    n = len(faces)

    h, w = img.shape[:2]
    s = min(1.0, 1024 / max(h, w))
    small = img if s == 1.0 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    image_flags = quality_flags(grey_stats(small))

    face_rows, people = [], {}
    for f in faces:
        person = describe_person(img, f)
        person["age"] = apply_age_answers(person["age"], age_answers.get(f.label))
        people[f.label] = person
        face_rows.append({
            "index": f.index,
            "label": f.label,
            "letter": f.letter,
            "colour": colour_hex(f.index),
            "box": {"x": f.x, "y": f.y, "w": f.w, "h": f.h},
            "confidence": round(f.score, 3),
            "quality_flags": quality_flags(grey_stats(face_region(img, f)), face=f),
            "features": person,
        })

    # "Side by side" is judged among the adults only, so a child standing
    # between two parents does not count against them.
    adult_order = [f.label for f in faces if is_eligible(people[f.label])]
    scored = []
    for a, b in unique_pairs(faces):
        feat = pair_features(a, b)
        both_adults = a.label in adult_order and b.label in adult_order
        adjacent = both_adults and abs(adult_order.index(a.label) - adult_order.index(b.label)) == 1
        row = score_pair(a, b, people[a.label], people[b.label], {**feat, "adjacent_in_row": int(adjacent)})
        row["geometry"] = feat
        scored.append(row)
    assert len(scored) == pair_count(n)

    prediction = predict(scored, people, n)

    return {
        "app_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "research_question": RESEARCH_QUESTION,
        "method": METHOD,
        "image": {
            "width": decoded.width,
            "height": decoded.height,
            "format": decoded.fmt.upper(),
            "quality_flags": image_flags,
        },
        "detector": {
            "name": "YuNet (OpenCV FaceDetectorYN)",
            "opencv_version": cv2.__version__,
            "total_detected": detection.total_detected,
            "capped_at": config.MAX_FACES if detection.capped else None,
        },
        "face_count": n,
        "pair_count": pair_count(n),
        "prediction": prediction,
        "age_answers_used": {k: v for k, v in age_answers.items() if k in people},
        "faces": face_rows,
        "pair_scores": sorted(scored, key=lambda p: -p["match_score"]),
        "limitations": LIMITATIONS,
    }
