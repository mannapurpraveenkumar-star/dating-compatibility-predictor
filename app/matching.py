"""Predict the single most likely pair in a group photo.

Method
------
1. Every person's features are read in the background (app.attributes):
   estimated age, apparent gender, outfit colour, and head height in the frame.
   (Smiles are logged but not scored: nearly everyone smiles in a posed photo.)
2. Every unique pair, n(n-1)/2 of them, is scored from 0 to 100 on two things:
     * Similarity  – people at a similar "level": similar apparent age, similar
                     height in the frame, similar outfit colour.
                     (Based on assortative mating: partners tend to resemble
                     each other on traits such as age and height.)
     * Togetherness – how they are posed: close together, side by side,
                     head tilted towards the other.
3. Pairs are chosen jointly: the set of non-overlapping pairs with the highest
   total of (score - MIN_MATCH_SCORE) wins. Choosing jointly matters in family
   photos, where couples often stand at opposite ends: grabbing the single best
   pair first can leave the remaining people wrongly paired. Only pairs scoring at
   least MIN_MATCH_SCORE are ever chosen.

Safeguards
----------
* Skin colour is never measured or used.
* Pairs must be one man and one woman (apparent gender), applied only when the
  model is at least GENDER_CONFIDENT sure about both people. See config.py.
* Only people the age check marks as adults are matched (app.attributes):
  estimated age 21+, no sign of a child from the second age model, and a face
  big enough to check. Anyone else is left out, so the system errs on the side
  of excluding rather than ever pairing a child.
* If no pair scores at least MIN_MATCH_SCORE, no pair is predicted.

The weights are a documented starting point chosen for this project, not
learned from data. app.research can test them against consented, labelled data.
"""

from __future__ import annotations

from app import config
from app.attributes import delta_e
from app.vision import Face, roll_angle

# Lowered from 50 to 35 once the apparent-gender rule removed the main source of
# false matches (same-age friends or siblings of the same gender). 35-49 = "Weak match".
MIN_MATCH_SCORE = 35

# Weight of each component in the 0-100 score. Missing components are skipped
# and the remaining weights are rescaled, so a missing model never zeroes a score.
# Age carries the most weight because partners' ages are strongly correlated
# (r around 0.8 in population studies), far more than height (around 0.2).
# Pose cues are weaker evidence: photographers often seat children between parents.
WEIGHTS = {
    # similarity
    "age": 40,
    "height": 5,
    "outfit": 10,
    # togetherness
    "closeness": 20,
    "side_by_side": 10,
    "leaning_in": 10,
}

FEATURE_NAMES = {
    "age": "Similar age",
    "height": "Similar height in photo",
    "outfit": "Similar outfit colour",
    "closeness": "Standing close",
    "side_by_side": "Next to each other",
    "leaning_in": "Leaning in",
}


def is_eligible(person: dict) -> bool:
    age = person.get("age") or {}
    return bool(age.get("adult"))


def exclusion_reason(person: dict) -> str:
    age = person.get("age") or {}
    return age.get("reason") or "age could not be checked"


def _age_similarity(pa: dict, pb: dict) -> float | None:
    ya, yb = (pa.get("age") or {}).get("years"), (pb.get("age") or {}).get("years")
    if ya is None or yb is None:
        return None
    return max(0.0, 1.0 - abs(ya - yb) / 20.0)     # same age = 1, 20+ years apart = 0


def _height_similarity(a: Face, b: Face, size_ratio: float) -> float | None:
    if size_ratio < config.HEIGHT_COMPARABLE_SIZE_RATIO:
        return None  # different distances from the camera: not comparable
    offset = abs(b.y - a.y) / ((a.h + b.h) / 2)
    return max(0.0, 1.0 - offset)


def _outfit_similarity(pa: dict, pb: dict) -> float | None:
    oa, ob = pa.get("outfit") or {}, pb.get("outfit") or {}
    if not oa.get("visible") or not ob.get("visible"):
        return None
    return max(0.0, 1.0 - delta_e(oa["lab"], ob["lab"]) / 50.0)


def _closeness(features: dict) -> float:
    d = features["centre_distance_face_widths"]
    return float(min(1.0, max(0.0, (4.0 - d) / (4.0 - 1.2))))


def _leaning_in(a: Face, b: Face) -> float | None:
    ra, rb = roll_angle(a), roll_angle(b)   # a is left of b
    if ra is None or rb is None:
        return None
    # Positive roll tilts towards image-right (towards b for the left person).
    lean = max(0.0, ra) + max(0.0, -rb)
    return min(1.0, lean / (2 * config.LEAN_DEGREES))


def same_apparent_gender(pa: dict, pb: dict) -> bool:
    """True only when the model is confident both people have the same apparent gender."""
    ga = (pa.get("age") or {}).get("gender")
    gb = (pb.get("age") or {}).get("gender")
    if not config.GENDER_RULE_ENABLED or not ga or not gb:
        return False
    confident = min(ga["confidence"], gb["confidence"]) >= config.GENDER_CONFIDENT
    return confident and ga["apparent"] == gb["apparent"]


def score_pair(a: Face, b: Face, pa: dict, pb: dict, features: dict) -> dict:
    parts = {
        "age": _age_similarity(pa, pb),
        "height": _height_similarity(a, b, features["size_ratio"]),
        "outfit": _outfit_similarity(pa, pb),
        "closeness": _closeness(features),
        "side_by_side": float(features["adjacent_in_row"]),
        "leaning_in": _leaning_in(a, b),
    }
    used = {k: v for k, v in parts.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in used)
    # Points each feature adds to the 0-100 score (they sum to the score).
    points = {k: (100.0 * WEIGHTS[k] * v / total_w if total_w else 0.0) for k, v in used.items()}
    score = sum(points.values())
    breakdown = [
        {"feature": k, "name": FEATURE_NAMES[k], "points": round(points[k], 1),
         "max_points": round(100.0 * WEIGHTS[k] / total_w, 1) if total_w else 0.0}
        for k in sorted(used, key=lambda k: -points[k])
    ] + [{"feature": k, "name": FEATURE_NAMES[k], "points": None, "max_points": None}
         for k, v in parts.items() if v is None]
    reasons = [f"{b['name'].lower()} (+{round(b['points'])})" for b in breakdown if b["points"]][:3]
    return {
        "a": a.label, "b": b.label,
        "blocked": "same apparent gender" if same_apparent_gender(pa, pb) else None,
        "match_score": round(score),
        "components": {k: (None if v is None else round(v, 2)) for k, v in parts.items()},
        "breakdown": breakdown,
        "reasons": reasons,
    }


def match_level(score: int) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 50:
        return "Possible match"
    return "Weak match"


def _pair_entry(p: dict) -> dict:
    return {"pair": [p["a"], p["b"]], "match_score": p["match_score"],
            "level": match_level(p["match_score"]), "reasons": p["reasons"],
            "breakdown": p.get("breakdown", [])}


def people_status(people: dict[str, dict], excluded: list[dict], pairs: list[dict],
                  scored_pairs: list[dict]) -> list[dict]:
    """One line per detected person: paired, excluded (and why), or not paired (and why)."""
    reasons = {e["label"]: e["reason"] for e in excluded}
    partner = {}
    for p in pairs:
        a, b = p["pair"]
        partner[a], partner[b] = (b, p["match_score"]), (a, p["match_score"])
    out = []
    for label in people:
        if label in partner:
            other, score = partner[label]
            out.append({"label": label, "status": "paired", "detail": f"Paired with {other} ({score}/100)"})
        elif label in reasons:
            age = people[label].get("age") or {}
            extra = ""
            if age.get("confirmed_by_user"):
                extra = ""
            elif age.get("reason") == "face too small to check age" and "face_px" in age:
                extra = f" ({age['face_px']} px wide; needs {config.MIN_FACE_FOR_AGE})"
            elif age.get("reason") == "face too blurry to check age" and "sharpness" in age:
                extra = f" (sharpness {age['sharpness']}; needs {config.MIN_SHARPNESS_FOR_AGE})"
            out.append({"label": label, "status": "excluded", "detail": f"Not matched: {reasons[label]}{extra}"})
        else:
            options = [p for p in scored_pairs if label in (p["a"], p["b"])
                       and not ({p["a"], p["b"]} & set(reasons)) and not p.get("blocked")]
            if not options:
                detail = "Not paired: no other adult to pair with"
            else:
                best = max(options, key=lambda p: p["match_score"])
                other = best["b"] if best["a"] == label else best["a"]
                if best["match_score"] < MIN_MATCH_SCORE:
                    detail = (f"Not paired: best option {other} scored {best['match_score']}, "
                              f"below {MIN_MATCH_SCORE}")
                else:
                    detail = (f"Not paired: best option {other} ({best['match_score']}) "
                              "fits better with someone else")
            out.append({"label": label, "status": "unpaired", "detail": detail})
    return out


MAX_JOINT_PEOPLE = 16   # exact joint search up to this many adults; greedy beyond


def best_matching(candidates: list[dict]) -> list[dict]:
    """Non-overlapping pairs with the highest total (score - MIN_MATCH_SCORE).

    Exact search over subsets of people (bitmask dynamic programming). Subtracting
    the threshold means a strong pair is never broken up just to make two weak ones.
    """
    people = sorted({p["a"] for p in candidates} | {p["b"] for p in candidates})
    if len(people) > MAX_JOINT_PEOPLE:          # very large groups: best-first instead
        chosen, used = [], set()
        for p in candidates:                    # already sorted best-first
            if p["a"] not in used and p["b"] not in used:
                chosen.append(p)
                used.update((p["a"], p["b"]))
        return chosen
    index = {name: i for i, name in enumerate(people)}
    edges = {}
    for p in candidates:
        edges[(index[p["a"]], index[p["b"]])] = p

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(free: int) -> tuple[float, tuple]:
        if free == 0:
            return 0.0, ()
        i = (free & -free).bit_length() - 1      # lowest free person
        rest = free & ~(1 << i)
        best = solve(rest)                       # option 1: leave person i unpaired
        for j in range(i + 1, len(people)):
            if rest & (1 << j):
                p = edges.get((i, j)) or edges.get((j, i))
                if p is not None:
                    value, picked = solve(rest & ~(1 << j))
                    value += p["match_score"] - MIN_MATCH_SCORE
                    if value > best[0]:
                        best = (value, picked + ((i, j),))
        return best

    _, picked = solve((1 << len(people)) - 1)
    return [edges.get((i, j)) or edges.get((j, i)) for i, j in picked]


def predict(scored_pairs: list[dict], people: dict[str, dict], n_faces: int) -> dict:
    """Choose likely pairs jointly, never using a person twice."""
    excluded = [{"label": label, "reason": exclusion_reason(p)} for label, p in people.items() if not is_eligible(p)]
    out_labels = {e["label"] for e in excluded}
    ranked = sorted((p for p in scored_pairs if p["a"] not in out_labels and p["b"] not in out_labels
                     and not p.get("blocked")),
                    key=lambda p: (-p["match_score"], p["a"], p["b"]))

    chosen = best_matching([p for p in ranked if p["match_score"] >= MIN_MATCH_SCORE])
    pairs = [_pair_entry(p) for p in sorted(chosen, key=lambda p: -p["match_score"])]

    result = {"pairs": pairs, "pair": None, "match_score": None, "level": None, "reasons": [],
              "excluded_people": excluded, "pairs_compared": len(scored_pairs),
              "people_status": people_status(people, excluded, pairs, scored_pairs)}
    if pairs:
        result.update({k: pairs[0][k] for k in ("pair", "match_score", "level", "reasons")})

    eligible = len(people) - len(excluded)
    unreadable = sum(1 for e in excluded if e["reason"] in ("face too small to check age", "face too blurry to check age"))
    if n_faces < 2:
        result["message"] = "At least two people are needed to predict a pair."
    elif eligible < 2 and unreadable and unreadable == len(excluded):
        result["message"] = ("The faces are too small for the app to tell adults from children. "
                             "Mark who is an adult under “Check ages” and press Find pairs.")
        result["needs_age_check"] = True
    elif eligible < 2:
        result["message"] = "Not enough adults to predict a pair. People who may be under 18 are never matched."
        if unreadable:
            result["message"] += " Some faces were too small to check: mark them under “Check ages”."
            result["needs_age_check"] = True
    elif not pairs:
        result["message"] = "No likely pair found in this photo."
    else:
        names = [f"{p['pair'][0]} & {p['pair'][1]}" for p in pairs]
        result["message"] = (f"{names[0]} are the most likely pair." if len(pairs) == 1
                             else f"{len(pairs)} likely pairs found: {', '.join(names)}.")
        if len(ranked) > 1 and ranked[1]["match_score"] == ranked[0]["match_score"] \
                and {ranked[1]["a"], ranked[1]["b"]} & set(pairs[0]["pair"]):
            result["tie"] = True
            result["message"] += " The top pair tied with another option, so it's a close call."
    return result
