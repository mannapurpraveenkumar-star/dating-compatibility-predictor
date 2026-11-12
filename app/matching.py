"""Score every pair of people and predict the likely couples.

Method
------
1. Every person's features are read in the background (app.attributes):
   estimated age, apparent gender, attractiveness rating (if its model is
   installed), outfit colour and position in the photo.
2. Every unique pair, n(n-1)/2 of them, gets a 0-1 RATING on each FACTOR below
   (1 = as alike / as close as possible). A pair's SCORE (0-100) is the weighted
   average of its ratings; the weights can be changed per request.
3. Pairs are chosen jointly: the set of non-overlapping pairs with the highest
   total of (score - MIN_MATCH_SCORE) wins, so a strong pair is never broken up
   to make two weak ones. Only pairs scoring at least MIN_MATCH_SCORE are chosen.

ADDING OR REMOVING A FACTOR
---------------------------
All pair factors live in the FACTORS list below. To add one:
  1. write a rating function `def _my_factor(ctx: PairContext) -> float | None`
     returning 0-1, or None when it can't be measured for this pair;
  2. add a Factor(...) entry to FACTORS with a key, a display name, a group
     ("similarity" or "proximity"), a default weight and a one-line description.
The website, the sliders, the ratings tables, the report and the log all pick
it up automatically. If it needs a new per-person measurement, add it in
app.attributes.describe_person first and read it from ctx.pa / ctx.pb.
To switch a factor off, set its default weight to 0 (it is still rated and
shown, just not scored); to remove it entirely, delete its entry.

Safeguards
----------
* Skin colour is never measured or used.
* Pairs must be one man and one woman (apparent gender), applied only when the
  model is at least GENDER_CONFIDENT sure about both people. See config.py.
* Only people the age check marks as adults are matched (app.attributes), or
  people the user has marked as adults. Anyone else is left out.

The weights are a documented starting point chosen for this project, not
learned from data. app.research can test them against consented, labelled data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app import config
from app.attributes import delta_e
from app.vision import Face, roll_angle

# Lowered from 50 to 35 once the apparent-gender rule removed the main source of
# false matches (same-age friends or siblings of the same gender). 35-49 = "Weak match".
MIN_MATCH_SCORE = 35
MAX_WEIGHT = 100


@dataclass
class PairContext:
    """Everything a rating function may look at for one pair."""
    a: Face                 # the two faces (a is left of b when in the same photo)
    b: Face
    pa: dict                # person features from app.attributes.describe_person
    pb: dict
    geometry: dict | None   # app.analysis.pair_features; None when in different photos


@dataclass(frozen=True)
class Factor:
    key: str
    name: str
    group: str              # "similarity" (who they are) or "proximity" (how they stand)
    default_weight: int
    rate: Callable[[PairContext], float | None]
    description: str


# ---------------------------------------------------------------------------
# Rating functions: each returns 0-1, or None when it can't be measured
# ---------------------------------------------------------------------------
def _age(ctx: PairContext) -> float | None:
    ya, yb = (ctx.pa.get("age") or {}).get("years"), (ctx.pb.get("age") or {}).get("years")
    if ya is None or yb is None:
        return None
    return max(0.0, 1.0 - abs(ya - yb) / 20.0)          # same age = 1, 20+ years apart = 0


def _attractiveness(ctx: PairContext) -> float | None:
    ra = (ctx.pa.get("attractiveness") or {}).get("score")
    rb = (ctx.pb.get("attractiveness") or {}).get("score")
    if ra is None or rb is None:
        return None
    return max(0.0, 1.0 - abs(ra - rb) / 1.5)          # 1-5 scale: same = 1, 1.5+ apart = 0


def _outfit(ctx: PairContext) -> float | None:
    oa, ob = ctx.pa.get("outfit") or {}, ctx.pb.get("outfit") or {}
    if not oa.get("visible") or not ob.get("visible"):
        return None
    return max(0.0, 1.0 - delta_e(oa["lab"], ob["lab"]) / 50.0)


def _height(ctx: PairContext) -> float | None:
    g = ctx.geometry
    if g is None or g["size_ratio"] < config.HEIGHT_COMPARABLE_SIZE_RATIO:
        return None  # different photos, or different distances from the camera
    offset = abs(ctx.b.y - ctx.a.y) / ((ctx.a.h + ctx.b.h) / 2)
    return max(0.0, 1.0 - offset)


def _closeness(ctx: PairContext) -> float | None:
    if ctx.geometry is None:
        return None
    d = ctx.geometry["centre_distance_face_widths"]
    return float(min(1.0, max(0.0, (4.0 - d) / (4.0 - 1.2))))


def _side_by_side(ctx: PairContext) -> float | None:
    return None if ctx.geometry is None else float(ctx.geometry["adjacent_in_row"])


def _leaning_in(ctx: PairContext) -> float | None:
    if ctx.geometry is None:
        return None
    ra, rb = roll_angle(ctx.a), roll_angle(ctx.b)       # a is left of b
    if ra is None or rb is None:
        return None
    # Positive roll tilts towards image-right (towards b for the left person).
    lean = max(0.0, ra) + max(0.0, -rb)
    return min(1.0, lean / (2 * config.LEAN_DEGREES))


# ---------------------------------------------------------------------------
# THE FACTOR LIST. Defaults: similarity only; proximity starts at 0 (off),
# because in this project's use case where people stand is not relevant.
# Age weighs most: partners' ages are strongly correlated (r about 0.8).
# ---------------------------------------------------------------------------
FACTORS: list[Factor] = [
    Factor("age", "Similar age", "similarity", 40, _age,
           "Estimated ages: same age = full rating, 20+ years apart = zero."),
    Factor("attractiveness", "Similar attractiveness", "similarity", 30, _attractiveness,
           "Model rating on a 1-5 scale (SCUT-FBP5500); same rating = full, 1.5+ apart = zero."),
    Factor("outfit", "Similar outfit colour", "similarity", 10, _outfit,
           "Main colour of the chest area, compared by CIELAB colour distance."),
    Factor("height", "Similar height in photo", "similarity", 5, _height,
           "Whose head is higher; same photo and similar camera distance only."),
    Factor("closeness", "Standing close", "proximity", 0, _closeness,
           "Distance between faces: full at 1.2 face-widths or closer, zero at 4."),
    Factor("side_by_side", "Next to each other", "proximity", 0, _side_by_side,
           "Neighbours in the left-to-right order, counting adults only."),
    Factor("leaning_in", "Leaning in", "proximity", 0, _leaning_in,
           "Head tilted towards the other person, from the angle between the eyes."),
]
FACTOR_KEYS = [f.key for f in FACTORS]
DEFAULT_WEIGHTS = {f.key: f.default_weight for f in FACTORS}


def factor_catalogue() -> list[dict]:
    """The factor list as plain data, for the website's sliders and the report."""
    return [{"key": f.key, "name": f.name, "group": f.group,
             "default_weight": f.default_weight, "description": f.description} for f in FACTORS]


def parse_weights(raw: dict | None) -> dict[str, int]:
    """Defaults, overridden by any known key in ``raw`` (clamped to 0-MAX_WEIGHT)."""
    weights = dict(DEFAULT_WEIGHTS)
    for key, value in (raw or {}).items():
        if key in weights:
            try:
                weights[key] = int(min(MAX_WEIGHT, max(0, round(float(value)))))
            except (TypeError, ValueError):
                pass
    return weights


def is_eligible(person: dict) -> bool:
    if person.get("duplicate_of"):             # same person as someone already counted
        return False
    age = person.get("age") or {}
    return bool(age.get("adult"))


def exclusion_reason(person: dict) -> str:
    if person.get("duplicate_of"):
        return f"same person as {person['duplicate_of']}"
    age = person.get("age") or {}
    return age.get("reason") or "age could not be checked"


def same_apparent_gender(pa: dict, pb: dict) -> bool:
    """True only when the model is confident both people have the same apparent gender."""
    ga = (pa.get("age") or {}).get("gender")
    gb = (pb.get("age") or {}).get("gender")
    if not config.GENDER_RULE_ENABLED or not ga or not gb:
        return False
    confident = min(ga["confidence"], gb["confidence"]) >= config.GENDER_CONFIDENT
    return confident and ga["apparent"] == gb["apparent"]


def score_pair(a: Face, b: Face, pa: dict, pb: dict, geometry: dict | None,
               weights: dict[str, int] | None = None) -> dict:
    """Rate a pair on every factor and combine the ratings into a 0-100 score.

    Factors that can't be measured, or whose weight is 0, are left out and the
    remaining weights are rescaled, so the points always add up to the score.
    """
    weights = weights or DEFAULT_WEIGHTS
    ctx = PairContext(a, b, pa, pb, geometry)
    ratings = {f.key: f.rate(ctx) for f in FACTORS}
    used = {k: v for k, v in ratings.items() if v is not None and weights.get(k, 0) > 0}
    total_w = sum(weights[k] for k in used)
    points = {k: (100.0 * weights[k] * v / total_w if total_w else 0.0) for k, v in used.items()}
    score = sum(points.values())
    breakdown = []
    for f in sorted(FACTORS, key=lambda f: -points.get(f.key, -1)):
        v = ratings[f.key]
        entry = {"feature": f.key, "name": f.name, "group": f.group,
                 "rating": None if v is None else round(v, 2), "weight": weights.get(f.key, 0)}
        if f.key in used:
            entry.update(points=round(points[f.key], 1), max_points=round(100.0 * weights[f.key] / total_w, 1))
        else:
            entry.update(points=None, max_points=None,
                         status="off" if weights.get(f.key, 0) == 0 else "not measurable")
        breakdown.append(entry)
    reasons = [f"{b['name'].lower()} (+{round(b['points'])})" for b in breakdown if b["points"]][:3]
    return {
        "a": a.label, "b": b.label,
        "same_photo": geometry is not None,
        "blocked": "same apparent gender" if same_apparent_gender(pa, pb) else None,
        "match_score": round(score),
        "components": {k: (None if v is None else round(v, 2)) for k, v in ratings.items()},
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
        elif people[label].get("duplicate_of"):
            keep = people[label]["duplicate_of"]
            out.append({"label": label, "status": "duplicate",
                        "detail": f"Same person as {keep}, so counted once"})
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

    duplicates = sum(1 for p in people.values() if p.get("duplicate_of"))
    distinct = n_faces - duplicates
    excluded_real = len(excluded) - duplicates        # excluded for age, not as repeats
    eligible = len(people) - len(excluded)
    unreadable = sum(1 for e in excluded if e["reason"] in ("face too small to check age", "face too blurry to check age"))
    if duplicates:
        result["same_person_count"] = duplicates
    if distinct < 2:
        result["message"] = ("At least two people are needed to predict a pair."
                             + (" The photos show the same person." if duplicates else ""))
    elif eligible < 2 and unreadable and unreadable == excluded_real:
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
