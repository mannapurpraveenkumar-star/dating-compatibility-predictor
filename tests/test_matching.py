from __future__ import annotations

import pytest

from app.analysis import pair_features
from app.matching import MIN_MATCH_SCORE, is_eligible, match_level, predict, score_pair
from app.vision import Face


def face(i, x, y=100, w=80, h=100):
    letter = "ABCDEFGH"[i]
    return Face(i, f"Person {letter}", letter, x, y, w, h, 0.95,
                [(x + 25, y + 40), (x + 55, y + 40), (x + 40, y + 60), (x + 28, y + 80), (x + 52, y + 80)])


def person(age=30, smile=0.9, lab=(50.0, 10.0, 10.0), visible=True, adult=None, gender=None):
    adult = (age is not None and age >= 21) if adult is None else adult
    return {
        "age": {"years": age, "adult": adult, "reason": None if adult else "may be under 18",
                "gender": None if gender is None else {"apparent": gender, "confidence": 0.99}},
        "smile": None if smile is None else {"smiling": smile >= 0.5, "score": smile, "expression": "happiness"},
        "outfit": {"visible": visible, "lab": list(lab)} if visible else {"visible": False},
    }


def scored(fa, fb, pa, pb):
    return score_pair(fa, fb, pa, pb, pair_features(fa, fb))


def test_similar_people_score_higher_than_different_people():
    a, b = face(0, 100), face(1, 200)
    same = scored(a, b, person(), person())["match_score"]
    diff = scored(a, b, person(), person(age=68, lab=(90, -40, 60)))["match_score"]
    assert same > diff


def test_closer_pair_scores_higher_than_distant_pair():
    a, near, far = face(0, 100), face(1, 190), face(2, 700)
    p = person()
    assert scored(a, near, p, p)["match_score"] > scored(a, far, p, p)["match_score"]


def test_missing_features_do_not_zero_the_score():
    a, b = face(0, 100), face(1, 190)
    blank = person(age=None, smile=None, visible=False)
    s = scored(a, b, blank, blank)
    assert s["match_score"] > 0
    assert s["components"]["age"] is None and s["components"]["outfit"] is None


def test_score_is_bounded():
    a, b = face(0, 100), face(1, 185)
    s = scored(a, b, person(), person())["match_score"]
    assert 0 <= s <= 100


def test_minors_are_never_matched():
    assert not is_eligible(person(age=16))
    assert not is_eligible(person(age=30, adult=False))     # adult check failed for another reason
    assert not is_eligible(person(age=None))                # unknown age: never matched
    assert is_eligible(person(age=30))

    faces = [face(0, 100), face(1, 190), face(2, 280)]
    people = {"Person A": person(), "Person B": person(age=9), "Person C": person()}
    rows = [scored(faces[0], faces[1], people["Person A"], people["Person B"]),
            scored(faces[0], faces[2], people["Person A"], people["Person C"]),
            scored(faces[1], faces[2], people["Person B"], people["Person C"])]
    p = predict(rows, people, 3)
    assert [e["label"] for e in p["excluded_people"]] == ["Person B"]
    assert all("Person B" not in x["pair"] for x in p["pairs"])


def test_two_couples_give_two_pairs_without_reusing_anyone():
    rows = [{"a": "Person A", "b": "Person B", "match_score": 90, "reasons": []},
            {"a": "Person A", "b": "Person C", "match_score": 85, "reasons": []},   # A already used
            {"a": "Person C", "b": "Person D", "match_score": 70, "reasons": []},
            {"a": "Person B", "b": "Person D", "match_score": 65, "reasons": []}]
    people = {k: person() for k in ("Person A", "Person B", "Person C", "Person D")}
    p = predict(rows, people, 4)
    assert [x["pair"] for x in p["pairs"]] == [["Person A", "Person B"], ["Person C", "Person D"]]
    assert p["pair"] == ["Person A", "Person B"] and "2 likely pairs" in p["message"]


def test_small_faces_message():
    small = {"years": None, "adult": False, "reason": "face too small to check age"}
    people = {"Person A": {"age": small}, "Person B": {"age": small}}
    p = predict([{"a": "Person A", "b": "Person B", "match_score": 90, "reasons": []}], people, 2)
    assert p["pairs"] == [] and "too small" in p["message"]


def test_not_enough_adults():
    faces = [face(0, 100), face(1, 190)]
    people = {"Person A": person(), "Person B": person(age=5)}
    p = predict([scored(faces[0], faces[1], people["Person A"], people["Person B"])], people, 2)
    assert p["pair"] is None and "under 18" in p["message"]


def test_low_scores_give_no_prediction():
    rows = [{"a": "Person A", "b": "Person B", "match_score": MIN_MATCH_SCORE - 1, "reasons": []}]
    p = predict(rows, {"Person A": person(), "Person B": person()}, 2)
    assert p["pair"] is None and "No likely pair" in p["message"]


def test_prediction_picks_highest_score_and_flags_ties():
    rows = [{"a": "Person A", "b": "Person B", "match_score": 70, "reasons": []},
            {"a": "Person A", "b": "Person C", "match_score": 82, "reasons": ["x"]},
            {"a": "Person B", "b": "Person C", "match_score": 82, "reasons": []}]
    people = {k: person() for k in ("Person A", "Person B", "Person C")}
    p = predict(rows, people, 3)
    assert p["pair"] in (["Person A", "Person C"], ["Person B", "Person C"]) and p["match_score"] == 82
    assert p.get("tie") is True
    assert len(p["pairs"]) == 1                              # B can't pair: A and C are taken


def test_joint_pairing_beats_grabbing_the_best_pair_first():
    # Greedy would take B-C (75) and leave A and D unpaired.
    # Jointly, A-B + C-D (70 + 70) is the better whole-photo answer.
    rows = [{"a": "Person B", "b": "Person C", "match_score": 75, "reasons": []},
            {"a": "Person A", "b": "Person B", "match_score": 70, "reasons": []},
            {"a": "Person C", "b": "Person D", "match_score": 70, "reasons": []}]
    people = {k: person() for k in ("Person A", "Person B", "Person C", "Person D")}
    p = predict(rows, people, 4)
    assert sorted(map(tuple, (x["pair"] for x in p["pairs"]))) == [("Person A", "Person B"), ("Person C", "Person D")]


def test_joint_pairing_never_breaks_a_strong_pair_for_two_weak_ones():
    rows = [{"a": "Person A", "b": "Person B", "match_score": 90, "reasons": []},
            {"a": "Person A", "b": "Person C", "match_score": 38, "reasons": []},
            {"a": "Person B", "b": "Person D", "match_score": 38, "reasons": []}]
    people = {k: person() for k in ("Person A", "Person B", "Person C", "Person D")}
    p = predict(rows, people, 4)
    assert [x["pair"] for x in p["pairs"]] == [["Person A", "Person B"]]


@pytest.mark.parametrize("score,level", [(90, "Strong match"), (75, "Strong match"), (65, "Good match"),
                                         (55, "Possible match"), (40, "Weak match")])
def test_match_levels(score, level):
    assert match_level(score) == level


def test_breakdown_points_add_up_to_the_score():
    a, b = face(0, 100), face(1, 190)
    s = scored(a, b, person(), person(age=40))
    measured = [x for x in s["breakdown"] if x["points"] is not None]
    assert abs(sum(x["points"] for x in measured) - s["match_score"]) <= 1
    assert all(x["points"] <= x["max_points"] + 1e-9 for x in measured)
    assert round(sum(x["max_points"] for x in measured)) == 100


def test_every_person_gets_a_status():
    faces = [face(0, 100), face(1, 190), face(2, 280), face(3, 900)]
    people = {"Person A": person(), "Person B": person(), "Person C": person(age=8), "Person D": person(age=70)}
    rows = []
    for i in range(4):
        for j in range(i + 1, 4):
            fa, fb = faces[i], faces[j]
            rows.append(scored(fa, fb, people[fa.label], people[fb.label]))
    p = predict(rows, people, 4)
    status = {s["label"]: s for s in p["people_status"]}
    assert set(status) == set(people)
    assert status["Person C"]["status"] == "excluded"
    for label in ("Person A", "Person B", "Person D"):
        assert status[label]["status"] in ("paired", "unpaired")
        assert status[label]["detail"]
    assert status["Person D"]["status"] == "unpaired" and "below" in status["Person D"]["detail"]


def test_gender_rule_blocks_only_confident_same_gender_pairs(monkeypatch):
    from app import config
    a, b = face(0, 100), face(1, 190)
    assert scored(a, b, person(gender="female"), person(gender="female"))["blocked"] == "same apparent gender"
    assert scored(a, b, person(gender="female"), person(gender="male"))["blocked"] is None
    unsure = person(gender="female")
    unsure["age"]["gender"]["confidence"] = 0.7          # below GENDER_CONFIDENT: never blocks
    assert scored(a, b, unsure, person(gender="female"))["blocked"] is None
    monkeypatch.setattr(config, "GENDER_RULE_ENABLED", False)
    assert scored(a, b, person(gender="female"), person(gender="female"))["blocked"] is None


def test_blocked_pairs_are_never_predicted():
    rows = [{"a": "Person A", "b": "Person B", "match_score": 90, "reasons": [], "blocked": "same apparent gender"},
            {"a": "Person A", "b": "Person C", "match_score": 60, "reasons": []}]
    p = predict(rows, {k: person() for k in ("Person A", "Person B", "Person C")}, 3)
    assert [x["pair"] for x in p["pairs"]] == [["Person A", "Person C"]]
