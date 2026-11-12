from __future__ import annotations

import json

import pytest

from app.analysis import (
    PAIR_FEATURE_NAMES,
    analyse,
    feature_vector,
    pair_count,
    pair_features,
    unique_pairs,
)
from app.vision import Detection, Face, decode_image, get_detector
from tests.conftest import load


def run(name: str) -> dict:
    d = decode_image(load(name))
    return analyse(d, get_detector().detect(d.image))


# ---- pair maths ---------------------------------------------------------
@pytest.mark.parametrize("n", range(0, 12))
def test_pair_count_formula(n):
    assert pair_count(n) == n * (n - 1) // 2
    assert len(unique_pairs(list(range(n)))) == pair_count(n)


def test_pair_count_rejects_negative():
    with pytest.raises(ValueError):
        pair_count(-1)


def test_no_duplicate_or_self_pairs():
    pairs = unique_pairs(["A", "B", "C", "D", "E"])
    as_sets = [frozenset(p) for p in pairs]
    assert len(as_sets) == len(set(as_sets)) == 10
    assert all(len(s) == 2 for s in as_sets)
    assert ("B", "A") not in pairs


def test_pair_features_are_symmetric():
    a = Face(0, "Person A", "A", 10, 10, 50, 60, 0.9)
    b = Face(1, "Person B", "B", 200, 40, 40, 50, 0.9)
    assert pair_features(a, b) == pair_features(b, a)
    assert len(feature_vector(pair_features(a, b))) == len(PAIR_FEATURE_NAMES)


# ---- full analysis ------------------------------------------------------
def test_group_compares_every_pair_and_predicts_one(no_gender_rule):
    r = run("group_four.jpg")
    assert r["face_count"] == 4
    assert r["pair_count"] == 6 == len(r["pair_scores"])
    ids = {frozenset((p["a"], p["b"])) for p in r["pair_scores"]}
    assert len(ids) == 6                                    # every pair once
    p = r["prediction"]
    assert p["pair"] is not None and len(p["pair"]) == 2
    assert p["pair"] == p["pairs"][0]["pair"]
    people = [label for x in p["pairs"] for label in x["pair"]]
    assert len(people) == len(set(people))                  # nobody in two pairs
    scores = {frozenset((x["a"], x["b"])): x["match_score"] for x in r["pair_scores"]}
    for x in p["pairs"]:
        assert scores[frozenset(x["pair"])] == x["match_score"]
    json.dumps(r)


def test_scores_sorted_and_in_range():
    scores = [p["match_score"] for p in run("group_four.jpg")["pair_scores"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores)


def test_skin_colour_never_measured():
    r = run("group_four.jpg")
    text = json.dumps(r).lower()
    assert "skin" not in text.replace("skin colour is never used", "")


def test_people_features_are_read():
    for f in run("group_four.jpg")["faces"]:
        feats = f["features"]
        assert set(feats) == {"age", "smile", "outfit", "attractiveness"}
        assert feats["smile"]["smiling"] is True             # the test portrait is smiling


def test_one_face_has_no_prediction():
    r = run("single_face.jpg")
    assert r["face_count"] == 1 and r["pair_count"] == 0
    assert r["prediction"]["pair"] is None
    assert "two people" in r["prediction"]["message"]


def test_no_faces():
    r = run("no_face.jpg")
    assert r["face_count"] == 0 and r["prediction"]["pair"] is None


def test_capped_detection_is_reported():
    d = decode_image(load("group_four.jpg"))
    det = get_detector().detect(d.image)
    r = analyse(d, Detection(det.faces, total_detected=50))
    assert r["detector"]["total_detected"] == 50


def test_limitations_present():
    r = run("group_four.jpg")
    assert any("not prove" in s for s in r["limitations"])
    assert any("under 18" in s for s in r["limitations"])


def test_same_gender_pairs_are_blocked():
    r = run("group_four.jpg")          # four copies of the same woman
    assert all(p["blocked"] == "same apparent gender" for p in r["pair_scores"])
    assert r["prediction"]["pairs"] == []


def test_smile_is_not_scored():
    r = run("group_four.jpg")
    assert "smile" not in r["pair_scores"][0]["components"]
