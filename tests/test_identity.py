"""Same person in several photos is counted once (app.identity)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, identity, main
from app.identity import find_same_people
from app.matching import exclusion_reason, is_eligible, predict
from app.vision import Face
from tests.conftest import load

client = TestClient(main.app)
needs_model = pytest.mark.skipif(not identity.recognizer.available, reason="SFace model not downloaded")


def face(label: str, photo: int, size: int = 100, index: int = 0) -> Face:
    return Face(index, label, label[-1], 0, 0, size, size, 0.9, [], photo)


def unit(*v) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def post(*names):
    files = [("photo", (f"{i}.jpg", load(n), "image/jpeg")) for i, n in enumerate(names)]
    return client.post("/api/analyse", files=files).json()


# ---- grouping rules (no model needed) ----------------------------------------
def test_same_face_in_two_photos_is_merged_and_larger_face_kept():
    faces = [face("Person A", 0, 60), face("Person B", 0, 80), face("Person C", 1, 120)]
    emb = {"Person A": unit(1, 0, 0), "Person B": unit(0, 1, 0), "Person C": unit(1, 0.05, 0)}
    out = find_same_people(faces, emb)
    assert out["duplicate_of"] == {"Person A": "Person C"}
    assert out["groups"] == [["Person C", "Person A"]]
    assert out["similarity"]["Person A"] > 0.99


def test_similar_sized_faces_keep_the_earliest_photo():
    faces = [face("Person A", 0, 100), face("Person B", 1, 105)]
    out = find_same_people(faces, {"Person A": unit(1, 0), "Person B": unit(1, 0)})
    assert out["duplicate_of"] == {"Person B": "Person A"}


def test_two_faces_in_one_photo_are_never_merged():
    faces = [face("Person A", 0), face("Person B", 0)]
    out = find_same_people(faces, {"Person A": unit(1, 0), "Person B": unit(1, 0)})
    assert out["duplicate_of"] == {}


def test_a_person_matches_at_most_one_face_per_photo():
    # C (photo 2) looks like both A and B (photo 1): only the closer one is merged.
    faces = [face("Person A", 0), face("Person B", 0), face("Person C", 1)]
    emb = {"Person A": unit(1, 0.1), "Person B": unit(1, 0.3), "Person C": unit(1, 0.12)}
    out = find_same_people(faces, emb)
    assert out["duplicate_of"] == {"Person C": "Person A"}


def test_three_photos_of_one_person_form_one_group():
    faces = [face("Person A", 0), face("Person B", 1), face("Person C", 2)]
    out = find_same_people(faces, {k: unit(1, 0) for k in ("Person A", "Person B", "Person C")})
    assert out["duplicate_of"] == {"Person B": "Person A", "Person C": "Person A"}


def test_different_people_below_threshold_stay_separate():
    faces = [face("Person A", 0), face("Person B", 1)]
    out = find_same_people(faces, {"Person A": unit(1, 0), "Person B": unit(0.3, 1)})
    assert out["duplicate_of"] == {}


def test_missing_embedding_is_not_merged():
    faces = [face("Person A", 0), face("Person B", 1)]
    assert find_same_people(faces, {"Person A": unit(1, 0), "Person B": None})["duplicate_of"] == {}


def test_repeat_is_never_matched_and_explained():
    adult = {"age": {"adult": True, "years": 35}}
    people = {"Person A": adult, "Person B": adult, "Person C": {**adult, "duplicate_of": "Person A"}}
    assert not is_eligible(people["Person C"])
    assert exclusion_reason(people["Person C"]) == "same person as Person A"
    pair = lambda a, b: {"a": a, "b": b, "match_score": 90, "components": {}, "breakdown": [], "reasons": []}
    result = predict([pair("Person A", "Person C"), pair("Person A", "Person B"), pair("Person B", "Person C")],
                     people, 3)
    assert [p["pair"] for p in result["pairs"]] == [["Person A", "Person B"]]
    status = {s["label"]: s for s in result["people_status"]}
    assert status["Person C"]["status"] == "duplicate"
    assert "Same person as Person A" in status["Person C"]["detail"]


# ---- real photos (needs the SFace model) ---------------------------------------
@needs_model
def test_same_photo_twice_is_one_person():
    body = post("single_face.jpg", "single_face.jpg")
    assert body["face_count"] == 2 and body["people_count"] == 1
    assert body["faces"][1]["features"]["duplicate_of"] == "Person A"
    assert body["prediction"]["pairs"] == []
    assert "same person" in body["prediction"]["message"]


@needs_model
def test_person_in_group_and_alone_is_counted_once(no_gender_rule):
    # group_four.jpg is four copies of the woman in single_face.jpg: exactly one
    # of them is merged with the single photo, because one photo can't hold her twice.
    body = post("group_four.jpg", "single_face.jpg")
    dups = {f["label"]: f["features"]["duplicate_of"] for f in body["faces"] if f["features"].get("duplicate_of")}
    assert body["face_count"] == 5 and body["people_count"] == 4 and len(dups) == 1
    photo = {f["label"]: f["photo"] for f in body["faces"]}
    [(dup, kept)] = dups.items()
    assert {photo[dup], photo[kept]} == {1, 2}            # merged across the two photos
    for p in body["prediction"]["pairs"]:
        assert not set(p["pair"]) & set(dups)
    assert "same person as" in body["report_markdown"]


@needs_model
def test_one_photo_is_not_checked():
    body = post("group_four.jpg")
    assert body["people_count"] == body["face_count"] == 4
    assert body["same_person"]["checked"] is False


@needs_model
def test_check_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "SAME_PERSON_CHECK_ENABLED", False)
    body = post("single_face.jpg", "single_face.jpg")
    assert body["people_count"] == 2 and not body["same_person"]["groups"]


def test_health_reports_same_person_model():
    assert "same_person_model" in client.get("/api/health").json()
