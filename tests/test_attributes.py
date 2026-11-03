from __future__ import annotations

import numpy as np
import pytest

from app import attributes
from app.attributes import (
    colour_name,
    delta_e,
    describe_person,
    dominant_colour,
    measure_outfit,
)
from app.vision import Face, decode_image, get_detector
from tests.conftest import load


@pytest.mark.parametrize("bgr,name", [
    ((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "grey"),
    ((0, 0, 220), "red"), ((220, 0, 0), "blue"), ((0, 200, 0), "green"),
    ((0, 230, 230), "yellow"), ((0, 140, 255), "orange"),
])
def test_colour_names(bgr, name):
    assert colour_name(bgr) == name


def test_delta_e():
    assert delta_e([50, 0, 0], [50, 0, 0]) == 0
    assert delta_e([50, 0, 0], [53, 4, 0]) == pytest.approx(5.0)


def test_dominant_colour_ignores_minority_pixels():
    px = np.zeros((1000, 3), np.uint8)
    px[:800] = (200, 30, 30)        # 80 % blue
    px[800:] = (30, 30, 200)        # 20 % red
    lab = dominant_colour(px)
    assert colour_name(attributes.lab_to_bgr(lab)) == "blue"


def test_outfit_not_visible_when_body_out_of_frame():
    img = np.full((200, 300, 3), 120, np.uint8)
    f = Face(0, "Person A", "A", 100, 120, 60, 75, 0.9)   # face near the bottom edge
    assert measure_outfit(img, f) == {"visible": False}


def test_real_photo_features():
    d = decode_image(load("single_face.jpg"))
    f = get_detector().detect(d.image).faces[0]
    p = describe_person(d.image, f)
    assert 30 <= p["age"]["years"] <= 60          # the astronaut was about 42
    assert p["age"]["adult"] is True
    assert p["age"]["gender"]["apparent"] == "female" and p["age"]["gender"]["confidence"] >= 0.85
    assert p["smile"]["smiling"] is True
    assert p["outfit"]["visible"] is True and p["outfit"]["name"] == "black"   # black collar under the suit ring


def test_missing_models_are_optional(monkeypatch, tmp_path):
    monkeypatch.setattr(attributes, "age_model", attributes._OnnxModel(tmp_path / "x.onnx", "Age"))
    monkeypatch.setattr(attributes, "age_regressor", attributes._OnnxModel(tmp_path / "z.onnx", "Age"))
    monkeypatch.setattr(attributes, "expression_model", attributes._OnnxModel(tmp_path / "y.onnx", "Expression"))
    d = decode_image(load("single_face.jpg"))
    f = get_detector().detect(d.image).faces[0]
    p = describe_person(d.image, f)
    assert p["age"]["adult"] is False and p["age"]["reason"] == "age could not be checked"
    assert p["smile"] is None
    assert p["outfit"]["visible"] is True


def test_small_faces_are_never_marked_adult():
    d = decode_image(load("single_face.jpg"))
    import cv2
    tiny = cv2.resize(d.image, (100, 100), interpolation=cv2.INTER_AREA)   # face about 30 px wide
    faces = get_detector().detect(tiny).faces
    assert faces and min(faces[0].w, faces[0].h) < 40
    age = describe_person(tiny, faces[0])["age"]
    assert age["adult"] is False and age["reason"] == "face too small to check age"


def test_enlarged_thumbnails_are_refused():
    import cv2
    d = decode_image(load("single_face.jpg"))
    small = cv2.resize(d.image, (110, 110), interpolation=cv2.INTER_AREA)
    blown_up = cv2.resize(small, (440, 440), interpolation=cv2.INTER_CUBIC)   # big face, no real detail
    f = get_detector().detect(blown_up).faces[0]
    assert min(f.w, f.h) >= 40
    age = describe_person(blown_up, f)["age"]
    assert age["adult"] is False and age["reason"] == "face too blurry to check age"
