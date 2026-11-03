from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config, main
from tests.conftest import load

client = TestClient(main.app)


def post(data: bytes, name: str = "photo.jpg", mime: str = "image/jpeg"):
    return client.post("/api/analyse", files={"photo": (name, data, mime)})


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["detector_ready"] is True
    assert body["age_model"] is True and body["expression_model"] is True
    assert body["limits"]["formats"] == ["jpeg", "png", "webp"]


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "THE DATING GAME" in r.text
    assert client.get("/static/script.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


@pytest.mark.parametrize("name,mime", [
    ("group_four.jpg", "image/jpeg"), ("group_four.png", "image/png"), ("group_four.webp", "image/webp"),
])
def test_analyse_group_photo(name, mime, no_gender_rule):
    r = post(load(name), name, mime)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["face_count"] == 4 and body["pair_count"] == 6
    pair = body["prediction"]["pair"]
    assert pair and len(pair) == 2
    matched = sorted(label for x in body["prediction"]["pairs"] for label in x["pair"])
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert len(body["crops"]) == 4
    assert sorted(c["label"] for c in body["crops"] if c["in_pair"]) == matched
    assert all(c["image"].startswith("data:image/png;base64,") for c in body["crops"])
    assert body["report_markdown"].startswith("# THE DATING GAME")
    assert body["report_html"].startswith("<!doctype html>")
    assert r.headers["cache-control"] == "no-store"


def test_analyse_single_face():
    body = post(load("single_face.jpg")).json()
    assert body["face_count"] == 1 and body["pair_count"] == 0
    assert body["prediction"]["pair"] is None


def test_analyse_no_faces_is_not_an_error():
    r = post(load("no_face.jpg"))
    assert r.status_code == 200
    assert r.json()["face_count"] == 0


def test_unsupported_format():
    r = post(b"GIF89a" + b"\x00" * 200, "a.gif", "image/gif")
    assert r.status_code == 415
    assert r.json()["detail"]["code"] == "unsupported_format"


def test_filename_and_mime_are_not_trusted():
    # A text file pretending to be a JPEG is still rejected.
    r = post(b"not an image at all" * 20, "fake.jpg", "image/jpeg")
    assert r.status_code == 415


def test_corrupted_image():
    r = post(load("group_four.jpg")[:400])
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "corrupted_image"


def test_empty_file():
    r = post(b"")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "empty_file"


def test_missing_photo_field():
    r = client.post("/api/analyse", files={"other": ("a.jpg", b"123", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "missing_file"


def test_not_multipart():
    r = client.post("/api/analyse", json={"photo": "x"})
    assert r.status_code == 400


def test_file_too_large(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10_000)
    r = post(load("group_four.jpg"))       # ~80 KB
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "file_too_large"


def test_unexpected_error_returns_500_without_crashing(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(main, "analyse", boom)
    r = post(load("group_four.jpg"))
    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "internal_error"
    assert client.get("/api/health").status_code == 200   # server still alive


def test_age_answers_override_the_models(no_gender_rule):
    # Mark everyone in the test photo as under 18: nobody may be paired.
    labels = ",".join(f"Person {c}" for c in "ABCD")
    r = client.post("/api/analyse", files={"photo": ("g.jpg", load("group_four.jpg"), "image/jpeg")},
                    data={"children": labels})
    body = r.json()
    assert r.status_code == 200 and body["prediction"]["pairs"] == []
    assert all(f["features"]["age"]["reason"] == "marked as under 18 by you" for f in body["faces"])


def test_age_answers_let_small_faces_be_paired(no_gender_rule):
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(load("group_four.jpg"), np.uint8), cv2.IMREAD_COLOR)
    small = cv2.resize(img, (420, 170), interpolation=cv2.INTER_AREA)       # faces about 30 px
    data = cv2.imencode(".png", small)[1].tobytes()
    first = post(data, "s.png", "image/png").json()
    assert first["prediction"]["pairs"] == [] and first["prediction"].get("needs_age_check")
    labels = ",".join(f["label"] for f in first["faces"])
    second = client.post("/api/analyse", files={"photo": ("s.png", data, "image/png")},
                         data={"adults": labels}).json()
    assert second["prediction"]["pairs"], second["prediction"]["message"]
    assert all(f["features"]["age"]["confirmed_by_user"] for f in second["faces"])


def test_bad_age_labels_are_ignored():
    from app.main import parse_age_answers
    assert parse_age_answers("Person A, nonsense,<script>", "Person B,Person A") == {
        "Person A": "child", "Person B": "child"}
