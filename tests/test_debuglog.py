from __future__ import annotations

import json

from app import config
from app.main import app
from fastapi.testclient import TestClient
from tests.conftest import load

client = TestClient(app)


def analyse(name: str) -> dict:
    return client.post("/api/analyse", files={"photo": (name, load(name), "image/jpeg")}).json()


def test_each_analysis_is_logged_without_images():
    body = analyse("group_four.jpg")
    lines = config.DEBUG_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["id"] == body["log_id"]
    assert entry["face_count"] == 4 and len(entry["people"]) == 4 and len(entry["pair_scores"]) == 6
    assert entry["people"][0]["age"]["years"] is not None
    assert "components" in entry["pair_scores"][0]
    text = lines[0]
    assert "base64" not in text and "group_four" not in text      # no image data, no filename


def test_log_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "DEBUG_LOG_ENABLED", False)
    body = analyse("group_four.jpg")
    assert body["log_id"] is None and not config.DEBUG_LOG_PATH.exists()


def test_log_failure_does_not_break_analysis(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(config, "DEBUG_LOG_PATH", blocker / "log.jsonl")   # parent is a file: write fails
    r = client.post("/api/analyse", files={"photo": ("a.jpg", load("group_four.jpg"), "image/jpeg")})
    assert r.status_code == 200 and r.json()["log_id"] is None
