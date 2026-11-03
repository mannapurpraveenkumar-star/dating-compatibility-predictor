from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture(autouse=True)
def _debug_log_to_tmp(tmp_path, monkeypatch):
    """Tests never write to the real logs/ folder."""
    from app import config
    monkeypatch.setattr(config, "DEBUG_LOG_PATH", tmp_path / "analysis_log.jsonl")


@pytest.fixture
def no_gender_rule(monkeypatch):
    """group_four.jpg is four copies of the same woman, so the gender rule would block every pair."""
    from app import config
    monkeypatch.setattr(config, "GENDER_RULE_ENABLED", False)


@pytest.fixture
def group_jpg() -> bytes:
    return load("group_four.jpg")


@pytest.fixture
def single_jpg() -> bytes:
    return load("single_face.jpg")


@pytest.fixture
def no_face_jpg() -> bytes:
    return load("no_face.jpg")
