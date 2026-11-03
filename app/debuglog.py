"""Background log of every analysis: features and scores, never images.

Each analysis appends one JSON line to logs/analysis_log.jsonl. When a prediction
looks wrong, the log shows which feature caused it (for example, an age read as
14 instead of 40). Read it with:  .\\.venv\\Scripts\\python.exe scripts\\show_log.py

Stored: image size, face boxes and detection confidence, each person's age
details, smile score and outfit colour, every pair's score and components, and
the final prediction. Not stored: the photo, face crops or the filename.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid

from app import config

log = logging.getLogger(__name__)
_lock = threading.Lock()


def build_entry(result: dict) -> dict:
    people = []
    for f in result["faces"]:
        feats = f["features"]
        outfit = feats.get("outfit") or {}
        smile = feats.get("smile") or {}
        people.append({
            "label": f["label"],
            "box": f["box"],
            "detection_confidence": f["confidence"],
            "quality_flags": f["quality_flags"],
            "age": feats.get("age"),
            "smile_score": smile.get("score"),   # logged for reference; not used in the score
            "outfit": {k: outfit.get(k) for k in ("visible", "name", "hex", "lab")},
        })
    return {
        "id": uuid.uuid4().hex[:8],
        "time": result["generated_at"],
        "app_version": result["app_version"],
        "image": {k: result["image"][k] for k in ("width", "height", "format")},
        "face_count": result["face_count"],
        "people": people,
        "pair_scores": [
            {"a": p["a"], "b": p["b"], "match_score": p["match_score"], "components": p["components"],
             "blocked": p.get("blocked")}
            for p in result["pair_scores"]
        ],
        "prediction": {k: result["prediction"].get(k)
                       for k in ("pairs", "excluded_people", "people_status", "message")},
    }


def record(result: dict) -> str | None:
    """Append one entry; returns its id. Never raises: logging must not break analysis."""
    if not config.DEBUG_LOG_ENABLED:
        return None
    try:
        entry = build_entry(result)
        path = config.DEBUG_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write debug log: %s", type(exc).__name__)
        return None
