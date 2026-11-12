"""Download the face, age and expression models and verify their checksums.

The models already ship in models/. Use this only if one is missing or damaged:
    .\\.venv\\Scripts\\python.exe scripts\\download_model.py [--force]
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

MODELS = [
    ("Face detector (YuNet)", config.YUNET_MODEL_PATH, config.YUNET_MODEL_SHA256, config.YUNET_MODEL_URL),
    ("Age bands", config.AGE_MODEL_PATH, config.AGE_MODEL_SHA256, config.AGE_MODEL_URL),
    ("Age regressor", config.AGE_REGRESSOR_PATH, config.AGE_REGRESSOR_SHA256, config.AGE_REGRESSOR_URL),
    ("Expression (FER+)", config.EXPRESSION_MODEL_PATH, config.EXPRESSION_MODEL_SHA256, config.EXPRESSION_MODEL_URL),
    ("Same person (SFace)", config.SFACE_MODEL_PATH, config.SFACE_MODEL_SHA256, config.SFACE_MODEL_URL),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, path: Path, digest: str, url: str, force: bool) -> bool:
    if path.is_file() and not force and sha256(path) == digest:
        print(f"[OK]   {name}: already present and verified")
        return True
    print(f"[...]  {name}: downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    if hashlib.sha256(data).hexdigest() != digest:
        print(f"[FAIL] {name}: checksum mismatch, file NOT saved")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"[OK]   {name}: saved {path.name} ({len(data):,} bytes)")
    return True


def main() -> int:
    force = "--force" in sys.argv
    ok = all([fetch(*m, force) for m in MODELS])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
