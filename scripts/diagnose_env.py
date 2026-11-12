"""Diagnose the Python / OpenCV environment for The Dating Game.

Run from F:\\THE_DATING_GAME:
    .\\.venv\\Scripts\\python.exe scripts\\diagnose_env.py

It explains the most common cause of
    AttributeError: module 'cv2' has no attribute 'CascadeClassifier'
and checks everything the face detector needs. It changes nothing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
OPENCV_DISTS = ["opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"]

problems: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  [FAIL] {msg}")
    problems.append(fix)


print("\n1. Python")
print(f"  executable: {sys.executable}")
print(f"  version:    {sys.version.split()[0]}")
if sys.version_info[:2] == (3, 11):
    ok("Python 3.11")
else:
    print("  [WARN] Not Python 3.11. The pinned packages were chosen for 3.11.")
if ".venv" in sys.executable:
    ok("Running inside the project's .venv")
else:
    print("  [WARN] Not running from .venv. Use .\\.venv\\Scripts\\python.exe")

print("\n2. Files that could hide the real cv2 package")
shadows = [p for p in ROOT.rglob("cv2*") if ".venv" not in p.parts and "__pycache__" not in p.parts]
if shadows:
    for p in shadows:
        bad(f"found {p}", f"Rename or delete {p}: it hides the real OpenCV module.")
else:
    ok("no cv2.py or cv2 folder in the project")

print("\n3. Installed OpenCV packages")
installed = {}
for d in OPENCV_DISTS:
    try:
        installed[d] = md.version(d)
    except md.PackageNotFoundError:
        pass
for d, v in installed.items():
    print(f"  {d}=={v}")
if len(installed) == 0:
    bad("no OpenCV package installed", "Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
elif len(installed) > 1:
    names = " ".join(installed)
    bad(f"{len(installed)} OpenCV packages installed; they overwrite each other's files",
        f"Run: .\\.venv\\Scripts\\python.exe -m pip uninstall -y {names}\n"
        "       then: .\\.venv\\Scripts\\python.exe -m pip install --force-reinstall --no-cache-dir opencv-python-headless==4.10.0.84")
else:
    ok("exactly one OpenCV package")

print("\n4. Importing cv2")
try:
    import cv2  # noqa: E402
    print(f"  cv2 file:    {cv2.__file__}")
    print(f"  cv2 version: {getattr(cv2, '__version__', 'MISSING')}")
    if not getattr(cv2, "__version__", None):
        bad("cv2 imported but has no version (broken or shadowed install)",
            "Reinstall OpenCV as shown in step 3.")
    for attr in ("FaceDetectorYN", "imdecode", "CascadeClassifier"):
        print(f"  cv2.{attr}: {'present' if hasattr(cv2, attr) else 'absent'}")
    if hasattr(cv2, "FaceDetectorYN"):
        ok("FaceDetectorYN available (this app does not need CascadeClassifier)")
    else:
        bad("cv2.FaceDetectorYN missing (needs OpenCV 4.8 or newer)",
            "Reinstall: .\\.venv\\Scripts\\python.exe -m pip install --force-reinstall opencv-python-headless==4.10.0.84")
    if not hasattr(cv2, "CascadeClassifier"):
        print("  note: CascadeClassifier is absent. In OpenCV 5.x it moved to the contrib modules,")
        print("        which is the usual cause of the AttributeError you saw. This app uses YuNet instead.")
except Exception as exc:  # noqa: BLE001
    cv2 = None
    bad(f"import cv2 failed: {type(exc).__name__}: {exc}", "Reinstall OpenCV as shown in step 3.")

print("\n5. Face model")
if MODEL.is_file():
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest == MODEL_SHA256:
        ok(f"{MODEL.name} present and checksum matches")
    else:
        bad("model checksum does not match (corrupted or wrong file)",
            "Run: .\\.venv\\Scripts\\python.exe scripts\\download_model.py --force")
else:
    bad(f"{MODEL} not found", "Run: .\\.venv\\Scripts\\python.exe scripts\\download_model.py")

print("\n5b. Age and expression models (optional; features are skipped without them)")
for fname in ("age_googlenet.onnx", "genderage.onnx", "emotion-ferplus-8.onnx", "face_recognition_sface_2021dec.onnx"):
    if (ROOT / "models" / fname).is_file():
        ok(f"{fname} present")
    else:
        print(f"  [WARN] {fname} missing. Run: .\\.venv\\Scripts\\python.exe scripts\\download_model.py")

print("\n6. Live detection test")
if cv2 is not None and hasattr(cv2, "FaceDetectorYN") and MODEL.is_file():
    sample = ROOT / "tests" / "fixtures" / "group_four.jpg"
    try:
        import numpy as np
        img = cv2.imdecode(np.fromfile(str(sample), dtype=np.uint8), cv2.IMREAD_COLOR)
        det = cv2.FaceDetectorYN.create(str(MODEL), "", (img.shape[1], img.shape[0]), 0.6, 0.3, 5000)
        _, faces = det.detect(img)
        n = 0 if faces is None else len(faces)
        if n == 4:
            ok(f"detected {n} faces in {sample.name} (expected 4)")
        else:
            bad(f"detected {n} faces in {sample.name}, expected 4", "Reinstall OpenCV and re-download the model.")
    except Exception as exc:  # noqa: BLE001
        bad(f"detection failed: {type(exc).__name__}: {exc}", "Reinstall OpenCV and re-download the model.")
else:
    print("  skipped (fix the problems above first)")

print("\n7. Web packages")
for pkg in ("fastapi", "uvicorn", "python-multipart", "numpy", "pytest", "httpx"):
    try:
        print(f"  {pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        bad(f"{pkg} not installed", "Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")

print("\n" + "=" * 60)
if problems:
    print("PROBLEMS FOUND. Suggested fixes, in order:")
    for i, fix in enumerate(dict.fromkeys(problems), 1):
        print(f"  {i}. {fix}")
    sys.exit(1)
print("Everything looks good. Start the app with:")
print("  .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload")
