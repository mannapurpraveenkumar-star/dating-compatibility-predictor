"""Central configuration: limits, paths and detector thresholds.

Every limit lives here so it can be changed in one place and referenced by tests.
"""

from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
MODEL_DIR = PROJECT_ROOT / "models"

# YuNet face detector (OpenCV Zoo, MIT licence). See models/README.md.
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Optional attribute models (ONNX Model Zoo). The app still runs without them;
# the matching attribute is then shown as "Not available".
AGE_MODEL_PATH = MODEL_DIR / "age_googlenet.onnx"
AGE_MODEL_SHA256 = "fa2a3228e425056aa2b080b3afd3cf607327c86616e952602ed67b5fc16ab356"
AGE_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/"
    "validated/vision/body_analysis/age_gender/models/age_googlenet.onnx"
)
EXPRESSION_MODEL_PATH = MODEL_DIR / "emotion-ferplus-8.onnx"
EXPRESSION_MODEL_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
EXPRESSION_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/"
    "validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

# Numeric age regressor (InsightFace "genderage", via github.com/yakhyo/facial-analysis).
# Only its age output is used; the gender output is ignored.
AGE_REGRESSOR_PATH = MODEL_DIR / "genderage.onnx"
AGE_REGRESSOR_SHA256 = "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"
AGE_REGRESSOR_URL = "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/genderage.onnx"

# ---- Adult check (safeguard: possible children are never matched) --------
MIN_FACE_FOR_AGE = 40          # px; smaller faces are too small to check age, so never matched
MIN_SHARPNESS_FOR_AGE = 90     # Laplacian variance at 128 px; blurrier faces (incl. enlarged thumbnails) are refused
ADULT_MIN_AGE = 21             # estimated age must be at least this (margin above 18)
CHILD_SIGNAL_LIMIT = 0.35      # band model's average probability of age 12 or under, across 4 views

# ---- Apparent gender rule -------------------------------------------------
# Predicted pairs must be one man and one woman, because in this project's photos
# the couples are parents. This rules out same-sex couples, and gender read from a
# face can be wrong, so it only applies when the model is confident for BOTH people.
GENDER_RULE_ENABLED = True
GENDER_CONFIDENT = 0.85

# ---- Attribute thresholds (documented in README) -------------------------
SMILE_THRESHOLD = 0.5          # FER+ "happiness" share needed to call a face smiling
OUTFIT_MIN_VISIBLE = 0.4       # fraction of the clothing region that must be inside the photo
LEAN_DEGREES = 7.0             # head tilt (each) that counts as fully leaning in
HEIGHT_COMPARABLE_SIZE_RATIO = 0.7  # faces must be similar in size (similar distance) to compare height

# ---- Upload limits -------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024          # 10 MB
MAX_IMAGE_PIXELS = 16_000_000                # 16 MP: keeps memory under the free plan's 512 MB
MIN_IMAGE_SIDE = 64                          # smaller images cannot hold usable faces
ALLOWED_FORMATS = ("jpeg", "png", "webp")

# ---- Detection -----------------------------------------------------------
DETECTION_MAX_SIDE = 1600       # images are downscaled to this for detection speed
DISPLAY_MAX_SIDE = 1600         # annotated image returned to the browser
SCORE_THRESHOLD = 0.6           # YuNet confidence needed to accept a face
NMS_THRESHOLD = 0.3
TOP_K = 5000
MAX_FACES = 40                  # 40 faces -> 780 pairs; more is kept out of the report
MIN_FACE_SIDE_FOR_QUALITY = 40  # px in the original image; smaller faces are flagged
CROP_MARGIN = 0.25              # extra context around each crop, as a fraction of box size
CROP_SIZE = 192                 # crops are returned as squares of this side (px)

# ---- Quality heuristics (documented in README) ---------------------------
BLUR_LAPLACIAN_VAR = 60.0       # below this, a face crop is flagged as possibly blurred
DARK_MEAN = 50.0                # mean grey level (0-255) below this is flagged dark
BRIGHT_MEAN = 210.0             # above this is flagged over-exposed
LOW_CONTRAST_STD = 25.0         # grey-level std below this is flagged low contrast
LOW_CONFIDENCE = 0.75           # detections between SCORE_THRESHOLD and this are flagged

# ---- Debug log (features and scores only; never images) -------------------
# Each analysis appends one JSON line to logs/analysis_log.jsonl so that wrong
# predictions can be traced to the feature that caused them. Set to False to stop.
DEBUG_LOG_ENABLED = os.getenv("DEBUG_LOG", "1") == "1"   # on locally, off on Render (DEBUG_LOG=0)
DEBUG_LOG_PATH = PROJECT_ROOT / "logs" / "analysis_log.jsonl"
