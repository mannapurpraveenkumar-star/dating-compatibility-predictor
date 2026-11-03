"""Per-person visual features, read in the background for pair matching.

Each attribute is an *appearance measurement* from one photo, never a fact about
the person:

* Estimated age       – average of an InsightFace age regressor and an age-band
                        classifier; the band classifier also checks for children.
                        Faces under MIN_FACE_FOR_AGE pixels are too small to check.
* Smiling             – FER+ facial-expression model, "happiness" share. A smile is
                        an expression, not a feeling.
* Outfit colour       – dominant colour of the region below the face.

Models are optional: if a model file is missing, that attribute is reported as
not available instead of failing the whole analysis.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from app import config
from app.vision import Face, crop_face, face_region, grey_stats

log = logging.getLogger(__name__)

AGE_BANDS = ["0–2", "4–6", "8–12", "15–20", "25–32", "38–43", "48–53", "60+"]
EXPRESSIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]



# ---------------------------------------------------------------------------
# Lazy, thread-safe model loading
# ---------------------------------------------------------------------------
class _OnnxModel:
    def __init__(self, path: Path, name: str):
        self.path = Path(path)
        self.name = name
        self._net = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.path.is_file() and not self._failed

    def forward(self, blob: np.ndarray) -> np.ndarray | None:
        if not self.available:
            return None
        with self._lock:
            try:
                if self._net is None:
                    self._net = cv2.dnn.readNetFromONNX(str(self.path))
                    log.info("%s model loaded", self.name)
                self._net.setInput(blob)
                return np.asarray(self._net.forward(), dtype=np.float64).ravel()
            except cv2.error:
                log.warning("%s model failed; attribute disabled", self.name)
                self._failed = True
                return None


age_model = _OnnxModel(config.AGE_MODEL_PATH, "Age bands")
age_regressor = _OnnxModel(config.AGE_REGRESSOR_PATH, "Age regressor")
expression_model = _OnnxModel(config.EXPRESSION_MODEL_PATH, "Expression")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Age (and apparent gender, from the same model)
# ---------------------------------------------------------------------------
def _square_crop(image: np.ndarray, face: Face, size: int, scale: float) -> np.ndarray:
    side = max(face.w, face.h) * scale
    k = size / side
    m = np.float32([[k, 0, size / 2 - face.cx * k], [0, k, size / 2 - face.cy * k]])
    return cv2.warpAffine(image, m, (size, size), borderValue=0)


def regress_age(image: np.ndarray, face: Face) -> dict | None:
    """Median age over 6 views (3 crop sizes x mirror), plus the spread between views.

    The same model also gives apparent gender (outputs 0-1: female/male scores).
    """
    ages, male = [], []
    for scale in (1.3, 1.5, 1.7):
        crop = _square_crop(image, face, 96, scale)
        for view in (crop, cv2.flip(crop, 1)):
            out = age_regressor.forward(cv2.dnn.blobFromImage(view, 1.0, (96, 96), (0, 0, 0), swapRB=True))
            if out is None or out.size != 3:
                return None
            ages.append(float(out[2]) * 100)
            male.append(float(_softmax(out[:2])[1]))
    p_male = float(np.mean(male))
    return {"years": int(round(float(np.median(ages)))), "low": int(min(ages)), "high": int(round(max(ages))),
            "gender": {"apparent": "male" if p_male >= 0.5 else "female",
                       "confidence": round(max(p_male, 1 - p_male), 2)}}


BAND_MIDPOINTS = np.array([1, 5, 10, 17.5, 28.5, 40.5, 50.5, 65.0])


def band_views(image: np.ndarray, face: Face) -> dict | None:
    """Age-band model over 4 views: expected age and probability of age 12 or under."""
    expected, child = [], []
    for margin in (0.25, 0.4):
        crop = crop_face(image, face, 224, margin=margin)
        for view in (crop, cv2.flip(crop, 1)):
            out = age_model.forward(cv2.dnn.blobFromImage(view, 1.0, (224, 224), (104, 117, 123), swapRB=False))
            if out is None or out.size != len(AGE_BANDS):
                return None
            expected.append(float(out @ BAND_MIDPOINTS))   # outputs are already probabilities
            child.append(float(out[:3].sum()))            # bands 0-2, 4-6, 8-12
    return {"years": float(np.median(expected)), "child_signal": float(np.mean(child)),
            "child_views": [round(c, 2) for c in child]}


def estimate_age(image: np.ndarray, face: Face) -> dict:
    """Age from two models combined, plus an adult check that is True only when safe.

    The two models fail in different places: in testing, the regressor read young
    children as 23-26, while the band model read adults 10-20 years too young. The
    combined estimate is the average of the two; the adult check also requires the
    band model's average child probability to stay below CHILD_SIGNAL_LIMIT.
    """
    face_px = min(face.w, face.h)
    sharpness = round(grey_stats(face_region(image, face))["sharpness"])
    reg = regress_age(image, face) if age_regressor.available else None
    bands = band_views(image, face) if age_model.available else None
    if reg is None or bands is None:
        return {"years": None, "adult": False, "reason": "age could not be checked",
                "face_px": face_px, "sharpness": sharpness}
    years = int(round((reg["years"] + bands["years"]) / 2))
    detail = {"years": years, "gender": reg["gender"],
              "regressor_years": reg["years"], "regressor_range": [reg["low"], reg["high"]],
              "band_years": round(bands["years"]), "child_signal": round(bands["child_signal"], 2),
              "child_views": bands["child_views"], "face_px": face_px, "sharpness": sharpness}
    # Small or blurry faces: the estimates above are kept (they still help compare
    # two people once the user has confirmed who is an adult) but are never trusted
    # to decide who is a child. In testing, a boy in a 239 px photo read as age 23-25.
    if face_px < config.MIN_FACE_FOR_AGE:
        return {**detail, "adult": False, "reason": "face too small to check age"}
    # An enlarged thumbnail passes the size check without adding detail; this catches it.
    if sharpness < config.MIN_SHARPNESS_FOR_AGE:
        return {**detail, "adult": False, "reason": "face too blurry to check age"}
    if years < config.ADULT_MIN_AGE or bands["child_signal"] >= config.CHILD_SIGNAL_LIMIT:
        return {**detail, "adult": False, "reason": "may be under 18"}
    return {**detail, "adult": True, "reason": None}


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------
def detect_smile(image: np.ndarray, face: Face) -> dict | None:
    region = image[face.y : face.y + face.h, face.x : face.x + face.w]
    grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blob = cv2.resize(grey, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)[None, None]
    out = expression_model.forward(blob)   # raw scores; FER+ needs a softmax
    if out is None or out.size != len(EXPRESSIONS):
        return None
    p = _softmax(out)
    score = float(p[1])
    return {"smiling": score >= config.SMILE_THRESHOLD, "score": round(score, 2),
            "expression": EXPRESSIONS[int(p.argmax())]}


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR pixels (N x 3) -> true CIELAB floats (L 0-100, a/b about -128..127)."""
    px = (np.asarray(bgr, dtype=np.float32).reshape(-1, 1, 3)) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_BGR2Lab).reshape(-1, 3)


def lab_to_bgr(lab) -> tuple[int, int, int]:
    px = np.asarray(lab, dtype=np.float32).reshape(1, 1, 3)
    bgr = cv2.cvtColor(px, cv2.COLOR_Lab2BGR).reshape(3)
    return tuple(int(round(v)) for v in np.clip(bgr * 255.0, 0, 255))


def bgr_hex(bgr) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def delta_e(lab1, lab2) -> float:
    """CIE76 colour difference. About 2.3 is just noticeable; 18+ is clearly different."""
    return float(np.linalg.norm(np.asarray(lab1, float) - np.asarray(lab2, float)))


def colour_name(bgr) -> str:
    """Plain-English name for a colour, from its hue, saturation and value."""
    px = np.uint8([[list(bgr)]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV_FULL)[0, 0].astype(float)
    hue, sat, val = h * 360 / 256, s / 255, v / 255
    if val < 0.2:
        return "black"
    if sat < 0.15:
        return "white" if val > 0.85 else "grey"
    if (hue < 15 or hue >= 345):
        return "pink" if sat < 0.45 and val > 0.7 else ("maroon" if val < 0.45 else "red")
    if hue < 45:
        if sat < 0.35 and val > 0.6:
            return "beige"
        return "brown" if val < 0.6 else "orange"
    if hue < 70:
        return "olive" if val < 0.5 else ("beige" if sat < 0.35 else "yellow")
    if hue < 170:
        return "green"
    if hue < 200:
        return "teal"
    if hue < 255:
        return "navy" if val < 0.45 else "blue"
    if hue < 290:
        return "purple"
    return "pink"


def dominant_colour(pixels_bgr: np.ndarray, k: int = 3) -> np.ndarray:
    """Lab centre of the largest k-means cluster (robust to patterns and shadows)."""
    lab = bgr_to_lab(pixels_bgr)
    if len(lab) > 4000:
        lab = lab[np.random.default_rng(0).choice(len(lab), 4000, replace=False)]
    k = min(k, len(lab))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centres = cv2.kmeans(lab.astype(np.float32), k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    return centres[int(counts.argmax())]


# ---------------------------------------------------------------------------
# Outfit
# ---------------------------------------------------------------------------
def outfit_region(face: Face, width: int, height: int) -> tuple[int, int, int, int, float]:
    """Box below the chin where the upper body usually is, plus how much of it is in frame."""
    x0 = face.cx - 0.7 * face.w
    x1 = face.cx + 0.7 * face.w
    y0 = face.y + 1.3 * face.h
    y1 = face.y + 2.1 * face.h
    wanted = (x1 - x0) * (y1 - y0)
    cx0, cy0 = max(0, int(x0)), max(0, int(y0))
    cx1, cy1 = min(width, int(x1)), min(height, int(y1))
    got = max(0, cx1 - cx0) * max(0, cy1 - cy0)
    return cx0, cy0, cx1, cy1, (got / wanted if wanted else 0.0)


def measure_outfit(image: np.ndarray, face: Face) -> dict:
    h, w = image.shape[:2]
    x0, y0, x1, y1, visible = outfit_region(face, w, h)
    if visible < config.OUTFIT_MIN_VISIBLE or (x1 - x0) * (y1 - y0) < 150:
        return {"visible": False}
    centre = dominant_colour(image[y0:y1, x0:x1].reshape(-1, 3))
    bgr = lab_to_bgr(centre)
    return {"visible": True, "name": colour_name(bgr), "hex": bgr_hex(bgr),
            "lab": [round(float(c), 1) for c in centre], "box": [x0, y0, x1 - x0, y1 - y0]}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def describe_person(image: np.ndarray, face: Face) -> dict:
    return {
        "age": estimate_age(image, face),
        "smile": detect_smile(image, face) if expression_model.available else None,
        "outfit": measure_outfit(image, face),
    }
