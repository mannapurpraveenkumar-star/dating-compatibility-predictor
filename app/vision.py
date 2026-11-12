"""Image validation, face detection, annotation and cropping.

Detector: YuNet (``cv2.FaceDetectorYN``), a small pretrained CNN face detector
from the OpenCV Model Zoo. It is part of OpenCV's core ``objdetect`` module
from OpenCV 4.8 onwards, so it does not depend on Haar cascade XML files or on
``cv2.CascadeClassifier`` (which OpenCV 5 moved into the contrib modules).

Everything happens in memory: uploads are decoded from bytes and never written
to disk, so there are no temporary files to clean up.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class VisionError(Exception):
    """Base error with an HTTP status and a stable machine-readable code."""

    status_code = 400
    code = "vision_error"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class ImageValidationError(VisionError):
    status_code = 422
    code = "invalid_image"


class DetectorUnavailableError(VisionError):
    status_code = 503
    code = "detector_unavailable"


class DetectionError(VisionError):
    status_code = 500
    code = "detection_failed"


# ---------------------------------------------------------------------------
# Format sniffing and header dimension parsing
# ---------------------------------------------------------------------------
def sniff_format(data: bytes) -> str | None:
    """Identify the format from magic bytes (never trust the filename)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def describe_unsupported(data: bytes) -> str:
    """Give a friendlier name for common formats we do not accept."""
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if data[:2] == b"BM":
        return "BMP"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftypavif"):
        return "HEIC/AVIF"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if data[:5] == b"%PDF-":
        return "PDF"
    return "an unrecognised format"


def _png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i, n = 2, len(data)
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:            # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return None
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in sof:
            if i + 9 > n:
                return None
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return w, h
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return w, h
    if chunk == b"VP8 ":
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if chunk == b"VP8L":
        b = data[21:25]
        bits = int.from_bytes(b, "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def header_dimensions(data: bytes, fmt: str) -> tuple[int, int] | None:
    """Read width/height from the file header *before* decoding.

    This blocks "decompression bombs" (tiny files that decode to enormous
    images) without spending memory on a full decode.
    """
    try:
        return {"png": _png_size, "jpeg": _jpeg_size, "webp": _webp_size}[fmt](data)
    except (struct.error, KeyError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
@dataclass
class DecodedImage:
    image: np.ndarray          # BGR, uint8, H x W x 3
    fmt: str
    width: int
    height: int
    byte_size: int


def decode_image(data: bytes) -> DecodedImage:
    """Validate and decode upload bytes into a BGR image held in memory."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.", code="empty_file", status_code=400)

    if len(data) > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
        raise ImageValidationError(
            f"The file is larger than the {mb:.0f} MB limit.", code="file_too_large", status_code=413
        )

    fmt = sniff_format(data)
    if fmt is None:
        raise ImageValidationError(
            f"Unsupported file type ({describe_unsupported(data)}). Please upload a JPEG, PNG or WebP photo.",
            code="unsupported_format",
            status_code=415,
        )

    dims = header_dimensions(data, fmt)
    if dims is not None:
        w, h = dims
        if w * h > config.MAX_IMAGE_PIXELS:
            raise ImageValidationError(
                f"The image is {w} x {h} pixels, which exceeds the "
                f"{config.MAX_IMAGE_PIXELS / 1e6:.0f}-megapixel limit.",
                code="too_many_pixels",
                status_code=413,
            )

    buf = np.frombuffer(data, dtype=np.uint8)
    try:
        # IMREAD_COLOR applies EXIF orientation, so phone photos appear upright.
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None or image.size == 0:
        raise ImageValidationError(
            "The file could not be read as an image. It may be corrupted or incomplete.",
            code="corrupted_image",
        )

    h, w = image.shape[:2]
    if w * h > config.MAX_IMAGE_PIXELS:
        raise ImageValidationError(
            f"The image is {w} x {h} pixels, which exceeds the "
            f"{config.MAX_IMAGE_PIXELS / 1e6:.0f}-megapixel limit.",
            code="too_many_pixels",
            status_code=413,
        )
    if min(w, h) < config.MIN_IMAGE_SIDE:
        raise ImageValidationError(
            f"The image is only {w} x {h} pixels. Please use a photo at least "
            f"{config.MIN_IMAGE_SIDE} pixels on each side.",
            code="image_too_small",
        )
    # Work on a copy no larger than WORKING_MAX_SIDE: keeps memory low on small
    # servers (several photos at once) without losing useful face detail.
    s = config.WORKING_MAX_SIDE / max(w, h)
    if s < 1:
        image = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return DecodedImage(image=image, fmt=fmt, width=w, height=h, byte_size=len(data))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass
class Face:
    index: int                                  # position after left-to-right sort (0-based)
    label: str                                  # "Person A", ...
    letter: str                                 # "A", ...
    x: int
    y: int
    w: int
    h: int
    score: float                                # YuNet detection confidence (0-1)
    landmarks: list[tuple[float, float]] = field(default_factory=list)  # 5 points
    photo: int = 0                              # which uploaded photo (0-based) in multi-photo mode

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h


def person_letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, 27 -> AB ... (spreadsheet-style)."""
    if index < 0:
        raise ValueError("index must be non-negative")
    letters = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FaceDetector:
    """Thread-safe lazy wrapper around ``cv2.FaceDetectorYN``."""

    def __init__(self, model_path: Path = config.YUNET_MODEL_PATH):
        self.model_path = Path(model_path)
        self._detector = None
        self._lock = threading.Lock()

    def check(self) -> None:
        """Raise DetectorUnavailableError with a clear fix if anything is missing."""
        if not hasattr(cv2, "FaceDetectorYN"):
            raise DetectorUnavailableError(
                f"This OpenCV build ({cv2.__version__}, {cv2.__file__}) has no FaceDetectorYN. "
                "Reinstall a single OpenCV package: see README 'Troubleshooting OpenCV'."
            )
        if not self.model_path.is_file():
            raise DetectorUnavailableError(
                f"Face model not found at {self.model_path}. Run: "
                r".\.venv\Scripts\python.exe scripts\download_model.py"
            )

    def _load(self):
        self.check()
        try:
            det = cv2.FaceDetectorYN.create(
                str(self.model_path), "", (320, 320),
                config.SCORE_THRESHOLD, config.NMS_THRESHOLD, config.TOP_K,
            )
        except cv2.error as exc:
            raise DetectorUnavailableError(
                f"OpenCV could not load the face model ({exc.err if hasattr(exc, 'err') else exc}). "
                "The file may be corrupted: run scripts\\download_model.py again."
            ) from exc
        log.info("YuNet face detector loaded (OpenCV %s)", cv2.__version__)
        return det

    def detect(self, image: np.ndarray) -> "Detection":
        """Detect faces, returning them sorted left-to-right with labels."""
        h, w = image.shape[:2]
        scale = min(1.0, config.DETECTION_MAX_SIDE / max(h, w))
        work = image if scale == 1.0 else cv2.resize(
            image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
        wh, ww = work.shape[:2]

        with self._lock:  # YuNet keeps internal state (input size), so serialise calls
            if self._detector is None:
                self._detector = self._load()
            try:
                self._detector.setInputSize((ww, wh))
                _, raw = self._detector.detect(work)
            except cv2.error as exc:
                raise DetectionError("The face detector failed on this image. Please try another photo.") from exc

        return self._to_faces(raw, scale, w, h)

    @staticmethod
    def _to_faces(raw, scale: float, w: int, h: int) -> "Detection":
        if raw is None or len(raw) == 0:
            return Detection([], 0)
        faces: list[Face] = []
        for row in np.asarray(raw, dtype=np.float64):
            x, y, bw, bh = (row[:4] / scale).tolist()
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue  # degenerate box after clipping
            pts = [(float(row[4 + 2 * k] / scale), float(row[5 + 2 * k] / scale)) for k in range(5)]
            faces.append(Face(0, "", "", x0, y0, x1 - x0, y1 - y0, float(row[14]), pts))

        total = len(faces)
        if len(faces) > config.MAX_FACES:
            faces.sort(key=lambda f: f.score, reverse=True)
            faces = faces[: config.MAX_FACES]

        # Left-to-right by box centre; ties (same column) broken top-to-bottom.
        faces.sort(key=lambda f: (round(f.cx), round(f.cy)))
        for i, f in enumerate(faces):
            f.index, f.letter = i, person_letter(i)
            f.label = f"Person {f.letter}"
        return Detection(faces, total)


@dataclass
class Detection:
    faces: list[Face]
    total_detected: int          # before the MAX_FACES cap

    @property
    def capped(self) -> bool:
        return self.total_detected > len(self.faces)


def label_across_photos(per_photo: list[list[Face]]) -> list[Face]:
    """Give faces from several photos one continuous labelling: photo 1's faces
    left to right (Person A, B, ...), then photo 2's, and so on."""
    faces: list[Face] = []
    for p, photo_faces in enumerate(per_photo):
        for f in photo_faces:
            f.photo = p
            f.index = len(faces)
            f.letter = person_letter(f.index)
            f.label = f"Person {f.letter}"
            faces.append(f)
    return faces


_default_detector: FaceDetector | None = None


def get_detector() -> FaceDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = FaceDetector()
    return _default_detector


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def grey_stats(bgr: np.ndarray) -> dict[str, float]:
    """Brightness, contrast and sharpness of an image region.

    Sharpness is the variance of the Laplacian measured after resizing to a
    fixed width (128 px), so values are comparable between large and small faces.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if grey.shape[1] != 128:
        new_h = max(1, round(grey.shape[0] * 128 / grey.shape[1]))
        grey = cv2.resize(grey, (128, new_h), interpolation=cv2.INTER_AREA)
    return {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
    }


def roll_angle(face: Face) -> float | None:
    """In-plane head tilt in degrees from the two eye landmarks (0 = level)."""
    if len(face.landmarks) < 2:
        return None
    (ax, ay), (bx, by) = sorted(face.landmarks[:2])  # sort by x: image-left eye first
    if bx - ax < 1e-6:
        return None
    return float(math.degrees(math.atan2(by - ay, bx - ax)))


def face_region(image: np.ndarray, face: Face) -> np.ndarray:
    return image[face.y : face.y + face.h, face.x : face.x + face.w]


def crop_face(image: np.ndarray, face: Face, size: int = config.CROP_SIZE, margin: float | None = None) -> np.ndarray:
    """Square crop around the face with a margin, padded if it runs off the edge."""
    margin = config.CROP_MARGIN if margin is None else margin
    side = max(face.w, face.h) * (1 + 2 * margin)
    half = side / 2
    x0, y0 = int(round(face.cx - half)), int(round(face.cy - half))
    x1, y1 = int(round(face.cx + half)), int(round(face.cy + half))
    h, w = image.shape[:2]
    pad = (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))
    region = image[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    if any(pad):
        region = cv2.copyMakeBorder(region, *pad, cv2.BORDER_CONSTANT, value=(40, 30, 35))
    return cv2.resize(region, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
# BGR colours, chosen to stay distinguishable on photos and to each other.
BOX_COLOURS = [
    (110, 64, 230),   # rose
    (199, 141, 46),   # teal-blue
    (49, 166, 245),   # amber
    (164, 96, 152),   # plum
    (90, 170, 80),    # green
    (60, 100, 200),   # coral
    (200, 120, 120),  # slate blue
    (40, 140, 180),   # ochre
]


def box_colour(index: int) -> tuple[int, int, int]:
    return BOX_COLOURS[index % len(BOX_COLOURS)]


def colour_hex(index: int) -> str:
    b, g, r = box_colour(index)
    return f"#{r:02x}{g:02x}{b:02x}"


MATCH_COLOUR = (79, 32, 179)     # BGR rose, for the predicted pair
OTHER_COLOUR = (150, 140, 150)   # BGR muted grey, for everyone else when a pair is shown


def annotate(image: np.ndarray, faces: list[Face], max_side: int = config.DISPLAY_MAX_SIDE,
             highlight: list[list[str]] | None = None) -> np.ndarray:
    """Return a (possibly downscaled) copy with labelled boxes.

    With ``highlight`` (a list of label pairs), each pair is drawn in rose and
    joined by a line; everyone else is drawn in grey.
    """
    h, w = image.shape[:2]
    # Small photos are enlarged for display only (never for analysis) so labels stay readable.
    s = min(max_side / max(h, w), max(1.0, 960 / max(h, w)))
    if s == 1.0:
        out = image.copy()
    else:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        out = cv2.resize(image, (round(w * s), round(h * s)), interpolation=interp)
    oh, ow = out.shape[:2]
    thickness = max(2, round(max(oh, ow) / 450))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, max(oh, ow) / 1500)

    matched = {label for pair in (highlight or []) for label in pair}
    for labels in highlight or []:
        pair = [f for f in faces if f.label in labels]
        if len(pair) == 2:
            left, right = sorted(pair, key=lambda f: f.cx)
            p0 = (round((left.x + left.w) * s), round(left.cy * s))   # edge to edge, not across faces
            p1 = (round(right.x * s), round(right.cy * s))
            cv2.line(out, p0, p1, MATCH_COLOUR, thickness + 1, cv2.LINE_AA)

    for f in faces:
        if highlight:
            c = MATCH_COLOUR if f.label in matched else OTHER_COLOUR
        else:
            c = box_colour(f.index)
        x0, y0 = round(f.x * s), round(f.y * s)
        x1, y1 = round((f.x + f.w) * s), round((f.y + f.h) * s)
        cv2.rectangle(out, (x0, y0), (x1, y1), c, thickness, cv2.LINE_AA)

        text = f.label
        (tw, th), base = cv2.getTextSize(text, font, font_scale, max(1, thickness - 1))
        if tw > (x1 - x0) * 1.4:      # crowded faces: just the letter, so labels don't overlap
            text = f.letter
            (tw, th), base = cv2.getTextSize(text, font, font_scale, max(1, thickness - 1))
        pad = max(3, thickness * 2)
        ly1 = y0 - thickness if y0 - th - 2 * pad - thickness >= 0 else min(oh, y1 + th + 2 * pad + thickness)
        ly0 = ly1 - th - 2 * pad
        lx0 = min(max(0, x0 - thickness // 2), max(0, ow - tw - 2 * pad))
        cv2.rectangle(out, (lx0, ly0), (lx0 + tw + 2 * pad, ly1), c, -1)
        cv2.putText(out, text, (lx0 + pad, ly1 - pad), font, font_scale, (255, 255, 255),
                    max(1, thickness - 1), cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Encoding for the browser
# ---------------------------------------------------------------------------
def to_data_url(image: np.ndarray, fmt: str = "png", quality: int = 90) -> str:
    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", image)
        mime = "image/png"
    if not ok:
        raise DetectionError("Could not encode the result image.")
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")
