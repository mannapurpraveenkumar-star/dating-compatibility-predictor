"""FastAPI application: page, health check and photo analysis endpoint.

Run (from F:\\THE_DATING_GAME):
    .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
import logging
import re
import time

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import __version__, config, debuglog
from app.analysis import analyse_photos
from app.attributes import age_model, age_regressor, attractiveness_model, expression_model
from app.identity import recognizer
from app.matching import MIN_MATCH_SCORE, factor_catalogue
from app.report import build_html, build_markdown
from app.vision import (
    VisionError,
    annotate,
    colour_hex,
    crop_face,
    decode_image,
    get_detector,
    to_data_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dating_game")

app = FastAPI(
    title="The Dating Game",
    version=__version__,
    description="Detects faces in one or more group photos, rates every pair and predicts likely couples.",
)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

NO_STORE = {"Cache-Control": "no-store"}
FORM_OVERHEAD = 64 * 1024  # allowance for multipart boundaries and headers


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    detector_ready, detector_error = True, None
    try:
        get_detector().check()
    except VisionError as exc:
        detector_ready, detector_error = False, exc.message
    return JSONResponse({
        "status": "ok" if detector_ready else "degraded",
        "version": __version__,
        "opencv_version": cv2.__version__,
        "detector": "YuNet (cv2.FaceDetectorYN)",
        "detector_ready": detector_ready,
        "age_model": age_model.available and age_regressor.available,
        "expression_model": expression_model.available,
        "attractiveness_model": attractiveness_model.available,
        "same_person_model": recognizer.available,
        "detector_error": detector_error,
        "limits": {
            "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_megapixels": config.MAX_IMAGE_PIXELS // 1_000_000,
            "formats": list(config.ALLOWED_FORMATS),
            "max_photos": config.MAX_PHOTOS,
        },
    })


LABEL_RE = re.compile(r"^Person [A-Z]{1,3}$")


def parse_age_answers(adults: str | None, children: str | None) -> dict[str, str]:
    """Comma-separated labels from the form, e.g. adults="Person A,Person C"."""
    answers: dict[str, str] = {}
    for value, kind in ((adults, "adult"), (children, "child")):
        for label in (value or "").split(","):
            label = label.strip()
            if label and LABEL_RE.match(label):
                answers[label] = kind          # "child" wins if a label is in both lists
    return answers


def parse_weights_field(raw: str | None) -> dict | None:
    """The optional "weights" form field: JSON like {"age": 40, "closeness": 0}."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _process(datas: list[bytes], age_answers: dict[str, str] | None = None,
             weights: dict | None = None) -> dict:
    """CPU-bound work, run in a worker thread. Everything stays in memory."""
    photos = []
    for i, data in enumerate(datas):
        try:
            decoded = decode_image(data)
        except VisionError as exc:
            if len(datas) > 1:
                exc.message = f"Photo {i + 1}: {exc.message}"
            raise
        photos.append((decoded, get_detector().detect(decoded.image)))
    result = analyse_photos(photos, age_answers, weights)

    pairs = [p["pair"] for p in result["prediction"]["pairs"]]
    matched = {label for pair in pairs for label in pair}
    annotated = [to_data_url(annotate(decoded.image, det.faces, highlight=pairs or None), fmt="jpeg", quality=88)
                 for decoded, det in photos]
    crops = [
        {"label": f.label, "letter": f.letter, "photo": f.photo + 1, "in_pair": f.label in matched,
         "colour": ("#b3204f" if f.label in matched else "#968c96") if pairs else colour_hex(f.index),
         "image": to_data_url(crop_face(photos[f.photo][0].image, f), fmt="png")}
        for _, det in photos for f in det.faces
    ]
    result["annotated_images"] = annotated
    result["annotated_image"] = annotated[0]
    result["crops"] = crops
    result["log_id"] = debuglog.record(result)   # features and scores only, never the image
    result["report_markdown"] = build_markdown(result)
    result["report_html"] = build_html(result, annotated, crops)
    return result


@app.get("/api/factors")
def factors():
    """The pair factors and their default weights (for the website's sliders)."""
    return JSONResponse({"factors": factor_catalogue(), "min_match_score": MIN_MATCH_SCORE})


@app.post("/api/analyse")
async def analyse_photo(request: Request):
    started = time.perf_counter()
    max_total = config.MAX_UPLOAD_BYTES * config.MAX_PHOTOS

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_total + FORM_OVERHEAD:
        raise _error(413, "file_too_large",
                     f"The upload is larger than {config.MAX_PHOTOS} x {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    if "multipart/form-data" not in request.headers.get("content-type", ""):
        raise _error(400, "bad_request", "Send the photo as multipart/form-data in a field named 'photo'.")
    try:
        form = await request.form(max_files=config.MAX_PHOTOS + 1, max_fields=20)
    except Exception:  # malformed multipart body
        raise _error(400, "bad_request", "The upload could not be read. Please try again.")

    uploads = [u for u in form.getlist("photo") if not isinstance(u, str) and hasattr(u, "read")]
    adults, children, weights_raw = form.get("adults"), form.get("children"), form.get("weights")
    age_answers = parse_age_answers(adults if isinstance(adults, str) else None,
                                    children if isinstance(children, str) else None)
    weights = parse_weights_field(weights_raw if isinstance(weights_raw, str) else None)
    if not uploads:
        await form.close()
        raise _error(400, "missing_file", "No photo was uploaded. Choose a JPEG, PNG or WebP image.")
    if len(uploads) > config.MAX_PHOTOS:
        await form.close()
        raise _error(400, "too_many_photos", f"Upload at most {config.MAX_PHOTOS} photos at once.")

    try:
        datas = [await u.read(config.MAX_UPLOAD_BYTES + 1) for u in uploads]
    finally:
        await form.close()  # releases any spooled temporary file immediately

    try:
        result = await run_in_threadpool(_process, datas, age_answers, weights)
    except VisionError as exc:
        log.info("Rejected upload: %s (%d photos)", exc.code, len(datas))
        raise _error(exc.status_code, exc.code, exc.message)
    except Exception as exc:  # never crash the server on one bad photo
        log.error("Unexpected analysis failure: %s", type(exc).__name__)
        raise _error(500, "internal_error", "Something went wrong while analysing the photo. Please try another image.")
    finally:
        del datas

    log.info("Analysed %d photo(s): %d faces, %d pairs compared, %d predicted, %.0f ms",
             len(result["images"]), result["face_count"], result["pair_count"],
             len(result["prediction"]["pairs"]), (time.perf_counter() - started) * 1000)
    return JSONResponse(result, headers=NO_STORE)
