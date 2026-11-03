from __future__ import annotations

import struct
import zlib

import cv2
import numpy as np
import pytest

from app import config, vision
from app.vision import (
    DetectionError,
    DetectorUnavailableError,
    FaceDetector,
    ImageValidationError,
    annotate,
    crop_face,
    decode_image,
    get_detector,
    header_dimensions,
    person_letter,
    sniff_format,
)
from tests.conftest import load


def png_with_header_size(w: int, h: int) -> bytes:
    """A PNG whose IHDR claims w x h (body is not a valid image of that size)."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return b"\x89PNG\r\n\x1a\n" + chunk + b"\x00" * 64


# ---- format detection ---------------------------------------------------
@pytest.mark.parametrize("name,fmt", [
    ("group_four.jpg", "jpeg"), ("group_four.png", "png"), ("group_four.webp", "webp"),
])
def test_sniff_and_header_dimensions_match_decoded(name, fmt):
    data = load(name)
    assert sniff_format(data) == fmt
    decoded = decode_image(data)
    assert header_dimensions(data, fmt) == (decoded.width, decoded.height)


def test_unknown_bytes_are_not_an_image_format():
    assert sniff_format(b"hello world, this is text") is None


# ---- validation errors --------------------------------------------------
def test_empty_upload_rejected():
    with pytest.raises(ImageValidationError) as e:
        decode_image(b"")
    assert e.value.code == "empty_file" and e.value.status_code == 400


def test_text_file_rejected_as_unsupported():
    with pytest.raises(ImageValidationError) as e:
        decode_image(b"this is definitely not an image" * 10)
    assert e.value.code == "unsupported_format" and e.value.status_code == 415


def test_gif_rejected_with_named_format():
    with pytest.raises(ImageValidationError) as e:
        decode_image(b"GIF89a" + b"\x00" * 100)
    assert e.value.code == "unsupported_format"
    assert "GIF" in e.value.message


def test_truncated_jpeg_reported_as_corrupted(group_jpg):
    with pytest.raises(ImageValidationError) as e:
        decode_image(group_jpg[:300])
    assert e.value.code == "corrupted_image" and e.value.status_code == 422


def test_file_size_limit(monkeypatch, group_jpg):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    with pytest.raises(ImageValidationError) as e:
        decode_image(group_jpg)
    assert e.value.code == "file_too_large" and e.value.status_code == 413


def test_pixel_limit_checked_from_header_before_decoding():
    bomb = png_with_header_size(20000, 20000)   # 400 MP claimed in a ~100-byte file
    with pytest.raises(ImageValidationError) as e:
        decode_image(bomb)
    assert e.value.code == "too_many_pixels" and e.value.status_code == 413


def test_tiny_image_rejected():
    ok, buf = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))
    with pytest.raises(ImageValidationError) as e:
        decode_image(buf.tobytes())
    assert e.value.code == "image_too_small"


def test_valid_formats_decode_to_bgr():
    for name in ("group_four.jpg", "group_four.png", "group_four.webp"):
        d = decode_image(load(name))
        assert d.image.ndim == 3 and d.image.shape[2] == 3 and d.image.dtype == np.uint8


# ---- detection ----------------------------------------------------------
def test_no_faces_in_cat_photo(no_face_jpg):
    det = get_detector().detect(decode_image(no_face_jpg).image)
    assert det.faces == [] and det.total_detected == 0


def test_one_face(single_jpg):
    faces = get_detector().detect(decode_image(single_jpg).image).faces
    assert len(faces) == 1
    assert faces[0].label == "Person A"
    assert 0.6 <= faces[0].score <= 1.0
    assert len(faces[0].landmarks) == 5


@pytest.mark.parametrize("name", ["group_four.jpg", "group_four.png", "group_four.webp"])
def test_group_faces_sorted_left_to_right(name):
    d = decode_image(load(name))
    faces = get_detector().detect(d.image).faces
    assert len(faces) == 4
    assert [f.label for f in faces] == ["Person A", "Person B", "Person C", "Person D"]
    centres = [f.cx for f in faces]
    assert centres == sorted(centres)
    for f in faces:  # boxes lie inside the image
        assert 0 <= f.x and 0 <= f.y and f.x + f.w <= d.width and f.y + f.h <= d.height


def test_large_image_is_downscaled_for_detection_but_boxes_use_original_scale(group_jpg):
    img = decode_image(group_jpg).image
    big = cv2.resize(img, None, fx=2.5, fy=2.5)          # 3200 px wide > DETECTION_MAX_SIDE
    small_faces = get_detector().detect(img).faces
    big_faces = get_detector().detect(big).faces
    assert len(big_faces) == len(small_faces) == 4
    for s, b in zip(small_faces, big_faces):
        assert abs(b.cx - 2.5 * s.cx) < 0.06 * big.shape[1]


def test_face_cap_keeps_most_confident(monkeypatch):
    monkeypatch.setattr(config, "MAX_FACES", 3)
    rows = []
    for i in range(5):  # five 20x20 boxes left to right, confidence rising
        x = 10 + i * 40
        rows.append([x, 10, 20, 20] + [x + 5, 15] * 5 + [0.6 + 0.05 * i])
    det = FaceDetector._to_faces(np.array(rows, dtype=np.float32), 1.0, 400, 100)
    assert det.total_detected == 5 and det.capped
    assert len(det.faces) == 3
    assert [round(f.score, 2) for f in det.faces] == [0.7, 0.75, 0.8]
    assert [f.letter for f in det.faces] == ["A", "B", "C"]


def test_missing_model_gives_clear_error(tmp_path):
    det = FaceDetector(model_path=tmp_path / "missing.onnx")
    with pytest.raises(DetectorUnavailableError) as e:
        det.detect(np.zeros((100, 100, 3), np.uint8))
    assert "download_model.py" in e.value.message and e.value.status_code == 503


def test_detector_failure_does_not_crash(monkeypatch, group_jpg):
    det = FaceDetector()
    det.detect(decode_image(group_jpg).image)  # load the real model first

    class Broken:
        def setInputSize(self, size):
            pass

        def detect(self, img):
            raise cv2.error("simulated failure")

    det._detector = Broken()
    with pytest.raises(DetectionError):
        det.detect(decode_image(group_jpg).image)


# ---- labels, crops, annotation ------------------------------------------
@pytest.mark.parametrize("i,letters", [(0, "A"), (1, "B"), (25, "Z"), (26, "AA"), (27, "AB"), (51, "AZ"), (52, "BA"), (701, "ZZ"), (702, "AAA")])
def test_person_letters(i, letters):
    assert person_letter(i) == letters


def test_crops_are_square_and_fixed_size(group_jpg):
    d = decode_image(group_jpg)
    for f in get_detector().detect(d.image).faces:
        c = crop_face(d.image, f)
        assert c.shape == (config.CROP_SIZE, config.CROP_SIZE, 3)


def test_crop_near_edge_is_padded_not_failing():
    img = np.full((200, 200, 3), 128, np.uint8)
    face = vision.Face(0, "Person A", "A", 0, 0, 60, 60, 0.9)
    assert crop_face(img, face).shape == (config.CROP_SIZE, config.CROP_SIZE, 3)


def test_annotation_keeps_size_and_draws(group_jpg):
    d = decode_image(group_jpg)
    faces = get_detector().detect(d.image).faces
    out = annotate(d.image, faces)
    assert out.shape == d.image.shape
    assert not np.array_equal(out, d.image)
    assert np.array_equal(decode_image(group_jpg).image, d.image)  # original untouched


def test_annotation_downscales_very_large_images():
    img = np.zeros((1000, 4000, 3), np.uint8)
    out = annotate(img, [], max_side=1600)
    assert max(out.shape[:2]) == 1600


def test_annotation_enlarges_small_images_for_display():
    img = np.zeros((240, 240, 3), np.uint8)
    out = annotate(img, [])
    assert max(out.shape[:2]) == 960
