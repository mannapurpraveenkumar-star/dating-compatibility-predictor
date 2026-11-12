from __future__ import annotations

from app.analysis import analyse
from app.report import build_html, build_markdown
from app.vision import decode_image, get_detector
from tests.conftest import load


def result_for(name: str) -> dict:
    d = decode_image(load(name))
    return analyse(d, get_detector().detect(d.image))


def test_markdown_report_shows_the_prediction(no_gender_rule):
    r = result_for("group_four.jpg")
    md = build_markdown(r)
    a, b = r["prediction"]["pair"]
    assert md.startswith("# THE DATING GAME")
    assert f"{a} & {b}" in md.split("**Predicted")[1].split("**")[0]
    assert f"{a} & {b}: match score {r['prediction']['match_score']} / 100" in md
    assert "Pairs compared: 6" in md
    assert "## Limitations" in md and "## Privacy" in md


def test_html_report_is_self_contained(no_gender_rule):
    r = result_for("group_four.jpg")
    a, b = r["prediction"]["pair"]
    crops = [{"label": a, "image": "data:image/png;base64,AAAA"}, {"label": b, "image": "data:image/png;base64,BBBB"},
             {"label": "Person Z", "image": "data:image/png;base64,ZZZZ"}]
    html = build_html(r, "data:image/jpeg;base64,PHOTO", crops)
    assert html.startswith("<!doctype html>")
    assert "AAAA" in html and "BBBB" in html and "ZZZZ" not in html   # only the pair's faces
    assert "PHOTO" in html
    assert "<script" not in html.lower()


def test_report_when_no_pair():
    r = result_for("no_face.jpg")
    assert "No pair predicted" in build_markdown(r)
    assert "No pair predicted" in build_html(r)


def test_html_escapes_text():
    r = result_for("single_face.jpg")
    r["prediction"]["message"] = "<script>alert(1)</script>"
    html = build_html(r)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_report_includes_ratings_and_weights(no_gender_rule):
    r = result_for("group_four.jpg")
    md = build_markdown(r)
    assert "## Everyone's ratings" in md and "## Every pair's ratings" in md
    assert "Similar attractiveness" in md and "| Standing close | proximity | 0 |" in md
    assert md.count("PREDICTED") == len(r["prediction"]["pairs"])
    html = build_html(r, ["data:image/jpeg;base64,ONE", "data:image/jpeg;base64,TWO"])
    assert "ONE" in html and "TWO" in html and "Every pair's ratings" in html
