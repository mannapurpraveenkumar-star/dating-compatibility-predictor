"""Report generation: Markdown and a self-contained HTML report.

Both are built only from the analysis result, so the downloaded report always
matches what the website shows: the face count and the single predicted pair.
"""

from __future__ import annotations

from html import escape


def _names(entry: dict) -> str:
    return f"{entry['pair'][0]} & {entry['pair'][1]}"


def _headline(p: dict) -> str:
    if not p["pairs"]:
        return "No pair predicted"
    return ", ".join(_names(x) for x in p["pairs"])


def _excluded(p: dict) -> str:
    return ", ".join(f"{e['label']} ({e['reason']})" for e in p["excluded_people"])


def build_markdown(result: dict) -> str:
    p = result["prediction"]
    out = [
        "# THE DATING GAME — Match Report",
        "*Beauty Is in the Eye of the Beholder*\n",
        f"Generated: {result['generated_at']} (UTC) · App version {result['app_version']}\n",
        "## Result",
        f"- Faces detected: {result['face_count']}",
        f"- Pairs compared: {result['pair_count']}",
        f"- **Predicted {'pairs' if len(p['pairs']) > 1 else 'pair'}: {_headline(p)}**",
    ]
    for x in p["pairs"]:
        out.append(f"- {_names(x)}: match score {x['match_score']} / 100 ({x['level']})")
        for b in x.get("breakdown", []):
            if b["points"] is not None:
                out.append(f"    - {b['name']}: +{b['points']:.0f} of {b['max_points']:.0f}")
    out.append(f"- {p['message']}")
    if p["excluded_people"]:
        out.append(f"- Not matched: {_excluded(p)}")
    if p.get("people_status"):
        out += ["", "## Everyone in the photo"]
        out += [f"- {s['label']}: {s['detail']}" for s in p["people_status"]]
    out += ["", "## How it works", result["method"], "", "## Limitations"]
    out += [f"- {line}" for line in result["limitations"]]
    out += ["", "## Privacy",
            "The photo was processed in memory and was not stored by the server. This report contains the "
            "photo, so share it only with the consent of the people pictured."]
    return "\n".join(out) + "\n"


_CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#2b1631;background:#f7f1f3;margin:0;padding:32px 16px;line-height:1.55}
main{max-width:860px;margin:auto;background:#fff;border:1px solid #e6d9df;border-radius:14px;padding:36px}
h1{font-size:28px;letter-spacing:.06em;margin:0}
.tag{font-style:italic;color:#5e4a62;margin:4px 0 20px}
h2{font-size:19px;border-bottom:1px solid #e6d9df;padding-bottom:6px;margin-top:32px}
.photo{max-width:100%;border-radius:8px;display:block;margin:12px 0}
.match{margin-bottom:12px;display:flex;flex-wrap:wrap;align-items:center;gap:18px;background:#fbeef2;border-radius:12px;padding:18px}
.match img{width:110px;height:110px;border-radius:10px;border:4px solid #b3204f}
.score{font-size:44px;color:#b3204f;line-height:1}
.meta,ul,p{font:15px/1.6 system-ui,sans-serif}
"""


def build_html(result: dict, annotated_data_url: str | None = None, crops: list[dict] | None = None) -> str:
    """Self-contained HTML report (images embedded as data URLs)."""
    e = escape
    p = result["prediction"]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>The Dating Game — Match Report</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>THE DATING GAME</h1>",
        "<p class='tag'>Beauty Is in the Eye of the Beholder — Match Report</p>",
        f"<p class='meta'>Generated {e(result['generated_at'])} (UTC) · version {e(result['app_version'])} · "
        f"{result['face_count']} faces detected · {result['pair_count']} pairs compared</p>",
        f"<h2>Predicted {'pairs' if len(p['pairs']) > 1 else 'pair'}: {e(_headline(p))}</h2>",
    ]
    by_label = {c["label"]: c for c in (crops or [])}
    for x in p["pairs"]:
        parts.append("<div class='match'>")
        for label in x["pair"]:
            if label in by_label:
                parts.append(f"<img src='{e(by_label[label]['image'])}' alt='{e(label)}'>")
        parts.append(f"<div><strong>{e(_names(x))}</strong><div class='score'>{x['match_score']}"
                     f"<small style='font-size:18px'>/100</small></div><p>{e(x['level'])}</p><ul>")
        for b in x.get("breakdown", []):
            if b["points"] is not None:
                parts.append(f"<li>{e(b['name'])}: +{b['points']:.0f} of {b['max_points']:.0f}</li>")
        parts.append("</ul></div></div>")
    parts.append(f"<p>{e(p['message'])}</p>")
    if p["excluded_people"]:
        parts.append(f"<p>Not matched: {e(_excluded(p))}.</p>")
    if p.get("people_status"):
        parts.append("<h2>Everyone in the photo</h2><ul>")
        parts += [f"<li>{e(s['label'])}: {e(s['detail'])}</li>" for s in p["people_status"]]
        parts.append("</ul>")
    if annotated_data_url:
        parts.append(f"<img class='photo' src='{e(annotated_data_url)}' alt='Photo with detected faces; the predicted pair is highlighted'>")
    parts.append(f"<h2>How it works</h2><p>{e(result['method'])}</p><h2>Limitations</h2><ul>")
    parts += [f"<li>{e(s)}</li>" for s in result["limitations"]]
    parts.append("</ul><h2>Privacy</h2><p>The photo was processed in memory and was not stored by the server. "
                 "This report contains the photo, so share it only with the consent of the people pictured.</p>")
    parts.append("</main></body></html>")
    return "".join(parts)
