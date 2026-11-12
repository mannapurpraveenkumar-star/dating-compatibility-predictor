"""Report generation: Markdown and a self-contained HTML report.

Both are built only from the analysis result, so the downloaded report always
matches what the website shows: the predicted pairs, every person's ratings,
every pair's ratings and the weights used.
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


def _gender(age: dict) -> str:
    g = age.get("gender")
    return "-" if not g else f"{g['apparent']} ({g['confidence']:.0%})"


def _attract(person: dict) -> str:
    a = person.get("attractiveness")
    if not a:
        return "not installed"
    return f"{a['score']:.2f} / 5" + ("" if a.get("reliable", True) else " (small face)")


def people_rows(result: dict) -> list[list[str]]:
    """One row per person: label, photo, age, gender, attractiveness, outfit, adult."""
    rows = []
    for f in result["faces"]:
        feats = f["features"]
        age = feats.get("age") or {}
        outfit = feats.get("outfit") or {}
        adult = (f"same person as {feats['duplicate_of']}" if feats.get("duplicate_of")
                 else "yes" if age.get("adult") else f"no: {age.get('reason') or 'unknown'}")
        if age.get("confirmed_by_user"):
            adult += " (your answer)"
        rows.append([f["label"], str(f.get("photo", 1)), "-" if age.get("years") is None else str(age["years"]),
                     _gender(age), _attract(feats), outfit.get("name") or "not visible", adult])
    return rows


PEOPLE_HEADER = ["Person", "Photo", "Age (est.)", "Apparent gender", "Attractiveness", "Outfit", "Adult"]


def pair_status(result: dict, pair: dict) -> str:
    p = result["prediction"]
    chosen = {frozenset(x["pair"]) for x in p["pairs"]}
    excluded = {x["label"] for x in p["excluded_people"]}
    if frozenset((pair["a"], pair["b"])) in chosen:
        return "PREDICTED"
    dup = {f["label"]: f["features"].get("duplicate_of") for f in result["faces"]}
    if dup.get(pair["a"]) == pair["b"] or dup.get(pair["b"]) == pair["a"]:
        return "same person"
    if dup.get(pair["a"]) or dup.get(pair["b"]):
        return "repeat of a person (counted once)"
    if {pair["a"], pair["b"]} & excluded:
        return "person not matched"
    if pair.get("blocked"):
        return pair["blocked"]
    return "below cut-off" if pair["match_score"] < result.get("min_match_score", 35) else "other pair fits better"


def pair_rows(result: dict) -> tuple[list[str], list[list[str]]]:
    """Every pair with its rating (0-100) on each factor, its score and status."""
    factors = result.get("factors") or []
    header = ["Pair", "Score"] + [f["name"] for f in factors] + ["Status"]
    rows = []
    for pair in result["pair_scores"]:
        comp = pair["components"]
        cells = ["-" if comp.get(f["key"]) is None else str(round(100 * comp[f["key"]])) for f in factors]
        rows.append([f"{pair['a']} & {pair['b']}", str(pair["match_score"])] + cells + [pair_status(result, pair)])
    return header, rows


def weight_rows(result: dict) -> list[list[str]]:
    return [[f["name"], f["group"], str(result["weights"].get(f["key"], 0))] for f in result.get("factors") or []]


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


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
    if result.get("factors"):
        out += ["", "## Importance of each factor (weights used)"] + _md_table(["Factor", "Group", "Weight"], weight_rows(result))
    out += ["", "## Everyone's ratings"] + _md_table(PEOPLE_HEADER, people_rows(result))
    if result["pair_scores"]:
        header, rows = pair_rows(result)
        out += ["", "## Every pair's ratings (0-100 per factor; '-' = not measurable)"] + _md_table(header, rows)
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
table{border-collapse:collapse;font:13px/1.4 system-ui,sans-serif;margin:8px 0}
th,td{border-bottom:1px solid #e6d9df;padding:5px 8px;text-align:left}
th{background:#f7eef1}.scroll{overflow-x:auto}
"""


def _html_table(header: list[str], rows: list[list[str]]) -> str:
    e = escape
    head = "".join(f"<th>{e(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build_html(result: dict, annotated: str | list[str] | None = None, crops: list[dict] | None = None) -> str:
    """Self-contained HTML report (images embedded as data URLs)."""
    e = escape
    annotated_list = [annotated] if isinstance(annotated, str) else list(annotated or [])
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
    if result.get("factors"):
        parts.append("<h2>Importance of each factor (weights used)</h2>" + _html_table(["Factor", "Group", "Weight"], weight_rows(result)))
    parts.append("<h2>Everyone's ratings</h2>" + _html_table(PEOPLE_HEADER, people_rows(result)))
    if result["pair_scores"]:
        header, rows = pair_rows(result)
        parts.append("<h2>Every pair's ratings</h2><p>0-100 per factor; '-' = not measurable.</p>" + _html_table(header, rows))
    for i, url in enumerate(annotated_list):
        parts.append(f"<img class='photo' src='{e(url)}' alt='Photo {i + 1} with detected faces; predicted pairs highlighted'>")
    parts.append(f"<h2>How it works</h2><p>{e(result['method'])}</p><h2>Limitations</h2><ul>")
    parts += [f"<li>{e(s)}</li>" for s in result["limitations"]]
    parts.append("</ul><h2>Privacy</h2><p>The photo was processed in memory and was not stored by the server. "
                 "This report contains the photo, so share it only with the consent of the people pictured.</p>")
    parts.append("</main></body></html>")
    return "".join(parts)
