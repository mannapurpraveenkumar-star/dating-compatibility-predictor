"""Show the background analysis log as readable tables.

    .\\.venv\\Scripts\\python.exe scripts\\show_log.py            # latest analysis
    .\\.venv\\Scripts\\python.exe scripts\\show_log.py --last 5   # last 5 analyses
    .\\.venv\\Scripts\\python.exe scripts\\show_log.py --id 3fa9c2d1
    .\\.venv\\Scripts\\python.exe scripts\\show_log.py --csv people.csv   # every person, for Excel

Use it to find what went wrong: e.g. an adult excluded as "may be under 18"
because one model read them as a child, or a couple missed because their
estimated ages are far apart.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def load() -> list[dict]:
    path = config.DEBUG_LOG_PATH
    if not path.is_file():
        print(f"No log yet at {path}. Analyse a photo first.")
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fmt(v) -> str:
    return "-" if v is None else str(v)


def gender_text(age: dict) -> str:
    g = age.get("gender")
    return "-" if not g else f"{g['apparent'][0].upper()} {g['confidence']:.2f}"


def show(e: dict) -> None:
    print("=" * 96)
    img = e["image"]
    print(f"Analysis {e['id']}  {e['time']}  {img['width']}x{img['height']} {img['format']}  "
          f"{e['face_count']} faces  (app {e['app_version']})")
    print(f"Result: {e['prediction']['message']}")
    print("\nPEOPLE")
    print(f"{'Who':10}{'Face px':>9}{'Age':>5}{'Model 1':>9}{'Model 2':>9}{'Child?':>8}  "
          f"{'Adult':6}{'Gender':>10}  {'Outfit':10}Result for this person")
    status = {x["label"]: x["detail"] for x in e["prediction"].get("people_status") or []}
    for p in e["people"]:
        age = p["age"] or {}
        face_px = min(p["box"]["w"], p["box"]["h"])
        flags = status.get(p["label"]) or age.get("reason") or ""
        if p["quality_flags"]:
            flags += f"  [{', '.join(p['quality_flags'])}]"
        print(f"{p['label']:10}{face_px:>9}{fmt(age.get('years')):>5}{fmt(age.get('regressor_years')):>9}"
              f"{fmt(age.get('band_years')):>9}{fmt(age.get('child_signal')):>8}  "
              f"{'yes' if age.get('adult') else 'NO':6}{gender_text(age):>10}  "
              f"{fmt(p['outfit'].get('name')):10}{flags}")
    print("\nPAIRS (best first; components are 0-1, '-' = not measurable)")
    keys = ["age", "height", "outfit", "closeness", "side_by_side", "leaning_in"]
    excluded = {x["label"] for x in e["prediction"]["excluded_people"]}
    chosen = {frozenset(x["pair"]) for x in e["prediction"]["pairs"]}
    print(f"{'Pair':22}{'Score':>6}  " + "".join(f"{k[:9]:>10}" for k in keys) + "  Note")
    for p in e["pair_scores"][:15]:
        comp = p["components"]
        note = ("PREDICTED" if frozenset((p["a"], p["b"])) in chosen
                else "not used: person excluded" if {p["a"], p["b"]} & excluded
                else f"not used: {p['blocked']}" if p.get("blocked") else "")
        print(f"{p['a'] + ' & ' + p['b']:22}{p['match_score']:>6}  "
              + "".join(f"{fmt(comp.get(k)):>10}" for k in keys) + f"  {note}")
    if len(e["pair_scores"]) > 15:
        print(f"... {len(e['pair_scores']) - 15} more pairs")


def write_csv(entries: list[dict], out: str) -> None:
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["analysis_id", "time", "person", "face_px", "age", "model1_age", "model2_age",
                    "child_signal", "adult", "reason", "smile", "outfit", "result"])
        for e in entries:
            for p in e["people"]:
                a = p["age"] or {}
                w.writerow([e["id"], e["time"], p["label"], min(p["box"]["w"], p["box"]["h"]), a.get("years"),
                            a.get("regressor_years"), a.get("band_years"), a.get("child_signal"),
                            a.get("adult"), a.get("reason"), p["smile_score"], p["outfit"].get("name"),
                            e["prediction"]["message"]])
    print(f"Wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--last", type=int, default=1)
    ap.add_argument("--id")
    ap.add_argument("--csv")
    args = ap.parse_args()
    entries = load()
    if not entries:
        return 1
    if args.csv:
        write_csv(entries, args.csv)
        return 0
    chosen = [e for e in entries if e["id"] == args.id] if args.id else entries[-args.last:]
    if not chosen:
        print(f"No analysis with id {args.id}")
        return 1
    for e in chosen:
        show(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
