"""Render <out>/tables.md from metrics.json, baseline.json, shortcut.json, lofo.json, analysis.json,
data_summary.json and (optionally) repro.json in the same directory.
  $PY scripts/render_tables.py --out results/v1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fla_terrain.models.report import render  # noqa: E402


def load(out, name):
    p = os.path.join(out, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1")
    args = ap.parse_args()
    extra = {k: load(args.out, f"{k}.json") for k in ("shortcut", "lofo", "analysis", "data_summary")}
    rp = load(args.out, "repro.json")
    if rp:
        extra["repro"] = rp["summary"]
    sn = load(args.out, "support_note.json")
    if sn:
        extra["support_note"] = sn["text"]
    from fla_terrain.models.provenance import git_state
    g = git_state()
    txt = render(load(args.out, "metrics.json"), load(args.out, "baseline.json"), extra)
    txt += f"\n\n_tables rendered by commit `{g['commit']}` (clean: {not g['dirty']}); numbers from the commits recorded above._\n"
    open(os.path.join(args.out, "tables.md"), "w").write(txt)
    print(f"wrote {args.out}/tables.md")


if __name__ == "__main__":
    main()
