"""Re-render results/v2/<run>/tables.md and metrics.json from the committed job JSONs, course.json and selection
record (no models, no data). The commit printed in the tables is the one recorded in metrics.json (the commit that
produced the numbers), so a re-render in a clean checkout must give an empty git diff.
  $PY scripts/render_v2_tables.py --out results/v2/src3
"""
import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fla_terrain.models.v2report import build_tables  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/v2/src3")
a = ap.parse_args()
m = json.load(open(f"{a.out}/metrics.json"))
build_tables(a.out, m["sources"], {"commit": m["git"], "dirty": m["git_dirty"]})
print(f"re-rendered {a.out}/tables.md and metrics.json (recorded commit {m['git'][:10]})")
