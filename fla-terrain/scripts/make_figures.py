"""Slide figures from an existing results directory (no training, no new numbers).

  $PY scripts/make_figures.py --results results/v1 --out /path/to/fla-terrain/figures_v1 --print-values --check-tables

  $PY scripts/make_figures.py --v2 --results results/v2/src3 --out /path/to/fla-terrain/figures_v2 --check-tables

--v2            v2 figures (fla_terrain/viz/figures_v2.py) from the committed v2 job JSONs and course.json
--print-values  print every plotted value (bar means, seed values, matrix cells)
--check-tables  compare each plotted value of figures 1-5 with <results>/tables.md at the table's precision
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.viz import figures as F  # noqa: E402

NUM = r"-?\d+\.\d+"


def _cell(c):
    """'0.802 ± 0.001 (0.800, 0.803, 0.802)' -> (mean, [seeds], ndp); '0.730' -> (0.730, [], 3)."""
    nums = re.findall(NUM, c)
    if not nums:
        return None
    ndp = len(nums[0].split(".")[1])
    return float(nums[0]), [float(x) for x in nums[2:]] if "±" in c else [], ndp


def _rows(md, header_start):
    """Rows (list of cells) of the first markdown table whose header line starts with header_start."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header_start):
            out = []
            for r in lines[i + 2:]:
                if not r.startswith("|"):
                    break
                out.append([c.strip() for c in r.strip().strip("|").split("|")])
            return out
    raise KeyError(header_start)


def table_values(md):
    """key -> (value, ndp), keys identical to the ones the figure functions emit."""
    T = {}

    def put(key, cell):
        p = _cell(cell)
        if p is None:
            return
        m, seeds, ndp = p
        T[key] = (m, ndp)
        for i, v in enumerate(seeds):
            T[f"{key} [seed {i}]"] = (v, ndp)
    for r in _rows(md, "| input set | terrain macro-F1 | accuracy"):
        put(f"terrain macro-F1 | {r[0]}", r[1])
        put(f"support-state F1 | {r[0]}", r[6])
    for r in _rows(md, "| input set | airborne | partial | full"):
        for c, name in enumerate(("airborne", "partial", "full")):
            put(f"support F1 {name} | {r[0]}", r[1 + c])
    titles = [t.replace("\n", " ") for t, _, _ in F.DELAY_PANELS]
    for r in _rows(md, "| input set | strip latency s"):
        for t, ci in zip(titles, (1, 2, 4, 5, 6)):
            put(f"{t} | {r[0]}", r[ci])
        put(f"Lifted miss rate | {r[0]}", r[7])
    for r in _rows(md, "| held out | input set | terrain F1 unseen"):
        put(f"LOFO unseen (held-out) | {r[0]} | {r[1]}", r[2])
        put(f"LOFO seen (in-distribution) | {r[0]} | {r[1]}", r[3])
        p = _cell(r[3])
        for i, v in enumerate(p[1]):
            T[f"LOFO seen | {r[0]} | {r[1]} [seed {i}]"] = (v, p[2])
    # confusion matrices (counts) -> row-normalized
    for s in F.MAIN:
        m = re.search(rf"\*\*{re.escape(s)}\*\*\n\n(\|.*\n)+", md)
        if not m:
            continue
        rows = [ln for ln in m.group(0).splitlines()[4:] if ln.startswith("|")]
        names = [c.strip() for c in m.group(0).splitlines()[2].strip("|").split("|")][1:]
        cm = np.array([[float(c) for c in ln.strip().strip("|").split("|")[1:]] for ln in rows])
        cm = cm / cm.sum(1, keepdims=True)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                T[f"{s} row-norm [{a} -> {b}]"] = (cm[i, j], 3)
    return T


def table_values_v2(md):
    """key -> (value, ndp) for the v2 tables.md, keys identical to fla_terrain/viz/figures_v2.py."""
    T = {}

    def put(key, cell):
        p = _cell(cell)
        if p is None:
            return
        m, seeds, ndp = p
        T[key] = (m, ndp)
        for i, v in enumerate(seeds):
            T[f"{key} [seed {i}]"] = (v, ndp)
    for r in _rows(md, "| input set | M1 F1 | acc | M2 F1"):
        put(f"M1 | {r[0]}", r[1])
        put(f"M2 | {r[0]}", r[3])
        put(f"M2_partial | {r[0]}", r[4])
        put(f"M5_long_miss | {r[0]}", r[10])
    for r in _rows(md, "| input set | walk_"):
        put(f"LOGO mean held-out M1 | {r[0]}", r[-2])
        put(f"LOGO in-dist mean | {r[0]}", r[-1])
    for r in _rows(md, "| detector | source |"):
        if r[1] == "pooled":
            put(f"course acc | {r[0]}", r[3])
            put(f"course macro_f1 | {r[0]}", r[4])
            put(f"course latency_median_s | {r[0]}", r[6])
    hdr = [ln for ln in md.splitlines() if ln.startswith("| input set | metric | nominal")][0]
    sets = [c.strip() for c in hdr.strip().strip("|").split("|")][2:]
    for r in _rows(md, "| input set | metric | nominal"):
        if r[1] == "M1":
            for st, c in zip(sets, r[2:]):
                put(f"sens M1 | {r[0]} | {st}", c)
    for r in _rows(md, "| input set | M1 | M2 | M3 |"):
        put(f"sens M1 | {r[0]} | trained+tested nosat", r[1])
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-values", action="store_true")
    ap.add_argument("--check-tables", action="store_true")
    ap.add_argument("--v2", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.v2:
        return main_v2(args)
    src = {"metrics": json.load(open(os.path.join(args.results, "metrics.json"))),
           "lofo": json.load(open(os.path.join(args.results, "lofo.json")))}
    T = table_values(open(os.path.join(args.results, "tables.md")).read()) if args.check_tables else {}
    n_ok = n_bad = n_missing = 0
    for name, need, fn in F.FIGURES:
        fig, vals = fn(src[need]) if need else fn()
        path = os.path.join(args.out, name)
        fig.savefig(path)
        F.plt.close(fig)
        print(f"wrote {path}")
        for key, v in vals:
            line = f"  {key}: " + (f"{v:.4f}" if np.isscalar(v) else str(tuple(round(float(x), 2) for x in v)))
            if args.check_tables and need:
                if key in T:
                    tv, ndp = T[key]
                    ok = round(float(v), ndp) == round(tv, ndp) or abs(float(v) - tv) <= 0.5 * 10 ** -ndp + 1e-9
                    n_ok += ok
                    n_bad += not ok
                    line += f"   table {tv:.{ndp}f} ({ndp} dp) {'MATCH' if ok else 'MISMATCH'}"
                else:
                    n_missing += 1
                    line += "   NOT IN TABLE"
            if args.print_values or (args.check_tables and "MISMATCH" in line):
                print(line)
    if args.check_tables:
        print(f"check vs tables.md: {n_ok} match, {n_bad} mismatch, {n_missing} plotted values not in tables.md")
        sys.exit(1 if n_bad else 0)


def main_v2(args):
    from fla_terrain.viz import figures_v2 as F2
    T = table_values_v2(open(os.path.join(args.results, "tables.md")).read()) if args.check_tables else {}
    n_ok = n_bad = n_missing = 0
    for name, fn in F2.FIGURES_V2:
        fig, vals = fn(args.results)
        path = os.path.join(args.out, name)
        fig.savefig(path)
        F2.plt.close(fig)
        print(f"wrote {path}")
        for key, v in vals:
            line = f"  {key}: {float(v):.4f}"
            if args.check_tables:
                if key in T:
                    tv, ndp = T[key]
                    ok = abs(float(v) - tv) <= 0.5 * 10 ** -ndp + 1e-9
                    n_ok += ok
                    n_bad += not ok
                    line += f"   table {tv:.{ndp}f} {'MATCH' if ok else 'MISMATCH'}"
                else:
                    n_missing += 1
                    line += "   NOT IN TABLE"
            if args.print_values or (args.check_tables and "MISMATCH" in line):
                print(line)
    if args.check_tables:
        print(f"check vs tables.md: {n_ok} match, {n_bad} mismatch, {n_missing} plotted values not in tables.md")
        sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
