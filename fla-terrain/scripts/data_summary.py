"""Summary of a generated dataset (T5 check).

  $PY scripts/data_summary.py --data DIR [--workers 32]
Prints: validator result; class x mode x source balance (episodes, hours, falls, label -1 fraction, support
fractions, partial frames per episode); hours per class (terrain label, frame-level); distinct split_keys per split;
fall check (every frame after fall_time labelled -1, fell flag set); command-sampler check (schedule kinds per
class for the velocity policies); strip episodes that never touch the new surface; edge episodes without partial
support.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fla_terrain.schema import TERRAIN_LABELS, validate_episode  # noqa: E402


def one(p):
    errs = validate_episode(p, raise_on_error=False)
    meta = json.load(open(p[:-4] + ".json"))
    z = np.load(p)
    t = z["t"]
    lab = z["label_terrain"]
    sup = np.stack([z["label_support_L"], z["label_support_R"]])
    tp = meta["terrain_params"]
    ft = meta.get("fall_time")
    fall_ok = True
    if ft is not None:
        fall_ok = bool(meta.get("fell")) and bool(np.all(lab[t > ft] == -1))
    td = meta.get("touchdowns", [])
    new_touch = any(1 in r for _, _, r in td) if tp.get("class", "").startswith("strip") else None
    return {
        "bad": bool(errs), "cls": tp.get("class", "?") + (f"-{tp['variant']}" if tp.get("variant") else ""),
        "mode": meta["mode"], "source": tp.get("source", meta["motion_source"]), "split": tp.get("range_split", ""),
        "split_key": meta["split_key"], "fell": bool(meta.get("fell")), "fall_ok": fall_ok, "T": len(t),
        "unk": int((lab == -1).sum()), "air": float((sup == 0).sum() / 2), "part": float((sup == 1).sum() / 2),
        "full": float((sup == 2).sum() / 2), "labels": np.bincount(lab.astype(int) + 1, minlength=7),
        "cmd_kind": (tp.get("cmd_schedule") or {}).get("kind"), "new_touch": new_touch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--json", default=None, help="also write the tables as JSON (for render_tables.py)")
    args = ap.parse_args()
    paths = sorted(glob.glob(f"{args.data}/**/episode_*.npz", recursive=True))
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        R = pool.map(one, paths, chunksize=8)
    bad = sum(r["bad"] for r in R)
    print(f"validator: {len(R)} episodes, {bad} bad -> {'PASS' if not bad else 'FAIL'}")
    agg = defaultdict(lambda: defaultdict(float))
    for r in R:
        a = agg[(r["cls"], r["mode"], r["source"])]
        for k in ("T", "unk", "air", "part", "full"):
            a[k] += r[k]
        a["eps"] += 1
        a["falls"] += r["fell"]
    print(f"\n{'class':22s} {'mode':7s} {'source':14s} {'eps':>4s} {'falls':>5s} {'hours':>6s} {'lab-1':>6s} "
          f"{'air':>5s} {'part':>5s} {'full':>5s} {'part/ep':>7s}")
    for k in sorted(agg):
        a = agg[k]
        f = a["T"]
        print(f"{k[0]:22s} {k[1]:7s} {k[2]:14s} {int(a['eps']):4d} {int(a['falls']):5d} {f/50/3600:6.3f} "
              f"{a['unk']/f:6.3f} {a['air']/f:5.2f} {a['part']/f:5.2f} {a['full']/f:5.2f} {a['part']/a['eps']:7.1f}")
    # hours per terrain label (frames), per mode
    print("\nhours per terrain label (frames) by mode:")
    modes = sorted({r["mode"] for r in R})
    lab = {m: np.sum([r["labels"] for r in R if r["mode"] == m], 0) for m in modes}
    for i in range(7):
        name = TERRAIN_LABELS[i - 1]
        print(f"  {name:10s} " + "  ".join(f"{m} {lab[m][i]/50/3600:6.3f}" for m in modes))
    # split keys
    sk = defaultdict(set)
    for r in R:
        sk[r["split"]].add(r["split_key"])
    print("\ndistinct split_keys per split: " + ", ".join(f"{s} {len(v)}" for s, v in sorted(sk.items())))
    inter = [(a, b) for a in sk for b in sk if a < b and sk[a] & sk[b]]
    print(f"split_key in two splits: {inter or 'none'} -> {'PASS' if not inter else 'FAIL'}")
    # falls
    nf = sum(r["fell"] for r in R)
    fok = all(r["fall_ok"] for r in R)
    print(f"falls: {nf} episodes ({100*nf/len(R):.1f} %); frames after fall labelled -1 and flagged: "
          f"{'PASS' if fok else 'FAIL'}")
    # command sampler balance
    ck = defaultdict(Counter)
    for r in R:
        if r["cmd_kind"]:
            ck[r["cls"]][r["cmd_kind"]] += 1
    kinds = sorted({k for c in ck.values() for k in c})
    print("\ncommand schedule kinds per class (velocity policies, same sampler everywhere):")
    for c in sorted(ck):
        tot = sum(ck[c].values())
        print(f"  {c:22s} " + " ".join(f"{k} {ck[c][k]/tot:.2f}" for k in kinds) + f"  (n={tot})")
    # strips / edge
    st = [r for r in R if r["new_touch"] is not None]
    if st:
        miss = [r for r in st if not r["new_touch"]]
        by = Counter(r["source"] for r in miss)
        print(f"\nstrip episodes never touching the new surface: {len(miss)}/{len(st)} {dict(by)}")
    ed = [r for r in R if r["cls"].startswith("edge")]
    if ed:
        nop = Counter((r["cls"], r["mode"]) for r in ed if r["part"] == 0)
        print(f"edge episodes without partial support: {sum(nop.values())}/{len(ed)} {dict(nop)}")
    if args.json:
        st_miss = [r for r in st if not r["new_touch"]] if st else []
        json.dump({"episodes": len(R), "bad": bad, "falls": nf, "fall_labels_ok": fok,
                   "split_keys": {s: len(v) for s, v in sk.items()},
                   "cells": [{"class": k[0], "mode": k[1], "source": k[2], "episodes": int(a["eps"]),
                              "falls": int(a["falls"]), "hours": a["T"] / 50 / 3600} for k, a in sorted(agg.items())],
                   "strip_never_touched": len(st_miss), "strip_total": len(st),
                   "edge_without_partial": int(sum(1 for r in R if r["cls"].startswith("edge") and r["part"] == 0)),
                   "edge_total": len(ed)}, open(args.json, "w"), indent=1)
    return 0 if not bad and not inter and fok else 1


if __name__ == "__main__":
    sys.exit(main())
