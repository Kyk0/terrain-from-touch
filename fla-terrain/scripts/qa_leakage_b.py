#!/usr/bin/env python
"""B15: own leakage test from the episode files (json + npz), independent of fla_terrain.models.
  python scripts/qa_leakage_b.py --data data/v1
Checks: split_key / episode_id in two splits; per split the actual mu (grip vs slip regions) and soft stiffness
values: min distance from any val/test value to the nearest train value; band names vs split; duplicate episodes
across splits (hash of the raw foot-zone + joint arrays)."""
import argparse, glob, hashlib, json, os
from collections import defaultdict
from multiprocessing import Pool
import numpy as np


def one(p):
    j = json.load(open(p))
    z = np.load(p[:-5] + ".npz")
    h = hashlib.sha1(np.ascontiguousarray(z["foot_zone_L"][:200]).tobytes() + np.ascontiguousarray(z["joint_pos"][:200]).tobytes()).hexdigest()
    tp = j["terrain_params"]
    bands = []

    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif k.endswith("band"):
                bands.append(v)
    walk(tp)
    return {"split": tp.get("range_split"), "key": j["split_key"], "id": j["episode_id"], "hash": h, "bands": bands,
            "regions": [(r["class"], r["mu"], r["stiffness"]) for r in tp.get("regions", [])], "mode": j["mode"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/path/to/fla-terrain/data/v1"); a = ap.parse_args()
    with Pool(32) as p:
        R = p.map(one, sorted(glob.glob(os.path.join(a.data, "*.json"))), chunksize=16)
    sp = ("train", "val", "test")
    keys = {s: {r["key"] for r in R if r["split"] == s} for s in sp}
    ids = {s: {r["id"] for r in R if r["split"] == s} for s in sp}
    print(f"[B15 leakage] {len(R)} episodes; per split " + ", ".join(f"{s} {len(keys[s])}" for s in sp))
    print("  split_key shared: " + ", ".join(f"{x}-{y} {len(keys[x] & keys[y])}" for x, y in (("train", "val"), ("train", "test"), ("val", "test"))) +
          "; episode_id shared: " + str(sum(len(ids[x] & ids[y]) for x, y in (("train", "val"), ("train", "test"), ("val", "test")))))
    hs = defaultdict(set)
    for r in R:
        hs[r["hash"]].add(r["split"])
    print(f"  identical raw arrays across splits: {sum(len(v) > 1 for v in hs.values())} hashes "
          f"(duplicate hashes within a split: {len(R) - len(hs)})")
    badb = [r["id"] for r in R if any(not b.startswith(r["split"]) for b in r["bands"])]
    print(f"  band names not starting with their split: {len(badb)}")
    # values: grip mu (non-slip regions), slip mu, soft stiffness
    vals = {s: defaultdict(list) for s in sp}
    for r in R:
        for c, mu, k in r["regions"]:
            vals[r["split"]]["mu_slip" if c == "hard_slip" else "mu_grip"].append(mu)
            if c == "soft":
                vals[r["split"]]["stiffness_soft"].append(k)
    for q in ("mu_grip", "mu_slip", "stiffness_soft"):
        tr = np.unique(vals["train"][q])
        line = []
        for s in ("val", "test"):
            v = np.unique(vals[s][q])
            if not len(v) or not len(tr):
                continue
            gap = np.min(np.abs(v[:, None] - tr[None, :]))
            shared = len(np.intersect1d(v, tr))
            rel = gap / np.median(tr)
            line.append(f"{s}: n {len(v)}, range {v.min():.4g}-{v.max():.4g}, min gap to train {gap:.4g} ({100*rel:.1f} %), shared {shared}")
        print(f"  {q:15s} train n {len(tr)} range {tr.min():.4g}-{tr.max():.4g}; " + "; ".join(line))


if __name__ == "__main__":
    main()
