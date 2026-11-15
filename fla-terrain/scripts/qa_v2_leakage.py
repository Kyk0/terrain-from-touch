#!/usr/bin/env python
"""B17 item 3: own leakage test on data/v2, data/v2_sens/*, data/v2_course (json metadata + raw arrays).
split_key / episode_id in two splits (within each set and across sets); band names in their own split; actual mu (grip /
slip / hard_mid) and soft stiffness values: shared values and min gap val/test -> train; hard_mid and course episodes
only in test and their split_keys absent from train/val; identical raw arrays across splits (hash)."""
import glob, hashlib, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

ROOT = "/path/to/fla-terrain/data"


def one(p):
    j = json.load(open(p))
    z = np.load(p[:-5] + ".npz")
    h = hashlib.sha1(np.ascontiguousarray(z["joint_pos"][:200]).tobytes() + np.ascontiguousarray(z["imu_gyro"][:200]).tobytes()).hexdigest()
    tp = j["terrain_params"]
    bands = []

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif k.endswith("band") and isinstance(v, str):
                bands.append(v)
    walk(tp)
    return {"set": os.path.relpath(os.path.dirname(p), ROOT), "split": tp.get("range_split"), "key": j["split_key"],
            "id": j["episode_id"], "hash": h, "bands": bands, "cls": str(tp.get("class")), "mode": j["mode"],
            "regions": [(r.get("class"), r.get("mu"), r.get("stiffness")) for r in tp.get("regions", [])]}


def main():
    files = sorted(glob.glob(f"{ROOT}/v2/*.json") + glob.glob(f"{ROOT}/v2_sens/*/*.json") + glob.glob(f"{ROOT}/v2_course/*.json"))
    with Pool(40) as p:
        R = p.map(one, files, chunksize=16)
    sp = ("train", "val", "test")
    out = []
    by = defaultdict(list)
    for r in R:
        by[r["set"]].append(r)
    out.append("episodes per set: " + ", ".join(f"{s} {len(v)}" for s, v in sorted(by.items())))
    keys = {s: {r["key"] for r in R if r["split"] == s} for s in sp}
    out.append("split_key shared between splits (all sets pooled): " + ", ".join(
        f"{a}-{b} {len(keys[a] & keys[b])}" for a, b in (("train", "val"), ("train", "test"), ("val", "test"))))
    hs = defaultdict(set)
    for r in R:
        hs[r["hash"]].add(r["split"])
    out.append(f"identical raw arrays (joint_pos+gyro first 4 s) in two splits: {sum(len(v) > 1 for v in hs.values())}")
    out.append(f"band names not starting with their split: {sum(any(not b.startswith(r['split']) for b in r['bands']) for r in R)}")
    hm = [r for r in R if r["cls"] == "hard_mid"]
    co = [r for r in R if r["set"] == "v2_course"]
    out.append(f"hard_mid episodes {len(hm)}: splits {sorted({r['split'] for r in hm})}; their split_keys in train/val: "
               f"{len({r['key'] for r in hm} & (keys['train'] | keys['val']))}")
    out.append(f"course episodes {len(co)}: splits {sorted({r['split'] for r in co})}; their split_keys in train/val: "
               f"{len({r['key'] for r in co} & (keys['train'] | keys['val']))}")
    sens = [r for r in R if r["set"].startswith("v2_sens")]
    out.append(f"sensitivity episodes {len(sens)}: splits {dict((s, sum(r['split'] == s for r in sens)) for s in sp)}; "
               f"test split_keys of v2_sens shared with v2 train/val: "
               f"{len({r['key'] for r in sens if r['split'] == 'test'} & {r['key'] for r in by['v2'] if r['split'] in ('train', 'val')})}")
    vals = {s: defaultdict(list) for s in sp}
    for r in R:
        for c, mu, k in r["regions"]:
            if mu is None:
                continue
            q = "mu_slip" if c == "hard_slip" else ("mu_mid" if c == "hard_mid" else "mu_grip")
            vals[r["split"]][q].append(mu)
            if c == "soft" and k is not None:
                vals[r["split"]]["stiffness_soft"].append(k)
    for q in ("mu_grip", "mu_slip", "stiffness_soft", "mu_mid"):
        tr = np.unique(vals["train"][q])
        line = f"{q}: train n {len(tr)}" + (f" range {tr.min():.4g}-{tr.max():.4g}" if len(tr) else "")
        for s in ("val", "test"):
            v = np.unique(vals[s][q])
            if len(v) and len(tr):
                line += f"; {s} n {len(v)} range {v.min():.4g}-{v.max():.4g}, shared with train {len(np.intersect1d(v, tr))}, " \
                        f"min gap {np.min(np.abs(v[:, None] - tr[None])):.4g}"
            elif len(v):
                line += f"; {s} n {len(v)} range {v.min():.4g}-{v.max():.4g} (no train values)"
        out.append(line)
    print("[B17-3] leakage, own test\n  " + "\n  ".join(out))
    os.makedirs("results/v2/qa_b", exist_ok=True)
    open("results/v2/qa_b/leakage.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
