"""A8 checks and tables for the v2 datasets.

  $PY scripts/v2_dataset_report.py [--root /path/to/fla-terrain/data] [--out results/v2/dataset_report.json]
Directories: v2 (main), v2_sens/{nominal,shifted,g1,pads}, v2_course.
Prints: validator per directory; leakage (no split_key in two splits; hard_mid and courses only in test; every
friction / stiffness value inside a band of its own split; the sensitivity test episodes identical across the four
settings); hours by class x mode x source x split (main); falls by class and source; course runs per source (fell /
crossed every chunk); pad twin trajectory divergence; disk used.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.schema import validate_episode  # noqa: E402
from fla_terrain.sim.terrains import BANDS, RIGID_STIFFNESS, stiffness_from_tc  # noqa: E402


def one(p):
    errs = validate_episode(p, raise_on_error=False)
    m = json.load(open(p[:-4] + ".json"))
    tp = m["terrain_params"]
    z = np.load(p)
    return {"path": p, "bad": len(errs), "err": errs[:2], "id": m["episode_id"], "split_key": m["split_key"],
            "split": tp.get("range_split", ""), "cls": tp.get("class", "") + (f"-{tp['variant']}" if tp.get("variant") else ""),
            "mode": m["mode"], "source": tp.get("source", ""), "fell": bool(m.get("fell")), "T": len(z["t"]),
            "mus": [r["mu"] for r in tp.get("regions", [])], "ks": [r["stiffness"] for r in tp.get("regions", [])],
            "xy": tp.get("base_xy_1hz"), "end_x": tp.get("course_end_x"), "chunks": len(tp.get("chunks", []) or []),
            "boundaries": tp.get("boundaries"), "has_jt": "joint_target" in z.files,
            "jt_nan": bool(np.isnan(z["joint_target"]).any()) if "joint_target" in z.files else None,
            "ground": np.bincount(np.concatenate([z["label_ground_L"], z["label_ground_R"]]), minlength=5).tolist()}


def band_owner(v, kind):
    out = set()
    if kind == "mu":
        for name in ("mu_grip", "mu_slip", "mu_mid"):
            for s, bands in BANDS[name].items():
                for _, lo, hi in bands:
                    if lo - 1e-9 <= v <= hi + 1e-9:
                        out.add(s)
    else:
        for s, bands in BANDS["soft_tc"].items():
            for _, lo, hi in bands:
                a, b = sorted((stiffness_from_tc(hi), stiffness_from_tc(lo)))
                if a * (1 - 1e-6) <= v <= b * (1 + 1e-6):
                    out.add(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/path/to/fla-terrain/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dirs = {"main": "v2", "sens_nominal": "v2_sens/nominal", "sens_shifted": "v2_sens/shifted", "sens_g1": "v2_sens/g1",
            "sens_pads": "v2_sens/pads", "course": "v2_course"}
    from multiprocessing import Pool
    R = {}
    with Pool(32) as pool:
        for k, d in dirs.items():
            R[k] = pool.map(one, sorted(glob.glob(os.path.join(args.root, d, "episode_*.npz"))), chunksize=8)
    rep = {}
    print("validator:")
    for k, rs in R.items():
        bad = [r for r in rs if r["bad"]]
        print(f"  {k:13s} {len(rs):5d} episodes, {len(bad)} bad" + (f"  e.g. {bad[0]['err']}" if bad else ""))
        rep[f"validator_{k}"] = {"episodes": len(rs), "bad": len(bad)}
    # leakage, main
    errs = []
    keys = defaultdict(set)
    for r in R["main"]:
        keys[r["split"]].add(r["split_key"])
    shared = [(a, b) for a in keys for b in keys if a < b and keys[a] & keys[b]]
    if shared:
        errs.append(f"split_key shared {shared}")
    notest = [r["id"] for r in R["main"] + R["course"] if (r["cls"] == "hard_mid" or r["cls"] == "course")
              and r["split"] != "test"]
    if notest:
        errs.append(f"{len(notest)} hard_mid/course episodes outside test")
    badv = 0
    for r in R["main"] + R["sens_nominal"]:
        if r["cls"] == "course":
            continue
        for mu in r["mus"]:
            badv += band_owner(mu, "mu") != {r["split"]}
        for kv in r["ks"]:
            if kv != RIGID_STIFFNESS:
                badv += band_owner(kv, "k") != {r["split"]}
    if badv:
        errs.append(f"{badv} friction/stiffness values outside their split's bands")
    tests = {k: {r["id"] for r in R[k] if r["split"] == "test"} for k in ("sens_nominal", "sens_shifted", "sens_g1", "sens_pads")}
    same = len({frozenset(v) for v in tests.values()}) == 1
    if not same:
        errs.append("sensitivity test sets differ across settings")
    sens_tv = {r["split_key"] for r in R["sens_nominal"] if r["split"] in ("train", "val")}
    sens_te = tests["sens_nominal"]
    if sens_tv & sens_te:
        errs.append("sensitivity split_key in train/val and test")
    print(f"split_keys main: " + ", ".join(f"{s} {len(v)}" for s, v in sorted(keys.items())) +
          f"; sens test episodes per setting {[len(v) for v in tests.values()]} identical {same}")
    print(f"LEAKAGE {'PASS' if not errs else 'FAIL: ' + '; '.join(errs)}")
    rep["leakage"] = errs or "PASS"
    # hours table (main)
    h = Counter()
    for r in R["main"]:
        h[(r["cls"], r["mode"], r["source"], r["split"])] += r["T"] / 50 / 3600
    print(f"\nmain dataset hours (class, mode, source): train / val / test")
    cells = sorted({k[:3] for k in h})
    for c in cells:
        print(f"  {c[0]:20s} {c[1]:6s} {c[2]:14s} " + " / ".join(f"{h[c + (s,)]:.2f}" for s in ("train", "val", "test")))
    tot = {s: sum(v for k, v in h.items() if k[3] == s) for s in ("train", "val", "test")}
    print(f"  TOTAL {sum(tot.values()):.2f} h: " + ", ".join(f"{s} {v:.2f}" for s, v in tot.items()))
    rep["hours_main"] = {"|".join(k): v for k, v in h.items()}
    # falls
    f = defaultdict(lambda: [0, 0])
    for r in R["main"]:
        if r["mode"] in ("walk", "stand", "lifted"):
            f[(r["cls"], r["source"])][0] += r["fell"]
            f[(r["cls"], r["source"])][1] += 1
    print("\nfalls (main, walk/stand/lifted) by class and source (only cells with falls):")
    for k, (a, b) in sorted(f.items()):
        if a:
            print(f"  {k[0]:20s} {k[1]:14s} {a}/{b}")
    print(f"  total {sum(v[0] for v in f.values())}/{sum(v[1] for v in f.values())}")
    # joint io
    jt = Counter((r["source"], r["has_jt"], r["jt_nan"]) for r in R["main"])
    print("joint_target/torque recorded (source, present, any NaN): " + ", ".join(f"{k}: {v}" for k, v in sorted(jt.items(), key=str)))
    # courses
    print("\ncourse runs per source: fell / crossed every chunk (base x beyond the last chunk) / total")
    cr = {}
    for src in sorted({r["source"] for r in R["course"]}):
        rs = [r for r in R["course"] if r["source"] == src]
        crossed = sum(1 for r in rs if r["xy"] and r["boundaries"] and r["xy"][-1][0] >= r["boundaries"][-1])
        cr[src] = {"runs": len(rs), "fell": sum(r["fell"] for r in rs), "crossed_all": crossed}
        print(f"  {src:12s} fell {cr[src]['fell']:2d} / crossed {crossed:2d} / {len(rs)}")
    rep["course"] = cr
    # pads divergence
    nom = {r["id"]: r["xy"] for r in R["sens_nominal"] if r["split"] == "test"}
    dv = defaultdict(list)
    for r in R["sens_pads"]:
        a, b = nom.get(r["id"]), r["xy"]
        if a and b:
            for t in (5, 10, 19):
                if len(a) > t and len(b) > t:
                    dv[t].append(float(np.hypot(a[t][0] - b[t][0], a[t][1] - b[t][1])))
    print("\npad twin vs nominal base-position divergence (m): " + ", ".join(
        f"t={t}s median {np.median(v):.3f} p90 {np.percentile(v, 90):.3f}" for t, v in sorted(dv.items())))
    rep["pads_divergence"] = {t: [float(np.median(v)), float(np.percentile(v, 90))] for t, v in dv.items()}
    du = subprocess.run(["du", "-sh"] + [os.path.join(args.root, d) for d in ("v2", "v2_sens", "v2_course")],
                        capture_output=True, text=True).stdout
    print("\ndisk:\n" + du)
    rep["disk"] = du
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
