#!/usr/bin/env python
"""B11: independent QA of walking episodes (schema arrays + metadata only).

  python scripts/walk_qa_dataset.py --data /path/to/fla-terrain/data/v1

Prints per class x source (walk mode): episodes, falls, step frequency (contact_gt rising edges, >= 0.08 s
airborne, before any fall), cmd_vel stats; strip and edge checks; constant channels; and a shortcut test:
a random forest on trivial episode-level features (length, command stats, first-frame joint pose; no sensors,
no IMU) predicting the class, scored on the test split. Speed is not derivable: the schema has no base velocity.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

SRC = {"G113_m2200": "G113", "G80_m4200": "G80", "sonic_k1_it50000": "SONIC"}


def touchdowns(c, min_air=4):
    n, air, on = 0, min_air, False
    for v in c:
        if v and not on:
            if air >= min_air:
                n += 1
            on, air = True, 0
        elif not v:
            on = False
            air += 1
    return n


def one(path):
    j = json.load(open(path))
    z = np.load(path[:-5] + ".npz")
    tp = j["terrain_params"]
    T = len(z["t"])
    lt = z["label_terrain"]
    ft = j.get("fall_time")
    end = int(round(ft * 50)) if ft is not None else T
    walk0 = 50
    steps = sum(touchdowns(z[f"contact_gt_{s}"][walk0:end].astype(bool)) for s in "LR")
    dur = max((end - walk0) / 50.0, 1e-6)
    cmd = z["cmd_vel"]
    chans = {"joint_pos": z["joint_pos"], "joint_vel": z["joint_vel"], "imu_gyro": z["imu_gyro"],
             "imu_gravity": z["imu_gravity"], "foot_L": z["foot_zone_L"].reshape(T, -1),
             "foot_R": z["foot_zone_R"].reshape(T, -1)}
    const = [f"{k}[{i}]" for k, v in chans.items() for i in np.nonzero(v.std(0) < 1e-6)[0]]
    td = j.get("touchdowns") or []
    regions = sorted({r for _, _, rs in td for r in rs})
    first_new = next((t for t, _, rs in td if 1 in rs), None)
    return {
        "mode": j["mode"], "cls": tp.get("class"), "src": SRC.get(j["motion_source"], j["motion_source"]),
        "split": tp.get("range_split"), "T": T, "fell": bool(j.get("fell")), "steps_s": steps / dur,
        "cmd_vx": float(cmd[min(walk0, T - 1):, 0].mean()), "cmd_wz": float(np.abs(cmd[min(walk0, T - 1):, 2]).mean()),
        "cmd_zero": float(np.mean(np.abs(cmd).max(1) < 1e-9)), "kind": (tp.get("cmd_schedule") or {}).get("kind"),
        "unk": float(np.mean(lt == -1)), "partial": bool(((z["label_support_L"] == 1) | (z["label_support_R"] == 1))[:end].any()),
        "partial_frac": float(np.mean((z["label_support_L"] == 1) | (z["label_support_R"] == 1))),
        "regions": regions, "n_regions": len(tp.get("regions", [])), "first_new": first_new,
        "q0": z["joint_pos"][0].tolist(), "const": const, "labels": sorted(set(lt.tolist())),
        "fz0": float(z["foot_zone_L"][:25, :, 2].sum() + z["foot_zone_R"][:25, :, 2].sum()) / 25,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data, "*.json")))
    with Pool(32) as p:
        R = p.map(one, files, chunksize=8)
    W = [r for r in R if r["mode"] == "walk"]
    modes = defaultdict(int)
    for r in R:
        modes[r["mode"]] += 1
    print(f"[B11] {a.data}: {len(R)} episodes {dict(modes)}; walk table (speed not derivable from the schema):")
    print(f"{'class':17s} {'src':5s} {'n':>4s} {'fall':>5s} {'steps/s':>7s} {'cmd vx':>11s} {'cmd |wz|':>8s} {'T':>9s} {'unk':>5s}")
    by = defaultdict(list)
    for r in W:
        by[(r["cls"], r["src"])].append(r)
    for (c, s), rs in sorted(by.items()):
        vx = np.array([r["cmd_vx"] for r in rs])
        Ts = [r["T"] for r in rs]
        print(f"{c:17s} {s:5s} {len(rs):4d} {np.mean([r['fell'] for r in rs]):5.2f} {np.mean([r['steps_s'] for r in rs]):7.2f} "
              f"{vx.mean():5.3f}+-{vx.std():.3f} {np.mean([r['cmd_wz'] for r in rs]):8.3f} {min(Ts):4d}-{max(Ts):4d} "
              f"{np.mean([r['unk'] for r in rs]):5.2f}")
    # strips
    st = [r for r in W if str(r["cls"]).startswith("strip")]
    if st:
        print("strip: both surfaces touched / first touchdown on new surface (median s), per source:")
        for s in sorted({r["src"] for r in st}):
            rs = [r for r in st if r["src"] == s]
            both = [r for r in rs if len(r["regions"]) >= 2]
            fn = [r["first_new"] for r in rs if r["first_new"] is not None]
            print(f"  {s:5s} {len(both)}/{len(rs)} = {len(both)/len(rs):.2f}; first new {np.median(fn) if fn else float('nan'):.1f} s "
                  f"(p10 {np.percentile(fn, 10) if fn else float('nan'):.1f}, p90 {np.percentile(fn, 90) if fn else float('nan'):.1f})")
    ed = [r for r in W if r["cls"] == "edge"]
    if ed:
        print("edge: episodes with partial support (before fall) / mean fraction of partial frames, per source: " + "; ".join(
            f"{s} {np.mean([r['partial'] for r in ed if r['src'] == s]):.2f} / {np.mean([r['partial_frac'] for r in ed if r['src'] == s]):.3f}"
            for s in sorted({r["src"] for r in ed})))
        non = [r for r in W if r["cls"] != "edge"]
        print(f"      non-edge walk episodes with partial-support frames: {np.mean([r['partial'] for r in non]):.2f} "
              f"(mean fraction {np.mean([r['partial_frac'] for r in non]):.3f})")
    # cmd distribution per class (velocity policies)
    print("cmd_vel per class, velocity policies (mean vx | mean |wz| | zero-command frames | schedule kinds):")
    for c in sorted({r["cls"] for r in W}):
        rs = [r for r in W if r["cls"] == c and r["src"] != "SONIC"]
        kinds = defaultdict(int)
        for r in rs:
            kinds[r["kind"]] += 1
        print(f"  {c:17s} {np.mean([r['cmd_vx'] for r in rs]):.3f} | {np.mean([r['cmd_wz'] for r in rs]):.3f} | "
              f"{np.mean([r['cmd_zero'] for r in rs]):.3f} | {dict(kinds)}")
    sc = [r for r in W if r["src"] == "SONIC"]
    print(f"  SONIC cmd_vel all zero: {all(r['cmd_zero'] == 1.0 for r in sc)}")
    const = defaultdict(int)
    for r in W:
        for c in r["const"]:
            const[c] += 1
    print(f"constant channels in walk episodes (count of episodes): {dict(sorted(const.items(), key=lambda x: -x[1])[:8])}")
    # shortcut test
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    def feats(r):
        return [r["T"], r["cmd_vx"], r["cmd_wz"], r["cmd_zero"]] + r["q0"]

    for name, f in (("T+cmd+start pose", feats), ("T+cmd", lambda r: [r["T"], r["cmd_vx"], r["cmd_wz"], r["cmd_zero"]]), ("start pose only", lambda r: r["q0"]), ("T only", lambda r: [r["T"]])):
        tr = [r for r in W if r["split"] == "train"]
        te = [r for r in W if r["split"] == "test"]
        if not tr or not te:
            continue
        clf = RandomForestClassifier(300, random_state=0, n_jobs=8).fit([f(r) for r in tr], [r["cls"] for r in tr])
        pr = clf.predict([f(r) for r in te])
        f1 = f1_score([r["cls"] for r in te], pr, average="macro")
        print(f"shortcut test ({name}) -> class, test macro-F1 {f1:.2f} (chance ~{1/len({r['cls'] for r in W}):.2f}); "
              f"train {len(tr)} test {len(te)}")
    # per-class start-up load (first 0.5 s mean total Fz)
    print("first 0.5 s mean total Fz per class (N): " + ", ".join(
        f"{c} {np.mean([r['fz0'] for r in W if r['cls'] == c]):.0f}" for c in sorted({r['cls'] for r in W})))


if __name__ == "__main__":
    main()
