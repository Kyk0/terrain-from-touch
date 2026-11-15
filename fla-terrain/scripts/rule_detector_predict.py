#!/usr/bin/env python
"""Write rule-based detector predictions (walking/rule_detector.predict_episode format: t_end, terrain, sup_L, sup_R,
one npz per episode, windows as fla_terrain.models.dataset) with fixed, already fitted thresholds.
  python scripts/rule_detector_predict.py --data data/v2_course --thresholds results/v2/rule_detector/thresholds.json \
      --out results/v2/rule_detector/course
No fitting and no labels are read here (only joint/IMU/command arrays and the metadata fall time)."""
import argparse, glob, json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from walking import rule_detector as R  # noqa: E402

_K = {}


def one(a):
    path, th, out = a
    if "k" not in _K:
        _K["k"] = R.Kinematics()
    j = json.load(open(path))
    z = np.load(path[:-5] + ".npz")
    arr = {k: z[k] for k in ("t", "joint_pos", "joint_vel", "imu_gyro", "imu_gravity", "cmd_vel", "joint_target",
                             "joint_torque") if k in z.files}
    p = R.predict_episode(arr, {"joint_names": j["joint_names"], "fall_time": j.get("fall_time")}, th, _K["k"])
    np.savez(os.path.join(out, f"{j['episode_id']}.npz"), episode=j["episode_id"], **p)
    return j["episode_id"], len(p["terrain"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    th = R.load_thresholds(a.thresholds)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, "*.json")))
    with Pool(40) as p:
        res = p.map(one, [(f, th, a.out) for f in files], chunksize=1)
    json.dump({"thresholds": a.thresholds, "data": a.data, "episodes": {e: n for e, n in res}},
              open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"[rule predict] {len(res)} episodes, {sum(n for _, n in res)} windows -> {a.out}")
