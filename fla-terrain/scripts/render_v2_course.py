#!/usr/bin/env python
"""B10: course video layout test. Simulates G113 on course seed >= 100 (never an evaluation course 0-39) on the v2
foot, writes per-window prediction files (truth, rule-based), renders render/v2/course_test_seed<N>.mp4.
  python scripts/render_v2_course.py [--seed 100] [--seconds 60]"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from render_v2 import clips as C  # noqa: E402
from walking import rule_detector as R  # noqa: E402
from walking.course import make_course  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--thresholds", default="results/v2/rule_detector_pilot/thresholds.json")
a = ap.parse_args()
assert a.seed >= 100, "seeds 0-39 are evaluation courses"
cfg = make_course(a.seed)
m, info, run, rec = C.walk_run(cfg, a.seconds + 1.0)
arr = rec.arrays()
meta = {"joint_names": list(info["joint_names"]), "fall_time": run["fell"]}
pdir = os.path.join(C.OUT, "preds")
os.makedirs(pdir, exist_ok=True)
pr = R.predict_episode(arr, meta, R.load_thresholds(a.thresholds))
st = R.window_starts(len(arr["t"]), run["fell"], arr["t"])
truth = arr["label_terrain"][st + R.W - 1]
np.savez(os.path.join(pdir, f"course{a.seed}_rule.npz"), t_end=pr["t_end"], terrain=pr["terrain"], sup_L=pr["sup_L"], sup_R=pr["sup_R"])
np.savez(os.path.join(pdir, f"course{a.seed}_truth.npz"), t_end=pr["t_end"], terrain=truth)
run["win_t"], run["win_truth"] = pr["t_end"], truth
lab = truth >= 0
print(f"[course {a.seed}] fell {run['fell']}, {len(truth)} windows; rule-based acc on labelled windows "
      f"{np.mean(pr['terrain'][lab] == truth[lab]):.3f}; classes {sorted(set(truth.tolist()))}")
path = C.course_video(run, m, info, {"rule-based": os.path.join(pdir, f"course{a.seed}_rule.npz")},
                      os.path.join(C.OUT, f"course_test_seed{a.seed}.mp4"), title=f"G113, course {a.seed} (test of layout)")
import subprocess  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_v2_clips import frames  # noqa: E402
dur, fr = frames(path)
print(f"[render] {os.path.basename(path)} {dur:.1f} s; frames {', '.join(map(os.path.basename, fr))}")
