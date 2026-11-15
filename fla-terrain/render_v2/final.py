"""B18 final v2 course videos: recorded course runs of data/v2_course, selected by render_v2.select_final (first run
of each source, in seed order, that crosses every chunk), poses from a bit-identical re-simulation
(render_v2.rollout_final, 30 fps), sole panel = the RECORDED 50 Hz point map of that episode (raw, with points at the
sensor's per-point limit marked), strips = stored per-window predictions of that exact run:
  truth       label of the window's last frame (stored with A's course predictions; checked against the episode)
  sole model  P+C2 = P+Nreal+Sblk seed 0   (a/results/v2/src3/course_preds/P+Nreal+Sblk_s0/<episode>.npz)
  P only      P seed 0                     (a/results/v2/src3/course_preds/P_s0/<episode>.npz)
  rule-based  walking/rule_detector, thresholds fitted on data/v2 train (results/v2/rule_detector/course/<ep>.npz)
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .core import FPS, W, Writer, camera, smooth, style
from .panel import CLASS_HEX, CLASS_NAMES, SolePanel, StripPanel

DATA = "/path/to/fla-terrain/data/v2_course"
PRED = "/path/to/fla-terrain/a/results/v2/src3/course_preds"
RULE = "/path/to/fla-terrain/b/results/v2/rule_detector/course"
ROLL = "/path/to/fla-terrain/b_runs/render_final"
OUT = "/path/to/fla-terrain/render/v2/final"
ROWS = (("truth", None), ("sole model P+C2", os.path.join(PRED, "P+Nreal+Sblk_s0")), ("P only", os.path.join(PRED, "P_s0")),
        ("rule-based", RULE))
NAMES = {"walk_g113": "G113", "walk_g80": "G80", "walk_sonic": "SONIC"}


def _font(size):
    import matplotlib
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), size)


def course_video(ep: str, source: str, crossed_of_40: int, nofall_of_40: int):
    from fla_terrain.sim.build import build_model_v2
    from walking.course import make_course
    meta = json.load(open(os.path.join(DATA, f"episode_{ep}.json")))
    z = np.load(os.path.join(DATA, f"episode_{ep}.npz"))
    roll = np.load(os.path.join(ROLL, f"{ep}.npz"))
    qpos, tv = roll["qpos"], roll["t_video"]
    seed = int(meta["terrain_params"]["course_seed"])
    m, info = build_model_v2(make_course(seed))          # same bodies/joints; terrain of this course seed
    style(m, info)
    F = np.stack([z["foot_points_L"], z["foot_points_R"]], 1)          # [T, 2, P, 3] recorded
    t50 = z["t"]
    sat = float(meta["sensor_model"]["params"]["sat_normal"])
    tmax = float(tv[-1])
    strip = StripPanel(W, 270, [r[0] for r in ROWS], tmax, title="terrain class per window (1 s causal window, new every 0.2 s)",
                       legend=list(zip(CLASS_NAMES, CLASS_HEX)) + [("unknown / transition", "#000000")])
    data = {}
    truth = None
    for k, (row, d) in enumerate(ROWS):
        if d is None:
            continue
        p = np.load(os.path.join(d, f"{ep}.npz"))
        data[row] = (np.asarray(p["t_end"], float), np.asarray(p["terrain"]))
        if truth is None and "truth" in p.files:
            truth = (np.asarray(p["t_end"], float), np.asarray(p["truth"]).astype(int))
    data["truth"] = truth
    # check the stored truth against the episode's own labels at the window ends
    fi = np.round(truth[0] * 50).astype(int) - 1
    truth_ok = bool(np.array_equal(z["label_terrain"][fi].astype(int), truth[1]))
    for k, (row, _) in enumerate(ROWS):
        te, cl = data[row]
        strip.add_segments(k, te - 0.2, te, [CLASS_HEX[c] if c >= 0 else "#000000" for c in cl])
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 450, 860)
    pan = SolePanel(info["point_layout"], 420, 450, "", sat=sat)
    look = smooth(qpos[:, :3], 15)
    caption = (f"{NAMES[source]} on course seed {seed}: recorded test run, no fall, every chunk crossed",
               f"{NAMES[source]} crossed every chunk in {crossed_of_40} of 40 courses ({nofall_of_40} of 40 without a fall)")
    wr = Writer(os.path.join(OUT, f"course_{NAMES[source].lower()}_seed{seed:03d}.mp4"))
    font = _font(13)
    n = len(tv)
    for j in range(n):
        t = float(tv[j])
        d.qpos[:] = qpos[j]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera(look[j] - [0, 0, 0.2], 2.6, 120, -18))
        view = Image.fromarray(r.render())
        dr = ImageDraw.Draw(view)
        dr.rectangle([0, 0, 860, 40], fill=(21, 23, 28))
        dr.text((8, 3), caption[0], fill=(232, 232, 232), font=font)
        dr.text((8, 21), caption[1], fill=(170, 175, 185), font=font)
        i = max(0, int(np.searchsorted(t50, t + 1e-9, side="right")) - 1)   # latest finished 50 Hz sensor frame
        right = pan.draw(F[i], f"t = {t50[i]:5.2f} s")
        top = np.concatenate([np.asarray(view), right], 1)
        wr.write(np.concatenate([top, strip.render(t)], 0))
    wr.close()
    return {"file": os.path.basename(wr.path), "frames": n, "seconds": n / FPS, "truth_matches_episode_labels": truth_ok,
            "sat_normal_N": sat, "caption": " / ".join(caption)}
