#!/usr/bin/env python
"""B18 checks: (1) press-down clip: recorded sum over points vs applied load; (2) edge clip: points over the void
read noise only; (3) course video: the frame where the sole-model strip changes class matches the boundary latency of
that run (course_score definition), read from the rendered MP4 pixels. Writes render/v2/final/checks.json (merged)."""
import json, os, subprocess, sys
import mujoco, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.sim.recorder import EpisodeRecorder
from fla_terrain.sim.rig import rig_proprio
from render_v2.clips import RigScript, _edge, _flat, _rig
from render_v2.core import nominal_params

OUT = "/path/to/fla-terrain/render/v2/final"


def rig_run(kind):
    cfg = _flat(1.0) if kind == "press" else _edge()
    m, info = _rig(cfg, nominal_params())
    d = mujoco.MjData(m); sc = RigScript(kind, m, info); sc.reset(d)
    rec = EpisodeRecorder(m, info, motion_source="rig_script", mode="rig", split_key=f"render_{kind}",
                          rng=np.random.default_rng(0), proprio_fn=rig_proprio)
    rec.reset(d)
    press, pos = [], []
    nf = 0
    lay = info["point_layout"]
    while d.time < 8.0:
        sc.pre_step(d); mujoco.mj_step(m, d); rec.step(d)
        if len(rec.frames["F"]) > nf:
            nf = len(rec.frames["F"])
            press.append(-d.xfrc_applied[info["carrier"]["L"], 2])
            b = info["foot_bodies"]["L"]; R = d.xmat[b].reshape(3, 3)
            o = d.xpos[b] + R @ info["sole_cfg"].sole_origin_body()
            pos.append(o[None, :] + lay.xy @ R[:, :2].T)
    return m, info, np.asarray(rec.frames["F"]), np.asarray(press), np.asarray(pos), nf


def check_press():
    m, info, F, press, _, n = rig_run("press")
    W = m.body_subtreemass[info["carrier"]["L"]] * 9.81          # carrier + foot weight [N]
    t = (np.arange(n) + 1) / 50
    sel = (t > 1.5) & (t < 7.5)
    rec = F[sel, 0, :, 2].sum(1)
    exp = W + press[sel]
    err = rec / exp - 1
    return {"weight_N": round(float(W), 2), "applied_max_N": round(float(exp.max()), 1),
            "recorded_sum_at_max_N": round(float(rec[np.argmax(exp)]), 1),
            "rel_error_median": round(float(np.median(err)), 4), "rel_error_p95_abs": round(float(np.percentile(np.abs(err), 95)), 4),
            "pass_2pct": bool(np.percentile(np.abs(err), 95) < 0.02)}


def check_edge():
    m, info, F, _, pos, n = rig_run("edge")
    t = (np.arange(n) + 1) / 50
    sel = t > 2.0
    void = pos[sel, :, 0] > 0.026 + 0.002            # ground ends at x = 0.026 (2 mm margin)
    fz = F[sel, 0, :, 2]
    v, onp = fz[void], fz[~void]
    prm = nominal_params()
    return {"void_points_per_frame_mean": round(float(void.sum(1).mean()), 1),
            "void_Fz_mean_N": round(float(v.mean()), 3), "void_Fz_std_N": round(float(v.std()), 3),
            "void_absFz_p99_N": round(float(np.percentile(np.abs(v), 99)), 3), "void_absFz_max_N": round(float(np.abs(v).max()), 3),
            "sensor_noise_normal_N": prm.noise_normal, "sensor_offset_std_N": prm.offset_std,
            "supported_points_Fz_mean_N": round(float(onp.mean()), 2),
            "pass_noise_only": bool(np.percentile(np.abs(v), 99) < 4 * np.hypot(prm.noise_normal, prm.offset_std))}


def check_strip():
    """G113 run: per-boundary latency (course_score definition) from the stored P+C2 predictions, then the rendered
    strip pixel of the sole-model row at the detection window, in the frames just before and after detection."""
    from render_v2.final import DATA, PRED
    from render_v2.select_final import select
    ep = select()["walk_g113"]["episode"]
    meta = json.load(open(os.path.join(DATA, f"episode_{ep}.json")))
    tp = meta["terrain_params"]
    p = np.load(os.path.join(PRED, "P+Nreal+Sblk_s0", f"{ep}.npz"))
    te, pt = np.asarray(p["t_end"], float), np.asarray(p["terrain"])
    tds = meta["touchdowns"]
    chunks = tp["chunks"]
    t0s = []
    for c in chunks:
        ts = [tt for tt, _, rr in tds if c["region"] in rr]
        t0s.append(min(ts) if ts else None)
    prev, rows = 0, []
    for k, c in enumerate(chunks):
        new = int(c["label"])
        if t0s[k] is None:
            break
        if new == prev:
            continue
        t0 = t0s[k]
        t_stop = min([x for x in t0s[k + 1:] if x is not None] + [te[-1], t0 + 5.0])
        det = next((te[j] for j in range(len(te) - 1) if t0 - 1e-6 <= te[j] <= t_stop + 1e-6 and pt[j] == new and pt[j + 1] == new), None)
        rows.append({"boundary": k, "to": new, "t0": round(float(t0), 2), "detected_window_end": None if det is None else round(float(det), 2),
                     "latency_s": None if det is None else round(float(det - t0), 2)})
        prev = new
    # pixel check on the first detected boundary: strip x for t = det - 0.1 (inside the detection window), sole-model row
    b = next(r for r in rows if r["detected_window_end"] is not None)
    det = b["detected_window_end"]
    path = os.path.join(OUT, f"course_g113_seed{int(tp['course_seed']):03d}.mp4")
    tmax = len(np.load(f"/path/to/fla-terrain/b_runs/render_final/{ep}.npz")["t_video"]) / 30.0
    x = int(round(1280 * (0.13 + 0.85 * (det - 0.1) / tmax)))
    y = 450 + int(round(270 * (1 - (0.40 + 0.42 * (2.5 / 4)))))     # row 1 (sole model) centre of 4 rows
    from render_v2.panel import CLASS_HEX
    want = np.array([int(CLASS_HEX[b["to"]][i:i + 2], 16) for i in (1, 3, 5)])
    px = {}
    for name, tt in (("before", det - 2 / 30), ("after", det + 2 / 30)):
        o = f"/tmp/b18_px_{name}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{tt:.3f}", "-i", path, "-frames:v", "1", o], check=True)
        from PIL import Image
        img = np.asarray(Image.open(o).convert("RGB")).astype(int)
        px[name] = img[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).mean(0)
    dist = {k: float(np.abs(v - want).max()) for k, v in px.items()}
    return {"episode": ep, "boundaries": rows, "checked_boundary": b, "pixel_xy": [x, y],
            "new_class_colour": want.tolist(), "max_channel_diff_before": round(dist["before"], 1),
            "max_channel_diff_after": round(dist["after"], 1),
            "pass": bool(dist["after"] < 25 and dist["before"] > 25)}


if __name__ == "__main__":
    what = sys.argv[1:] or ["press", "edge", "strip"]
    path = os.path.join(OUT, "checks.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for w in what:
        res[w] = {"press": check_press, "edge": check_edge, "strip": check_strip}[w]()
        print(w, json.dumps(res[w])[:700])
    json.dump(res, open(path, "w"), indent=1, default=float)
