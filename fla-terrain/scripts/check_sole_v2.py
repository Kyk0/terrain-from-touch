"""A6 checks for sole v2 (V2_BRIEF section 1). Prints one pass/fail table with the measured numbers.

v1 checks 1-5 + recorder: scripts/check_sole.py --sole v2 (unchanged thresholds).
New checks (true point map = noise-free per-point forces averaged over each 150 Hz sample, nominal parameters):
 N1 standing on flat rigid ground (PD stand, last 1 s): no single point > 15 % of the foot's load; middle part < 5 %;
    'shared by many points' operationalized as: >= 50 % of the points of the back part and of the front part each carry
    >= 1/2 of the uniform share (1 / number of load-bearing points). (A first rule, >= 10 points with >= 1 % each,
    was miscalibrated for 152 points where the uniform share is 0.96 %; it is still printed.)
 N2 CoP of the point map vs CoP of the simulator's foot wrench (impulse average over the same sample), standing and
    walking (G113, 10 s): max error over samples with foot Fz > 20 N <= 5 mm.
 N3 standing with a 5 mm bump under the middle of the left foot: middle share rises clearly above flat ground
    (operationalized: >= 10 % and >= 10x the flat value).
 N4 foot-only rig, foot held flat, half of the foot over an edge (lateral and longitudinal): points over the void read
    exactly 0; support fraction from the map (loaded points of the load-bearing parts) within 10 % (relative) of the
    geometric ground truth for the same points.
 N5 sum over points = foot total contact force, all three axes, within 1 % (of |F|) before the sensor model,
    walking G113 10 s.
Plus timing: wall time per simulated second, v1 recorder vs v2 recorder vs v2 pads, walking G113.
  $PY scripts/check_sole_v2.py
"""
from __future__ import annotations

import subprocess
import sys
import time

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from fla_terrain.sim import EpisodeRecorder, SoleSensor, StandMotionSource, build_model, build_model_v2  # noqa: E402
from fla_terrain.sim.motion import set_pose  # noqa: E402
from fla_terrain.sim.pointmap import SoleV2Params  # noqa: E402

ROWS = []


def row(name, measured, threshold, ok):
    ROWS.append((name, measured, threshold, ok))


def walker():
    from walking.k1_velocity import K1VelocityPolicy
    return K1VelocityPolicy("G113_m2200")


def record(m, info, src, T, cmd=None, ref=False, rng=0):
    """Run src for T s with a v2 recorder keeping the true samples; optionally the reference impulse-averaged foot
    wrench per sample (Fz, CoP, F) from the simulator."""
    d = mujoco.MjData(m)
    if hasattr(src, "base_height"):
        src.reset(m, d, np.random.default_rng(rng))
    else:
        src.reset(m, d)
    rec = EpisodeRecorder(m, info, motion_source="check", mode="stand", split_key="c", rng=np.random.default_rng(rng))
    rec.keep_true = True
    rec.reset(d)
    S = SoleSensor(m, info["sole_cfg"]) if ref else None
    refs, acc, nst = [], {s: [0.0, np.zeros(2), np.zeros(3)] for s in "LR"}, 0
    nb = len(rec.true_samples)
    for _ in range(int(T / m.opt.timestep)):
        src.pre_step(m, d, cmd)
        mujoco.mj_step(m, d)
        rec.step(d, cmd if cmd is not None else (0, 0, 0))
        if ref:
            nst += 1
            w = S.read_full(d)["wrench"]
            for s in "LR":
                if w[s] is not None:
                    acc[s][0] += w[s][0][2]
                    acc[s][1] += w[s][0][2] * w[s][1]
                    acc[s][2] += w[s][0]
            if len(rec.true_samples) > nb:  # a sample was closed by this step
                refs.append({s: (acc[s][0] / nst, acc[s][1] / acc[s][0] if acc[s][0] > 0 else np.full(2, np.nan),
                                 acc[s][2] / nst) for s in "LR"})  # mean Fz, impulse-averaged CoP, mean F
                acc, nst = {s: [0.0, np.zeros(2), np.zeros(3)] for s in "LR"}, 0
                nb = len(rec.true_samples)
    return rec, refs, d


def v1_checks():
    out = subprocess.run([sys.executable, f"{ROOT}/scripts/check_sole.py", "--sole", "v2"], capture_output=True,
                         text=True).stdout
    for ln in out.splitlines():
        if ln.startswith(("PASS", "FAIL")):
            name = ln.split(":")[0][5:]
            row(f"v1 {name}", ln.split(":", 1)[1].strip()[:110], "as v1", ln.startswith("PASS"))


def n1_n3():
    prm = SoleV2Params()
    m, info = build_model_v2({"type": "flat", "mu": 1.0}, prm)
    L = info["point_layout"]
    rec, _, _ = record(m, info, StandMotionSource(info), 4.0)
    true = np.mean([t for tt, t in rec.true_samples if tt > 3.0], 0)  # [2, P, 3]
    fz = true[0, :, 2]
    share = fz / fz.sum()
    mid_flat = share[L.part == 1].sum()
    nb = int((share[L.part == 0] >= 0.01).sum())
    nf = int((share[L.part == 2] >= 0.01).sum())
    uni = 1.0 / int((L.part != 1).sum())  # uniform share over the load-bearing points
    fb = float((share[L.part == 0] >= 0.5 * uni).mean())
    ff = float((share[L.part == 2] >= 0.5 * uni).mean())
    row("N1 standing: max point share", f"{100 * share.max():.1f} % (L); R {100 * (true[1, :, 2] / true[1, :, 2].sum()).max():.1f} %",
        "< 15 %", share.max() < 0.15)
    row("N1 standing: middle share", f"{100 * mid_flat:.2f} %", "< 5 %", mid_flat < 0.05)
    row("N1 standing: load shared by many points",
        f"points with >= 1/2 of the uniform share ({100 * 0.5 * uni:.2f} %): back {100 * fb:.0f} %, front {100 * ff:.0f} % "
        f"[first rule '>=10 points with >=1 %' gave back {nb}, front {nf}: miscalibrated for P={L.P} (uniform share "
        f"{100 * uni:.2f} %), replaced]", ">= 50 % of the points of each part", fb >= 0.5 and ff >= 0.5)
    # N3: 5 mm bump under the middle of the left foot
    d0 = mujoco.MjData(m)
    set_pose(m, d0, info)
    b = info["foot_bodies"]["L"]
    R = d0.xmat[b].reshape(3, 3)
    c = d0.xpos[b] + R @ info["sole_cfg"].sole_origin_body()
    xm = np.mean(L.xy[L.part == 1, 0])
    pos = c + R[:, 0] * xm
    h = 0.005
    terrain = {"geoms": [{"type": "plane", "mu": 1.0, "region": 0},
                         {"type": "box", "pos": [pos[0], pos[1], h / 2], "size": [0.02, 0.03, h / 2], "mu": 1.0,
                          "region": 0}],
               "regions": [{"label": 3, "mu": 1.0, "stiffness": 7.7e8, "slope": 0.0}]}
    m2, info2 = build_model_v2(terrain, prm)
    rec2, _, d2 = record(m2, info2, StandMotionSource(info2), 4.0)
    t2 = np.mean([t for tt, t in rec2.true_samples if tt > 3.0], 0)
    sh2 = t2[0, :, 2] / t2[0, :, 2].sum()
    mid_bump = sh2[L.part == 1].sum()
    row("N3 bump under middle: middle share", f"{100 * mid_bump:.1f} % (flat {100 * mid_flat:.2f} %); robot fallen "
        f"{bool(d2.qpos[2] < 0.35)}", ">= 10 % and >= 10x flat", mid_bump >= 0.10 and mid_bump >= 10 * max(mid_flat, 1e-9))


def n2_n5():
    prm = SoleV2Params()
    m, info = build_model_v2({"type": "flat", "mu": 1.0}, prm)
    L = info["point_layout"]
    errs, worst, n5 = [], 0.0, 0
    for name, src, T, cmd in (("stand", StandMotionSource(info), 3.0, None), ("walk", walker(), 10.0, (0.4, 0, 0))):
        rec, refs, _ = record(m, info, src, T, cmd, ref=True)
        for (tt, true), rf in zip(rec.true_samples, refs):
            for f, s in enumerate("LR"):
                Fz, cop, F = rf[s]
                p = true[f]
                if Fz > 20.0 and np.all(np.isfinite(cop)):
                    cmap = (p[:, 2] @ L.xy) / p[:, 2].sum()
                    errs.append((name, np.linalg.norm(cmap - cop) * 1e3))
                if np.linalg.norm(F) > 20.0:
                    worst = max(worst, np.abs(p.sum(0) - F).max() / np.linalg.norm(F))
                    n5 += 1
    e = np.array([x for _, x in errs])
    es = np.array([x for n, x in errs if n == "walk"])
    row("N2 CoP map vs simulator", f"max {e.max():.3f} mm, p99 {np.percentile(e, 99):.3f} mm (walk max {es.max():.3f} mm, "
        f"{len(e)} samples)", "<= 5 mm", e.max() <= 5.0)
    row("N5 sum over points = foot force (3 axes)", f"max error {100 * worst:.4f} % of |F| ({n5} samples)", "<= 1 %",
        worst <= 0.01)


def n4():
    from fla_terrain.sim.rig import RigMotion, build_foot_rig
    prm = SoleV2Params()
    for case in ("lateral", "longitudinal"):
        if case == "lateral":   # left foot: +y half on the platform, -y half over the void
            a = {"type": "box", "pos": [0, 0.15 + 0.5, -0.05], "size": [1.0, 0.5, 0.05], "mu": 1.0, "region": 0}
        else:                   # left foot: heel half on the platform, toe half over the void
            a = {"type": "box", "pos": [0.026 - 0.5, 0.65, -0.05], "size": [0.5, 0.55, 0.05], "mu": 1.0, "region": 0}
        bR = {"type": "box", "pos": [0, -0.6, -0.05], "size": [1.0, 0.5, 0.05], "mu": 1.0, "region": 0}
        terrain = {"geoms": [{"type": "plane", "pos": [0, 0, -0.2], "mu": 1.0, "region": 0}, a, bR],
                   "regions": [{"label": 5, "mu": 1.0, "stiffness": 7.7e8, "slope": 0.0}]}
        m, info = build_foot_rig(terrain, v2_params=prm, lock_tilt=True)
        rng = np.random.default_rng(0)
        src = RigMotion(info, rng, 2.0)
        still = {"v": [0.0, 0.0], "tilt": [0.0, 0.0], "load_mod": 0.0, "lift": False}
        src.phases = [{"t": 0.0, "dur": 1e9, "L": still, "R": still}]
        from fla_terrain.sim.rig import rig_proprio
        d = mujoco.MjData(m)
        src.reset(m, d)
        rec = EpisodeRecorder(m, info, motion_source="rig", mode="rig", split_key="c", rng=rng, proprio_fn=rig_proprio)
        rec.keep_true = True
        rec.reset(d)
        for _ in range(int(1.5 / m.opt.timestep)):
            src.pre_step(m, d)
            mujoco.mj_step(m, d)
            rec.step(d)
        L = info["point_layout"]
        p = np.mean([t for tt, t in rec.true_samples if tt > 1.0], 0)[0, :, 2]
        g = rec.recon.gaps(d, "L")
        over = g < 0.005 + np.where(L.part == 1, prm.mid_clearance, 0.0)  # geometric GT: ground within 5 mm
        bear = L.part != 1
        void_max = np.abs(p[~over]).max() if (~over).any() else 0.0
        frac_map = float((p[bear] > 1e-3 * p.sum()).mean())
        frac_gt = float(over[bear].mean())
        row(f"N4 {case} edge: points over void", f"max |Fz| {void_max:.2e} N over {int((~over).sum())} void points "
            f"(foot Fz {p.sum():.1f} N)", "= 0", void_max == 0.0)
        row(f"N4 {case} edge: support fraction", f"map {frac_map:.3f} vs GT {frac_gt:.3f} "
            f"({100 * abs(frac_map - frac_gt) / max(frac_gt, 1e-9):.1f} %)", "within 10 %",
            abs(frac_map - frac_gt) <= 0.10 * frac_gt)


def timing():
    res = {}
    for name, build in (("v1 (2x4 zones)", lambda: build_model({"type": "flat", "mu": 1.0})),
                        ("v2 reconstruction", lambda: build_model_v2({"type": "flat", "mu": 1.0})),
                        ("v2 pads", lambda: build_model_v2({"type": "flat", "mu": 1.0}, pads=True))):
        m, info = build()
        d = mujoco.MjData(m)
        src = walker()
        src.reset(m, d, np.random.default_rng(0))
        rec = EpisodeRecorder(m, info, motion_source="t", mode="walk", split_key="t", rng=np.random.default_rng(0))
        rec.reset(d)
        T = 10.0
        t0 = time.time()
        for _ in range(int(T / m.opt.timestep)):
            src.pre_step(m, d, (0.4, 0, 0))
            mujoco.mj_step(m, d)
            rec.step(d, (0.4, 0, 0))
        res[name] = (time.time() - t0) / T
    base = res["v1 (2x4 zones)"]
    row("timing (walk G113, recorder on)", ", ".join(f"{k} {v:.3f} s/sim-s ({v / base:.2f}x)" for k, v in res.items()),
        "report", True)


def main():
    v1_checks()
    n1_n3()
    n2_n5()
    n4()
    timing()
    w = max(len(r[0]) for r in ROWS)
    print(f"{'check':{w}s} | result | threshold | measured")
    for name, meas, thr, ok in ROWS:
        print(f"{name:{w}s} | {'PASS' if ok else 'FAIL'} | {thr} | {meas}")
    ok = all(r[3] for r in ROWS)
    print(f"OVERALL {'PASS' if ok else 'FAIL'} ({sum(r[3] for r in ROWS)}/{len(ROWS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
