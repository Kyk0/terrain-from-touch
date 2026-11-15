"""T3 checks of the virtual sole on the K1 (MuJoCo). Prints one PASS/FAIL line per check.

1 standing:     sum Fz both feet = m g within 2 %                 (PD stand, flat, mu 1.0)
2 single-leg:   load on the stance foot (= m g within 2 %, other foot 0), CoP inside the sole
3 drag:         foot-only rig (K1 foot box -> same add_sole zones and contact params) loaded with
                10 kg per foot, dragged at 0.2 m/s: |shear| / Fz within 10 % of mu for mu = 0.3, 0.6, 0.9
4 slope:        PD stand on a theta slope (feet flat on it): |shear| / Fz = tan(theta) within 10 %
5 lifted/swing: base held 0.5 m above ground with motion running: all zone forces exactly 0;
                and swing foot in check 2: all forces 0
Forces are the true zone forces (sensor noise off) averaged over the stated window; one
extra line runs the full sensor model + recorder + writer and validates the episode.
"""
from __future__ import annotations

import argparse
import sys
import tempfile

import mujoco
import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fla_terrain.schema import validate_episode  # noqa: E402
from fla_terrain.sim import EpisodeRecorder, SoleSensor, StandMotionSource, build_model  # noqa: E402
from fla_terrain.sim.motion import PDController, set_pose  # noqa: E402

G = 9.81
RESULTS = []
from fla_terrain.sim import SoleConfig  # noqa: E402
SOLE = SoleConfig()


def report(name, ok, msg):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")


def inside_sole(cop, cfg, margin=0.0):
    hx, hy, _ = cfg.box_half
    return bool(np.all(np.isfinite(cop)) and abs(cop[0]) <= hx - margin and abs(cop[1]) <= hy - margin)


def simulate(m, d, src, T, t_avg, sensor, extra=None):
    """Run motion source for T seconds; return list of read_full dicts over the last t_avg seconds."""
    out = []
    n = int(round(T / m.opt.timestep))
    for k in range(n):
        if extra is not None:
            extra(m, d, k)
        src.pre_step(m, d, np.zeros(3))
        mujoco.mj_step(m, d)
        if k >= n - int(round(t_avg / m.opt.timestep)):
            out.append(sensor.read_full(d))
    return out


def stack(reads, side):
    return np.array([r["F"][side] for r in reads])  # [N, Z, 3]


def check_standing():
    m, info = build_model({"type": "flat", "mu": 1.0}, sole_cfg=SOLE)
    d = mujoco.MjData(m)
    src = StandMotionSource(info)
    src.reset(m, d)
    reads = simulate(m, d, src, 4.0, 2.0, SoleSensor(m, SOLE))
    fz = stack(reads, "L")[..., 2].sum(1) + stack(reads, "R")[..., 2].sum(1)
    mg = info["mass"] * G
    err = fz.mean() / mg - 1
    report("1 standing", abs(err) <= 0.02 and not src.fallen(m, d),
           f"sum Fz = {fz.mean():.2f} N, m g = {mg:.2f} N, err {100*err:+.3f} % (per-step spread {100*fz.std()/mg:.3f} %)")


def single_leg_pose(m, d, info):
    """Lean pose: both legs rolled in parallel (feet flat) to put the CoM over the left sole centre,
    right leg lifted. Solved by Newton on (hip roll a, left hip/ankle pitch c)."""
    o = info["sole_cfg"].sole_origin_body()
    bL = info["foot_bodies"]["L"]

    def pose(a, c):
        return {"left_hip_roll_joint": a, "left_ankle_roll_joint": -a, "right_hip_roll_joint": a,
                "right_ankle_roll_joint": -a, "left_hip_pitch_joint": -0.2 + c, "left_ankle_pitch_joint": -0.2 - c,
                "right_hip_pitch_joint": -0.7, "right_knee_pitch_joint": 1.0, "right_ankle_pitch_joint": -0.3}

    def err(x):
        set_pose(m, d, info, pose(*x))
        return (d.subtree_com[info["base_body"]] - (d.xpos[bL] + d.xmat[bL].reshape(3, 3) @ o))[:2]

    x = np.zeros(2)
    for _ in range(8):
        e = err(x)
        J = np.stack([(err(x + h) - e) / 1e-4 for h in np.eye(2) * 1e-4], 1)
        x = x - np.linalg.solve(J, e)
    return pose(*x), x


def check_single_leg():
    m, info = build_model({"type": "flat", "mu": 1.0}, sole_cfg=SOLE)
    d = mujoco.MjData(m)
    pose, x = single_leg_pose(m, d, info)
    pose.update({"right_hip_pitch_joint": -1.0, "right_knee_pitch_joint": 1.6, "right_ankle_pitch_joint": -0.6})
    src = StandMotionSource(info, pose=pose)
    src.reset(m, d)
    # PD alone cannot balance on one foot at the policy gains (wobbles, swing foot touches down);
    # this is a static posture test, so leg gains x4 and ankles x8.
    sc = {j: (8.0 if "ankle" in j else 4.0) for j in info["actuator_joints"] if any(k in j for k in ("hip", "knee", "ankle"))}
    src.pd = PDController(m, info, pose, kp_scale=sc)
    S = SoleSensor(m, SOLE)
    reads = simulate(m, d, src, 4.0, 2.0, S)
    FL, FR = stack(reads, "L"), stack(reads, "R")
    mg = info["mass"] * G
    fzL = FL[..., 2].sum(1).mean()
    fR_max = np.abs(FR).max()
    cop_zone = np.array([S.zone_cop(f) for f in FL])
    cop_pts = np.array([r["cop"]["L"] for r in reads])
    cfg = info["sole_cfg"]
    ins = all(inside_sole(c, cfg) for c in cop_zone) and all(inside_sole(c, cfg) for c in cop_pts)
    ok = abs(fzL / mg - 1) <= 0.02 and fR_max == 0.0 and ins and not src.fallen(m, d)
    report("2 single-leg", ok,
           f"Fz stance = {fzL:.2f} N ({100*(fzL/mg-1):+.2f} % of m g), max |F| swing foot = {fR_max:.1e} N, "
           f"CoP zone-based mean ({cop_zone[:,0].mean()*1e3:+.1f}, {cop_zone[:,1].mean()*1e3:+.1f}) mm, "
           f"contact-point ({cop_pts[:,0].mean()*1e3:+.1f}, {cop_pts[:,1].mean()*1e3:+.1f}) mm, all inside "
           f"+-{cfg.box_half[0]*1e3:.0f} x +-{cfg.box_half[1]*1e3:.0f} mm: {ins} (lean a={x[0]:+.3f}, c={x[1]:+.3f} rad)")
    return ok, fR_max


def build_foot_rig(mu: float, load_kg: float = 10.0):
    """Foot-only rig: two K1 foot bodies (same collision box -> same add_sole zones, same contact and
    solver parameters as the K1 MJCF) on x/y/z slide joints (orientation held flat), each loaded with
    ``load_kg`` and dragged in +x by a velocity actuator. Returns (model, info-like dict)."""
    from fla_terrain.sim import SoleConfig
    from fla_terrain.sim import k1 as K1
    from fla_terrain.sim.build import add_terrain
    from fla_terrain.sim.sole import add_sole

    base = mujoco.MjSpec.from_file(K1.K1_XML)
    bm = base.compile()
    foot_geom = [g for g in range(bm.ngeom) if bm.geom_bodyid[g] == mujoco.mj_name2id(
        bm, mujoco.mjtObj.mjOBJ_BODY, K1.FOOT_BODIES["L"]) and bm.geom_contype[g]][0]
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    for k in ("impratio", "tolerance", "cone", "solver", "integrator", "iterations"):
        setattr(spec.option, k, getattr(base.option, k))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1])
    add_terrain(spec, {"type": "flat", "mu": mu})
    for s, y in (("L", 0.15), ("R", -0.15)):
        b = spec.worldbody.add_body(name=K1.FOOT_BODIES[s], pos=[0, y, 0.038])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(K1.FOOT_BOX_POS), size=list(K1.FOOT_BOX_HALF),
                   friction=list(bm.geom_friction[foot_geom]), solref=list(bm.geom_solref[foot_geom]),
                   solimp=list(bm.geom_solimp[foot_geom]), condim=int(bm.geom_condim[foot_geom]))
        b.add_joint(name=f"x{s}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        b.add_joint(name=f"y{s}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
        b.add_joint(name=f"z{s}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        b.mass = load_kg
        b.inertia = [0.01, 0.01, 0.01]
        b.explicitinertial = True
        a = spec.add_actuator(name=f"drag{s}", target=f"x{s}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_velocity(kv=5000.0)
        add_sole(spec, s, SOLE)
    m = spec.compile()
    return m


def check_drag(speed=0.2, T=3.0, t_avg=1.5, info_speeds=(0.05,)):
    lines, info_lines = [], []
    ok_all = True
    for v in (speed,) + tuple(info_speeds):
        for mu in (0.3, 0.6, 0.9):
            m = build_foot_rig(mu)
            d = mujoco.MjData(m)
            S = SoleSensor(m, SOLE)
            n = int(T / m.opt.timestep)
            reads = []
            for k in range(n):
                d.ctrl[:] = v if k * m.opt.timestep > 0.5 else 0.0
                mujoco.mj_step(m, d)
                if k >= n - int(t_avg / m.opt.timestep):
                    reads.append(S.read_full(d))
            res = []
            for s in "LR":
                F = stack(reads, s).sum(1)
                res.append((np.hypot(F[:, 0].mean(), F[:, 1].mean()) / F[:, 2].mean(), F[:, 2].mean() / (10 * G),
                            np.mean(F[:, 2] <= 0)))
            vx = d.qvel[0]
            ok = all(abs(r - mu) / mu <= 0.10 for r, _, _ in res)
            txt = (f"mu={mu}: L {res[0][0]:.3f} R {res[1][0]:.3f} (Fz/mg {res[0][1]:.3f}, "
                   f"Fz=0 in {100*res[0][2]:.0f}% of steps)")
            if v == speed:
                ok_all &= ok
                lines.append(txt)
            else:
                info_lines.append(txt)
    report("3 drag", ok_all, f"foot-only rig (2 K1 feet, 10 kg each, held flat), slip {speed} m/s, |shear|/Fz per foot: "
           + "; ".join(lines))
    print(f"INFO 3 drag at {info_speeds[0]} m/s (soft-friction regime, no PASS/FAIL): " + "; ".join(info_lines))


def check_slope(theta_deg=(5.0, 10.0)):
    lines = []
    ok_all = True
    for th in theta_deg:
        m, info = build_model({"type": "slope", "slope_deg": th, "mu": 1.0}, sole_cfg=SOLE)
        d = mujoco.MjData(m)
        t = np.deg2rad(th)
        # trunk upright, feet pitched toe-up by theta so the soles lie flat on the slope
        pose = {"left_ankle_pitch_joint": -0.2 - t, "right_ankle_pitch_joint": -0.2 - t}
        src = StandMotionSource(info, pose=pose)
        src.reset(m, d)
        S = SoleSensor(m, SOLE)
        reads = simulate(m, d, src, 4.0, 2.0, S)
        F = stack(reads, "L").sum(1) + stack(reads, "R").sum(1)
        ratio = np.hypot(F[:, 0].mean(), F[:, 1].mean()) / F[:, 2].mean()
        R = d.xmat[info["foot_bodies"]["L"]].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(R[:, 2] @ np.array([-np.sin(t), 0, np.cos(t)]), -1, 1)))
        ok = abs(ratio / np.tan(t) - 1) <= 0.10 and not src.fallen(m, d)
        ok_all &= ok
        lines.append(f"theta={th:.0f} deg: |shear|/Fz {ratio:.4f} vs tan {np.tan(t):.4f} ({100*(ratio/np.tan(t)-1):+.1f} %), "
                     f"|F|/mg {np.linalg.norm(F.mean(0))/(info['mass']*G):.3f}, sole-vs-slope tilt {tilt:.2f} deg")
    report("4 slope", ok_all, "; ".join(lines))


def check_lifted(fR_swing):
    m, info = build_model({"type": "flat", "mu": 1.0}, base_rig=True, sole_cfg=SOLE)
    d = mujoco.MjData(m)
    z = set_pose(m, d, info)
    d.qpos[2] = z + 0.5
    d.mocap_pos[0] = [0, 0, z + 0.5]
    mujoco.mj_forward(m, d)
    # motion keeps running: sinusoidal leg swing on top of the default pose
    src = StandMotionSource(info, offset_fn=lambda t: {"left_hip_pitch_joint": 0.4 * np.sin(6 * t),
                                                       "right_hip_pitch_joint": -0.4 * np.sin(6 * t),
                                                       "left_knee_pitch_joint": 0.3 * (1 + np.sin(6 * t)),
                                                       "right_knee_pitch_joint": 0.3 * (1 - np.sin(6 * t))})
    src.pd = PDController(m, info)
    src.q0 = src.pd.q_target.copy()
    reads = simulate(m, d, src, 3.0, 3.0, SoleSensor(m, SOLE))
    fmax = max(np.abs(stack(reads, s)).max() for s in "LR")
    nc = sum(r["contact"]["L"] or r["contact"]["R"] for r in reads)
    ok = fmax == 0.0 and fR_swing == 0.0
    report("5 lifted/swing", ok, f"lifted 3 s with legs swinging: max |F| = {fmax:.1e} N, frames with contact {nc}; "
                                 f"swing foot in single-leg: max |F| = {fR_swing:.1e} N")


def check_recorder():
    m, info = build_model({"type": "flat", "mu": 0.8, "label": 0}, sole_cfg=SOLE)
    d = mujoco.MjData(m)
    src = StandMotionSource(info)
    src.reset(m, d)
    rec = EpisodeRecorder(m, info, motion_source=src.name, mode="stand", split_key="check_sole",
                          rng=np.random.default_rng(0))
    rec.reset(d)
    for _ in range(int(3.0 / m.opt.timestep)):
        src.pre_step(m, d, np.zeros(3))
        mujoco.mj_step(m, d)
        rec.step(d)
    with tempfile.TemporaryDirectory() as td:
        p = rec.write(td, 0, notes="check_sole recorder smoke")
        errs = validate_episode(p, raise_on_error=False)
        with np.load(p) as zf:
            T = len(zf["t"])
            fz = zf["foot_zone_L"][..., 2].sum(1) + zf["foot_zone_R"][..., 2].sum(1)
            sup = (zf["label_support_L"][-50:] == 2).mean()
    mg = info["mass"] * G
    ok = not errs and T == 150
    report("recorder", ok, f"3 s -> T={T} frames at 50 Hz, validator errors {len(errs)}, noisy sum Fz last 1 s "
                           f"{fz[-50:].mean():.1f} N ({100*(fz[-50:].mean()/mg-1):+.2f} %, gain/offset errors on), "
                           f"support=full {100*sup:.0f} %")


def main():
    global SOLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--sole", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()
    SOLE = SoleConfig(version=args.sole)
    print(f"mujoco {mujoco.__version__}, K1 22dof + sole {SOLE.version} ({SOLE.Z} regions/foot), dt 0.002, base MJCF contact params, terrain priority 1")
    check_standing()
    _, fR = check_single_leg()
    check_drag()
    check_slope()
    check_lifted(fR)
    check_recorder()
    ok = all(RESULTS)
    print(f"OVERALL {'PASS' if ok else 'FAIL'} ({sum(RESULTS)}/{len(RESULTS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
