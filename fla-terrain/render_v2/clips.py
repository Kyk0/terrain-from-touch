"""Clip generators for FLA-0 v2 (ticket B10). All 1280x720, 30 fps, H.264.

rig_clip(kind):   foot rig (the training implementation's fla_terrain.sim.rig.build_foot_rig with the v2 sole, nominal parameters,
                  10 kg per foot), scripted protocol, ~8 s; left: 3D view of the left foot, right: sole panel.
    press      settle 1 s, extra downward force ramps 0 -> 150 N over 4 s, released over 1.5 s
    drag03 / drag09  friction 0.3 / 0.9: settle, press +50 N, then a sideways (+y) drive whose force limit ramps
               0 -> 150 N over 5 s (target 0.1 m/s): the foot sticks until the drive exceeds mu * Fz, then slides
    twist      as press, then the foot is twisted about the vertical axis (position servo ramping to 0.6 rad,
               kp 20 Nm/rad): needs a yaw hinge that the training implementation's rig does not have, so this clip uses a copy of
               build_foot_rig with one extra hinge (render_v2.clips._rig, yaw=True); everything else identical
    edge       ground ends at the sole centre (half the sole over a 15 cm drop), settle, press +60 N, hold
closeup(cls):     G113 walking on the v2 foot, 2 s of simulation shown at 0.25x (8 s); left: close camera on the feet,
                  right: sole panel + gait-phase strip (stance / swing per foot from ground-truth contact).
course_video():   robot view on top (left) + sole panel (right), timeline strips under it: truth, sole model,
                  proprioception only, rule-based; each read from a per-window prediction file (npz with t_end,
                  terrain); rows without a file are drawn as "not available". Strips fill in as the robot walks.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from fla_terrain.sim.recorder import EpisodeRecorder
from fla_terrain.sim.rig import ANKLE_H, build_foot_rig, rig_proprio

from .core import FPS, H, W, Tap, Writer, camera, nominal_params, smooth, style
from .panel import CLASS_HEX, CLASS_NAMES, SUP_HEX, SolePanel, StripPanel

OUT = "/path/to/fla-terrain/render/v2"


# ------------------------------------------------------------------------------------------------ rig
def _flat(mu):
    return {"type": "flat", "mu": mu, "geoms": [{"type": "plane", "mu": mu, "region": 0}],
            "regions": [{"label": 0 if mu > 0.5 else 1, "mu": mu, "stiffness": 7.7e8, "slope": 0.0}]}


def _edge():
    e = 0.026   # sole centre (foot box centre) is 26 mm ahead of the ankle: ground ends there
    return {"type": "edge", "geoms": [{"type": "box", "pos": [e - 1.0, 0, -0.05], "size": [1.0, 1.0, 0.05], "region": 0,
                                       "mu": 1.0},
                                      {"type": "plane", "pos": [0, 0, -0.15], "region": 0, "mu": 1.0}],
            "regions": [{"label": 5, "mu": 1.0, "stiffness": 7.7e8, "slope": 0.0}]}


def _rig(terrain_cfg, params, yaw=False):
    """the training implementation's build_foot_rig, unchanged except (render only, through a compile hook on the rig spec): a checker
    texture on the terrain geoms (so sliding / twisting is visible) and, for the twist clip, one extra yaw hinge
    (+ position servo) per foot (the rig has no yaw DOF)."""
    import fla_terrain.sim.rig as R
    spec_cls = mujoco.MjSpec
    orig_compile = spec_cls.compile
    holder = {}

    def compile_hooked(self):
        if not holder.get("done") and "rig_L" in [b.name for b in self.bodies]:   # only the rig spec
            holder["done"] = True
            t = self.add_texture()
            t.name, t.type, t.builtin = "viz_floor", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
            t.rgb1, t.rgb2 = [0.62, 0.62, 0.60], [0.48, 0.48, 0.46]
            t.width = t.height = 256
            mat = self.add_material()
            mat.name = "viz_floor_mat"
            tex = list(mat.textures)
            tex[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
            mat.textures = tex
            mat.texrepeat = [20.0, 20.0]     # 5 cm checker cells
            mat.texuniform = True
            for g in self.worldbody.geoms:
                if g.name.startswith("terrain_") and not (g.type == mujoco.mjtGeom.mjGEOM_PLANE and g.pos[2] < -0.01):
                    g.material = mat.name
            if yaw:
                for side in "LR":
                    f = self.body(R.k1.FOOT_BODIES[side])
                    f.add_joint(name=f"rig_{side}_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                                range=[-1.2, 1.2], damping=0.5)
                    act = self.add_actuator(name=f"p_{side}_yaw", target=f"rig_{side}_yaw",
                                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
                    act.set_to_position(kp=20.0, kv=0.5)
        return orig_compile(self)

    spec_cls.compile = compile_hooked
    try:
        m, info = R.build_foot_rig(terrain_cfg, v2_params=params, load_kg=(10.0, 10.0))
    finally:
        spec_cls.compile = orig_compile
    return m, info


class RigScript:
    def __init__(self, kind, m, info):
        self.kind, self.m, self.info = kind, m, info
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        self.vy = [aid(f"v_{s}_y") for s in "LR"]
        self.vx = [aid(f"v_{s}_x") for s in "LR"]
        self.yaw = [aid(f"p_{s}_yaw") for s in "LR"]
        self.carrier = [info["carrier"][s] for s in "LR"]

    def reset(self, d):
        from fla_terrain.sim.rig import RigMotion
        RigMotion(self.info, np.random.default_rng(0), 1.0).reset(self.m, d)   # lowers the feet onto the terrain

    def pre_step(self, d):
        t = d.time
        m = self.m
        d.ctrl[:] = 0.0
        d.xfrc_applied[:] = 0.0
        ramp = lambda a, b: float(np.clip((t - a) / (b - a), 0, 1))  # noqa: E731
        press = 0.0
        if self.kind == "press":
            press = 150.0 * ramp(1.0, 5.0) * (1 - ramp(5.0, 6.5))
        elif self.kind.startswith("drag"):
            press = 50.0 * ramp(1.0, 2.0)
            lim = 150.0 * ramp(2.0, 7.0) if t < 7.2 else 150.0
            for a in self.vy:
                m.actuator_forcerange[a] = (-max(lim, 1e-3), max(lim, 1e-3))
                d.ctrl[a] = 0.1 if 2.0 <= t < 7.2 else 0.0
        elif self.kind == "twist":
            press = 50.0 * ramp(1.0, 2.0)
            for a in self.yaw:
                d.ctrl[a] = 0.6 * ramp(2.0, 6.0)
        elif self.kind == "edge":
            press = 60.0 * ramp(1.0, 3.0)
        for c in self.carrier:
            d.xfrc_applied[c, 2] = -press


def rig_clip(kind, seconds=8.0, out=OUT):
    params = nominal_params()
    cfg = {"press": _flat(1.0), "drag03": _flat(0.3), "drag09": _flat(0.9), "twist": _flat(1.0), "edge": _edge()}[kind]
    m, info = _rig(cfg, params, yaw=(kind == "twist"))
    d = mujoco.MjData(m)
    sc = RigScript(kind, m, info)
    sc.reset(d)
    rec = EpisodeRecorder(m, info, motion_source="rig_script", mode="rig", split_key=f"render_{kind}",
                          rng=np.random.default_rng(0), proprio_fn=rig_proprio)
    rec.reset(d)
    tap = Tap(rec)
    while d.time < seconds:
        sc.pre_step(d)
        mujoco.mj_step(m, d)
        tap.step(d)
    a = tap.arrays()
    style(m, info)
    titles = {"press": "rig: press straight down (10 kg + up to 150 N)",
              "drag03": "rig: press and drag sideways until it slides, friction 0.3",
              "drag09": "rig: press and drag sideways until it slides, friction 0.9",
              "twist": "rig: press and twist about the vertical axis",
              "edge": "rig: half of the sole over an edge (15 cm drop)"}
    path = os.path.join(out, f"rig_{kind}.mp4")
    _render_side_by_side(m, a, path, titles[kind], info["point_layout"],
                         cam=lambda i: camera(_feet_mid(m, d, a["qpos"][i], info), 0.85 if kind != "edge" else 0.7,
                                              *((60, -22) if kind == "edge" else (140, -38))))
    return path, a


def _feet_mid(m, d, q, info):
    """midpoint between the two sole centres (camera target: both feet and the ground in view)."""
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return np.mean([d.xpos[info["foot_bodies"][s]] for s in "LR"], 0) - [0, 0, ANKLE_H]


def _foot_pos(m, d, q, info):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.xpos[info["foot_bodies"]["L"]].copy() - [0, 0, ANKLE_H]


def _render_side_by_side(m, a, path, title, layout, cam, speed=1.0, t0=0.0, t1=None, strip=None):
    """Left 640x720 3D view, right 640x720 panel (+ optional phase strip under the panel)."""
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, 640)
    ph = 110 if strip is not None else 0
    pan = SolePanel(layout, 640, H - ph, title, sat=nominal_params().sat_normal)
    t1 = a["t"][-1] if t1 is None else t1
    n = int(round((t1 - t0) / speed * FPS))
    wr = Writer(path)
    for k in range(n):
        t = t0 + k * speed / FPS
        i = max(0, int(np.searchsorted(a["t"], t, side="right")) - 1)     # latest finished 50 Hz frame
        d.qpos[:] = a["qpos"][i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam(i))
        view = r.render()
        sub = f"t = {a['t'][i]:5.2f} s" + (f"   (shown at {speed:g}x)" if speed != 1.0 else "")
        right = pan.draw(a["F"][i], sub)
        if strip is not None:
            right = np.concatenate([right, strip.render(t)], 0)
        wr.write(np.concatenate([view, right], 1))
    wr.close()
    return n


# ------------------------------------------------------------------------------------------------ close-ups
def walk_run(terrain_cfg, seconds, policy="G113_m2200", cmd=(0.35, 0, 0), seed=0, params=None, sched=None):
    """Simulate a walking source on the v2 foot with the recorder; returns (model, info, tap arrays, recorder)."""
    from fla_terrain.sim.build import build_model_v2
    from walking import K1VelocityPolicy
    m, info = build_model_v2(terrain_cfg, params=params or nominal_params())
    d = mujoco.MjData(m)
    src = K1VelocityPolicy(policy)
    sp = terrain_cfg.get("spawn", [0, 0, 0])
    src.reset(m, d, np.random.default_rng(seed), base_pos=(sp[0], sp[1], 0.58 + sp[2]), base_yaw=terrain_cfg.get("yaw", 0.0))
    rec = EpisodeRecorder(m, info, motion_source=policy, mode="walk", split_key="render",
                          rng=np.random.default_rng(seed))
    rec.reset(d)
    tap = Tap(rec)
    fell = None
    for i in range(int(round(seconds / m.opt.timestep))):
        t = i * m.opt.timestep
        c = (sched(t) if sched else (cmd if t >= 1.0 else (0, 0, 0)))
        src.pre_step(m, d, c)
        mujoco.mj_step(m, d)
        tap.step(d, c)
        if fell is None and (i + 1) % 10 == 0 and src.fallen(m, d):
            fell = t
            rec.mark_fall(d)
            break
    a = tap.arrays()
    a["fell"] = fell
    return m, info, a, rec


def closeup(cls, out=OUT, t0=3.0, dur=2.0, speed=0.25):
    from fla_terrain.sim.build import slope_normal
    if cls == "hard_grip":
        cfg = _flat(1.0)
    elif cls == "hard_slip":
        cfg = _flat(0.2)
    elif cls == "slope":
        cfg = {"type": "slope", "geoms": [{"type": "plane", "normal": list(slope_normal(8.0, 0.0)), "mu": 1.0, "region": 0}],
               "regions": [{"label": 4, "mu": 1.0, "stiffness": 7.7e8, "slope": float(np.deg2rad(8.0))}]}
    m, info, a, _ = walk_run(cfg, t0 + dur + 0.2)
    style(m, info)
    sel = (a["t"] >= t0) & (a["t"] <= t0 + dur)
    strip = StripPanel(640, 110, ["left foot", "right foot"], t0 + dur, title="gait phase (ground-truth contact)",
                       legend=[("stance", "#3fbf6f"), ("swing", "#6f7f95")])
    strip.ax.set_xlim(t0, t0 + dur)
    tt = a["t"]
    for k in range(2):
        col = np.where(a["contact"][:, k], "#3fbf6f", "#6f7f95")
        strip.add_segments(k, tt - 0.02, tt, col)
    names = {"hard_grip": "hard-grip (mu 1.0)", "hard_slip": "hard-slip (mu 0.2)", "slope": "slope 8 deg uphill"}
    look = smooth(a["qpos"][:, :3], 10)

    def cam(i):
        return camera(look[i] - [0, 0, 0.47], 0.75, 135, -12)
    path = os.path.join(out, f"closeup_{cls}.mp4")
    _render_side_by_side(m, a, path, f"G113 on {names[cls]}, sole v2 nominal", info["point_layout"], cam, speed=speed, t0=t0,
                         t1=t0 + dur, strip=strip)
    return path, a


# ------------------------------------------------------------------------------------------------ course
ROWS = ("truth", "sole model", "proprioception only", "rule-based")


def course_video(run: dict, m, info, preds: dict, path, title="mixed course", max_t=None):
    """run: Tap arrays (+ 'win_t', 'win_truth' per window); preds: {row name: npz path or None}."""
    style(m, info)
    tmax = float(run["t"][-1] if max_t is None else max_t)
    strip = StripPanel(W, 270, list(ROWS), tmax, title="terrain class per 0.2 s window (1 s causal windows)",
                       legend=[(n, c) for n, c in zip(CLASS_NAMES, CLASS_HEX)] + [("not available", "#3a3d45")])
    data = {"truth": (run["win_t"], run["win_truth"])}
    for row in ROWS[1:]:
        p = preds.get(row)
        if p and os.path.exists(p):
            z = np.load(p)
            data[row] = (z["t_end"], z["terrain"])
    for k, row in enumerate(ROWS):
        if row in data:
            te, cl = data[row]
            cols = [CLASS_HEX[c] if c >= 0 else "#000000" for c in cl]
            strip.add_segments(k, te - 0.2, te, cols)
        else:
            strip.add_segments(k, [0.0], [tmax], ["#3a3d45"])
            strip.ax.text(tmax / 2, len(ROWS) - 1 - k, "not available", color="#c0c4cc", ha="center", va="center",
                          fontsize=9, zorder=6)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 450, 860)
    pan = SolePanel(info["point_layout"], 420, 450, "")
    look = smooth(run["qpos"][:, :3], 15)
    wr = Writer(path)
    n = int(round(tmax * FPS))
    for k in range(n):
        t = k / FPS
        i = max(0, int(np.searchsorted(run["t"], t, side="right")) - 1)
        d.qpos[:] = run["qpos"][i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera(look[i] - [0, 0, 0.2], 2.6, 120, -18))
        top = np.concatenate([r.render(), pan.draw(run["F"][i], f"{title}   t = {run['t'][i]:5.2f} s")], 1)
        wr.write(np.concatenate([top, strip.render(t)], 0))
    wr.close()
    return path
