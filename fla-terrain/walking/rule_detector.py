"""Rule-based terrain / support detector, no learning (FLA-0 v2 brief section 5 item 3, D14 question 5c).

Inputs: only what a controller has: joint_pos, joint_vel, imu_gyro, imu_gravity, cmd_vel, and (schema v0.3,
when present) joint_target and joint_torque. Never the sole, never ground-truth contact or labels at run time.
Outputs on the standard window grid (1.0 s windows, stride 0.2 s, prediction for the window's last frame, windows
ending after a fall dropped; same enumeration as fla_terrain.models.dataset): terrain class (0-5) and support
state per foot (0 airborne, 1 partial, 2 full), as `predict_episode(...) -> dict(terrain, sup_L, sup_R, t_end)`.

Signals (per 50 Hz frame, from forward kinematics of the K1 MJCF; base orientation from projected gravity, base
yaw 0, base linear velocity unknown = 0, base angular velocity = gyro):
  foot_z      sole-centre height of each foot in the gravity-aligned base frame
  foot_v      foot velocity relative to the base, gravity-aligned frame (horizontal part used)
  foot_tilt   angle between the foot's z axis and vertical [rad]
  contact     estimated: foot within H_C of the lower foot AND |foot vertical speed| < V_C
              (+ when joint_torque exists: |ankle pitch torque| > TAU_C)
  dslip       in double support, |v_L - v_R| horizontal: the unknown base velocity cancels, so a non-zero
              difference means one foot slides (relative to the other) [m/s]
  dz_ds       when both feet are vertically still (|v_z| < V_C, any height), |z_L - z_R|: height difference
              between the two footholds [m]
  track_err   when joint_target exists: mean |joint_target - joint_pos| over leg joints [rad]

Window features (over the 50 frames of the window) and the decision list, first match wins:
  1 hard_slip  if  median dslip over double-support frames          > T_SLIP
               or  (joint_target present) mean track_err          > T_TRACK
  2 slope      if  median stance-foot tilt                         > T_SLOPE
  3 edge       if  max dz_ds                                       > T_EDGE
               or  fraction of stance frames predicted partial     > T_PFRAC
  4 rough      if  std of stance-foot tilt                         > T_ROUGH
  5 hard_grip  otherwise.
  soft is never output (no rule found that separates it without the sole): soft windows count as misses.
Support per foot at the window's last frame: airborne if not in contact; partial if in contact and the foot
tilt differs from the window's median stance tilt by more than T_PART (the foot rolls over an edge); else full.

Thresholds: fitted once on the TRAIN split by `fit(...)` (each threshold is a 1-D search maximising the F1 of
the terrain macro-F1 over the classes present on train windows, rules applied in the order above; contact thresholds H_C, V_C maximise
agreement of airborne vs not with label_support on train frames). They are stored in a json; the defaults below
are only placeholders for running without a fit. Command: scripts/rule_detector_fit.py.
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

W, STRIDE = 50, 10
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
DEFAULTS = {"H_C": 0.015, "V_C": 0.15, "TAU_C": 5.0, "T_SLIP": 0.08, "T_TRACK": 0.1, "T_SLOPE": 0.06,
            "T_EDGE": 0.03, "T_PFRAC": 0.2, "T_ROUGH": 0.03, "T_PART": 0.08}
SOLE_OFFSET = np.array([0.026, 0.0, -0.038])   # sole centre in the ankle_roll body frame (foot box centre, bottom)


class Kinematics:
    def __init__(self, model=None):
        if model is None:
            from .k1_velocity import load_k1_model
            model = load_k1_model()
        self.m = model
        self.d = mujoco.MjData(model)
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.jnames = [model.joint(i).name for i in range(1, model.njnt)]
        self.jp = np.zeros((3, model.nv))

    def episode(self, q, qd, gyro, grav):
        """Per-frame foot_z [T,2], foot_v [T,2,3] (gravity-aligned base frame), foot_tilt [T,2]."""
        m, d = self.m, self.d
        T = len(q)
        z = np.zeros((T, 2))
        v = np.zeros((T, 2, 3))
        tilt = np.zeros((T, 2))
        for t in range(T):
            g = grav[t] / max(np.linalg.norm(grav[t]), 1e-9)       # gravity direction in the base frame
            # base orientation R (base -> world) with yaw 0 such that R @ g = (0, 0, -1)
            zb = -g                                                 # world z expressed in base frame
            xb = np.array([1.0, 0, 0]) - zb[0] * zb
            xb /= max(np.linalg.norm(xb), 1e-9)
            yb = np.cross(zb, xb)
            Rwb = np.stack([xb, yb, zb])                             # rows: world axes in base coords
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, Rwb.reshape(-1))             # base -> world rotation
            d.qpos[:3] = 0.0
            d.qpos[3:7] = quat
            d.qpos[7:] = q[t]
            d.qvel[:3] = 0.0
            d.qvel[3:6] = gyro[t]
            d.qvel[6:] = qd[t]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            for k, b in enumerate(self.feet):
                R = d.xmat[b].reshape(3, 3)
                p = d.xpos[b] + R @ SOLE_OFFSET
                mujoco.mj_jac(m, d, self.jp, None, p, b)
                v[t, k] = self.jp @ d.qvel
                z[t, k] = p[2]
                tilt[t, k] = np.arccos(np.clip(R[2, 2], -1, 1))
        return z, v, tilt


def frame_signals(arr: dict, kin: Kinematics, th: dict, jnames=None):
    z, v, tilt = kin.episode(arr["joint_pos"], arr["joint_vel"], arr["imu_gyro"], arr["imu_gravity"])
    low = z.min(1, keepdims=True)
    contact = (z - low < th["H_C"]) & (np.abs(v[:, :, 2]) < th["V_C"])
    if "joint_torque" in arr and jnames is not None:
        ia = [jnames.index(f"{s}_ankle_pitch_joint") for s in ("left", "right")]
        contact &= np.abs(arr["joint_torque"][:, ia]) > th["TAU_C"]
    ds = contact.all(1)
    still = (np.abs(v[:, :, 2]) < th["V_C"]).all(1)          # both feet vertically still, any height
    dslip = np.where(ds, np.linalg.norm(v[:, 0, :2] - v[:, 1, :2], axis=1), np.nan)
    dz = np.where(still, np.abs(z[:, 0] - z[:, 1]), np.nan)
    track = None
    if "joint_target" in arr and jnames is not None:
        il = [i for i, n in enumerate(jnames) if any(k in n for k in LEG)]
        track = np.abs(arr["joint_target"][:, il] - arr["joint_pos"][:, il]).mean(1)
    return {"z": z, "v": v, "tilt": tilt, "contact": contact, "dslip": dslip, "dz": dz, "track": track}


def window_starts(T, fall_time=None, t=None):
    st = np.arange(0, T - W + 1, STRIDE) if T >= W else np.zeros(0, int)
    if fall_time is not None and t is not None:
        st = st[t[st + W - 1] <= fall_time]
    return st


def window_features(sig, s):
    sl = slice(s, s + W)
    c = sig["contact"][sl]
    tilt = sig["tilt"][sl]
    st_tilt = tilt[c]
    med = float(np.median(st_tilt)) if st_tilt.size else 0.0
    f = {
        "slip": float(np.nanmedian(sig["dslip"][sl])) if np.isfinite(sig["dslip"][sl]).any() else 0.0,
        "tilt_med": med,
        "tilt_std": float(np.std(st_tilt)) if st_tilt.size > 2 else 0.0,
        "dz_max": float(np.nanmax(sig["dz"][sl])) if np.isfinite(sig["dz"][sl]).any() else 0.0,
        "track": float(np.mean(sig["track"][sl])) if sig["track"] is not None else None,
        "_dev": np.abs(tilt - med), "_c": c,
    }
    return f


def decide(f, th):
    pfrac = float(np.mean((f["_dev"] > th["T_PART"])[f["_c"]])) if f["_c"].any() else 0.0
    if f["slip"] > th["T_SLIP"] or (f["track"] is not None and f["track"] > th["T_TRACK"]):
        cls = 1
    elif f["tilt_med"] > th["T_SLOPE"]:
        cls = 4
    elif f["dz_max"] > th["T_EDGE"] or pfrac > th["T_PFRAC"]:
        cls = 5
    elif f["tilt_std"] > th["T_ROUGH"]:
        cls = 3
    else:
        cls = 0
    c_last, dev_last = f["_c"][-1], f["_dev"][-1]
    sup = [0 if not c_last[k] else (1 if dev_last[k] > th["T_PART"] else 2) for k in (0, 1)]
    return cls, sup


def predict_episode(arr: dict, meta: dict, th: dict = None, kin: Kinematics = None):
    th = {**DEFAULTS, **(th or {})}
    kin = kin or Kinematics()
    sig = frame_signals(arr, kin, th, meta.get("joint_names"))
    st = window_starts(len(arr["t"]), meta.get("fall_time"), arr["t"])
    out = {"terrain": [], "sup_L": [], "sup_R": [], "t_end": arr["t"][st + W - 1] if len(st) else np.zeros(0)}
    for s in st:
        cls, sup = decide(window_features(sig, s), th)
        out["terrain"].append(cls)
        out["sup_L"].append(sup[0])
        out["sup_R"].append(sup[1])
    return {k: np.asarray(v) for k, v in out.items()}


def load_thresholds(path):
    return {**DEFAULTS, **json.load(open(path))["thresholds"]}
