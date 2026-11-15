"""Motion sources owned by the training implementation (stand / rig) and pose helpers.

MotionSource interface (shared with the independent evaluation): name, physics_dt, reset(model, data, rng),
pre_step(model, data, cmd_vel) writes data.ctrl before every mj_step, fallen(model, data).
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import k1


def pose_vector(info, pose: dict | None = None) -> np.ndarray:
    """Joint targets in actuator order from a {joint_name: angle} dict on top of DEFAULT_POSE."""
    p = dict(k1.DEFAULT_POSE)
    p.update(pose or {})
    return np.array([p.get(j, 0.0) for j in info["actuator_joints"]])


def gains(info):
    kp, kd = zip(*(k1.pd_gains(j) for j in info["actuator_joints"]))
    return np.array(kp), np.array(kd)


def sole_corner_points(model, data, info) -> np.ndarray:
    """World coordinates of the bottom corners of all zone boxes [N, 3]."""
    pts = []
    for s in "LR":
        for g in info["zone_geoms"][s]:
            hx, hy, hz = model.geom_size[g]
            R = data.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append(data.geom_xpos[g] + R @ np.array([sx * hx, sy * hy, -hz]))
    return np.array(pts)


def terrain_height_gap(model, data, info, pts) -> np.ndarray:
    """Vertical distance of points above the terrain (group-2 geoms), by ray casting from 0.5 m above."""
    from .build import TERRAIN_GROUP
    grp = np.zeros(6, np.uint8)
    grp[TERRAIN_GROUP] = 1
    gid = np.zeros(1, np.int32)
    d = np.full(len(pts), np.inf)
    for i, p in enumerate(pts):
        r = mujoco.mj_ray(model, data, p + [0.0, 0.0, 0.5], np.array([0.0, 0.0, -1.0]), grp, True, -1, gid)
        if r >= 0:
            d[i] = r - 0.5
    return d


def set_pose(model, data, info, pose: dict | None = None, base_xy=(0.0, 0.0), base_yaw: float = 0.0,
             clearance: float = 0.0005, base_z: float | None = None):
    """Set joint pose, upright base, and drop the base so the lowest sole corner is ``clearance`` above
    the terrain (planar terrains only). Zero velocities. Returns the base height."""
    mujoco.mj_resetData(model, data)
    data.qpos[info["act_qpos_adr"]] = pose_vector(info, pose)
    data.qpos[0:2] = base_xy
    data.qpos[2] = 1.0 if base_z is None else base_z
    data.qpos[3:7] = [np.cos(base_yaw / 2), 0, 0, np.sin(base_yaw / 2)]
    mujoco.mj_kinematics(model, data)
    if base_z is None:
        for _ in range(3):
            gap = terrain_height_gap(model, data, info, sole_corner_points(model, data, info)).min()
            data.qpos[2] -= gap - clearance
            mujoco.mj_kinematics(model, data)
    mujoco.mj_forward(model, data)
    return float(data.qpos[2])


class PDController:
    """Joint PD to fixed targets (actuators are torque motors; ctrl clipped to forcerange by MuJoCo)."""

    def __init__(self, model, info, pose: dict | None = None, kp_scale: dict | None = None):
        self.info = info
        self.q_target = pose_vector(info, pose)
        self.kp, self.kd = gains(info)
        for j, s in (kp_scale or {}).items():
            i = info["actuator_joints"].index(j)
            self.kp[i] *= s
            self.kd[i] *= s

    def __call__(self, model, data):
        q = data.qpos[self.info["act_qpos_adr"]]
        qd = data.qvel[self.info["act_qvel_adr"]]
        data.ctrl[:] = self.kp * (self.q_target - q) - self.kd * qd


class StandMotionSource:
    """K1 standing under PD at the default pose (optionally with a scripted target offset function)."""

    name = "stand_pd"
    physics_dt = 0.002

    # Pure PD at the policy gains cannot stand: ankle stiffness 2 x 30 Nm/rad < m g h ~ 96 Nm/rad
    # (inverted pendulum), the robot slowly tips forward. The stand source uses ankles x4 (120/2.4).
    ANKLE_SCALE = 4.0
    # Ankle strategy: ankle pitch/roll targets += KB * projected-gravity tilt + KD * trunk rate. Needed on soft
    # ground (tc >= 0.029 tips over within 4 s without it); values found by a sign/gain sweep.
    KB, KD = 1.0, 0.02

    def __init__(self, info, pose: dict | None = None, offset_fn=None, ankle_scale: float | None = None,
                 balance: bool = True):
        self.info = info
        self.balance = balance
        aj = info["actuator_joints"]
        self._ip = [aj.index(f"{s}_ankle_pitch_joint") for s in ("left", "right")]
        self._ir = [aj.index(f"{s}_ankle_roll_joint") for s in ("left", "right")]
        self.pose = pose
        self.ankle_scale = self.ANKLE_SCALE if ankle_scale is None else ankle_scale
        self.offset_fn = offset_fn  # (t) -> {joint: delta to the stand pose}
        self.pd = None

    def reset(self, model, data, rng=None):
        set_pose(model, data, self.info, self.pose)
        sc = {j: self.ankle_scale for j in self.info["actuator_joints"] if "ankle" in j}
        self.pd = PDController(model, self.info, self.pose, kp_scale=sc)
        self.q0 = self.pd.q_target.copy()

    def pre_step(self, model, data, cmd_vel=None):
        qt = self.q0.copy()
        if self.offset_fn is not None:
            off = self.offset_fn(data.time)  # {joint: delta} added to the stand pose
            qt += np.array([off.get(j, 0.0) for j in self.info["actuator_joints"]])
        if self.balance:
            R = data.xmat[self.info["base_body"]].reshape(3, 3)
            g = R.T @ np.array([0.0, 0.0, -1.0])
            w = data.qvel[3:6]
            qt[self._ip] += self.KB * g[0] + self.KD * w[1]
            qt[self._ir] += -self.KB * g[1] + self.KD * w[0]
        self.pd.q_target = qt
        self.pd(model, data)

    def fallen(self, model, data):
        gz = data.xmat[self.info["base_body"]].reshape(3, 3)[2, 2]
        return bool(data.qpos[2] < 0.35 or gz < 0.5)
