"""K1 velocity-command walking policy as a MotionSource (MuJoCo, headless, numpy only).

Reproduces the observation / action pipeline of k1_gym policies (G113, G80) exactly as
`b_ext/k1_deploy/k1_policy.py` and `b_ext/k1_play_kit/play.py` do:

  obs (61) = gravity_b(3), ang_vel_b(3), cmd vx,vy,wz(3), gait clock [cos,sin]*gate(2),
             q - q0 (16), qd * 0.1 (16), last_action (16), heading error [sin,cos] (2)
  action (16) -> target = q0 + scale * clip(a, -1, 1)  (scale 1.0, shoulders 0.5)
  PD torque at the physics rate (500 Hz), policy at 50 Hz (decimation = 0.02 / timestep).

Works on any MjModel that keeps the base K1 MJCF joint and actuator names (motor actuators,
ctrl = torque). Extra geoms / a different ground are fine.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.environ.get("FLA_B_EXT", "/path/to/fla-terrain/b_ext")
K1_MJCF = os.path.join(EXT, "k1_play_kit", "assets", "K1_22dof.xml")

# Isaac Lab breadth-first order: the policy's joint order. Do not sort.
POLICY_JOINTS = (
    "aaleft_shoulder_pitch_joint", "aaright_shoulder_pitch_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_pitch_joint", "right_knee_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
)
# Held (not actioned) joints and their hold pose (k1_deploy convention: bent elbows).
HELD_JOINTS = {
    "aahead_yaw_joint": 0.0, "aahead_pitch_joint": 0.0,
    "left_elbow_pitch_joint": 0.0, "right_elbow_pitch_joint": 0.0,
    "left_elbow_yaw_joint": -0.35, "right_elbow_yaw_joint": 0.35,
}
HELD_KP, HELD_KD, HELD_TAU = 60.0, 2.0, 14.0  # play.py hold controller

DEFAULT_ANGLES = {  # substring -> angle; last match wins (same rule as envs/k1.py)
    "hip_pitch": -0.2, "knee_pitch": 0.4, "ankle_pitch": -0.2, "shoulder_pitch": -0.1,
    "left_shoulder_roll": -1.371, "right_shoulder_roll": 1.371,
}
STIFFNESS = {"hip": 140.0, "knee": 170.0, "ankle": 30.0, "shoulder": 20.0}
DAMPING = {"hip": 3.5, "knee": 4.25, "ankle": 0.6, "shoulder": 1.0}
ACTION_SCALE = {"hip": 1.0, "knee": 1.0, "ankle": 1.0, "shoulder": 0.5}
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
BASE_BODY = "trunk"


@dataclass(frozen=True)
class PolicySpec:
    name: str
    weights: str            # npz in walking/checkpoints (exported from k1_deploy TorchScript)
    cadence: tuple          # f = a + b*|v_xy| + c*|wz|
    cadence_range: tuple    # clip
    source: str             # original file


POLICIES = {
    "G113_m2200": PolicySpec("G113_m2200", "G113_m2200.npz", (0.75, 0.45, 0.7), (1.0, 1.6),
                             "b_ext/k1_deploy/policies/B_G113_m2200_soft.pt"),
    "G80_m4200": PolicySpec("G80_m4200", "G80_m4200.npz", (0.8, 0.8, 0.8), (0.8, 2.2),
                            "b_ext/k1_deploy/policies/C_G80_m4200_fallback.pt"),
}


K1V2_DIR = os.environ.get("FLA_K1V2_DIR", "/path/to/fla-terrain/b_runs/k1v2/policies")


def policy_spec(name):
    """Known policy, or a FLA-0 v2 checkpoint K1V2_* exported to K1V2_DIR/<name>.npz (G113 cadence law:
    the v2 recipe keeps G113's commands and gait clock)."""
    if name in POLICIES:
        return POLICIES[name]
    path = os.path.join(K1V2_DIR, f"{name}.npz")
    if name.startswith("K1V2") and os.path.exists(path):
        return PolicySpec(name, path, (0.75, 0.45, 0.7), (1.0, 1.6), path)
    raise KeyError(f"unknown policy {name!r}")


def _by_substring(table, name, default=None):
    out = default
    for k, v in table.items():
        if k in name:
            out = v
    if out is None:
        raise KeyError(f"no entry for joint {name}")
    return out


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


class MLPActor:
    def __init__(self, path):
        w = np.load(path)
        self.layers = [(w[f"{i}_weight"].astype(np.float64), w[f"{i}_bias"].astype(np.float64))
                       for i in (0, 2, 4, 6)]

    def __call__(self, x):
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = _elu(x)
        return x


def quat_rotate_inverse(q, v):
    """Rotate world vector v into the frame of quaternion q (w, x, y, z)."""
    w, xyz = q[0], q[1:4]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(xyz, v) * w * 2.0
    c = xyz * np.dot(xyz, v) * 2.0
    return a - b + c


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def load_k1_model(friction=1.0, timestep=0.002, path=K1_MJCF):
    """Base K1 MJCF with the two play.py contact fixes (condim>=3, friction set on ALL geoms).

    MuJoCo combines geom friction with max(), so a low-friction ground needs the robot geoms
    lowered too; this helper sets every geom's sliding friction to `friction`.
    """
    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = timestep
    model.geom_condim[:] = np.maximum(model.geom_condim, 3)
    model.geom_friction[:, 0] = friction
    return model


class K1VelocityPolicy:
    """MotionSource: K1 velocity-command walking (k1_gym policy)."""

    physics_dt = 0.002
    policy_dt = 0.02

    def __init__(self, policy="G113_m2200", init_noise=0.05, base_height=0.58,
                 fall_gz=-0.5, fall_height=0.25, hold_elbows=True):
        self.spec = policy_spec(policy)
        self.name = f"k1_velocity_{self.spec.name}"
        self.actor = MLPActor(os.path.join(HERE, "checkpoints", self.spec.weights))
        self.init_noise = init_noise
        self.base_height = base_height
        self.fall_gz = fall_gz
        self.fall_height = fall_height
        self.held_pose = dict(HELD_JOINTS)
        if not hold_elbows:
            self.held_pose.update(left_elbow_yaw_joint=0.0, right_elbow_yaw_joint=0.0)
        n = POLICY_JOINTS
        self.q0 = np.array([_by_substring(DEFAULT_ANGLES, j, 0.0) for j in n])
        self.kp = np.array([_by_substring(STIFFNESS, j) for j in n])
        self.kd = np.array([_by_substring(DAMPING, j) for j in n])
        self.scale = np.array([_by_substring(ACTION_SCALE, j) for j in n])
        self._bound = None
        # public state, useful for logging
        self.action = np.zeros(16)
        self.target = self.q0.copy()
        self.gait_phase = 0.0
        self.gait_freq = 0.0
        self.cmd = np.zeros(3)

    # ---- model binding -------------------------------------------------------------------
    def _bind(self, model):
        if self._bound is model:
            return
        J = mujoco.mjtObj.mjOBJ_JOINT
        A = mujoco.mjtObj.mjOBJ_ACTUATOR

        def ids(names):
            j = [mujoco.mj_name2id(model, J, n) for n in names]
            a = [mujoco.mj_name2id(model, A, n) for n in names]
            missing = [n for n, x, y in zip(names, j, a) if x < 0 or y < 0]
            if missing:
                raise ValueError(f"model lacks joints/actuators {missing}")
            return (np.array([model.jnt_qposadr[i] for i in j]),
                    np.array([model.jnt_dofadr[i] for i in j]), np.array(a))

        self.qadr, self.dadr, self.aid = ids(POLICY_JOINTS)
        held = list(self.held_pose)
        self.hqadr, self.hdadr, self.haid = ids(held)
        self.hq0 = np.array([self.held_pose[n] for n in held])
        self.tau_lim = model.actuator_forcerange[self.aid, 1].copy()
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        self.foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
        free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        self.free_q = model.jnt_qposadr[free[0]]
        self.free_d = model.jnt_dofadr[free[0]]
        self.decimation = max(1, int(round(self.policy_dt / model.opt.timestep)))
        if abs(model.opt.timestep - self.physics_dt) > 1e-9:
            # allowed (policy rate is kept), but training used 0.002
            pass
        self._bound = model

    # ---- MotionSource API ----------------------------------------------------------------
    def reset(self, model, data, rng=None, base_pos=None, base_yaw=0.0):
        """Default crouch pose at (x, y, z). base_pos None -> (0, 0, base_height).

        For raised ground pass base_pos with z = ground height + base_height.
        Writes qpos/qvel/ctrl and calls mj_forward. Does not reset other bodies.
        """
        self._bind(model)
        rng = np.random.default_rng() if rng is None else rng
        f = self.free_q
        pos = (0.0, 0.0, self.base_height) if base_pos is None else base_pos
        data.qpos[f:f + 3] = pos
        data.qpos[f + 3:f + 7] = (math.cos(base_yaw / 2), 0.0, 0.0, math.sin(base_yaw / 2))
        data.qvel[self.free_d:self.free_d + 6] = 0.0
        data.qpos[self.qadr] = self.q0 + rng.normal(0.0, self.init_noise, 16)
        data.qvel[self.dadr] = 0.0
        data.qpos[self.hqadr] = self.hq0
        data.qvel[self.hdadr] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        self.action = np.zeros(16)
        self.target = self.q0.copy()
        self.gait_phase = 0.0
        self.gait_freq = 0.0
        self.target_heading = None
        self._k = 0

    def _frequency(self, cmd):
        v = math.hypot(cmd[0], cmd[1])
        if v < 0.05 and abs(cmd[2]) < 0.05:
            return 0.0
        a, b, c = self.spec.cadence
        return float(np.clip(a + b * v + c * abs(cmd[2]), *self.spec.cadence_range))

    def observe(self, data, cmd):
        f, d = self.free_q, self.free_d
        quat = data.qpos[f + 3:f + 7]
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        omega = data.qvel[d + 3:d + 6]  # free joint angular velocity is already in body frame
        self.gait_freq = self._frequency(cmd)
        self.gait_phase = (self.gait_phase + self.policy_dt * self.gait_freq) % 1.0
        gate = 1.0 if self.gait_freq > 1e-8 else 0.0
        yaw = yaw_of(quat)
        if self.target_heading is None:
            self.target_heading = yaw
        self.target_heading = (self.target_heading + self.policy_dt * cmd[2] + math.pi) % (2 * math.pi) - math.pi
        herr = (yaw - self.target_heading + math.pi) % (2 * math.pi) - math.pi
        ph = 2 * math.pi * self.gait_phase
        return np.concatenate([
            grav, omega, cmd, [math.cos(ph) * gate, math.sin(ph) * gate],
            data.qpos[self.qadr] - self.q0, data.qvel[self.dadr] * 0.1, self.action,
            [math.sin(herr), math.cos(herr)],
        ])

    def pre_step(self, model, data, cmd_vel=None):
        if self._bound is not model:
            self._bind(model)
        if self._k % self.decimation == 0:
            self.cmd = np.zeros(3) if cmd_vel is None else np.asarray(cmd_vel, dtype=np.float64)[:3]
            obs = self.observe(data, self.cmd)
            self.action = np.clip(self.actor(obs), -1.0, 1.0)
            self.target = self.q0 + self.scale * self.action
        self._k += 1
        tau = self.kp * (self.target - data.qpos[self.qadr]) - self.kd * data.qvel[self.dadr]
        data.ctrl[self.aid] = np.clip(tau, -self.tau_lim, self.tau_lim)
        htau = HELD_KP * (self.hq0 - data.qpos[self.hqadr]) - HELD_KD * data.qvel[self.hdadr]
        data.ctrl[self.haid] = np.clip(htau, -HELD_TAU, HELD_TAU)

    def base_height_above_feet(self, data):
        return float(data.xpos[self.base_id, 2] - min(data.xpos[i, 2] for i in self.foot_ids))

    def fallen(self, model, data):
        self._bind(model)
        q = data.qpos[self.free_q + 3:self.free_q + 7]
        gz = quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))[2]
        return bool(gz > self.fall_gz or self.base_height_above_feet(data) < self.fall_height)


MotionSource = K1VelocityPolicy  # convenience alias
