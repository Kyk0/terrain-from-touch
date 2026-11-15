"""SONIC-K1 motion-tracking policy as a MotionSource (MuJoCo, numpy only, headless).

Plays retargeted human walking clips through SONIC-K1 universal-token policy (it50000, g1
encoder -> FSQ -> g1_dyn decoder), reproducing gear_sonic/scripts/eval_k1_mujoco.py of the SONIC repo:

  actor_obs (720) = 10-step history (oldest first, primed with the first sample) of
                    base_ang_vel_b (3), q - q0 (22), qd (22), last action (22), gravity_b (3), stacked per term
  tokenizer (500) = 10 future reference frames (every 5th 50 Hz frame = 1 s horizon):
                    [dof_pos (10x22) | dof_vel (10x22)] flattened then reshaped (10, 44), concatenated per row
                    with the 6D relative orientation mat(conj(q_robot) * q_ref)[:, :2]
  action (22, IsaacLab order) -> target = q0 + 0.25 * effort / kp * clip(a, -20, 20); explicit PD every
  physics step, torque clipped to the training effort limits (the MJCF forcerange still applies on top).

Only the reference *orientation* and joint trajectory enter the policy, not the root position, so clips
are chained: when a clip is about to end, playback jumps to the closest-pose frame (legs, pos + vel) of
another clip, with the reference yaw re-anchored to the robot's current heading.

Differences from SONIC's own MuJoCo eval, by design (documented in walking/README.md):
  * the robot model is the k1_play_kit MJCF (= SONIC k1.xml dynamically: same joints, bodies, masses,
    inertias, armature; it only adds two massless camera bodies); actuator forcerange stays as in the
    MJCF (hip_roll 43 Nm vs 76 in training);
  * physics dt default 0.002 (same as the velocity policies and the sole sensor); SONIC trained at
    0.005 s (decimation 4). Tested both: identical fall rate, speed and step rate. Policy rate stays 50 Hz.
"""

from __future__ import annotations

import glob
import math
import os

import mujoco
import numpy as np

SONIC_DIR = os.environ.get("FLA_SONIC_DIR", "/path/to/fla-terrain/b_runs/sonic")
WEIGHTS = os.path.join(SONIC_DIR, "sonic_k1_it50000.npz")
CLIPS = os.path.join(SONIC_DIR, "clips")
HIST = 10
ACTION_CLIP = 20.0


# ---- quaternion helpers (wxyz), copied from eval_k1_mujoco.py ------------------------------------
def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    u, w = q[..., 1:], q[..., :1]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_apply_inv(q, v):
    return quat_apply(quat_conj(q), v)


def quat_to_mat(q):
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ---- actor ------------------------------------------------------------------------------------------
def _silu(x):
    return x / (1.0 + np.exp(-x))


class SonicActor:
    def __init__(self, path=WEIGHTS):
        w = np.load(path)
        self.w = w

        def mlp(prefix):
            idx = sorted({int(k[len(prefix):].split("_")[0]) for k in w.files
                          if k.startswith(prefix) and k.endswith("_weight")})
            return [(w[f"{prefix}{i}_weight"].astype(np.float32), w[f"{prefix}{i}_bias"].astype(np.float32))
                    for i in idx]

        self.enc, self.dec = mlp("encoder_"), mlp("decoder_")
        self.n_tok, self.tok_dim = int(w["max_num_tokens"]), int(w["num_fsq_levels"])
        L = float(w["fsq_levels"])
        self.half_l = (L - 1) * (1 + 1e-3) / 2
        self.offset = 0.5 if int(L) % 2 == 0 else 0.0
        self.shift = math.atanh(self.offset / self.half_l)
        self.half_width = float(int(L) // 2)

    @staticmethod
    def _run(layers, x):
        for i, (W, b) in enumerate(layers):
            x = W @ x + b
            if i < len(layers) - 1:
                x = _silu(x)
        return x

    def fsq(self, z):
        return np.round(np.tanh(z + self.shift) * self.half_l - self.offset) / self.half_width

    def __call__(self, prop, tok):
        latent = self._run(self.enc, np.asarray(tok, np.float32))
        tokens = self.fsq(latent.astype(np.float32)).astype(np.float32)
        return self._run(self.dec, np.concatenate([tokens, np.asarray(prop, np.float32)]))


# ---- clips --------------------------------------------------------------------------------------------
class ClipLibrary:
    """Reference clips at 50 Hz (IsaacLab joint order), exported by scripts/walk_sonic_export.py."""

    def __init__(self, clip_dir=CLIPS, names=None):
        files = sorted(glob.glob(os.path.join(clip_dir, "*.npz")))
        if names is not None:
            files = [f for f in files if os.path.basename(f)[:-4] in set(names)]
        if not files:
            raise FileNotFoundError(f"no clips in {clip_dir}; run scripts/walk_sonic_export.py")
        self.names = [os.path.basename(f)[:-4] for f in files]
        self.clips = []
        for f in files:
            z = np.load(f)
            self.clips.append({k: z[k] for k in z.files})
        # pose index for clip chaining: legs (IsaacLab idx of hip/knee/ankle) pos + 0.1 * vel
        self._index = None

    def straight(self, min_fwd=1.0, t_fwd=3.0, max_lat=0.3, t_lat=6.0):
        """Names of clips that walk >= min_fwd m along their initial heading within t_fwd s and stay within
        max_lat m of that line for t_lat s (164 of 291 with the defaults). Use them, with base_yaw pointing
        at the feature, for edge / strip episodes: SonicK1(clip_names=lib.straight())."""
        out = []
        for n, c in zip(self.names, self.clips):
            y = yaw_of(c["root_quat"][0])
            R = np.array([[math.cos(y), math.sin(y)], [-math.sin(y), math.cos(y)]])
            p = (c["root_pos"][:, :2] - c["root_pos"][0, :2]) @ R.T
            T = int(c["T"])
            if p[min(int(t_fwd * 50), T - 1), 0] >= min_fwd and np.abs(p[:min(int(t_lat * 50), T), 1]).max() < max_lat:
                out.append(n)
        return out

    def build_index(self, leg_idx, runway):
        feats, ref = [], []
        for ci, c in enumerate(self.clips):
            T = int(c["T"])
            for f in range(0, max(0, T - runway)):
                ref.append((ci, f))
            n = max(0, T - runway)
            feats.append(np.concatenate([c["dof_pos"][:n, leg_idx], 0.1 * c["dof_vel"][:n, leg_idx]], 1))
        self._feats = np.concatenate(feats, 0)
        self._ref = np.array(ref)
        self._index = True

    def nearest(self, feat, exclude_clip, rng, k=8):
        d = np.sum((self._feats - feat) ** 2, 1)
        d[self._ref[:, 0] == exclude_clip] = np.inf
        best = np.argpartition(d, k)[:k]
        ci, f = self._ref[rng.choice(best)]
        return int(ci), int(f)


class SonicK1:
    """MotionSource: SONIC-K1 tracking retargeted walking clips (cmd_vel is ignored)."""

    policy_dt = 0.02

    def __init__(self, physics_dt=0.002, clips=None, clip_names=None, chain=True, runway_s=2.0,
                 fall_gz=-0.5, fall_height=0.25, weights=WEIGHTS):
        self.name = "sonic_k1_it50000"
        self.physics_dt = physics_dt
        self.actor = SonicActor(weights)
        w = self.actor.w
        self.il_joints = [str(j) for j in w["il_joints"]]
        self.kp, self.kd = w["kp"].astype(float), w["kd"].astype(float)
        self.effort, self.scale = w["effort"].astype(float), w["action_scale"].astype(float)
        self.q0 = w["default_pos"].astype(float)
        self.F, self.skip = int(w["num_future"]), int(w["frame_skips"])
        self.lib = clips or ClipLibrary(names=clip_names)
        self.chain, self.runway = chain, int(round(runway_s / self.policy_dt))
        self.leg_idx = [i for i, j in enumerate(self.il_joints) if any(s in j for s in ("hip", "knee", "ankle"))]
        if chain:
            self.lib.build_index(self.leg_idx, self.runway + self.F * self.skip)
        self.fall_gz, self.fall_height = fall_gz, fall_height
        self._bound = None
        self.clip_log = []          # (time, clip name, frame) of every clip start, for metadata

    def _bind(self, model):
        if self._bound is model:
            return
        J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
        jid = [mujoco.mj_name2id(model, J, n) for n in self.il_joints]
        aid = [mujoco.mj_name2id(model, A, n) for n in self.il_joints]
        if min(jid) < 0 or min(aid) < 0:
            raise ValueError("model lacks K1 joints/actuators")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.aid = np.array(aid)
        self.jrange = model.jnt_range[jid].copy()
        free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.fq, self.fd = model.jnt_qposadr[free], model.jnt_dofadr[free]
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                         for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.decimation = max(1, int(round(self.policy_dt / model.opt.timestep)))
        self._bound = model

    # ---- clip playback ------------------------------------------------------------------
    def _start_clip(self, ci, frame, robot_yaw, t):
        c = self.lib.clips[ci]
        self.ci, self.clip, self.frame = ci, c, frame
        # rotate the reference so its heading at `frame` equals the robot's heading
        self.yaw_off = yaw_quat(robot_yaw - yaw_of(c["root_quat"][frame]))
        self.clip_log.append((round(t, 3), self.lib.names[ci], frame))

    def ref_quat(self, idx):
        return quat_mul(self.yaw_off, self.clip["root_quat"][idx])

    def reference(self):
        """Current reference for logging: clip name, frame, and the clip root velocity (vx, vy in the
        reference heading frame, wz) -- the SONIC analogue of a velocity command (metadata, not cmd_vel)."""
        heading = yaw_quat(yaw_of(self.ref_quat(self.frame)))
        v = quat_apply_inv(heading, quat_apply(self.yaw_off, self.clip["root_lin_vel"][self.frame]))
        return {"clip": self.lib.names[self.ci], "frame": int(self.frame),
                "root_vel": [float(v[0]), float(v[1]), float(self.clip["root_ang_vel"][self.frame][2])]}

    # ---- MotionSource API -----------------------------------------------------------------
    def reset(self, model, data, rng=None, base_pos=None, base_yaw=0.0, clip=None):
        """RSI from a clip's first frame (joint pos/vel, root orientation and velocity).
        base_pos None -> (0, 0, clip root height); for raised ground add the ground height to z."""
        self._bind(model)
        self.rng = rng = np.random.default_rng() if rng is None else rng
        ci = int(rng.integers(len(self.lib.clips))) if clip is None else (
            self.lib.names.index(clip) if isinstance(clip, str) else int(clip))
        self.clip_log = []
        self._start_clip(ci, 0, base_yaw, 0.0)
        c = self.clip
        q = self.ref_quat(0)
        pos = (0.0, 0.0, float(c["root_pos"][0, 2])) if base_pos is None else base_pos
        f, d = self.fq, self.fd
        data.qpos[f:f + 3] = pos
        data.qpos[f + 3:f + 7] = q / np.linalg.norm(q)
        data.qvel[d:d + 3] = quat_apply(self.yaw_off, c["root_lin_vel"][0])
        data.qvel[d + 3:d + 6] = quat_apply_inv(q, quat_apply(self.yaw_off, c["root_ang_vel"][0]))
        data.qpos[self.qadr] = np.clip(c["dof_pos"][0], self.jrange[:, 0], self.jrange[:, 1])
        data.qvel[self.dadr] = c["dof_vel"][0]
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        self.hist = None
        self.last_action = np.zeros(22, np.float32)
        self.target = self.q0.copy()
        self._k = 0
        self.t = 0.0

    def _observe(self, data):
        q = data.qpos[self.fq + 3:self.fq + 7]
        terms = [data.qvel[self.fd + 3:self.fd + 6], data.qpos[self.qadr] - self.q0, data.qvel[self.dadr],
                 self.last_action, quat_apply_inv(q, np.array([0.0, 0.0, -1.0]))]
        terms = [np.asarray(x, np.float32) for x in terms]
        if self.hist is None:
            self.hist = [np.repeat(x[None], HIST, 0) for x in terms]
        else:
            self.hist = [np.concatenate([h[1:], x[None]], 0) for h, x in zip(self.hist, terms)]
        prop = np.concatenate([h.reshape(-1) for h in self.hist])
        T = int(self.clip["T"])
        idx = np.minimum(self.frame + np.arange(self.F) * self.skip, T - 1)
        jp, jv = self.clip["dof_pos"][idx], self.clip["dof_vel"][idx]
        cmd = np.concatenate([jp.reshape(-1), jv.reshape(-1)]).reshape(self.F, -1)
        rel = quat_mul(np.repeat(quat_conj(q)[None], self.F, 0), self.ref_quat(idx))
        ori = quat_to_mat(rel)[..., :2].reshape(self.F, 6)
        tok = np.concatenate([cmd, ori], -1).reshape(-1).astype(np.float32)
        return prop, tok

    def _advance(self, data):
        self.frame += 1
        T = int(self.clip["T"])
        if self.chain and self.frame >= T - 1 - self.F * self.skip:
            feat = np.concatenate([self.clip["dof_pos"][self.frame, self.leg_idx],
                                   0.1 * self.clip["dof_vel"][self.frame, self.leg_idx]])
            ci, f = self.lib.nearest(feat, self.ci, self.rng)
            robot_yaw = yaw_of(data.qpos[self.fq + 3:self.fq + 7])
            self._start_clip(ci, f, robot_yaw, self.t)
        elif self.frame >= T - 1:
            self.frame = T - 1      # no chaining: hold the last frame

    def pre_step(self, model, data, cmd_vel=None):
        if self._bound is not model:
            self._bind(model)
        if self._k % self.decimation == 0:
            if self._k > 0:
                self._advance(data)
            prop, tok = self._observe(data)
            a = np.clip(self.actor(prop, tok), -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
            self.last_action = a
            self.target = self.q0 + self.scale * a
        self._k += 1
        self.t = self._k * model.opt.timestep
        tau = self.kp * (self.target - data.qpos[self.qadr]) - self.kd * data.qvel[self.dadr]
        data.ctrl[self.aid] = np.clip(tau, -self.effort, self.effort)

    def base_height_above_feet(self, data):
        return float(data.xpos[self.base_id, 2] - min(data.xpos[i, 2] for i in self.foot_ids))

    def fallen(self, model, data):
        self._bind(model)
        q = data.qpos[self.fq + 3:self.fq + 7]
        gz = quat_apply_inv(q, np.array([0.0, 0.0, -1.0]))[2]
        return bool(gz > self.fall_gz or self.base_height_above_feet(data) < self.fall_height)
