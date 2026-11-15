# walking/ — K1 velocity walking motion source (the independent evaluation)

A tool for data generation, not a contribution. k1_gym velocity policies (trained in Isaac Lab,
flat ground only) replayed in MuJoCo, headless, numpy only (no torch, no viewer).

## Use (MotionSource interface)

```python
import mujoco, numpy as np
from walking import K1VelocityPolicy, load_k1_model

model = load_k1_model(friction=1.0)        # or fla_terrain.sim.build_model(...)
data = mujoco.MjData(model)
src = K1VelocityPolicy("G113_m2200")       # name -> metadata motion_source
src.reset(model, data, np.random.default_rng(0))      # optional: base_pos=(x, y, ground_z + 0.58), base_yaw=...
while not src.fallen(model, data):
    src.pre_step(model, data, (0.4, 0.0, 0.0))        # cmd_vel = (vx m/s, vy m/s, wz rad/s), body frame
    mujoco.mj_step(model, data)
```

- `physics_dt = 0.002` (training value). The policy runs every `round(0.02 / timestep)` calls of `pre_step`
  (50 Hz); PD torques are written to `data.ctrl` on every call. Call `reset` before the first `pre_step`.
- Requirements on the model: the joint and actuator names of `b_ext/k1_play_kit/assets/K1_22dof.xml`
  (motor actuators, ctrl = torque), one free joint, bodies `trunk`, `left/right_ankle_roll_link`.
  Extra geoms (sole zones), a different ground, other bodies are fine.
- Contact: the base MJCF has floor `condim=1` and friction 0.4. Use condim >= 3. MuJoCo combines friction
  with **max(geom1, geom2)**: for a low-friction ground, lower the robot's geoms (sole zones) as well.
- `fallen`: projected gravity z > -0.5 (tilt > 60 deg) or trunk less than 0.25 m above the lower foot body
  (standing ~0.49 m). Works on raised / sloped ground.
- Public state after each policy tick: `src.action`, `src.target`, `src.gait_phase`, `src.gait_freq`, `src.cmd`.

## Policy interface (identical to b_ext/k1_deploy/k1_policy.py)

- obs 61 = projected gravity (3), body angular velocity (3), command vx vy wz (3), gait clock
  [cos, sin](2 pi phase) * gate (2), q - q0 (16), 0.1 * qd (16), last action (16), heading error [sin, cos] (2).
- Gait clock f = a + b|v_xy| + c|wz|, clipped; f = 0 (gate 0, standing) if |v| < 0.05 and |wz| < 0.05.
  G113: (0.75, 0.45, 0.7) in [1.0, 1.6] Hz. G80: (0.8, 0.8, 0.8) in [0.8, 2.2] Hz.
- Heading target integrates the commanded wz from the yaw at the first tick.
- 16 actions, Isaac breadth-first order (see `POLICY_JOINTS`); target = q0 + scale * clip(a, -1, 1),
  scale 1 (shoulders 0.5). q0: hip_pitch -0.2, knee 0.4, ankle_pitch -0.2, shoulder_pitch -0.1,
  shoulder_roll -+1.371. PD hip 140/3.5, knee 170/4.25, ankle 30/0.6, shoulder 20/1.0, torque-clipped to
  the MJCF forcerange. Held: elbow_yaw -+0.35, elbow_pitch 0, head 0 (PD 60/2, clip 14 Nm).

## Checkpoints

| name | file here | original | notes |
|---|---|---|---|
| `G113_m2200` (default) | `checkpoints/G113_m2200.npz` | `b_ext/k1_deploy/policies/B_G113_m2200_soft.pt` | deployed slow policy |
| `G80_m4200` | `checkpoints/G80_m4200.npz` | `b_ext/k1_deploy/policies/C_G80_m4200_fallback.pt` | fast fallback |

npz = TorchScript weights (ELU MLP 61-256-128-128-16), max |diff| vs TorchScript 8e-7.

## Evaluate

```
PY=python3
$PY scripts/walk_eval_mujoco.py flat  --policy G113_m2200 --episodes 64   # B1
$PY scripts/walk_eval_mujoco.py sweep --policy G113_m2200 --episodes 32   # B3 terrain sweep
$PY -m pytest -q walking/tests
```
Results: `/path/to/fla-terrain/b_runs/eval/*.json`. Eval-only terrains: `walking/eval_terrains.py`
(the dataset terrains are the training implementation's `fla_terrain.sim`).

## Command schedules (`walking/scenarios.py`)

`sched = make_schedule(kind, rng, duration=20.0)`, then `cmd_vel = sched(t)`; deterministic given rng,
`sched.params` for metadata, 1 s of zero command first. Kinds: `steady` (vx 0.2-0.5), `speed_change`
(new vx every 3-6 s, 0.5 s ramps), `turn` (vx 0.25-0.4, alternating straight / |wz| 0.2-0.5 every 2-4 s),
`stop_go` (walk 3-5 s, stand 2 s), `straight` (vx 0.25-0.45, wz 0: heading hold, path = spawn yaw line).
No sideways commands: both policies under-track vy.

Tracking on flat ground (`scripts/walk_eval_scenarios.py`, 16 eps x 20 s each, 1 s averaged, 0.5 s lag):
0 falls everywhere; vx RMS error G113 0.012-0.021 m/s (stop_go 0.062), G80 0.016-0.038;
wz RMS error G113 0.07 (0.12 when turning), G80 0.08 (0.13 when turning).

## SONIC-K1 clip playback (`walking/sonic_k1.py`, MotionSource `SonicK1`)

Second gait source: SONIC-K1 universal-token tracking policy (it50000, g1 encoder -> FSQ ->
g1_dyn decoder) plays 291 retargeted human walking clips (forward, 180-360 deg turns, random direction;
from `sonic-k1/data/sonic_motionlib/unseen1985_10s`). `cmd_vel` is ignored; the motion comes from the clip.

```python
from walking.sonic_k1 import SonicK1
src = SonicK1()                       # physics_dt 0.002; ~1.4 ms per policy call (12 M params, numpy)
src.reset(model, data, rng)           # random clip, RSI from frame 0; clip="<name>" to pick one
src.pre_step(model, data)             # every physics step; src.clip_log = [(t, clip, frame), ...]
```

- Assets (not in git, 113 MB): `b_runs/sonic/{sonic_k1_it50000.npz, clips/*.npz}`, regenerate with
  `env_isaaclab/bin/python scripts/walk_sonic_export.py` (uses SONIC's own `eval_k1_mujoco.py` +
  MotionLibRobot; env_isaaclab only used, not modified). `FLA_SONIC_DIR` overrides the location.
- Model mapping: SONIC's `k1.xml` and `k1_play_kit/K1_22dof.xml` have the same 23 joints in the same order,
  same actuators, bodies, masses, inertias, armature and foot box (play kit adds 2 massless camera bodies).
  Policy order = IsaacLab breadth-first (22 joints incl. head and arms, all actioned); mapped by joint name.
  PD from SONIC `robots/k1.py`: kp = armature-based (hip pitch 30, knee 60, ankle 36, arms 4), target =
  q0 + 0.25 * effort / kp * clip(a, +-20), torque clipped to training effort; the MJCF forcerange still
  applies (hip_roll 43 Nm vs 76 in training; SONIC's own eval with Booster caps 35 Nm: 99 % success).
- Clips are <= 10 s; for longer episodes playback chains to the closest-pose frame (leg pos + 0.1 vel) of
  another clip, reference yaw re-anchored to the robot heading (only orientation enters the policy).
- Parity vs SONIC torch actor on SONIC's own setup: action diff 5e-7 at step 0, qpos diff 4 mm after 3 s.
- `scripts/walk_eval_sonic.py`: 32 eps x 20 s per condition, 0 falls everywhere (see NOTES_B.md A1).

SONIC helpers for data generation: `ClipLibrary().straight()` (164 clips that walk straight for >= 1 m) to aim at
edges/strips with `base_yaw`; `src.reference()` -> {clip, frame, root_vel} and `src.clip_log` for metadata;
cmd_vel for SONIC episodes = zeros. Throughput ~1.1 sim-s/wall-s per process (actor is memory-bound, 15 ms/call
single-threaded); see NOTES_B.md B8.
