#!/usr/bin/env python
"""One-off export of SONIC-K1 (it50000, g1 encoder) to numpy, run with env_isaaclab's python (read-only use):

  /path/to/.conda/envs/env_isaaclab/bin/python scripts/walk_sonic_export.py

Writes
  walking/checkpoints/sonic_k1_it50000.npz   encoder/decoder MLP weights + plant constants (IsaacLab order)
  b_runs/sonic/clips/<name>.npz             walking reference clips at 50 Hz from gear_sonic's MotionLibRobot
  b_runs/sonic/ref_rollout.npz              torch-actor rollout of clip 0 (actions, qpos) for parity tests
Uses gear_sonic/scripts/eval_k1_mujoco.py from the SONIC repo for everything (plant, clips, actor, sim).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

SONIC = Path("/path/to/sonic-k1")
REPO = SONIC / "release_eval/GR00T-WholeBodyControl-K1"
CKPT = SONIC / "release_eval/artifacts/sonic_k1_it50000_actor.pt"
CLIP_SRC = SONIC / "data/sonic_motionlib/unseen1985_10s"
HERE = Path(__file__).resolve().parents[1]
OUT_W = HERE / "walking/checkpoints/sonic_k1_it50000.npz"
OUT = Path("/path/to/fla-terrain/b_runs/sonic")
# forward / turning walking clips (no jog, sideways, moonwalk, injured, props)
WALK_RE = re.compile(r"^(neutral_walk_ff|neutral_walk_180|walk_180_R|proud_walk_ff|walk_the_dog_ff|"
                     r"Neutral_walk_forward|walk_forward_shoulder|walk_fast_forward_grab_normal|walking_random_direction)")

sys.path.insert(0, str(REPO / "gear_sonic/scripts"))
sys.path.insert(0, str(REPO))
import eval_k1_mujoco as E  # noqa: E402


def main():
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    plant = E.K1Plant()
    state = E.extract_actor_state(str(CKPT))
    cfg = E.load_run_config(str(CKPT))
    bb = cfg.algo.config.actor.backbone
    mc = cfg.manager_env.commands.motion
    w = {k.replace(".", "_"): v.numpy() for k, v in state.items()}
    w.update(il_joints=np.array(plant.il_joints), i2m=plant.i2m, m2i=plant.m2i, kp=plant.kp, kd=plant.kd,
             effort=plant.effort, armature=plant.armature, action_scale=plant.action_scale,
             default_pos=plant.default_pos, max_num_tokens=int(bb.max_num_tokens),
             fsq_levels=int(bb.fsq_level_list), num_fsq_levels=int(bb.num_fsq_levels),
             num_future=int(mc.num_future_frames),
             frame_skips=int(float(mc.dt_future_ref_frames) // (1.0 / float(mc.motion_lib_cfg.target_fps))))
    np.savez(OUT_W, **w)
    print(f"[export] weights {OUT_W} ({OUT_W.stat().st_size/1e6:.1f} MB) keys {sorted(k for k in w if '_weight' in k)}")

    # clips: symlink the walking subset and load it with MotionLibRobot (same arrays IsaacLab feeds)
    src = OUT / "clip_src"
    src.mkdir(exist_ok=True)
    names = sorted(f for f in os.listdir(CLIP_SRC) if WALK_RE.match(f))
    for f in names:
        if not (src / f).exists():
            (src / f).symlink_to(CLIP_SRC / f)
    clips = E.load_reference_clips(cfg, plant, str(src), None)
    (OUT / "clips").mkdir(exist_ok=True)
    for c in clips:
        np.savez(OUT / "clips" / f"{c['name']}.npz", T=c["T"], dof_pos=c["dof_pos"], dof_vel=c["dof_vel"],
                 root_pos=c["body_pos"][:, 0], root_quat=c["body_quat"][:, 0],
                 root_lin_vel=c["body_lin_vel"][:, 0], root_ang_vel=c["body_ang_vel"][:, 0])
    sp = [np.linalg.norm(c["body_pos"][-1, 0, :2] - c["body_pos"][0, 0, :2]) / (c["T"] / 50) for c in clips]
    print(f"[export] {len(clips)} walking clips -> {OUT/'clips'}; mean planar speed {np.mean(sp):.2f} m/s "
          f"(min {np.min(sp):.2f}, max {np.max(sp):.2f}); T {min(c['T'] for c in clips)}-{max(c['T'] for c in clips)}")

    # reference rollout with the torch actor on SONIC's own k1.xml (dt 0.005, decimation 4)
    ev = E.Evaluator(state, _cfg(cfg), {})
    torch.manual_seed(0)
    r = ev.run_clip(clips[0], max_steps=150)
    np.savez(OUT / "ref_rollout.npz", name=clips[0]["name"], action=r["action"], qpos=r["qpos"])
    print(f"[export] ref rollout {clips[0]['name']} steps {len(r['action'])} result {r['result']}")
    # FSQ/actor numpy parity on real observations
    hist = E.ProprioHistory()
    ev.reset(clips[0], 0)
    prop, tok = ev.observe(clips[0], 0, hist, np.zeros(22, np.float32))
    a_t = ev.actor(prop[None], tok[None])[0]
    np.savez(OUT / "actor_parity.npz", prop=prop, tok=tok, action=a_t)
    print("[export] done")


def _cfg(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=False)


if __name__ == "__main__":
    main()
