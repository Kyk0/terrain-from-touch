"""Generic episode generator (schema v0.2) for the FLA-0 dataset.

  PY=python3
  $PY scripts/gen_data.py --terrain hard_grip --source walk_g113 --episodes 8 --seconds 20 --seed 1 \
      --out /path/to/fla-terrain/data/e2e --workers 8
  --terrain: a class of fla_terrain.sim.terrains.CLASSES (edge takes --variant), --split train|val|test
  --source:  walk_g113 | walk_g80 | walk_sonic | stand | rig | lifted_<walker> | liftmid_<walker>
             (<walker> = g113 | g80 | sonic; 'lifted'/'lift_mid' alone mean g113)

Commands: velocity policies follow walking.scenarios.make_schedule with the kind drawn uniformly from KINDS, the
same sampler for every class (recorded in terrain_params.cmd_schedule); SONIC ignores commands (cmd_vel zeros),
its clip log is recorded. Every episode is its own split_key (terrain instance). Falls: recording continues 0.5 s
after the fall, then stops; frames after the fall get label_terrain -1; metadata fell / fall_time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from fla_terrain.schema import TERRAIN_LABELS  # noqa: E402
from fla_terrain.sim import EpisodeRecorder, StandMotionSource, build_model, build_model_v2  # noqa: E402
from fla_terrain.sim.terrains import sample_terrain  # noqa: E402

WALKERS = {"g113": "G113_m2200", "g80": "G80_m4200", "sonic": "sonic"}
POST_FALL_S = 0.5


def parse_source(src: str):
    """-> (kind, walker): kind in walk | stand | rig | lifted | liftmid."""
    if src.startswith("walk_"):
        return "walk", src[5:]
    if src in ("stand", "rig"):
        return src, None
    if src == "lifted":
        return "lifted", "g113"
    if src == "lift_mid":
        return "liftmid", "g113"
    for k in ("lifted_", "liftmid_"):
        if src.startswith(k):
            return k[:-1], src[len(k):]
    raise ValueError(f"unknown source {src!r}")


def mode_of(kind):
    return {"walk": "walk", "stand": "stand", "rig": "rig", "lifted": "lifted", "liftmid": "lifted"}[kind]


def make_walker(walker):
    if walker == "sonic":
        from walking.sonic_k1 import SonicK1
        return SonicK1()
    from walking.k1_velocity import K1VelocityPolicy
    return K1VelocityPolicy(WALKERS[walker])


def stand_offsets(rng, scale=1.0):
    """Slow weight shifts (hip/ankle roll), small squats and ankle rolls: random sinusoids."""
    f = rng.uniform(0.1, 0.4, 3)
    amp = scale * rng.uniform([0.04, 0.08, 0.03], [0.10, 0.25, 0.10])
    ph = rng.uniform(0, 2 * np.pi, 3)

    def off(t):
        r, sq, ar = amp * np.sin(2 * np.pi * f * t + ph)
        sq = abs(sq)
        return {"left_hip_roll_joint": r, "right_hip_roll_joint": r, "left_ankle_roll_joint": -r + ar,
                "right_ankle_roll_joint": -r + ar, "left_hip_pitch_joint": -sq, "right_hip_pitch_joint": -sq,
                "left_knee_pitch_joint": 2 * sq, "right_knee_pitch_joint": 2 * sq, "left_ankle_pitch_joint": -sq,
                "right_ankle_pitch_joint": -sq}  # deltas on top of the stand pose
    return off, {"f": f.tolist(), "amp": amp.tolist()}


def make_target_fn(src, info):
    """PD position target of the motion source in joint_pos order (velocity policies: 16 actioned joints + held
    joints; SONIC: its 22 joints; PD stand: its targets). None if the source does not expose one."""
    qa = {int(adr): i for i, adr in enumerate(info["qpos_adr"])}
    J = len(info["qpos_adr"])
    if hasattr(src, "hqadr"):          # K1VelocityPolicy
        def fn(d):
            out = np.full(J, np.nan)
            out[[qa[int(x)] for x in src.qadr]] = src.target
            out[[qa[int(x)] for x in src.hqadr]] = src.hq0
            return out
        return fn
    if hasattr(src, "qadr") and hasattr(src, "target"):   # SonicK1
        def fn(d):
            out = np.full(J, np.nan)
            out[[qa[int(x)] for x in src.qadr]] = src.target
            return out
        return fn
    if hasattr(src, "pd") and src.pd is not None:          # StandMotionSource
        def fn(d):
            out = np.full(J, np.nan)
            out[[qa[int(x)] for x in info["act_qpos_adr"]]] = src.pd.q_target
            return out
        return fn
    return None


def run_episode(task, hook=None):
    """hook(k, model, data, info): optional, called after every physics step (used by the renderer to log poses).
    It must not touch the simulation state; with hook=None behaviour is exactly as before."""
    t_wall = time.time()
    a = task
    rng = np.random.default_rng([a["seed"], a["index"], zlib.crc32(a["episode_id"].encode())])
    kind, walker = parse_source(a["source"])
    T = float(a["seconds"])
    if a["terrain"] == "course":  # mixed-terrain course (walking/course.py), test only
        from walking.course import make_course
        tcfg = make_course(int(a["course_seed"]), split="test")
        if a.get("course_auto_T"):  # long enough to cross every chunk: stand 1 s + start pad + planned chunks + 5 s
            from walking.course import START_PAD
            T = min(T, 1.0 + START_PAD / tcfg["v_nominal"] + tcfg["params"]["planned_s"] + 5.0)
    else:
        tcfg = sample_terrain(a["terrain"], rng, a["split"], a.get("variant"), stand=kind == "stand")
    # sole: v1 (2x4 zones, FLA-0 v1) or v2 (point map); v2 setting: random (per-episode randomization) | nominal |
    # shifted | pads | g1 (fla_terrain.sim.pointmap.sole_setting); parameters are recorded in the metadata
    v2 = a.get("sole", "v1") == "v2"
    prm, pads = None, False
    if v2:
        from fla_terrain.sim.pointmap import sole_setting
        prm, pads = sole_setting(a.get("sole_setting", "random"), rng)
    spawn = np.array(tcfg["spawn"], float)
    yaw = float(tcfg["yaw"])
    events = {}
    sched = None
    proprio_fn = None
    eq = -1
    lift = None

    if kind == "rig":
        from fla_terrain.sim.rig import RigMotion, build_foot_rig, rig_proprio
        load = rng.uniform(3.0, 15.0, 2)
        if v2 and pads:
            raise ValueError("the pad sole is not built for the foot rig")
        m, info = build_foot_rig(tcfg, load_kg=tuple(load), v2_params=prm if v2 else None)
        d = mujoco.MjData(m)
        src = RigMotion(info, rng, T)
        src.reset(m, d)
        proprio_fn = rig_proprio
        motion_name = src.name
        events["rig"] = {"load_kg": load.tolist(), "phases": src.phases}
    else:
        if v2:
            m, info = build_model_v2(tcfg, prm, pads=pads, base_rig=kind in ("lifted", "liftmid"))
        else:
            m, info = build_model(tcfg, base_rig=kind in ("lifted", "liftmid"))
        d = mujoco.MjData(m)
        if kind == "stand":
            # on slopes keep the trunk upright and pitch/roll the ankles so both soles lie on the slope
            pose = None
            if tcfg["class"] == "slope":
                th = np.deg2rad(tcfg["params"]["slope_deg"])
                rel = tcfg["params"]["uphill_rel_heading"]
                pose = {f"{s}_ankle_pitch_joint": -0.2 - th * np.cos(rel) for s in ("left", "right")}
                pose.update({f"{s}_ankle_roll_joint": th * np.sin(rel) for s in ("left", "right")})
            # very soft ground (test_out) tips the PD stand over with full-size motions: half amplitude on soft
            off, events["stand"] = stand_offsets(rng, 0.5 if tcfg["class"] == "soft" else 1.0)
            src = StandMotionSource(info, pose=pose, offset_fn=off)
            src.reset(m, d)
            from fla_terrain.sim.motion import set_pose
            set_pose(m, d, info, pose, base_xy=spawn[:2], base_yaw=yaw)
            motion_name = src.name
        else:
            src = make_walker(walker)
            if walker == "sonic":
                src.reset(m, d, rng, base_yaw=yaw)
                d.qpos[0:2] = spawn[:2]
                d.qpos[2] += spawn[2]
                mujoco.mj_forward(m, d)
            else:
                src.reset(m, d, rng, base_pos=(spawn[0], spawn[1], spawn[2] + src.base_height), base_yaw=yaw)
                from walking.scenarios import KINDS, Schedule, make_schedule
                if a["terrain"] == "course":  # straight along +x at the course's nominal speed after 1 s standing
                    v = float(tcfg["v_nominal"])
                    sched = Schedule("course", [(1.0, (v, 0.0, 0.0))], {"vx": v}, 1.0)
                else:
                    sk = str(rng.choice(KINDS))
                    sched = make_schedule(sk, rng, duration=T)
                events["cmd_schedule"] = sched.params
            motion_name = src.name if walker == "sonic" else WALKERS[walker]
        if kind in ("lifted", "liftmid"):
            eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "rig_weld")
            b = info["base_body"]
            if kind == "lifted":
                lift = {"t_up": 0.0, "t_down": None, "dz": float(rng.uniform(0.1, 0.4))}
                d.qpos[2] += lift["dz"]
                mujoco.mj_forward(m, d)
                d.mocap_pos[0] = d.xpos[b]
                d.mocap_quat[0] = d.xquat[b]
            else:
                t_up = float(rng.uniform(3.0, max(3.5, 0.4 * T)))
                lift = {"t_up": t_up, "t_down": t_up + float(rng.uniform(2.0, 4.0)), "dz": float(rng.uniform(0.1, 0.3)),
                        "ramp": 0.5}
                d.eq_active[eq] = 0
            events["lift"] = lift

    target_fn = None if kind == "rig" else make_target_fn(src, info)
    tparams = {**tcfg["params"], "regions": tcfg["regions"]}
    if a["terrain"] == "course":
        tparams.update({"chunks": tcfg["chunks"], "boundaries": tcfg["boundaries"], "course_seed": int(a["course_seed"]),
                        "course_end_x": tcfg["course_end_x"]})
    rec = EpisodeRecorder(m, info, motion_source=motion_name, mode=mode_of(kind), split_key=a["split_key"], rng=rng,
                          terrain_params=tparams, proprio_fn=proprio_fn, target_fn=target_fn)
    rec.reset(d)
    # extra sensor readouts of the SAME simulation (sensitivity sets): sensor-side settings only (nominal geometry)
    extra = []
    for name, out_dir in (a.get("readouts") or []):
        from fla_terrain.sim.pointmap import sole_setting
        if not v2 or pads or name not in ("nominal", "shifted", "g1", "nosat"):
            raise ValueError(f"readout {name!r} needs a v2 non-pad primary episode and a sensor-side setting")
        p_r, _ = sole_setting(name, None)
        if tuple(p_r.boundaries) != tuple(prm.boundaries) or p_r.mid_clearance != prm.mid_clearance:
            raise ValueError("readouts must share the foot geometry of the primary episode")
        info_r = dict(info, sole_v2_params=p_r,
                      point_layout=info["sole_cfg"].__class__(**{**info["sole_cfg"].to_dict(), "point_mask": p_r.mask})
                      .point_layout())
        rng_r = np.random.default_rng([a["seed"], a["index"], zlib.crc32((a["episode_id"] + name).encode())])
        r2 = EpisodeRecorder(m, info_r, motion_source=motion_name, mode=mode_of(kind), split_key=a["split_key"],
                             rng=rng_r, terrain_params=dict(tparams), proprio_fn=proprio_fn, target_fn=target_fn)
        r2.reset(d)
        extra.append((name, out_dir, r2))
    base_xy = []
    dt = m.opt.timestep
    t_fall, held = None, False
    for k in range(int(round(T / dt))):
        t = k * dt
        cmd = sched(t) if sched is not None else np.zeros(3)
        if kind == "liftmid":
            if not held and lift["t_up"] <= t < lift["t_down"]:
                b = info["base_body"]
                d.mocap_pos[0] = d.xpos[b]
                d.mocap_quat[0] = d.xquat[b]
                d.eq_active[eq] = 1
                held = True
                lift["z_ref"] = float(d.xpos[b][2])
            if held:
                s_ = np.clip((t - lift["t_up"]) / lift["ramp"], 0, 1) - np.clip(
                    (t - lift["t_down"] + lift["ramp"]) / lift["ramp"], 0, 1)
                d.mocap_pos[0, 2] = lift["z_ref"] + s_ * lift["dz"]
                if t >= lift["t_down"]:
                    d.eq_active[eq] = 0
                    held = False
        src.pre_step(m, d, cmd)
        mujoco.mj_step(m, d)
        rec.step(d, cmd)
        for _, _, r2 in extra:
            r2.step(d, cmd)
        if k % int(round(1.0 / dt)) == 0:
            base_xy.append([round(float(d.qpos[0]), 4), round(float(d.qpos[1]), 4)])
        if hook is not None:
            hook(k, m, d, info)
        if t_fall is None and k % 10 == 0 and kind in ("walk", "stand", "liftmid") and not held:
            if src.fallen(m, d):
                t_fall = t
                rec.mark_fall(d)
                for _, _, r2 in extra:
                    r2.mark_fall(d)
        if t_fall is not None and t - t_fall > POST_FALL_S:
            break
    for r_ in [rec] + [r2 for _, _, r2 in extra]:
        if kind == "lifted":
            r_.unknown_intervals.append((0.0, 1e9))
        elif kind == "liftmid":
            r_.unknown_intervals.append((lift["t_up"], lift["t_down"]))
    if walker == "sonic":
        events["sonic_clips"] = src.clip_log
    tp = rec.meta_base["terrain_params"]
    tp.update({"sole": {"version": a.get("sole", "v1"), "setting": a.get("sole_setting") if v2 else None},
               "fell": t_fall is not None, "fall_time": t_fall, "source": a["source"], "walker": walker,
               "spawn_yaw": yaw, "base_xy_1hz": base_xy, **events})
    notes = json.dumps({"fell": t_fall is not None, "fall_time": t_fall, "source": a["source"]})
    p = rec.write(a["out"], a["episode_id"], notes=notes)
    if extra:
        from fla_terrain.data.writer import write_episode
        arr0 = rec.arrays()
    for name, out_dir, r2 in extra:
        r2.meta_base["terrain_params"].update({**tp, "sole": {"version": "v2", "setting": name,
                                                              "readout_of": a.get("sole_setting")}})
        arr2 = r2.arrays()
        for k in ("joint_pos", "joint_vel", "imu_gyro", "imu_gravity"):  # same noisy proprio stream: pure sole difference
            arr2[k] = arr0[k]
        write_episode(out_dir, a["episode_id"], arr2, r2.meta(notes))
    arr = rec.arrays()
    return {"path": str(p), "terrain": a["terrain"], "mode": mode_of(kind), "source": a["source"],
            "fell": t_fall is not None, "sim_s": float(d.time), "wall_s": time.time() - t_wall,
            "labels": np.bincount(arr["label_terrain"].astype(int) + 1, minlength=7).tolist(),
            "support": [np.bincount(arr[f"label_support_{s}"].astype(int), minlength=3).tolist() for s in "LR"]}


def summarize(res, tag, wall, nw):
    sim = sum(r["sim_s"] for r in res)
    lab = np.sum([r["labels"] for r in res], 0)
    sup = np.sum([r["support"] for r in res], 0)
    print(f"gen_data {tag}: {len(res)} episodes, falls {sum(r['fell'] for r in res)}, sim {sim/3600:.3f} h "
          f"({sim:.0f} s), wall {wall:.0f} s, {nw} procs")
    print(f"  speed: {np.mean([r['sim_s']/r['wall_s'] for r in res]):.2f} sim-s per wall-s per process")
    print("  frames per terrain label: " + ", ".join(f"{TERRAIN_LABELS[i-1]}={c}" for i, c in enumerate(lab) if c))
    for si, s in enumerate("LR"):
        tot = sup[si].sum()
        print(f"  support {s}: airborne {sup[si][0]/tot:.3f} partial {sup[si][1]/tot:.3f} full {sup[si][2]/tot:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--source", default="walk_g113")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="episode id prefix (default terrain_source_split_seed)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sole", default="v1", choices=["v1", "v2"])
    ap.add_argument("--sole-setting", default="random", choices=["random", "nominal", "shifted", "pads", "g1"])
    ap.add_argument("--course-seed", type=int, default=0, help="with --terrain course: first course seed (one per episode)")
    args = ap.parse_args()
    parse_source(args.source)
    tag = args.tag or f"{args.terrain}{'-' + args.variant if args.variant else ''}_{args.source}_{args.split}_s{args.seed}"
    tasks = [{"terrain": args.terrain, "variant": args.variant, "source": args.source, "split": args.split,
              "seconds": args.seconds, "seed": args.seed, "index": i, "out": args.out,
              "episode_id": f"{tag}_{i:05d}", "split_key": f"{tag}_{i:05d}", "sole": args.sole,
              "sole_setting": args.sole_setting, "course_seed": args.course_seed + i} for i in range(args.episodes)]
    t0 = time.time()
    nw = max(1, min(args.workers, 40, len(tasks)))
    if nw == 1:
        res = [run_episode(t) for t in tasks]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(nw) as pool:
            res = list(pool.imap_unordered(run_episode, tasks))
    summarize(res, tag, time.time() - t0, nw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
