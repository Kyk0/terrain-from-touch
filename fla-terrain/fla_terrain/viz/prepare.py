"""Choose the test episodes shown in the videos and re-simulate them with pose logging.

Runs on the machine that holds dataset v1 (CPU only, no renderer). For every clip it writes a bundle
``<out>/<clip>.npz`` + ``<clip>.json`` with everything the renderer needs:

  qpos            [F, nq] robot pose at 30 fps over the whole 20 s episode (frame j = physics time t_video[j])
  episode arrays  the dataset episode itself (50 Hz sensor output, labels)
  pred_*          the P+N seed-0 test predictions for this episode's windows (1 s window, 0.2 s stride), read from
                  results/v1/pred_P+N_s0.npz - the same numbers the report is computed from
  json            terrain_cfg (to rebuild the scene), clip segment, events, the selection rule

No checkpoint was saved by the ablation, so the model is not re-run. Instead each episode is re-simulated from its
original task (same seed, same code) and every recorded array is compared with the dataset episode; the bundle is
written only if they are identical, i.e. the stored predictions are the model's output on exactly this rollout.

Selection rules are fixed here, in code, and applied in episode-index order (no hand picking):
  terrain clip   first test episode of (class, gait) without a fall that has a 10 s stretch where the commanded
                 forward speed is >= 0.15 m/s in >= 90 % of frames (earliest such stretch on a 1 s grid, from 1 s);
                 SONIC ignores commands: first test episode without a fall, 2-12 s.
  strip          first strip_grip_slip G113 test episode without a fall whose surface change is detected (not a
                 miss) and lies in 4-14 s; clip = 4 s before to 6 s after the change.
  lifted         first liftmid hard_grip G113 test episode without a fall; clip = 3 s before the lift to 7 s after.
  failure        first missed edge event (evaluate.py edge-delay definition: a stance that starts on partial
                 support and is never predicted partial) in the edge G113 test episodes, without a fall, with 4 s
                 before and 6 s after inside the episode, and with the partial-support part of the stance lasting
                 >= 0.3 s (shorter brushes end before a single 0.2 s-stride window can see them); the episode
                 already used for the edge terrain clip is skipped so the two clips show different footage.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

from ..schema import TERRAIN_LABELS

FPS = 30
CLIP_S = 10.0
GAIT_NAMES = {"g113": "G113 velocity policy", "g80": "G80 velocity policy", "sonic": "SONIC-K1 clip playback"}
TERRAIN_CLIPS = [("hard_grip", "walk_hard_grip"), ("hard_slip", "walk_hard_slip"), ("soft", "walk_soft"),
                 ("rough", "walk_rough"), ("slope", "walk_slope"), ("edge", "walk_edge-planks")]


class Source:
    """Dataset v1 frame cache + P+N seed-0 test predictions, indexed by episode."""

    def __init__(self, data: str, results: str, input_set: str = "P+N", seed: int = 0):
        from ..models.dataset import FrameData, cache_path
        self.data = data
        self.fd = FrameData.build(data, cache=cache_path(data))
        self.pred_file = os.path.join(results, f"pred_{input_set}_s{seed}.npz")
        p = np.load(self.pred_file)
        self.pred = {k: p[k] for k in p.files}
        self.idx = self.pred["idx"]
        self.ep_of = self.fd.ep_of_win[self.idx]
        self.t_end = self.fd.lab["t"][self.fd.last[self.idx]]
        self.by_ep = {}
        for i, e in enumerate(self.ep_of):
            self.by_ep.setdefault(int(e), []).append(i)
        self.by_ep = {e: sorted(ii, key=lambda i: self.t_end[i]) for e, ii in self.by_ep.items()}
        self.eps = self.fd.eps
        self.name2e = {em["episode"]: e for e, em in enumerate(self.eps)}

    def episodes(self, prefix: str):
        """Episode indices whose id is '<prefix>_test_NNNN', in index order."""
        out = [(em["episode"], e) for e, em in enumerate(self.eps) if em["episode"].startswith(prefix + "_test_")]
        return [e for _, e in sorted(out)]

    def lab(self, e, key):
        em = self.eps[e]
        return self.fd.lab[key][em["offset"]:em["offset"] + em["T"]]

    def cmd(self, e):
        em = self.eps[e]
        cols = self.fd.groups["P"][-3:]
        return self.fd.X[em["offset"]:em["offset"] + em["T"]][:, cols]

    def ep_pred(self, e):
        ii = np.array(self.by_ep[e])
        out = {"t": self.t_end[ii].astype(np.float64)}
        for k in ("terrain", "sup_L", "sup_R", "mu", "terrain_prob"):
            out[k] = self.pred[k][ii]
        return out


def edge_events(src: Source, e: int):
    """Edge events of one episode exactly as evaluate._edge_delay enumerates them.
    -> list of {side, t_p, t_stop, delay or None}."""
    t = src.lab(e, "t")
    pr = src.ep_pred(e)
    ev = []
    for s, key in (("L", "sup_L"), ("R", "sup_R")):
        sup = src.lab(e, key)
        on = sup > 0
        k = 0
        while k < len(on):
            if not on[k]:
                k += 1
                continue
            j = k
            while j < len(on) and on[j]:
                j += 1
            first = np.nonzero(sup[k:min(j, k + 5)] == 1)[0]
            if first.size:
                tp_, t_stop = float(t[k + first[0]]), float(t[j - 1])
                cand = np.nonzero((pr["t"] >= tp_ - 1e-6) & (pr["t"] <= t_stop + 1e-6) & (pr[key] == 1))[0]
                ev.append({"side": s, "t_p": tp_, "t_stop": t_stop,
                           "delay": float(pr["t"][cand].min() - tp_) if cand.size else None})
            k = j
    return sorted(ev, key=lambda d: d["t_p"])


def _walk_segment(src: Source, e: int):
    """Earliest 10 s stretch (1 s grid from 1 s) with forward command >= 0.15 m/s in >= 90 % of frames."""
    vx = src.cmd(e)[:, 0]
    t = src.lab(e, "t")
    for t0 in np.arange(1.0, t[-1] - CLIP_S + 1e-6, 1.0):
        m = (t > t0) & (t <= t0 + CLIP_S)
        if (vx[m] >= 0.15).mean() >= 0.9:
            return float(t0)
    return None


def select(src: Source) -> list[dict]:
    from ..models import evaluate as ev
    clips = []

    def add(clip, title, e, t0, group, **kw):
        em = src.eps[e]
        prefix = em["episode"].rsplit("_test_", 1)[0]
        cand = src.episodes(prefix)  # the pool the rule was applied to, with its falls (for the page)
        clips.append({"clip": clip, "title": title, "episode": em["episode"], "t0": float(t0),
                      "t1": float(t0 + CLIP_S), "group": group,
                      "pool": {"prefix": prefix, "n_test_episodes": len(cand),
                               "n_fell": sum(src.eps[c]["fall_time"] is not None for c in cand)}, **kw})

    for gait, classes in (("g113", TERRAIN_CLIPS), ("sonic", TERRAIN_CLIPS[:2])):
        for cls, prefix in classes:
            for e in src.episodes(f"{prefix}_walk_{gait}"):
                if src.eps[e]["fall_time"] is not None:
                    continue
                t0 = 2.0 if gait == "sonic" else _walk_segment(src, e)
                if t0 is None:
                    continue
                add(f"terrain_{cls}_{gait}", f"{cls.replace('_', '-')}, {GAIT_NAMES[gait]}", e, t0,
                    "terrain" if gait == "g113" else "sonic")
                break
            else:
                raise RuntimeError(f"no episode for {cls} {gait}")

    for e in src.episodes("strip_strip_grip_slip_walk_g113"):
        em = src.eps[e]
        if em["fall_time"] is not None:
            continue
        r = ev._strip_latency(src.fd, {e: src.by_ep[e]}, src.t_end, src.pred)
        if not r["n"] or r["miss_rate"] > 0:
            continue
        tc = min(t for t, _, reg in em["touchdowns"] if 1 in reg)
        if not 4.0 <= tc <= 14.0:
            continue
        add("strip_grip_slip_g113", "surface change grip -> slip, G113 velocity policy", e, round(tc - 4.0, 2), "special",
            events={"kind": "strip", "t_change": tc, "latency_s": r["median_latency_s"],
                    "latency_steps": r["median_latency_steps"], "t_detect": tc + r["median_latency_s"],
                    "new_class": int(em["regions"][1])})
        break
    else:
        raise RuntimeError("no strip episode")

    for e in src.episodes("liftmid_hard_grip_liftmid_g113"):
        em = src.eps[e]
        if em["fall_time"] is not None:
            continue
        r = ev._lifted_delay(src.fd, {e: src.by_ep[e]}, src.t_end, src.pred)
        t = src.lab(e, "t")
        air = (src.lab(e, "sup_L") == 0) & (src.lab(e, "sup_R") == 0) & (t >= em["lift"]["t_up"])
        ta = float(t[np.argmax(air)])
        t0 = max(0.0, min(round(em["lift"]["t_up"] - 3.0, 2), t[-1] - CLIP_S))
        add("lifted_mid_walk_g113", "lifted mid-walk, G113 velocity policy", e, t0, "special",
            events={"kind": "lift", "t_up": em["lift"]["t_up"], "t_down": em["lift"]["t_down"], "t_airborne": ta,
                    "ramp": em["lift"].get("ramp", 0.5),
                    "delay_s": None if r["miss_rate"] else r["median_delay_s"], "dz": em["lift"]["dz"]})
        break
    else:
        raise RuntimeError("no lifted episode")

    found = False
    for prefix in ("walk_edge-planks_walk_g113", "walk_edge-steps_walk_g113"):
        for e in src.episodes(prefix):
            em = src.eps[e]
            if em["fall_time"] is not None or em["episode"] in {c["episode"] for c in clips if c["group"] == "terrain"}:
                continue
            evs = edge_events(src, e)
            T = float(src.lab(e, "t")[-1])
            for x in evs:
                if x["delay"] is None and x["t_stop"] - x["t_p"] >= 0.3 and x["t_p"] - 4.0 >= 0.0 and x["t_p"] + 6.0 <= T:
                    add("failure_edge_miss_g113", "failure case: an edge touchdown the model misses, G113", e,
                        round(x["t_p"] - 4.0, 2), "special",
                        events={"kind": "edge_miss", "side": x["side"], "t_p": x["t_p"], "t_stop": x["t_stop"],
                                "episode_events": evs})
                    found = True
                    break
            if found:
                break
        if found:
            break
    if not found:
        raise RuntimeError("no missed edge event")
    return clips


def edge_miss_check(src: Source) -> dict:
    """Recount the walk edge events over all test episodes with edge_events(); must equal metrics.json.
    Also counts the missed events whose stance ends less than 0.3 s after the first partial frame."""
    n = miss = short = 0
    for e, em in enumerate(src.eps):
        if em["range_split"] == "test" and em["class"] == "edge" and em["mode"] == "walk" and e in src.by_ep:
            evs = edge_events(src, e)
            n += len(evs)
            miss += sum(x["delay"] is None for x in evs)
            short += sum(x["delay"] is None and x["t_stop"] - x["t_p"] < 0.3 for x in evs)
    return {"n_events": n, "miss_rate": miss / max(n, 1), "n_missed": miss, "n_missed_shorter_than_0.3s": short}


def rollout(task: dict, repo_root: str):
    """Re-run one dataset task, logging qpos at 30 fps. -> (written arrays, meta, qpos, t_video, info dict)."""
    sys.path.insert(0, os.path.join(repo_root, "scripts"))
    from gen_data import run_episode
    dt = 0.002
    n_frames = int(round(task["seconds"] * FPS))
    steps = {int(round(j / FPS / dt)): j for j in range(1, n_frames + 1)}
    log = {"qpos": [], "t": [], "info": None}

    def hook(k, m, d, info):
        if log["info"] is None:
            log["info"] = info
        if k + 1 in steps:
            log["qpos"].append(d.qpos.copy())
            log["t"].append(float(d.time))

    with tempfile.TemporaryDirectory() as tmp:
        r = run_episode({**task, "out": tmp}, hook=hook)
        z = np.load(r["path"])
        arrays = {k: z[k] for k in z.files}
        meta = json.load(open(r["path"][:-4] + ".json"))
    return arrays, meta, np.array(log["qpos"]), np.array(log["t"]), log["info"]


def identical(a: dict, b: dict) -> list[str]:
    """Keys whose arrays differ between two episodes (NaN-aware, exact)."""
    bad = [k for k in sorted(set(a) | set(b)) if k not in a or k not in b]
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            bad.append(k)
    return bad


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def make_bundle(src: Source, clip: dict, tasks: dict, repo_root: str, out: str) -> dict:
    e = src.name2e[clip["episode"]]
    em = src.eps[e]
    task = tasks[clip["episode"]]
    arrays, meta, qpos, tv, info = rollout(task, repo_root)
    z = np.load(os.path.join(src.data, f"episode_{clip['episode']}.npz"))
    stored = {k: z[k] for k in z.files}
    diff = identical(arrays, stored)
    pr = src.ep_pred(e)
    m = (pr["t"] > clip["t0"]) & (pr["t"] <= clip["t1"])
    lt, sl, sr = (src.lab(e, k) for k in ("terrain", "sup_L", "sup_R"))
    fi = np.round(pr["t"][m] * 50).astype(int) - 1  # frame index of each window end
    known = lt[fi] >= 0
    acc = {"n_windows": int(m.sum()), "terrain_known": int(known.sum()),
           "terrain_correct": int((pr["terrain"][m][known] == lt[fi][known]).sum()),
           "support_L_correct": int((pr["sup_L"][m] == sl[fi]).sum()),
           "support_R_correct": int((pr["sup_R"][m] == sr[fi]).sum())}
    regions = meta["terrain_params"]["regions"]
    walker = meta["terrain_params"].get("walker")
    j = {**clip, "identical_to_dataset": not diff, "differing_arrays": diff, "fps": FPS,
         "physics_dt": info["physics_dt"], "base_rig": bool(info["base_rig"]), "terrain_cfg": info["terrain_cfg"],
         "regions": regions, "class": em["class"], "variant": em.get("variant"), "band": em["band"],
         "gait": walker, "gait_name": GAIT_NAMES.get(walker, str(walker)), "motion_source": meta["motion_source"],
         "terrain_labels": {str(k): v for k, v in TERRAIN_LABELS.items()}, "mass": info["mass"],
         "zone_layout": meta["zone_layout"], "prediction_file": os.path.basename(src.pred_file),
         "segment_accuracy": acc, "cmd_schedule": meta["terrain_params"].get("cmd_schedule")}
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, clip["clip"] + ".npz"), qpos=qpos.astype(np.float64), t_video=tv,
                        pred_t=pr["t"], pred_terrain=pr["terrain"], pred_prob=pr["terrain_prob"].astype(np.float32),
                        pred_sup_L=pr["sup_L"], pred_sup_R=pr["sup_R"], pred_mu=pr["mu"], **stored)
    json.dump(_jsonable(j), open(os.path.join(out, clip["clip"] + ".json"), "w"), indent=1)
    return j
