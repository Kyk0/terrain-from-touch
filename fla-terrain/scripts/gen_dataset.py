"""Resumable dataset driver: expands a plan (configs/dataset_v1.yaml) into episodes and generates the missing ones.

  $PY scripts/gen_dataset.py --plan configs/dataset_v1.yaml --out /path/to/fla-terrain/data/v1 --workers 40
  --pilot 0.05   generate only the first ceil(5 %) of the episodes of every (group, class, source, split) cell
  --dry-run      print the plan table and exit
Detached: nohup setsid $PY scripts/gen_dataset.py ... > logs/wA_gen_v1.log 2>&1 & echo $! > logs/wA_gen_v1.pid
Episodes that already exist (npz + json) are skipped; failures are logged and the run continues.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from collections import Counter

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yaml  # noqa: E402

SPLITS = ("train", "val", "test")


def split_counts(n, ratios):
    raw = [n * r for r in ratios]
    c = [int(math.floor(x)) for x in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - c[i], reverse=True)[: n - sum(c)]:
        c[i] += 1
    return c


def expand(plan, out, pilot=None):
    """Plan -> one task per episode. Group keys:
      classes, sources, splits [train, val, test] ratios;
      hours (per class, spread over the sources; v1 plans) OR hours_per_source (per class per source: appending a
        source later adds new cells and changes no existing episode);
      seconds (episode length, default plan seconds); sole_setting; out_subdir (episodes go to <out>/<subdir>);
      readouts: [setting, ...] extra sensor readouts of every TEST episode (same simulation) -> <out>/<setting>;
      readout_splits: splits that get the readouts (default [test]);
      pads_twin: true -> every TEST episode is also simulated with the pad sole -> <out>/pads (same seed, commands);
      course_seeds: [..] (class 'course'): one test episode per seed and source."""
    tasks = []
    for g, spec in plan["groups"].items():
        T = float(spec.get("seconds", plan["seconds"]))
        gout = os.path.join(out, spec["out_subdir"]) if spec.get("out_subdir") else out
        setting = spec.get("sole_setting", plan.get("sole_setting", "random"))
        for cv in spec["classes"]:
            cls, _, var = cv.partition(":")
            for si, src in enumerate(spec["sources"]):
                if spec.get("course_seeds") is not None:
                    seeds = list(spec["course_seeds"])
                    cells = [("test", len(seeds))]
                else:
                    if "hours_per_source" in spec:
                        hps = spec["hours_per_source"]
                        n = int(round(hps.get(cv, hps["default"]) * 3600 / T))
                    else:  # v1 semantics
                        hours = spec["hours"].get(cv, spec["hours"]["default"])
                        n_cls = int(round(hours * 3600 / T))
                        n = split_counts(n_cls, [1 / len(spec["sources"])] * len(spec["sources"]) +
                                         [0] * (3 - len(spec["sources"])))[si]
                    cells = list(zip(SPLITS, split_counts(n, spec["splits"])))
                for split, k in cells:
                    kk = k if pilot is None else min(k, math.ceil(k * pilot))
                    for i in range(kk):
                        eid = f"{g}_{cls}{'-' + var if var else ''}_{src}_{split}_{i:04d}"
                        if spec.get("course_seeds") is not None:
                            eid = f"{g}_{src}_seed{seeds[i]:03d}"
                        t = {"terrain": cls, "variant": var or None, "source": src, "split": split,
                             "seconds": T, "seed": plan["seed"], "index": i, "out": gout, "episode_id": eid,
                             "split_key": eid, "group": g, "sole": plan.get("sole", "v1"), "sole_setting": setting}
                        if spec.get("course_seeds") is not None:
                            t["course_seed"] = int(seeds[i])
                            t["course_auto_T"] = bool(spec.get("course_auto_T", False))
                        if spec.get("readouts") and split in spec.get("readout_splits", ["test"]):
                            t["readouts"] = [(r, os.path.join(out, r)) for r in spec["readouts"]]
                        tasks.append(t)
                        if split == "test" and spec.get("pads_twin"):
                            tasks.append({**t, "sole_setting": "pads", "out": os.path.join(out, "pads"),
                                          "readouts": None, "group": g + "_pads"})
    return tasks


def _exists(t):
    outs = [t["out"]] + [o for _, o in (t.get("readouts") or [])]
    return all(os.path.exists(os.path.join(o, f"episode_{t['episode_id']}.npz")) and
               os.path.exists(os.path.join(o, f"episode_{t['episode_id']}.json")) for o in outs)


def _run(task):
    from gen_data import run_episode
    try:
        r = run_episode(task)
        r["ok"] = True
        return r
    except Exception:  # noqa: BLE001
        return {"ok": False, "episode_id": task["episode_id"], "err": traceback.format_exc(limit=3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--pilot", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pidfile", default=None)
    args = ap.parse_args()
    plan = yaml.safe_load(open(args.plan))
    tasks = expand(plan, args.out, args.pilot)
    hours = Counter()
    for t in tasks:
        hours[(t["group"], t["split"])] += t["seconds"] / 3600
    print(f"plan {args.plan}: {len(tasks)} episodes, {sum(hours.values()):.2f} h"
          + (f" (pilot {args.pilot})" if args.pilot else ""))
    for g in sorted({t["group"] for t in tasks}):
        print(f"  {g:10s} " + "  ".join(f"{s} {hours[(g, s)]:.2f} h" for s in SPLITS))
    if args.dry_run:
        return 0
    os.makedirs(args.out, exist_ok=True)
    if args.pidfile:
        open(args.pidfile, "w").write(f"{os.getpid()} pgid {os.getpgid(0)}\n")
    todo = [t for t in tasks if not _exists(t)]
    print(f"  existing {len(tasks) - len(todo)}, to generate {len(todo)}", flush=True)
    nw = max(1, min(args.workers, 40))
    t0 = time.time()
    done, fails, sim = 0, 0, 0.0
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(nw, maxtasksperchild=50) as pool:
        for r in pool.imap_unordered(_run, todo, chunksize=1):
            done += 1
            if r["ok"]:
                sim += r["sim_s"]
            else:
                fails += 1
                print(f"FAIL {r['episode_id']}: {r['err'].strip().splitlines()[-1]}", flush=True)
            if done % 100 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)} done, {fails} failed, sim {sim/3600:.2f} h, wall {el/60:.1f} min, "
                      f"ETA {el / done * (len(todo) - done) / 60:.1f} min", flush=True)
    print(f"GEN_DONE episodes {done - fails} ok, {fails} failed, wall {(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
