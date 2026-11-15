"""Step 1 of the videos (CPU, on the machine with dataset v1): choose the clips and write render bundles.

  PY=python3
  $PY scripts/render_prepare.py --data /path/to/fla-terrain/data/v1 --results <dir with pred_P+N_s0.npz> \
      --out /path/to/fla-terrain/render/bundles [--workers 12]

Selection rules and the identity check are described in fla_terrain/viz/prepare.py. Exit code 1 if a re-simulated
episode differs from the dataset episode (then its stored predictions must not be shown).
Step 2 is scripts/render_clips.py (needs a renderer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

_SRC = None


def _init(data, results):
    global _SRC
    from fla_terrain.viz.prepare import Source
    _SRC = Source(data, results)


def _one(args):
    clip, task, out = args
    from fla_terrain.viz.prepare import make_bundle
    return make_bundle(_SRC, clip, {clip["episode"]: task}, ROOT, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default=os.path.join(ROOT, "configs/dataset_v1.yaml"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    import yaml
    from gen_dataset import expand
    from fla_terrain.viz.prepare import Source, edge_miss_check, select
    src = Source(args.data, args.results)
    chk = edge_miss_check(src)
    ref = json.load(open(os.path.join(args.results, "metrics.json")))["runs"]["P+N"]["0"]["edge_delay"]["walk"]
    print(f"edge events recount: {chk} | metrics.json: n {ref['n_events']} miss {ref['miss_rate']:.6f}")
    if chk["n_events"] != ref["n_events"] or abs(chk["miss_rate"] - ref["miss_rate"]) > 1e-9:
        sys.exit("edge event recount does not match metrics.json")
    clips = select(src)
    tasks = {t["episode_id"]: t for t in expand(yaml.safe_load(open(args.plan)), args.data)}
    for c in clips:
        print(f"  {c['clip']:28s} {c['episode']:48s} {c['t0']:.2f}-{c['t1']:.2f} s  {c.get('events', {}).get('kind', '')}")
    os.makedirs(args.out, exist_ok=True)
    import multiprocessing as mp
    jobs = [(c, tasks[c["episode"]], args.out) for c in clips]
    with mp.get_context("fork").Pool(min(args.workers, len(jobs)), initializer=_init,
                                     initargs=(args.data, args.results)) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    bad = [r["clip"] for r in res if not r["identical_to_dataset"]]
    for r in res:
        a = r["segment_accuracy"]
        print(f"  {r['clip']:28s} identical {r['identical_to_dataset']}  terrain {a['terrain_correct']}/{a['terrain_known']}"
              f"  support L {a['support_L_correct']}/{a['n_windows']} R {a['support_R_correct']}/{a['n_windows']}")
    json.dump({"edge_events_P+N_s0_walk_test": chk, "clips": [{k: v for k, v in r.items() if k != "terrain_cfg"} for r in res]},
              open(os.path.join(args.out, "clips.json"), "w"), indent=1)
    print("PREPARE_DONE" if not bad else f"NOT IDENTICAL: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
