"""Re-simulate the selected course runs from their original tasks (configs/course_v2.yaml, gen_dataset.expand) with pose
logging at 30 fps (fla_terrain.viz.prepare.rollout, read-only) and compare every recorded array with the stored
episode bit for bit (prepare.identical). Bundles go to b_runs/render_final/<episode>.npz/.json only if identical."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = "/path/to/fla-terrain/b_runs/render_final"
DATA = "/path/to/fla-terrain/data/v2_course"


def one(ep):
    from gen_dataset import expand
    from fla_terrain.viz.prepare import identical, rollout
    tasks = {t["episode_id"]: t for t in expand(yaml.safe_load(open(os.path.join(REPO, "configs/course_v2.yaml"))), DATA)}
    task = tasks[ep]
    arrays, meta, qpos, tv, info = rollout(task, REPO)
    z = np.load(os.path.join(DATA, f"episode_{ep}.npz"))
    stored = {k: z[k] for k in z.files}
    diff = identical(arrays, stored)
    res = {"episode": ep, "identical_to_dataset": not diff, "differing_arrays": diff, "n_video_frames": int(len(qpos)),
           "seconds": float(task["seconds"]), "task": {k: v for k, v in task.items() if k != "out"}}
    os.makedirs(OUT, exist_ok=True)
    if not diff:
        np.savez_compressed(os.path.join(OUT, ep + ".npz"), qpos=qpos, t_video=tv)
    json.dump(res, open(os.path.join(OUT, ep + ".json"), "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    eps = sys.argv[1:]
    with Pool(len(eps)) as p:
        for r in p.map(one, eps):
            print({k: r[k] for k in ("episode", "identical_to_dataset", "differing_arrays", "n_video_frames", "seconds")})
