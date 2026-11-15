"""FLA-0 v2: every model and every table of DECISIONS.md D14 in one command, for a given list of walking sources.

  $PY scripts/run_v2.py --sources walk_g113,walk_g80,walk_sonic --out results/v2/src3 \
      --models-root /path/to/fla-terrain/models/v2_src3 [--workers 3] [--rule-dir DIR]
Stages (all by default, resumable: finished jobs are skipped):
  main    17 input sets (P, P+C1..5, C1..5, Sblk, Sreal) + diagnostic Pcmd, x seeds, on data/v2
  logo    leave-one-gait-out: the ten Ck / P+Ck, every held-out source, x seeds
  sens    P+C1..C5 trained on data/v2_sens/nominal, tested on nominal / shifted / pads / g1 (outline mask)
  select  best sole model by the D14 rule (highest mean val M1, walk windows, among the ten) -> course_selection.json,
          written BEFORE any course prediction exists
  course  selected model and P (all seeds) on data/v2_course; per-run window predictions saved
          (<out>/course_preds/<model>_s<seed>/<episode_id>.npz: t_end, terrain, sup_L, sup_R, terrain_prob); a
          rule-based detector directory (--rule-dir, files <episode_id>.npz in walking/rule_detector.py format) is
          scored by the same code
  tables  metrics.json + tables.md (main table, coded verdicts Q1-Q4, Q6, Q7, course table, v1 vs v2)
Episodes used: walking sources in --sources, their lifted / lifted-mid episodes, stand and foot rig.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

DATA = "/path/to/fla-terrain/data"
CK = {"C1": "Nreal+Sreal", "C2": "Nreal+Sblk", "C3": "Nreal", "C4": "Nblk+Sblk", "C5": "Nblk"}
SOLE10 = [f"P+{v}" for v in CK.values()] + list(CK.values())
MAIN_VIEWS = ["P"] + [f"P+{v}" for v in CK.values()] + list(CK.values()) + ["Sblk", "Sreal", "Pcmd"]
SENS_VIEWS = [f"P+{CK[k]}" for k in ("C2", "C3", "C5", "C1", "C4")]
SENS_SETS = ("nominal", "shifted", "pads", "g1", "nosat")
NOSAT_VIEWS = [f"P+{CK[k]}" for k in ("C2", "C3", "C5")]  # D19: trained and tested without per-point saturation
NAME = {**{v: k for k, v in CK.items()}, **{f"P+{v}": f"P+{k}" for k, v in CK.items()}, "P": "P", "Pcmd": "P (cmd)",
        "Sblk": "S_blk", "Sreal": "S_real"}

_FD = {}


def fd_of(path):
    from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2
    if path not in _FD:
        _FD.clear()
        _FD[path] = FrameDataV2.build(path, cache=cache_path_v2(path), workers=16)
    return _FD[path]


def allowed_eps(fd, sources):
    walkers = {s.split("_", 1)[1] for s in sources}
    ok = []
    for e in fd.eps:
        s = e["source"]
        ok.append(s in sources or s in ("stand", "rig") or
                  (s.startswith(("lifted_", "liftmid_")) and s.split("_", 1)[1] in walkers))
    return np.array(ok)


def splits_for(fd, sources, exclude_walker=None):
    sp = fd.split_indices()
    ok = allowed_eps(fd, sources)
    if exclude_walker:
        ok &= np.array([not e["source"].endswith("_" + exclude_walker) for e in fd.eps])
    return {k: v[ok[fd.ep_of_win[v]]] for k, v in sp.items()}


def _cap(sp, job):
    """Smoke tests only: subsample train/val windows (job['max_train']); the real run leaves them untouched."""
    if not job.get("max_train"):
        return sp
    rng = np.random.default_rng(0)
    return {k: (np.sort(rng.choice(v, min(len(v), job["max_train"]), replace=False)) if k != "test" else v)
            for k, v in sp.items()}


def run_job(job):
    import torch
    from fla_terrain.models.v2 import predict_saved, train_v2
    from fla_terrain.models.v2eval import evaluate_v2, walk_m1
    try:
        t0 = time.time()
        dev = "cuda"
        if job["kind"] in ("main", "logo"):
            fd = fd_of(f"{DATA}/v2")
            if job["kind"] == "main":
                sp = splits_for(fd, job["sources"])
            else:
                held = job["held"]
                tr = splits_for(fd, job["sources"], exclude_walker=held.split("_", 1)[1])
                te = fd.split_indices()["test"]
                srcs = np.array([e["source"] for e in fd.eps])
                modes = np.array([e["mode"] for e in fd.eps])
                te = te[(srcs[fd.ep_of_win[te]] == held) & (modes[fd.ep_of_win[te]] == "walk")]
                sp = {"train": tr["train"], "val": tr["val"], "test": te}
            sp = _cap(sp, job)
            _, pred, info = train_v2(fd, job["view"], job["seed"], sp, job["name"], f"{DATA}/v2", device=dev,
                                     models_root=job["models_root"], log=lambda *a: None, epochs=job.get("epochs", 25))
            res = {"test": evaluate_v2(fd, sp["test"], pred), "val_M1": walk_m1(fd, sp["val"], info["pred_val"]["terrain"])}
        elif job["kind"] == "sens_nosat":  # D19: train on the unsaturated readout, test on the unsaturated test set
            fd = fd_of(f"{DATA}/v2_sens/nosat")
            sp = _cap(splits_for(fd, job["sources"]), job)
            _, pred, info = train_v2(fd, job["view"], job["seed"], sp, job["name"], f"{DATA}/v2_sens/nosat",
                                     device=dev, models_root=job["models_root"], log=lambda *a: None,
                                     epochs=job.get("epochs", 25))
            res = {"val_M1": walk_m1(fd, sp["val"], info["pred_val"]["terrain"]),
                   "tests": {"nosat": evaluate_v2(fd, sp["test"], pred)}}
        else:  # sens
            fd = fd_of(f"{DATA}/v2_sens/nominal")
            sp = _cap(splits_for(fd, job["sources"]), job)
            _, pred, info = train_v2(fd, job["view"], job["seed"], sp, job["name"], f"{DATA}/v2_sens/nominal",
                                     device=dev, models_root=job["models_root"], log=lambda *a: None,
                                     epochs=job.get("epochs", 25))
            res = {"val_M1": walk_m1(fd, sp["val"], info["pred_val"]["terrain"]), "tests": {}}
            mdir = os.path.join(job["models_root"], job["name"])
            for s in SENS_SETS:
                from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2
                fs = FrameDataV2.build(f"{DATA}/v2_sens/{s}", cache=cache_path_v2(f"{DATA}/v2_sens/{s}"), workers=8)
                ok = allowed_eps(fs, job["sources"])
                idx = fs.split_indices()["test"]
                idx = idx[ok[fs.ep_of_win[idx]]]
                p = predict_saved(mdir, fs, idx, device=dev)
                res["tests"][s] = evaluate_v2(fs, idx, p)
        info.pop("pred_val", None)
        res.update({"job": {k: v for k, v in job.items() if k != "models_root"}, "train_info": info,
                    "model_dir": os.path.join(job["models_root"], job["name"]), "wall_s": time.time() - t0})
        os.makedirs(os.path.dirname(job["result"]), exist_ok=True)
        json.dump(res, open(job["result"], "w"), indent=1, default=float)
        torch.cuda.empty_cache()
        return job["name"], True, f"{time.time() - t0:.0f} s"
    except Exception:  # noqa: BLE001
        return job["name"], False, traceback.format_exc(limit=4)


def jobs_for(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    J = []
    base = {"sources": args.sources, "models_root": args.models_root}
    st = args.stages.split(",")
    if "main" in st:
        for v in MAIN_VIEWS:
            for s in seeds:
                J.append({**base, "kind": "main", "view": v, "seed": s, "name": f"main/{v}_s{s}",
                          "result": f"{args.out}/jobs/main/{v}_s{s}.json"})
    if "sens" in st:
        for v in SENS_VIEWS:
            for s in seeds:
                J.append({**base, "kind": "sens", "view": v, "seed": s, "name": f"sens/{v}_s{s}",
                          "result": f"{args.out}/jobs/sens/{v}_s{s}.json"})
    if "sens_nosat" in st:
        for v in NOSAT_VIEWS:
            for s in seeds:
                J.append({**base, "kind": "sens_nosat", "view": v, "seed": s, "name": f"sens_nosat/{v}_s{s}",
                          "result": f"{args.out}/jobs/sens_nosat/{v}_s{s}.json"})
    if "logo" in st:
        for held in args.sources:
            for v in SOLE10:
                for s in seeds:
                    J.append({**base, "kind": "logo", "held": held, "view": v, "seed": s,
                              "name": f"logo_{held}/{v}_s{s}", "result": f"{args.out}/jobs/logo_{held}/{v}_s{s}.json"})
    return [j for j in J if not os.path.exists(j["result"])]


# ----------------------------------------------------------------------------------------------- aggregation
def load_jobs(out, sub):
    R = {}
    for p in glob.glob(f"{out}/jobs/{sub}/*.json"):
        r = json.load(open(p))
        R.setdefault(r["job"]["view"], {})[str(r["job"]["seed"])] = r
    return R


def select(args):
    path = f"{args.out}/course_selection.json"
    if os.path.exists(path):
        return json.load(open(path))
    from fla_terrain.models.provenance import git_state
    M = load_jobs(args.out, "main")
    vals = {v: float(np.mean([r["val_M1"] for r in M[v].values()])) for v in SOLE10 if v in M and len(M[v]) == 3}
    if len(vals) < 10:
        raise SystemExit(f"selection needs all ten sole models with 3 seeds, have {sorted(vals)}")
    best = max(vals, key=vals.get)
    sel = {"rule": "D14 q5: sole model (ten Ck and P+Ck) with the highest mean M1 on the walk windows of the validation "
                   "split", "val_M1_mean": vals, "selected": best, "selected_name": NAME[best],
           "written_before_any_course_prediction": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "git": git_state()["commit"]}
    assert not glob.glob(f"{args.out}/course_preds/*"), "course predictions exist before the selection"
    json.dump(sel, open(path, "w"), indent=1)
    return sel


def course(args, sel):
    import torch  # noqa: F401
    from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2
    from fla_terrain.models.v2 import predict_saved
    from fla_terrain.models.v2eval import course_score
    fd = FrameDataV2.build(f"{DATA}/v2_course", cache=cache_path_v2(f"{DATA}/v2_course"), workers=8)
    ok = np.array([e["source"] in args.sources for e in fd.eps])
    idx = np.arange(len(fd.starts))
    idx = idx[ok[fd.ep_of_win[idx]]]
    res = {}
    for v in (sel["selected"], "P"):
        for s in args.seeds.split(","):
            mdir = os.path.join(args.models_root, f"main/{v}_s{s}")
            pdir = f"{args.out}/course_preds/{v}_s{s}"
            os.makedirs(pdir, exist_ok=True)
            p = predict_saved(mdir, fd, idx, device="cuda")
            by_ep = {}
            t_end = fd.lab["t"][fd.last[idx]]
            for e in np.unique(fd.ep_of_win[idx]):
                m = fd.ep_of_win[idx] == e
                d = {"t_end": t_end[m], "terrain": p["terrain"][m], "sup_L": p["sup_L"][m], "sup_R": p["sup_R"][m],
                     "terrain_prob": p["terrain_prob"][m], "truth": fd.lab["terrain"][fd.last[idx[m]]]}
                np.savez_compressed(os.path.join(pdir, f"{fd.eps[e]['episode']}.npz"), **d)
                by_ep[fd.eps[e]["episode"]] = d
            res.setdefault(v, {})[s] = course_score(fd, idx, by_ep)
    if args.rule_dir and os.path.isdir(args.rule_dir):
        by_ep = {os.path.basename(p)[:-4].replace("episode_", ""): dict(np.load(p))
                 for p in glob.glob(f"{args.rule_dir}/*.npz")}
        res["rule"] = {"0": course_score(fd, idx, by_ep)}
    json.dump(res, open(f"{args.out}/course.json", "w"), indent=1, default=float)
    return res


def sens_extra_tests(args):
    """Test sets added after the sensitivity models were trained (D19: 'nosat'): evaluate every nominal-trained
    sensitivity model on each test set it has not been evaluated on yet (saved weights, no retraining)."""
    import torch  # noqa: F401
    from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2
    from fla_terrain.models.v2 import predict_saved
    from fla_terrain.models.v2eval import evaluate_v2
    for st in SENS_SETS:
        todo = [p for p in glob.glob(f"{args.out}/jobs/sens/*.json") if st not in json.load(open(p))["tests"]]
        if not todo or not os.path.isdir(f"{DATA}/v2_sens/{st}"):
            continue
        fs = FrameDataV2.build(f"{DATA}/v2_sens/{st}", cache=cache_path_v2(f"{DATA}/v2_sens/{st}"), workers=16)
        ok = allowed_eps(fs, args.sources)
        idx = fs.split_indices()["test"]
        idx = idx[ok[fs.ep_of_win[idx]]]
        for p in todo:
            r = json.load(open(p))
            r["tests"][st] = evaluate_v2(fs, idx, predict_saved(r["model_dir"], fs, idx, device="cuda"))
            json.dump(r, open(p, "w"), indent=1, default=float)
        print(f"sensitivity models tested on '{st}': {len(todo)}", flush=True)


def fill_in_dist(args):
    """In-distribution M1 for every leave-one-gait-out job: the main model (same input set and seed) on the same
    held-out test windows (from the saved test predictions; labels read from the cache)."""
    from sklearn.metrics import f1_score
    from fla_terrain.models.dataset_v2 import cache_path_v2
    jobs = glob.glob(f"{args.out}/jobs/logo_*/*.json")
    todo = [p for p in jobs if "in_dist_M1" not in json.load(open(p))["test"]]
    if not todo:
        return
    c = np.load(cache_path_v2(f"{DATA}/v2"))
    lab_t, starts = c["lab_terrain"], c["starts"]
    for p in todo:
        r = json.load(open(p))
        v, s = r["job"]["view"], r["job"]["seed"]
        te = np.load(os.path.join(r["model_dir"], "test_pred.npz"))["idx"]
        mp = os.path.join(args.models_root, f"main/{v}_s{s}/test_pred.npz")
        if not os.path.exists(mp):
            continue
        d = np.load(mp)
        pos = {int(i): k for k, i in enumerate(d["idx"])}
        sel = [pos[int(i)] for i in te if int(i) in pos]
        y = lab_t[starts[te[[int(i) in pos for i in te]]] + 49].astype(int)
        pr = d["terrain"][sel]
        m = y >= 0
        r["test"]["in_dist_M1"] = float(f1_score(y[m], pr[m], labels=sorted(set(y[m].tolist())), average="macro"))
        json.dump(r, open(p, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="walk_g113,walk_g80,walk_sonic")
    ap.add_argument("--out", default="results/v2/src3")
    ap.add_argument("--models-root", default="/path/to/fla-terrain/models/v2_src3")
    ap.add_argument("--stages", default="main,sens,sens_nosat,logo,select,course,tables")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rule-dir", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    args.sources = args.sources.split(",")
    from fla_terrain.models.provenance import require_clean
    git = require_clean(args.allow_dirty)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    J = jobs_for(args)
    print(f"run_v2 sources {args.sources} -> {args.out}; {len(J)} training jobs to run; commit {git['commit'][:10]}",
          flush=True)
    if J:
        J.sort(key=lambda j: (j["kind"] != "main", j["kind"]))  # main first (needed for selection), data grouped
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for k, (name, ok, msg) in enumerate(pool.imap_unordered(run_job, J), 1):
                print(f"  [{k}/{len(J)}] {name}: {'ok ' + msg if ok else 'FAILED'} ({(time.time() - t0) / 60:.0f} min)",
                      flush=True)
                if not ok:
                    print(msg, flush=True)
    st = args.stages.split(",")
    if "sens" in st or "sens_nosat" in st:
        sens_extra_tests(args)
    sel = select(args) if "select" in st else None
    if sel:
        print(f"selected sole model: {sel['selected_name']} ({sel['selected']}), val M1 {sel['val_M1_mean'][sel['selected']]:.4f}",
              flush=True)
    if "course" in st:
        course(args, sel)
        print("course scored", flush=True)
    if "tables" in st:
        fill_in_dist(args)
        from fla_terrain.models.v2report import build_tables
        build_tables(args.out, args.sources, git)
        print(f"tables -> {args.out}/tables.md", flush=True)
    print(f"RUN_V2_DONE {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
