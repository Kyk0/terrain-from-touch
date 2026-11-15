"""A12 (EXPLORATORY, post hoc; not part of the D14 answers): why the point-map models fail on the pad sole and the
outline mask. No retraining.

1. The MAIN models (trained on per-episode randomized sole / sensor parameters) P+C2, P+C3, P+C5 x 3 seeds evaluated on
   the five sensitivity test sets (nominal, shifted, pads, g1 = outline mask, nosat), M1 / M2 / M3, next to the
   nominal-trained sensitivity models (results/v2/src3/jobs/sens).
2. Input statistics on the nominal, pads and g1 test sets (sensitivity test episodes, mode walk, foot frames with
   recorded load > 20 N): share of the foot's load in the four corner points and in the outer ring of the grid;
   fraction of loaded frames with a point at the 60 N limit; number of points carrying 90 % of the load; block sums
   (what the block models see) against nominal: per-frame correlation where the simulation is shared (g1) and
   distribution overlap (histogram intersection of the block load shares) for all.
Writes <out>/exploratory_robustness.md and .json.
  $PY scripts/exploratory_robustness.py --out results/v2/src3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402

from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2  # noqa: E402
from fla_terrain.models.provenance import git_state  # noqa: E402
from fla_terrain.models.v2 import predict_saved  # noqa: E402
from fla_terrain.models.v2eval import evaluate_v2  # noqa: E402

DATA = "/path/to/fla-terrain/data/v2_sens"
SETS = ("nominal", "shifted", "pads", "g1", "nosat")
VIEWS = (("P+C2", "P+Nreal+Sblk"), ("P+C3", "P+Nreal"), ("P+C5", "P+Nblk"))
MR = "/path/to/fla-terrain/models/v2_src3"


def ms(v, f="{:.3f}"):
    v = np.asarray(v, float)
    return (f + " ± " + f).format(v.mean(), v.std())


def test_idx(fd):
    idx = fd.split_indices()["test"]
    return idx


def part1():
    out, fds = {}, {s: FrameDataV2.build(f"{DATA}/{s}", cache=cache_path_v2(f"{DATA}/{s}"), workers=16) for s in SETS}
    for name, view in VIEWS:
        for st in SETS:
            fd = fds[st]
            idx = test_idx(fd)
            r = []
            for seed in (0, 1, 2):
                p = predict_saved(f"{MR}/main/{view}_s{seed}", fd, idx, device="cuda")
                e = evaluate_v2(fd, idx, p)
                r.append({k: e[k] for k in ("M1", "M2", "M3")})
            out.setdefault(name, {})[st] = r
            print(f"  main {name} on {st}: M1 {np.mean([x['M1'] for x in r]):.3f}", flush=True)
    return out


def walk_frames(fd):
    """Frame indices of walk test episodes and per-episode slices."""
    sl = []
    for e in fd.eps:
        if e["range_split"] == "test" and e["mode"] == "walk":
            sl.append((e["episode"], e["offset"], e["T"]))
    return sl


def part2(fds):
    rc = np.asarray(fds["nominal"].layout["rowcol"])
    gx, gy = rc[:, 0].max(), rc[:, 1].max()
    corner = ((rc[:, 0] == 0) | (rc[:, 0] == gx)) & ((rc[:, 1] == 0) | (rc[:, 1] == gy))
    ring = (rc[:, 0] == 0) | (rc[:, 0] == gx) | (rc[:, 1] == 0) | (rc[:, 1] == gy)
    res = {}
    shares = {}
    nom_blk = {}
    for st in ("nominal", "pads", "g1"):
        fd = fds[st]
        F, B, keys = [], [], []
        for ep, o, T in walk_frames(fd):
            F.append(fd.PTS[o:o + T, :, :, 2].astype(np.float32).reshape(-1, rc.shape[0]))   # [T*2, P]
            B.append(fd.BLK[o:o + T, :, :, 2].reshape(-1, 4))
            keys += [(ep, t, f) for t in range(T) for f in range(2)]
        F, B = np.concatenate(F), np.concatenate(B)
        tot = F.sum(1)
        m = tot > 20
        Fp = np.clip(F[m], 0, None)
        tp = Fp.sum(1)
        srt = -np.sort(-Fp, 1)
        n90 = (np.cumsum(srt, 1) < 0.9 * tp[:, None]).sum(1) + 1
        # spatial roughness: mean |F_i - mean of its 4 grid neighbours| over the loaded points, / mean loaded point
        G = F[m].reshape(-1, gx + 1, gy + 1)
        Gp = np.pad(G, ((0, 0), (1, 1), (1, 1)), mode="edge")
        nb = (Gp[:, :-2, 1:-1] + Gp[:, 2:, 1:-1] + Gp[:, 1:-1, :-2] + Gp[:, 1:-1, 2:]) / 4
        ld = G > 1.0
        rough = float((np.abs(G - nb)[ld]).mean() / G[ld].mean())
        res[st] = {"loaded_frames": int(m.sum()), "roughness": rough, "corner_share": float((Fp[:, corner].sum(1) / tp).mean()),
                   "ring_share": float((Fp[:, ring].sum(1) / tp).mean()),
                   "frames_at_60N": float((F[m] >= 59.9).any(1).mean()),
                   "points_for_90pct_median": float(np.median(n90)), "points_for_90pct_p10_p90":
                       [float(np.percentile(n90, 10)), float(np.percentile(n90, 90))]}
        shares[st] = B[m] / np.maximum(B[m].sum(1, keepdims=True), 1e-6)
        if st == "nominal":
            nom_blk = {k: b for k, b, mm in zip(keys, B, m) if mm}
        if st == "g1":
            pairs = [(nom_blk[k], b) for k, b, mm in zip(keys, B, m) if mm and k in nom_blk]
            a = np.array([x for x, _ in pairs]).ravel()
            c = np.array([y for _, y in pairs]).ravel()
            res[st]["block_corr_same_sim"] = float(np.corrcoef(a, c)[0, 1])
            res[st]["block_rel_diff_median"] = float(np.median(np.abs(c - a) / np.maximum(np.abs(a), 1.0)))
    bins = np.linspace(-0.2, 1.2, 57)
    for st in ("pads", "g1"):
        ov = []
        for k in range(4):
            h1, _ = np.histogram(np.clip(shares["nominal"][:, k], -0.2, 1.2), bins)
            h2, _ = np.histogram(np.clip(shares[st][:, k], -0.2, 1.2), bins)
            ov.append(float(np.minimum(h1 / h1.sum(), h2 / h2.sum()).sum()))
        res[st]["block_share_overlap"] = ov
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2/src3")
    a = ap.parse_args()
    git = git_state()
    p1 = part1()
    fds = {s: FrameDataV2.build(f"{DATA}/{s}", cache=cache_path_v2(f"{DATA}/{s}"), workers=16) for s in ("nominal", "pads", "g1")}
    p2 = part2(fds)
    sens = {}
    import glob
    for p in sorted(glob.glob(f"{a.out}/jobs/sens/*.json")):
        r = json.load(open(p))
        sens.setdefault(r["job"]["view"], []).append(r["tests"])
    L = ["# EXPLORATORY (A12, post hoc): why the point-map models fail on the pad sole and the outline mask", "",
         "Not part of the seven D14 answers; the pre-registered question-7 verdict stands. No retraining: saved weights "
         "evaluated on the sensitivity test sets (mode walk, test; mean ± population std over 3 seeds). "
         f"Commit `{git['commit']}` (clean tree: {not git['dirty']}). Command: `scripts/exploratory_robustness.py`.", "",
         "## 1. Trained with per-episode randomized sole/sensor parameters (main models) vs trained on the fixed nominal "
         "setting (sensitivity models)", "",
         "| model | training | metric | " + " | ".join(SETS) + " |", "|---|---|---|" + "---|" * len(SETS)]
    for name, view in VIEWS:
        for m in ("M1", "M2", "M3"):
            L.append(f"| {name} | randomized (main) | {m} | " + " | ".join(ms([x[m] for x in p1[name][st]]) for st in SETS) + " |")
        for m in ("M1", "M2", "M3"):
            L.append(f"| {name} | nominal only (sens) | {m} | " + " | ".join(
                ms([t[st][m] for t in sens[view]]) if all(st in t for t in sens[view]) else "n/a" for st in SETS) + " |")
    L += ["", "Note: the main models were trained on data/v2 (randomized), so their nominal column is a sensor setting "
          "inside their training distribution but other episodes (the sensitivity test episodes); shifted lies at the "
          "edges of the randomization ranges (spread 10 mm, compliance 0.4 mm, crosstalk 0.05 are range ends); pads, "
          "outline mask and no saturation are outside it.", "",
          "## 2. What differs in the inputs (walk test episodes, foot frames with recorded load > 20 N)", "",
          "| test set | loaded foot-frames | load share in 4 corner points | load share in outer ring | frames with a point at 60 N | points for 90 % of load, median [p10, p90] | spatial roughness | block sums vs nominal |",
          "|---|---|---|---|---|---|---|---|"]
    for st in ("nominal", "pads", "g1"):
        r = p2[st]
        bl = "reference"
        if st == "g1":
            bl = (f"same simulation: r = {r['block_corr_same_sim']:.3f}, median rel. diff {100 * r['block_rel_diff_median']:.1f} %; "
                  f"share overlap {', '.join(f'{x:.2f}' for x in r['block_share_overlap'])}")
        elif st == "pads":
            bl = f"different simulation (pad sole): share overlap per block {', '.join(f'{x:.2f}' for x in r['block_share_overlap'])}"
        L.append(f"| {st} | {r['loaded_frames']} | {100 * r['corner_share']:.1f} % | {100 * r['ring_share']:.1f} % | "
                 f"{100 * r['frames_at_60N']:.1f} % | {r['points_for_90pct_median']:.0f} [{r['points_for_90pct_p10_p90'][0]:.0f}, "
                 f"{r['points_for_90pct_p10_p90'][1]:.0f}] | {r['roughness']:.2f} | {bl} |")
    # 3. diagnostic: outline mask with the 'point exists' channel forced to all ones (absent points still read 0)
    fg = fds["g1"]
    idx = test_idx(fg)
    saved = [e["present"] for e in fg.eps]
    for e in fg.eps:
        e["present"] = [1] * len(e["present"])
    p3 = {}
    for name, view in VIEWS[:2]:
        for grp, root in (("randomized (main)", f"{MR}/main"), ("nominal only (sens)", f"{MR}/sens")):
            r = []
            for seed in (0, 1, 2):
                p = predict_saved(f"{root}/{view}_s{seed}", fg, idx, device="cuda")
                r.append(evaluate_v2(fg, idx, p)["M1"])
            p3[f"{name} {grp}"] = r
    for e, pr in zip(fg.eps, saved):
        e["present"] = pr
    L += ["", "## 3. Diagnostic: outline mask with the 'point exists' input channel forced to all ones", "",
          "The map encoder has a mask channel (1 = point exists); in training it was always all ones. Here the outline "
          "test set is re-predicted with the channel set to ones (the absent points still read 0 N), M1:", "",
          "| model | M1 outline, mask channel as recorded | M1 outline, mask channel all ones |", "|---|---|---|"]
    for name, view in VIEWS[:2]:
        for grp in ("randomized (main)", "nominal only (sens)"):
            base = [x["M1"] for x in p1[name]["g1"]] if grp.startswith("randomized") else [t["g1"]["M1"] for t in sens[view]]
            L.append(f"| {name} {grp} | {ms(base)} | {ms(p3[f'{name} {grp}'])} |")
    L += ["", "Block share overlap = histogram intersection (0..1) of each block's share of the foot load (front_l, "
          "front_r, back_l, back_r) against nominal; 1 = identical distributions."]
    open(f"{a.out}/exploratory_robustness.md", "w").write("\n".join(L) + "\n")
    json.dump({"part1_main_models": p1, "part2_inputs": p2, "part3_g1_mask_ones": p3, "git": git}, open(f"{a.out}/exploratory_robustness.json", "w"),
              indent=1, default=float)
    print("EXPLORATORY_DONE", flush=True)


if __name__ == "__main__":
    main()
