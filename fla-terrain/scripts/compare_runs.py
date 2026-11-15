"""Reproducibility: max absolute difference of the headline numbers between two result dirs (same seeds).
  $PY scripts/compare_runs.py --a results/v1_run1 --b results/v1   -> writes <b>/repro.json
Compared per input set and seed: walk terrain macro-F1, accuracy, friction MAE, log10 K MAE, slope MAE, support F1,
grip/slip F1 (G1-1), strip median latency steps and seconds.
"""
import argparse
import json
import os

KEYS = [("headline_walk", "terrain_f1"), ("headline_walk", "terrain_acc"), ("headline_walk", "mu_mae"),
        ("headline_walk", "logk_mae"), ("headline_walk", "slope_mae"), ("headline_walk", "support_f1"),
        ("g1_grip_slip_f1",), ("strip", "median_latency_steps"), ("strip", "median_latency_s")]


def get(d, k):
    for x in k:
        d = d[x]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    A = json.load(open(os.path.join(args.a, "metrics.json")))
    B = json.load(open(os.path.join(args.b, "metrics.json")))
    diffs = {}
    for s in B["runs"]:
        for seed in B["runs"][s]:
            if s in A["runs"] and seed in A["runs"][s]:
                for k in KEYS:
                    d = abs(get(A["runs"][s][seed], k) - get(B["runs"][s][seed], k))
                    name = "/".join(k)
                    diffs[name] = max(diffs.get(name, 0.0), d)
    worst = max(diffs.items(), key=lambda kv: kv[1]) if diffs else ("none", 0.0)
    summary = (f"max |difference| of headline numbers between {args.a} (commit {A.get('git_commit', '?')[:10]}) and "
               f"{args.b} (commit {B.get('git_commit', '?')[:10]}), same seeds: {worst[1]:.4f} ({worst[0]}); per metric: "
               + ", ".join(f"{k} {v:.4f}" for k, v in diffs.items()))
    print(summary)
    json.dump({"summary": summary, "max_abs_diff": diffs}, open(os.path.join(args.b, "repro.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
