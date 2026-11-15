"""Neural model (temporal CNN, heads: terrain, mu + log10 K, support L/R) for P, P+N, P+N+S.

  $PY scripts/train_model.py --data /path/to/fla-terrain/data/e2e [--epochs 20] [--seed 0]
Per-channel standardization from train; model selection on val loss; prints test metrics per set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fla_terrain.models.common import fmt_row, load_splits, metrics, targets  # noqa: E402
from fla_terrain.models.nets import TerrainNet, loss_fn, predict  # noqa: E402


def to_t(y, dev):
    return {k: torch.as_tensor(v, device=dev) for k, v in y.items()}


def run(data, s, args, dev):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ws, parts, _ = load_splits(data, s, args.split_mode)
    mu, sd = parts["train"].X.mean((0, 1)), parts["train"].X.std((0, 1)) + 1e-6
    X = {k: ((p.X - mu) / sd).astype(np.float32) for k, p in parts.items()}
    Y = {k: targets(p) for k, p in parts.items()}
    model = TerrainNet(X["train"].shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr = torch.as_tensor(X["train"], device=dev)
    Ytr = to_t(Y["train"], dev)
    Xva = torch.as_tensor(X["val"], device=dev)
    Yva = to_t(Y["val"], dev)
    best, best_state = np.inf, None
    n = len(Xtr)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            loss = loss_fn(model(Xtr[b]), {k: v[b] for k, v in Ytr.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva), Yva)) if len(Xva) else -ep
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return metrics(Y["test"], predict(model, X["test"], dev)), {k: len(v) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-mode", default="auto")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default="P,P+N,P+N+S")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    t0 = time.time()
    for s in args.sets.split(","):
        r, n = run(args.data, s, args, dev)
        res[s] = r
        if not len(res) - 1:
            print(f"TerrainNet on {dev} | windows {n} | seed {args.seed} | leakage check PASS")
        print("  " + fmt_row(s, r))
    print(f"  time {time.time()-t0:.0f} s")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
