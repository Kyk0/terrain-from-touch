"""Train TerrainNet on FrameData windows (GPU gather), select the epoch on val loss, predict test."""
from __future__ import annotations

import time

import numpy as np
import torch

from .dataset import W
from .nets import TerrainNet, loss_fn, predict_tensor

LAB_KEYS = ("terrain", "mu", "logk", "slope", "sup_L", "sup_R")


def train_net(fd, input_set, seed, splits, epochs=25, patience=5, batch=1024, lr=2e-3, device="cuda", log=print,
              drop_cols=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols = fd.columns(input_set)
    if drop_cols is not None:  # shortcut tests: remove some channels (e.g. the command)
        cols = np.array([c for c in cols if c not in set(drop_cols)])
    tr = splits["train"]
    # per-channel standardization from the frames of train windows' episodes
    tr_eps = np.unique(fd.ep_of_win[tr])
    fr = np.concatenate([np.arange(fd.eps[e]["offset"], fd.eps[e]["offset"] + fd.eps[e]["T"]) for e in tr_eps])
    Xc = fd.X[:, cols]
    mu, sd = Xc[fr].mean(0), Xc[fr].std(0) + 1e-3
    Xg = torch.as_tensor((Xc - mu) / sd, dtype=torch.float32, device=device)
    starts = torch.as_tensor(fd.starts, device=device)
    ar = torch.arange(W, device=device)
    lab = {}
    for k in LAB_KEYS:
        v = fd.lab[k][fd.last]
        lab[k] = torch.as_tensor(v.astype(np.int64) if k in ("terrain", "sup_L", "sup_R") else v.astype(np.float32),
                                 device=device)

    def get(idx):
        return Xg[starts[idx][:, None] + ar]

    def labels(idx):
        return {k: v[idx] for k, v in lab.items()}

    model = TerrainNet(len(cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr_t = torch.as_tensor(tr, device=device)
    va_t = torch.as_tensor(splits["val"], device=device)
    best, best_ep, best_state, bad = np.inf, -1, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = loss_fn(model(get(b)), labels(b))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for i in range(0, len(va_t), 8192):
                b = va_t[i:i + 8192]
                vl += float(loss_fn(model(get(b)), labels(b))) * len(b)
                n += len(b)
            vl /= max(n, 1)
        if vl < best - 1e-4:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    te_t = torch.as_tensor(splits["test"], device=device)
    pred = predict_tensor(model, lambda i0, i1: get(te_t[i0:i1]), len(te_t))
    info = {"best_epoch": best_ep, "val_loss": best, "epochs_run": ep + 1, "train_s": time.time() - t0,
            "n_params": sum(p.numel() for p in model.parameters()), "channels": int(len(cols))}
    log(f"    {input_set:12s} seed {seed}: best epoch {best_ep}, val loss {best:.4f}, {info['train_s']:.0f} s")
    return model, pred, info
