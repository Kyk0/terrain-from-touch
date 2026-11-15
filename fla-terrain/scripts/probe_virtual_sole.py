#!/usr/bin/env python
"""B13: "virtual sole" probe -- how much of the sole signal can proprioception predict?

  python scripts/probe_virtual_sole.py --data /path/to/fla-terrain/data/v1 --out results/v1/probe.json

Input: the P window (1.0 s, 50 Hz: joint pos/vel, gyro, gravity, cmd) from the training implementation's frame cache and splits
(FrameData, generation-time range_split, leakage-checked), walk-mode windows only. Target: sole signals at the
window's last frame (after the sensor model): 16 zone Fz, 32 zone Fx/Fy, 2 foot Fz, 4 foot Fx/Fy (all / body
weight), CoP x, y per foot from zone Fz and zone centres (stance only, masked loss). Causal 2-layer GRU, read-out at
the last step; trained on train, epoch selected on val loss, metrics on test only. No tuning on test.
R2 = 1 - SSE / SST pooled over the targets of a group (SST around each target's own test mean); MAE in body weights
(CoP in mm). "stance" = frames where that foot's true total Fz > 20 N.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fla_terrain.models.dataset import W, FrameData  # noqa: E402

BW = 193.0
FZ_STANCE = 20.0
CLASSES = {0: "hard_grip", 1: "hard_slip", 2: "soft", 3: "rough", 4: "slope", 5: "edge"}
# zone centres (sole frame, m), z = i_along * 2 + i_across (SoleConfig 2x4, 0.18 x 0.07)
ZC = np.array([(-0.09 + 0.045 * (i + 0.5), -0.035 + 0.035 * (j + 0.5)) for i in range(4) for j in range(2)])


class GRUProbe(nn.Module):
    def __init__(self, c_in, c_out, h=128):
        super().__init__()
        self.gru = nn.GRU(c_in, h, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, c_out))

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(y[:, -1])


def targets(fd, last):
    g, X = fd.groups, fd.X
    N = X[last][:, g["N"]] / BW                    # [n, 16] Fz zones L(8) R(8)
    S = X[last][:, g["S"]] / BW                    # [n, 32] FxL(8) FyL(8) FxR(8) FyR(8)
    Nt = X[last][:, g["Ntot"]] / BW                # [n, 2]
    St = X[last][:, g["Stot"]] / BW                # [n, 4] FxL FyL FxR FyR
    cop, mask = np.zeros((len(last), 4), np.float32), np.zeros((len(last), 4), np.float32)
    for f in range(2):
        fz = X[last][:, g["N"][8 * f:8 * f + 8]]
        tot = fz.sum(1)
        ok = tot > FZ_STANCE
        c = (fz @ ZC) / np.maximum(tot, 1e-6)[:, None]
        cop[:, 2 * f:2 * f + 2] = np.where(ok[:, None], c / 0.05, 0.0)   # /5 cm
        mask[:, 2 * f:2 * f + 2] = ok[:, None]
    Y = np.concatenate([N, S, Nt, St, cop], 1).astype(np.float32)
    M = np.concatenate([np.ones((len(last), 54), np.float32), mask], 1)
    stance = np.stack([Nt[:, 0] * BW > FZ_STANCE, Nt[:, 1] * BW > FZ_STANCE], 1)
    return Y, M, stance


# column groups in Y and which foot each column belongs to
def groups():
    foot = np.zeros(58, int)
    foot[8:16] = 1                                  # Fz zones R
    foot[16 + 16:48] = 1                            # Fx/Fy zones R
    foot[49] = 1                                    # Fz tot R
    foot[52:54] = 1                                 # Fx/Fy tot R
    foot[56:58] = 1                                 # CoP R
    return {"a_foot_Fz": np.arange(48, 50), "b_foot_shear": np.arange(50, 54), "c_zone_Fz": np.arange(0, 16),
            "d_zone_shear": np.arange(16, 48), "e_CoP": np.arange(54, 58)}, foot


def metrics(Yt, Yp, cols, rows):
    """pooled R2 and MAE over the given columns; rows: [n, len(cols)] bool selection per column."""
    sse = sst = ae = cnt = 0.0
    for k, c in enumerate(cols):
        r = rows[:, k]
        if r.sum() < 10:
            continue
        y, p = Yt[r, c], Yp[r, c]
        sse += np.sum((y - p) ** 2)
        sst += np.sum((y - y.mean()) ** 2)
        ae += np.sum(np.abs(y - p))
        cnt += r.sum()
    return (1 - sse / sst if sst > 0 else float("nan")), (ae / cnt if cnt else float("nan")), int(cnt)


def evaluate(Yt, Yp, M, stance, terrain):
    G, foot = groups()
    out = {}
    for cls_name, sel in [("all", np.ones(len(Yt), bool))] + [(n, terrain == k) for k, n in CLASSES.items()]:
        if sel.sum() < 50:
            continue
        res = {}
        for gname, cols in G.items():
            for scope in ("all", "stance"):
                if gname == "e_CoP" and scope == "all":
                    continue
                rows = np.ones((sel.sum(), len(cols)), bool) if scope == "all" else stance[sel][:, foot[cols]]
                if gname == "e_CoP":
                    rows &= M[sel][:, cols] > 0
                r2, mae, n = metrics(Yt[sel], Yp[sel], cols, rows)
                scale = 50.0 if gname == "e_CoP" else 1.0      # CoP -> mm
                res[f"{gname}/{scope}"] = {"r2": float(r2), "mae": float(mae * scale), "n": n}
        out[cls_name] = res
    # residual shear (b) during stance: RMS in N and as fraction of Fz, grip vs slip
    for k in (0, 1):
        sel = terrain == k
        res_n, ratio = [], []
        for f in range(2):
            st = sel & stance[:, f]
            cols = G["b_foot_shear"][2 * f:2 * f + 2]
            d = (Yt[st][:, cols] - Yp[st][:, cols]) * BW
            res_n.append(np.hypot(d[:, 0], d[:, 1]))
            ratio.append(np.hypot(d[:, 0], d[:, 1]) / (Yt[st][:, 48 + f] * BW))
        rn, rr = np.concatenate(res_n), np.concatenate(ratio)
        out.setdefault("residual_shear", {})[CLASSES[k]] = {
            "rms_N": float(np.sqrt(np.mean(rn ** 2))), "median_N": float(np.median(rn)),
            "p90_N": float(np.percentile(rn, 90)), "median_over_Fz": float(np.median(rr)), "n": int(len(rn))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/path/to/fla-terrain/data/v1")
    ap.add_argument("--out", default="results/v1/probe.json")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=1024)
    a = ap.parse_args()
    t0 = time.time()
    fd = FrameData.build(a.data, cache=a.data.rstrip("/") + "_cache.npz")
    sp = fd.split_indices()
    mode = np.array([e["mode"] for e in fd.eps])[fd.ep_of_win]
    idx = {s: i[mode[i] == "walk"] for s, i in sp.items()}
    dev = torch.device("cuda")
    Pcols = fd.groups["P"]
    Xp = fd.X[:, Pcols]
    mu, sd = Xp[fd.starts[idx["train"]]].mean(0), Xp[fd.starts[idx["train"]]].std(0) + 1e-6   # train stats
    Xg = torch.tensor((Xp - mu) / sd, dtype=torch.float32, device=dev)
    data = {}
    for s, i in idx.items():
        Y, M, stance = targets(fd, fd.last[i])
        data[s] = (torch.tensor(fd.starts[i], device=dev), torch.tensor(Y, device=dev), torch.tensor(M, device=dev),
                   stance, fd.lab["terrain"][fd.last[i]])
    ysd = data["train"][1].std(0) + 1e-3                                 # per-target loss scale (train)
    ar = torch.arange(W, device=dev)

    def batches(s, shuffle, g=None):
        st, Y, M = data[s][:3]
        order = torch.randperm(len(st), device=dev, generator=g) if shuffle else torch.arange(len(st), device=dev)
        for b in range(0, len(st), a.bs):
            j = order[b:b + a.bs]
            yield Xg[st[j][:, None] + ar[None]], Y[j], M[j]

    def predict(model, s):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x) for x, _, _ in batches(s, False)]).cpu().numpy()

    runs = {}
    for seed in [int(x) for x in a.seeds.split(",")]:
        torch.manual_seed(seed)
        g = torch.Generator(device=dev).manual_seed(seed)
        model = GRUProbe(len(Pcols), 58).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        best, best_state, hist = 1e9, None, []
        for ep in range(a.epochs):
            model.train()
            for x, y, m in batches("train", True, g):
                loss = (((model(x) - y) / ysd) ** 2 * m).sum() / m.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                vl = np.mean([float(((((model(x) - y) / ysd) ** 2 * m).sum() / m.sum()).item())
                              for x, y, m in batches("val", False)])
            hist.append(vl)
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        _, Yt, Mt, stance, terr = data["test"]
        runs[seed] = {"val_loss": hist, "best_epoch": int(np.argmin(hist)),
                      "test": evaluate(Yt.cpu().numpy(), predict(model, "test"), Mt.cpu().numpy(), stance, terr)}
        print(f"[seed {seed}] best epoch {runs[seed]['best_epoch']} val {best:.3f} ({time.time() - t0:.0f}s, "
              f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"cmd": " ".join(sys.argv), "windows": {s: int(len(i)) for s, i in idx.items()}, "runs": runs},
              open(a.out, "w"), indent=1)
    # summary: mean +- spread over seeds, test split, walk mode
    S = [runs[s]["test"] for s in runs]

    def ms(cls, key, f):
        v = [x[cls][key][f] for x in S if cls in x]
        return f"{np.mean(v):.2f}" + (f"+-{np.std(v):.2f}" if len(v) > 1 else "")

    print(f"[B13] P -> sole probe, test split walk ({len(idx['test'])} windows), mean+-sd over seeds {list(runs)}; "
          f"R2 | MAE (BW; CoP mm) all frames / stance frames")
    for gname in ("a_foot_Fz", "b_foot_shear", "c_zone_Fz", "d_zone_shear", "e_CoP"):
        al = (f"{ms('all', gname + '/all', 'r2')} | {ms('all', gname + '/all', 'mae')}  /  " if gname != "e_CoP" else "")
        print(f"  {gname:13s} {al}{ms('all', gname + '/stance', 'r2')} | {ms('all', gname + '/stance', 'mae')}")
    print("  stance R2 per class (foot Fz / foot shear / zone Fz / zone shear / CoP): " + "; ".join(
        f"{c} " + "/".join(ms(c, g + "/stance", "r2").split("+-")[0] for g in
                           ("a_foot_Fz", "b_foot_shear", "c_zone_Fz", "d_zone_shear", "e_CoP"))
        for c in CLASSES.values() if c in S[0]))
    for c in ("hard_grip", "hard_slip"):
        r = [x["residual_shear"][c] for x in S]
        print(f"  unexplained foot shear in stance, {c}: RMS {np.mean([x['rms_N'] for x in r]):.1f} N, median "
              f"{np.mean([x['median_N'] for x in r]):.1f} N, p90 {np.mean([x['p90_N'] for x in r]):.1f} N, "
              f"median residual/Fz {np.mean([x['median_over_Fz'] for x in r]):.3f}")
    print(f"  json -> {a.out}")


if __name__ == "__main__":
    main()
