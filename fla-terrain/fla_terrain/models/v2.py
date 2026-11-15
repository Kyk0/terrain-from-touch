"""v2 models: point-map views through a small masked 2D encoder per foot per frame, then the v1 causal temporal network;
block / proprioception views go straight into the temporal network. Three heads as in v1.

Every trained model is saved under <models_root>/<name>/: model.pt (state dict), config.json (view, seed, git commit,
data dir, architecture, normalization, training info), test_pred.npz (test predictions and window indices).
``load_model`` + ``predict`` reproduce the stored test predictions from the saved weights.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ..schema import N_SUPPORT_CLASSES, N_TERRAIN_CLASSES
from .dataset import W
from .dataset_v2 import FrameDataV2, parse_view
from .nets import CausalConv, loss_fn
from .provenance import git_state

MODELS_ROOT = "/path/to/fla-terrain/models"
LAB_KEYS = ("terrain", "mu", "logk", "slope", "sup_L", "sup_R")


class MapEncoder(nn.Module):
    """[N, C+1 (mask), gx, gy] -> [N, d]. The mask channel marks points that exist (foot-outline variant)."""

    def __init__(self, c_in, grid, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in + 1, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        gx, gy = (grid[0] + 1) // 2, (grid[1] + 1) // 2
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * gx * gy, d), nn.GELU())

    def forward(self, x):
        return self.fc(self.net(x))


class TerrainNetV2(nn.Module):
    def __init__(self, c_vec, c_map, grid, width=96, d_map=64, dropout=0.1):
        super().__init__()
        self.c_vec, self.c_map = c_vec, c_map
        self.enc = MapEncoder(c_map, grid, d_map) if c_map else None
        c_in = c_vec + (2 * d_map if c_map else 0)
        layers, c = [], c_in
        for dil in (1, 2, 4, 8):
            layers += [CausalConv(c, width, 5, dil), nn.BatchNorm1d(width), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.terrain = nn.Linear(2 * width, N_TERRAIN_CLASSES)
        self.props = nn.Linear(2 * width, 3)
        self.support = nn.Sequential(nn.Linear(width + c_in, 64), nn.GELU(), nn.Linear(64, 2 * N_SUPPORT_CLASSES))

    def frame_features(self, vec, maps):
        """vec [B, W, c_vec] or None; maps [B, W, 2, c_map+1, gx, gy] or None -> [B, W, c_in]."""
        parts = [vec] if vec is not None else []
        if maps is not None:
            B, Wn = maps.shape[:2]
            e = self.enc(maps.reshape(B * Wn * 2, *maps.shape[3:])).reshape(B, Wn, -1)
            parts.append(e)
        return torch.cat(parts, -1)

    def forward(self, vec, maps):
        x = self.frame_features(vec, maps)
        h = self.body(x.transpose(1, 2))
        last = h[:, :, -1]
        g = self.drop(torch.cat([h.mean(2), last], 1))
        sup = self.support(torch.cat([self.drop(last), x[:, -1]], 1))
        return {"terrain": self.terrain(g), "props": self.props(g), "support": sup.view(-1, 2, N_SUPPORT_CLASSES)}


AUG = {"gain_std": 0.1, "offset_std_n": 0.3}  # per window, foot, point, axis (training only)


class ViewInputs:
    """Builds normalized model inputs for windows of a FrameDataV2 on a device.

    Sole views are computed on the GPU from the per-point recording: maps (Nreal, Sreal) directly, block views
    (Nblk, Sblk) as per-episode weighted sums of the points (same as the cached BLK). With aug=True (training) every
    window gets a fresh random gain and offset per foot, point and axis BEFORE any view is formed (A9: without it the
    map models memorised the fixed per-episode sensor gain/offset pattern, an episode fingerprint, and overfit after
    one epoch: train F1 0.98, val 0.75). The same augmentation reaches every sole view."""

    def __init__(self, fd: FrameDataV2, view: str, device, norm: dict | None = None, train_idx=None):
        self.fd, self.view, self.dev = fd, view, device
        self.toks = parse_view(view)
        self.c_vec, self.c_map = fd.view_dims(view)
        if norm is None:
            norm = self.fit_norm(train_idx)
        self.norm = norm
        self.ar = torch.arange(W, device=device)
        self.starts = torch.as_tensor(fd.starts, device=device)
        self.ep = torch.as_tensor(fd.ep_of_win, device=device)
        self.vmu = torch.as_tensor(np.asarray(norm["vec_mu"], np.float32), device=device)
        self.vsd = torch.as_tensor(np.asarray(norm["vec_sd"], np.float32), device=device)
        self.np_ = (fd.n_p if "P" in self.toks else 0) + (fd.n_p + 3 if "Pcmd" in self.toks else 0)
        if self.np_:
            V = fd.V[:, :self.np_] if "P" in self.toks else fd.V[:, :fd.n_p + 3]
            self.vp = torch.as_tensor((V - norm["vec_mu"][:self.np_]) / norm["vec_sd"][:self.np_], dtype=torch.float32,
                                      device=device)
        self.sole = any(t in self.toks for t in ("Nreal", "Sreal", "Nblk", "Sblk"))
        if self.sole:
            self.pts = torch.as_tensor(fd.PTS, device=device)                         # fp16 [N, 2, P, 3]
            pres = np.array([e["present"] for e in fd.eps], np.float32)               # [episodes, P]
            self.present = torch.as_tensor(pres, device=device)
            w4 = np.array([e["block_w4"] for e in fd.eps], np.float32)                # [episodes, P, 4]
            self.w4 = torch.as_tensor(w4, device=device)
            self.mch = ([2] if "Nreal" in self.toks else []) + ([0, 1] if "Sreal" in self.toks else [])
            if self.mch:
                self.map_scale = torch.as_tensor(1.0 / np.array([norm["map_sd"][c] for c in self.mch], np.float32),
                                                 device=device)

    def _vec_np(self, frames):
        fd, parts = self.fd, []
        if "P" in self.toks:
            parts.append(fd.V[frames, :fd.n_p])
        if "Pcmd" in self.toks:
            parts.append(fd.V[frames, :fd.n_p + 3])
        if "Nblk" in self.toks:
            parts.append(fd.BLK[frames, :, :, 2].reshape(len(frames), -1))
        if "Sblk" in self.toks:
            parts.append(fd.BLK[frames, :, :, :2].reshape(len(frames), -1))
        return np.concatenate(parts, 1).astype(np.float32) if parts else np.zeros((len(frames), 0), np.float32)

    def fit_norm(self, train_idx):
        fd = self.fd
        eps = np.unique(fd.ep_of_win[train_idx])
        fr = np.concatenate([np.arange(fd.eps[e]["offset"], fd.eps[e]["offset"] + fd.eps[e]["T"]) for e in eps])
        v = self._vec_np(fr)
        pts = fd.PTS[fr[::5]].astype(np.float32)
        return {"vec_mu": v.mean(0), "vec_sd": v.std(0) + 1e-3,
                "map_sd": [float(pts[..., c].std() + 1e-3) for c in range(3)]}

    def batch(self, widx, aug=False):
        """widx: LongTensor of window indices -> (vec [B,W,c] | None, maps [B,W,2,c+1,gx,gy] | None)."""
        fr = self.starts[widx][:, None] + self.ar
        parts = [self.vp[fr]] if self.np_ else []
        maps = None
        if self.sole:
            ep = self.ep[widx]
            pts = self.pts[fr].float()                                                 # [B, W, 2, P, 3]
            pres = self.present[ep]                                                    # [B, P]
            if aug:
                B, P = pts.shape[0], pts.shape[3]
                g = 1 + AUG["gain_std"] * torch.randn(B, 1, 2, P, 3, device=self.dev)
                o = AUG["offset_std_n"] * torch.randn(B, 1, 2, P, 3, device=self.dev)
                pts = (pts * g + o) * pres[:, None, None, :, None]
            off = self.np_
            if "Nblk" in self.toks or "Sblk" in self.toks:
                blk = torch.einsum("bwfpc,bpk->bwfkc", pts, self.w4[ep])               # [B, W, 2, 4, 3]
                B = blk.shape[0]
                if "Nblk" in self.toks:
                    v = blk[..., 2].reshape(B, W, -1)
                    parts.append((v - self.vmu[off:off + 8]) / self.vsd[off:off + 8])
                    off += 8
                if "Sblk" in self.toks:
                    v = blk[..., :2].reshape(B, W, -1)
                    parts.append((v - self.vmu[off:off + 16]) / self.vsd[off:off + 16])
                    off += 16
            if self.mch:
                gx, gy = self.fd.grid
                m = pts[..., self.mch] * self.map_scale
                pm = pres[:, None, None, :, None].expand(-1, W, 2, -1, 1)
                m = torch.cat([m, pm], -1)
                B = m.shape[0]
                maps = m.permute(0, 1, 2, 4, 3).reshape(B, W, 2, m.shape[-1], gx, gy)
        vec = torch.cat(parts, -1) if parts else None
        return vec, maps


def labels_t(fd, device):
    out = {}
    for k in LAB_KEYS:
        v = fd.lab[k][fd.last]
        out[k] = torch.as_tensor(v.astype(np.int64) if k in ("terrain", "sup_L", "sup_R") else v.astype(np.float32),
                                 device=device)
    return out


def train_v2(fd, view, seed, splits, name, data_dir, epochs=25, patience=5, batch=512, lr=2e-3, device="cuda",
             models_root=MODELS_ROOT, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    vi = ViewInputs(fd, view, device, train_idx=splits["train"])
    model = TerrainNetV2(vi.c_vec, vi.c_map, fd.grid).to(device)
    lab = labels_t(fd, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = torch.as_tensor(splits["train"], device=device)
    va = torch.as_tensor(splits["val"], device=device)
    best, best_ep, best_state, bad, t0 = np.inf, -1, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = loss_fn(model(*vi.batch(b, aug=True)), {k: v[b] for k, v in lab.items()})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(float(loss_fn(model(*vi.batch(va[i:i + 2048])), {k: v[va[i:i + 2048]] for k, v in lab.items()}))
                     * len(va[i:i + 2048]) for i in range(0, len(va), 2048)) / max(len(va), 1)
        if vl < best - 1e-4:
            best, best_ep, bad = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    te = torch.as_tensor(splits["test"], device=device)
    pred = predict_v2(model, vi, te)
    pred_val = predict_v2(model, vi, va)
    info = {"best_epoch": best_ep, "val_loss": best, "epochs_run": ep + 1, "train_s": time.time() - t0,
            "n_params": sum(p.numel() for p in model.parameters())}
    cfg = {"name": name, "view": view, "seed": seed, "data_dir": data_dir, "git": git_state(), "window": W,
           "arch": {"class": "TerrainNetV2", "c_vec": vi.c_vec, "c_map": vi.c_map, "grid": list(fd.grid), "width": 96,
                    "d_map": 64}, "train": {"epochs": epochs, "patience": patience, "batch": batch, "lr": lr, **info},
           "aug": AUG, "split": "range_split", "n_windows": {k: int(len(v)) for k, v in splits.items()},
           "norm": {"vec_mu": np.asarray(vi.norm["vec_mu"]).tolist(), "vec_sd": np.asarray(vi.norm["vec_sd"]).tolist(),
                    "map_sd": list(vi.norm["map_sd"])}}
    d = os.path.join(models_root, name)
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, "model.pt"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(d, "test_pred.npz"), idx=splits["test"],
                        **{k: v for k, v in pred.items() if k != "terrain_prob"}, terrain_prob=pred["terrain_prob"])
    np.savez_compressed(os.path.join(d, "val_pred.npz"), idx=splits["val"], **pred_val)
    log(f"    {view:14s} seed {seed}: best epoch {best_ep}, val loss {best:.4f}, {info['train_s']:.0f} s -> {d}")
    info["pred_val"] = pred_val
    return model, pred, info


def predict_v2(model, vi, widx_t, batch=1024):
    return _predict(model, vi, widx_t, batch)


def _predict(model, vi, widx_t, batch):
    outs = {k: [] for k in ("terrain", "mu", "logk", "slope", "sup_L", "sup_R", "terrain_prob")}
    with torch.no_grad():
        for i in range(0, len(widx_t), batch):
            o = model(*vi.batch(widx_t[i:i + batch]))
            pr = torch.softmax(o["terrain"], 1)
            outs["terrain_prob"].append(pr.cpu().numpy().astype(np.float16))
            outs["terrain"].append(pr.argmax(1).cpu().numpy())
            for j, k in enumerate(("mu", "logk", "slope")):
                outs[k].append(o["props"][:, j].cpu().numpy())
            s = o["support"].argmax(2).cpu().numpy()
            outs["sup_L"].append(s[:, 0])
            outs["sup_R"].append(s[:, 1])
    return {k: np.concatenate(v) for k, v in outs.items()}


def load_model(model_dir, device="cpu"):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    a = cfg["arch"]
    model = TerrainNetV2(a["c_vec"], a["c_map"], tuple(a["grid"]), width=a["width"], d_map=a["d_map"]).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.eval()
    return model, cfg


def predict_saved(model_dir, fd, idx=None, device="cpu"):
    """Predictions of a saved model on windows idx of fd (default: the stored test windows)."""
    model, cfg = load_model(model_dir, device)
    n = cfg["norm"]
    norm = {"vec_mu": np.array(n["vec_mu"], np.float32), "vec_sd": np.array(n["vec_sd"], np.float32),
            "map_sd": n["map_sd"]}
    vi = ViewInputs(fd, cfg["view"], device, norm=norm)
    if idx is None:
        idx = np.load(os.path.join(model_dir, "test_pred.npz"))["idx"]
    return _predict(model, vi, torch.as_tensor(idx, device=device), 1024)
