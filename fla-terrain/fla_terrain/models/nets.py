"""Small temporal CNN with three heads: terrain class, properties (mu, log10 K), support state L/R."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import N_SUPPORT_CLASSES, N_TERRAIN_CLASSES


class CausalConv(nn.Module):
    def __init__(self, c_in, c_out, k, dil):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dil)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TerrainNet(nn.Module):
    """Causal dilated 1D-CNN (receptive field 61 frames >= window). Heads: terrain (pooled + last step),
    properties (same), support state per foot (last step + raw last input frame)."""

    def __init__(self, c_in: int, width: int = 96, dropout: float = 0.1):
        super().__init__()
        layers = []
        c = c_in
        for dil in (1, 2, 4, 8):
            layers += [CausalConv(c, width, 5, dil), nn.BatchNorm1d(width), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        h = 2 * width
        self.terrain = nn.Linear(h, N_TERRAIN_CLASSES)
        self.props = nn.Linear(h, 3)  # mu, log10 K, slope [rad]
        self.support = nn.Sequential(nn.Linear(width + c_in, 64), nn.GELU(), nn.Linear(64, 2 * N_SUPPORT_CLASSES))

    def forward(self, x):  # x [B, W, C]
        h = self.body(x.transpose(1, 2))  # [B, width, W]
        last = h[:, :, -1]
        g = self.drop(torch.cat([h.mean(2), last], 1))
        sup = self.support(torch.cat([self.drop(last), x[:, -1]], 1))
        return {"terrain": self.terrain(g), "props": self.props(g), "support": sup.view(-1, 2, N_SUPPORT_CLASSES)}


PROP_KEYS = ("mu", "logk", "slope")
PROP_SCALE = (0.1, 1.0, 0.05)
SUPPORT_WEIGHTS = (1.0, 3.0, 1.0)  # partial support is ~4 % of windows; fixed a priori, not tuned


def loss_fn(out, yb):
    t = yb["terrain"]
    known = t >= 0
    lt = F.cross_entropy(out["terrain"], t, ignore_index=-1) if known.any() else out["terrain"].sum() * 0
    lp = 0.0
    for i, (k, sc) in enumerate(zip(PROP_KEYS, PROP_SCALE)):
        y = yb[k]
        m = torch.isfinite(y) & known  # properties are unobservable when the terrain label is unknown
        if m.any():
            lp = lp + F.smooth_l1_loss(out["props"][m, i] / sc, y[m] / sc)
    w = torch.tensor(SUPPORT_WEIGHTS, device=t.device)
    ls = F.cross_entropy(out["support"].reshape(-1, N_SUPPORT_CLASSES),
                         torch.stack([yb["sup_L"], yb["sup_R"]], 1).reshape(-1), weight=w)
    return lt + lp + ls


def predict_tensor(model, get_batch, n, batch=4096):
    """get_batch(i0, i1) -> input tensor [B, W, C]. Returns numpy predictions for n windows."""
    model.eval()
    outs = {k: [] for k in ("terrain", "mu", "logk", "slope", "sup_L", "sup_R", "terrain_prob")}
    with torch.no_grad():
        for i in range(0, n, batch):
            o = model(get_batch(i, min(n, i + batch)))
            pr = torch.softmax(o["terrain"], 1)
            outs["terrain_prob"].append(pr.cpu().numpy().astype(np.float16))
            outs["terrain"].append(pr.argmax(1).cpu().numpy())
            for j, k in enumerate(PROP_KEYS):
                outs[k].append(o["props"][:, j].cpu().numpy())
            s = o["support"].argmax(2).cpu().numpy()
            outs["sup_L"].append(s[:, 0])
            outs["sup_R"].append(s[:, 1])
    return {k: np.concatenate(v) for k, v in outs.items()}
