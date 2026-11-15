"""Shared data preparation and metrics for the baseline and the neural model."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..data import check_no_leakage, load_windows, make_split_mapping, split_windows
from ..schema import N_TERRAIN_CLASSES, TERRAIN_LABELS


def load_splits(root, input_set: str, split_mode: str = "auto"):
    """Windows for one input set, split by split_key (range_split or hash) with the leakage check."""
    ws = load_windows(root, input_set)
    mapping = make_split_mapping(ws, split_mode)
    parts = split_windows(ws, mapping)  # calls check_no_leakage
    check_no_leakage({s: p.split_key for s, p in parts.items()})
    return ws, parts, mapping


def targets(ws):
    return {
        "terrain": ws.terrain.astype(np.int64),
        "mu": ws.mu.astype(np.float32),
        "logk": np.log10(np.asarray(ws.stiffness, np.float64)).astype(np.float32),
        "sup_L": ws.support[:, 0].astype(np.int64),
        "sup_R": ws.support[:, 1].astype(np.int64),
    }


def metrics(y: dict, p: dict) -> dict:
    """y/p dicts with terrain (labels), mu, logk (floats), sup_L, sup_R (labels). NaN/-1 targets ignored."""
    out = {}
    m = y["terrain"] >= 0
    if m.any():
        present = sorted(set(y["terrain"][m].tolist()))
        out["terrain_acc"] = float((p["terrain"][m] == y["terrain"][m]).mean())
        out["terrain_f1"] = float(f1_score(y["terrain"][m], p["terrain"][m], labels=present, average="macro"))
        out["n_eval"] = int(m.sum())
    for k in ("mu", "logk"):
        mk = np.isfinite(y[k]) & np.isfinite(p[k])
        out[f"{k}_mae"] = float(np.abs(p[k][mk] - y[k][mk]).mean()) if mk.any() else float("nan")
    f = []
    for s in ("sup_L", "sup_R"):
        present = sorted(set(y[s].tolist()))
        f.append(f1_score(y[s], p[s], labels=present, average="macro"))
    out["support_f1"] = float(np.mean(f))
    return out


def fmt_row(name, r):
    return (f"{name:8s} acc {r.get('terrain_acc', np.nan):.3f} F1 {r.get('terrain_f1', np.nan):.3f} "
            f"mu-MAE {r['mu_mae']:.3f} logK-MAE {r['logk_mae']:.3f} supF1 {r['support_f1']:.3f}")


def class_counts(t):
    return {TERRAIN_LABELS[c]: int((t == c).sum()) for c in range(-1, N_TERRAIN_CLASSES) if (t == c).any()}
