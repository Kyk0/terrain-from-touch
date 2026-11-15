"""Evaluation exactly as DECISIONS.md D7 (pre-registered) plus the brief's T8 metrics.

Inputs: FrameData, test window indices, predictions for those windows (terrain, mu, logk, slope, sup_L, sup_R).
All windows are the standard 1.0 s / 0.2 s-stride windows; delays therefore have 0.2 s resolution.

- terrain: macro-F1 over the classes present (label >= 0) + confusion matrix (rows true, cols predicted, 6x6),
  per mode and per band group (test_in / test_out); headline = mode walk, both bands pooled.
- G1-1: walk, non-strip episodes, true class hard_grip/hard_slip: F1 of each of the two classes with predictions as
  they are (a third class counts as a miss), averaged.
- friction MAE: windows with >= 1 foot in contact (GT support > 0) and label >= 0; all classes pooled and
  grip+slip only. Stiffness: MAE of log10 K (same subset; also soft only). Slope MAE in degrees.
- support F1: macro over (airborne, partial, full) per foot, mean of L and R.
- strip latency (G1-2): strip episodes; t0 = first touchdown of either foot on the new surface (recorder log);
  detection = first window end t >= t0 whose predicted class equals the new class for 2 consecutive windows;
  latency seconds and steps (touchdowns on the new surface in [t0, t]); not detected within 5 s -> 5 s / steps in
  5 s and counted as a miss. Episodes that never touch the new surface are excluded (counted).
- edge delay: edge episodes; for every stance (GT contact run) of a foot whose GT support is partial within its
  first 0.1 s: t_p = first partial frame; delay = first window end in [t_p, stance end] predicting partial for that
  foot, minus t_p; none -> miss.
- lifted delay: lifted-mid episodes; t_a = first frame at/after the lift start with both feet GT airborne;
  delay = first window end >= t_a predicting airborne for both feet for 2 consecutive windows, minus t_a; none
  before the put-down -> miss.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from ..schema import N_TERRAIN_CLASSES

STRIDE_S = 0.2
MAX_LAT = 5.0


def _f1(y, p, labels=None):
    labels = labels if labels is not None else sorted(set(y.tolist()))
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0)) if len(y) else float("nan")


def _block(y, pred, m):
    """Metrics for the windows in mask m."""
    out = {"n": int(m.sum())}
    t = m & (y["terrain"] >= 0)
    if t.any():
        out["terrain_f1"] = _f1(y["terrain"][t], pred["terrain"][t])
        out["terrain_acc"] = float((y["terrain"][t] == pred["terrain"][t]).mean())
        out["confusion"] = confusion_matrix(y["terrain"][t], pred["terrain"][t],
                                            labels=list(range(N_TERRAIN_CLASSES))).tolist()
    c = t & ((y["sup_L"] > 0) | (y["sup_R"] > 0))
    for k, sc in (("mu", 1.0), ("logk", 1.0), ("slope", 180 / np.pi)):
        mk = c & np.isfinite(y[k])
        out[f"{k}_mae"] = float(np.abs(pred[k][mk] - y[k][mk]).mean() * sc) if mk.any() else float("nan")
    gs = c & np.isin(y["terrain"], [0, 1]) & np.isfinite(y["mu"])
    out["mu_mae_grip_slip"] = float(np.abs(pred["mu"][gs] - y["mu"][gs]).mean()) if gs.any() else float("nan")
    so = c & (y["terrain"] == 2)
    out["logk_mae_soft"] = float(np.abs(pred["logk"][so] - y["logk"][so]).mean()) if so.any() else float("nan")
    if m.any():
        # D7 number (walk): macro over (airborne, partial, full) per foot, mean of L and R
        out["support_f1"] = float(np.mean([_f1(y[s][m], pred[s][m], [0, 1, 2]) if len(set(y[s][m].tolist())) > 1
                                           else _f1(y[s][m], pred[s][m]) for s in ("sup_L", "sup_R")]))
        # macro over the classes present in this subset only (for per-mode tables)
        out["support_f1_present"] = float(np.mean([_f1(y[s][m], pred[s][m]) for s in ("sup_L", "sup_R")]))
        # per class (airborne, partial, full), mean of L and R; NaN if the class is absent in the subset
        pc = []
        for s in ("sup_L", "sup_R"):
            f = f1_score(y[s][m], pred[s][m], labels=[0, 1, 2], average=None, zero_division=0)
            pc.append([f[c] if (y[s][m] == c).any() else np.nan for c in range(3)])
        out["support_f1_per_class"] = np.nanmean(np.array(pc, float), 0).tolist()
    return out


def evaluate(fd, idx, pred) -> dict:
    y = fd.win_labels(idx)
    y = {k: np.asarray(v) for k, v in y.items()}
    y["terrain"] = y["terrain"].astype(int)
    for s in ("sup_L", "sup_R"):
        y[s] = y[s].astype(int)
    ep = fd.ep_of_win[idx]
    E = fd.eps
    mode = np.array([E[e]["mode"] for e in ep])
    band = np.array([E[e]["band"] for e in ep])
    cls = np.array([E[e]["class"] for e in ep])
    res = {"by_mode": {}, "by_band_walk": {}}
    for md in ("walk", "stand", "rig", "lifted"):
        if (mode == md).any():
            res["by_mode"][md] = _block(y, pred, mode == md)
    for b in ("test_in", "test_out"):
        mm = (mode == "walk") & (band == b)
        if mm.any():
            res["by_band_walk"][b] = _block(y, pred, mm)
    res["headline_walk"] = res["by_mode"].get("walk", {})
    # G1-1
    g = (mode == "walk") & ~np.char.startswith(cls.astype(str), "strip") & np.isin(y["terrain"], [0, 1])
    res["g1_grip_slip_f1"] = float(np.mean(f1_score(y["terrain"][g], pred["terrain"][g], labels=[0, 1], average=None,
                                                    zero_division=0))) if g.any() else float("nan")
    res["g1_grip_slip_n"] = int(g.sum())
    # per-episode sequences
    t_end = fd.lab["t"][fd.last[idx]]
    by_ep = {}
    for i, e in enumerate(ep):
        by_ep.setdefault(int(e), []).append(i)
    res["strip"] = _strip_latency(fd, by_ep, t_end, pred)
    res["edge_delay"] = _edge_delay(fd, by_ep, t_end, pred)
    res["lifted_delay"] = _lifted_delay(fd, by_ep, t_end, pred)
    return res


def _strip_latency(fd, by_ep, t_end, pred):
    rows, never = [], 0
    for e, ii in by_ep.items():
        em = fd.eps[e]
        if not em["class"].startswith("strip") or em["mode"] != "walk":
            continue
        tds = [(t, r) for t, _, r in em["touchdowns"] if 1 in r]
        if not tds:
            never += 1
            continue
        t0 = min(t for t, _ in tds)
        new = em["regions"][1]
        ii = sorted(ii, key=lambda i: t_end[i])
        te = np.array([t_end[i] for i in ii])
        pc = np.array([pred["terrain"][i] for i in ii])
        det = None
        for k in range(len(ii) - 1):
            if te[k] >= t0 - 1e-6 and te[k] - t0 <= MAX_LAT and pc[k] == new and pc[k + 1] == new:
                det = te[k]
                break
        lat = MAX_LAT if det is None else float(det - t0)
        steps = sum(1 for t, _ in tds if t0 - 1e-6 <= t <= t0 + lat + 1e-6)
        rows.append({"episode": em["episode"], "type": em["class"], "latency_s": lat, "latency_steps": steps,
                     "miss": det is None})
    out = {"n": len(rows), "never_touched": never}
    if rows:
        out["median_latency_s"] = float(np.median([r["latency_s"] for r in rows]))
        out["median_latency_steps"] = float(np.median([r["latency_steps"] for r in rows]))
        out["miss_rate"] = float(np.mean([r["miss"] for r in rows]))
        out["per_type"] = {}
        for ty in sorted({r["type"] for r in rows}):
            rr = [r for r in rows if r["type"] == ty]
            out["per_type"][ty] = {"n": len(rr), "median_latency_s": float(np.median([r["latency_s"] for r in rr])),
                                   "median_latency_steps": float(np.median([r["latency_steps"] for r in rr])),
                                   "miss_rate": float(np.mean([r["miss"] for r in rr]))}
    return out


def _gt(fd, em, key):
    o = em["offset"]
    return fd.lab[key][o:o + em["T"]]


def _edge_delay(fd, by_ep, t_end, pred):
    res = {}
    for md in ("walk", "stand", "rig"):
        delays, miss = [], 0
        for e, ii in by_ep.items():
            em = fd.eps[e]
            if em["class"] != "edge" or em["mode"] != md:
                continue
            t = _gt(fd, em, "t")
            te = np.array([t_end[i] for i in ii])
            for s, key in (("L", "sup_L"), ("R", "sup_R")):
                sup = _gt(fd, em, key)
                ps = np.array([pred[key][i] for i in ii])
                on = sup > 0
                k = 0
                while k < len(on):
                    if not on[k]:
                        k += 1
                        continue
                    j = k
                    while j < len(on) and on[j]:
                        j += 1
                    first = np.nonzero(sup[k:min(j, k + 5)] == 1)[0]
                    if first.size:
                        tp_, t_stop = t[k + first[0]], t[j - 1]
                        cand = np.nonzero((te >= tp_ - 1e-6) & (te <= t_stop + 1e-6) & (ps == 1))[0]
                        if cand.size:
                            delays.append(float(te[cand].min() - tp_))
                        else:
                            miss += 1
                    k = j
        n = len(delays) + miss
        if n:
            res[md] = {"n_events": n, "median_delay_s": float(np.median(delays)) if delays else float("nan"),
                       "miss_rate": miss / n}
    return res


def _lifted_delay(fd, by_ep, t_end, pred):
    delays, miss, n = [], 0, 0
    for e, ii in by_ep.items():
        em = fd.eps[e]
        lift = em.get("lift")
        if em["mode"] != "lifted" or not lift or lift.get("t_down") is None:
            continue
        t = _gt(fd, em, "t")
        air = (_gt(fd, em, "sup_L") == 0) & (_gt(fd, em, "sup_R") == 0) & (t >= lift["t_up"])
        if not air.any():
            continue
        n += 1
        ta = t[np.argmax(air)]
        ii = sorted(ii, key=lambda i: t_end[i])
        te = np.array([t_end[i] for i in ii])
        both = np.array([(pred["sup_L"][i] == 0) & (pred["sup_R"][i] == 0) for i in ii])
        det = None
        for k in range(len(ii) - 1):
            if ta - 1e-6 <= te[k] <= lift["t_down"] and both[k] and both[k + 1]:
                det = te[k]
                break
        if det is None:
            miss += 1
        else:
            delays.append(float(det - ta))
    return {"n": n, "median_delay_s": float(np.median(delays)) if delays else float("nan"),
            "miss_rate": miss / n if n else float("nan")}
