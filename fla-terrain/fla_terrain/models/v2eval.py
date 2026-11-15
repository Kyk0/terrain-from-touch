"""FLA-0 v2 evaluation exactly as DECISIONS.md D14 (reusing the v1 D7 code in evaluate.py where D14 says so),
the coded 'real difference' rule and the verdicts of questions 1-4, 6, 7, and the course scorer (question 5).

Metrics (test split, mode walk unless stated): M1 terrain macro-F1 over the six classes; M2 support macro-F1 (mean of
feet) + partial F1; M3 friction MAE (windows with >= 1 foot in contact and a terrain label); M4 strip latency (median
s, steps, miss rate; v1 code); M5 edge events as D7 split by partial duration (< 0.3 s, >= 0.3 s): miss rate and
median delay; M6 lifted delay (v1 code); G1-1 grip/slip F1 (v1 code). hard_mid: friction MAE and assigned classes.
Ground-label analysis: recall of rough (edge) windows split by whether a foot was on a bump (partial support) in the
window.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..schema import TERRAIN_LABELS
from .evaluate import evaluate

LONG_S = 0.3
LONG_FRAMES = 15  # D14 '0.3 s or longer' = (index of the last stance frame) - (index of the first partial frame)
                  # >= 15 at 50 Hz (16 frames inclusive). Integer comparison: with float times the 21 events of
                  # exactly 0.30 s fall on either side by rounding (float32: 8 long, float64: 14 long; A11/A13).
F1_KEYS = {"M1", "M2", "M2_partial", "G1_1", "acc", "M1_macro", "recall"}


# ----------------------------------------------------------------------------------------------- metrics
def edge_events(fd, idx, pred, mode="walk"):
    """M5 (D7 definition, D14 split): stances of a foot in edge episodes whose GT support is partial within the first
    0.1 s; t_p = first partial frame; duration = t_p .. end of stance; detected if a window end inside [t_p, stance end]
    predicts partial for that foot; delay = first such window end - t_p."""
    t_end = fd.lab["t"][fd.last[idx]]
    by_ep = {}
    for i, e in enumerate(fd.ep_of_win[idx]):
        by_ep.setdefault(int(e), []).append(i)
    ev = {"short": [], "long": []}
    for e, ii in by_ep.items():
        em = fd.eps[e]
        if em["class"] != "edge" or em["mode"] != mode:
            continue
        o, T = em["offset"], em["T"]
        t = fd.lab["t"][o:o + T]
        te = t_end[ii]
        for key in ("sup_L", "sup_R"):
            sup = fd.lab[key][o:o + T]
            ps = pred[key][ii]
            on = sup > 0
            k = 0
            while k < T:
                if not on[k]:
                    k += 1
                    continue
                j = k
                while j < T and on[j]:
                    j += 1
                first = np.nonzero(sup[k:min(j, k + 5)] == 1)[0]
                if first.size:
                    tp, ts = t[k + first[0]], t[j - 1]
                    dur_frames = (j - 1) - (k + first[0])  # first partial frame .. last stance frame
                    cand = np.nonzero((te >= tp - 1e-6) & (te <= ts + 1e-6) & (ps == 1))[0]
                    d = float(te[cand].min() - tp) if cand.size else None
                    ev["long" if dur_frames >= LONG_FRAMES else "short"].append(d)
                k = j
    out = {}
    for g, v in ev.items():
        det = [x for x in v if x is not None]
        out[g] = {"n": len(v), "miss_rate": (1 - len(det) / len(v)) if v else float("nan"),
                  "median_delay_s": float(np.median(det)) if det else float("nan")}
    return out


def evaluate_v2(fd, idx, pred) -> dict:
    """All D14 metrics for predictions on windows idx of fd (test split)."""
    r = evaluate(fd, idx, pred)
    y = fd.win_labels(idx)
    ep = fd.ep_of_win[idx]
    mode = np.array([fd.eps[e]["mode"] for e in ep])
    cls = np.array([fd.eps[e]["class"] for e in ep])
    h = r.get("headline_walk", {})
    pc = h.get("support_f1_per_class", [np.nan] * 3)
    out = {"M1": h.get("terrain_f1", np.nan), "acc": h.get("terrain_acc", np.nan), "M2": h.get("support_f1", np.nan),
           "M2_partial": pc[1] if len(pc) > 1 else np.nan, "M3": h.get("mu_mae", np.nan),
           "M3_grip_slip": h.get("mu_mae_grip_slip", np.nan), "logk_mae": h.get("logk_mae", np.nan),
           "slope_mae_deg": h.get("slope_mae", np.nan), "G1_1": r.get("g1_grip_slip_f1", np.nan),
           "M4_s": r["strip"].get("median_latency_s", np.nan), "M4_steps": r["strip"].get("median_latency_steps", np.nan),
           "M4_miss": r["strip"].get("miss_rate", np.nan), "M4_n": r["strip"].get("n", 0),
           "M6_s": r["lifted_delay"].get("median_delay_s", np.nan), "M6_miss": r["lifted_delay"].get("miss_rate", np.nan),
           "support_f1_per_class": pc, "confusion_walk": h.get("confusion"), "by_mode": r["by_mode"],
           "by_band_walk": r["by_band_walk"], "strip_per_type": r["strip"].get("per_type", {})}
    m5 = edge_events(fd, idx, pred)
    out["M5_long_miss"] = m5["long"]["miss_rate"]
    out["M5_long_delay_s"] = m5["long"]["median_delay_s"]
    out["M5_short_miss"] = m5["short"]["miss_rate"]
    out["M5_short_delay_s"] = m5["short"]["median_delay_s"]
    out["M5_n"] = {g: m5[g]["n"] for g in m5}
    # hard_mid (test only, no terrain label)
    hm = (cls == "hard_mid") & (mode == "walk")
    c = hm & ((y["sup_L"] > 0) | (y["sup_R"] > 0)) & np.isfinite(y["mu"])
    out["hard_mid"] = {"windows": int(hm.sum()),
                       "mu_mae": float(np.abs(pred["mu"][c] - y["mu"][c]).mean()) if c.any() else float("nan"),
                       "assigned": {TERRAIN_LABELS[k]: float((pred["terrain"][hm] == k).mean()) for k in range(6)}
                       if hm.any() else {}}
    # per-foot ground-label analysis: recall of rough / edge windows by whether a foot touched a bump / partial
    gl = np.stack([fd.lab["ground_L"], fd.lab["ground_R"]], 1)
    st = fd.starts[idx]
    win = st[:, None] + np.arange(50)[None, :]
    gw = gl[win]                                                  # [N, 50, 2]
    ga = {}
    for name, lab_id, code in (("rough", 3, 3), ("edge", 5, 4)):
        m = (mode == "walk") & (y["terrain"] == lab_id)
        touched = (gw == code).any((1, 2))
        ga[name] = {f"{'with' if t_ else 'without'}": {
            "n": int((m & (touched == t_)).sum()),
            "recall": float((pred["terrain"][m & (touched == t_)] == lab_id).mean()) if (m & (touched == t_)).any()
            else float("nan")} for t_ in (True, False)}
    out["ground_analysis"] = ga
    return out


def walk_m1(fd, idx, pred_terrain):
    """M1 on the walk windows of idx (used for the D14 selection rule on the validation split)."""
    y = fd.win_labels(idx)["terrain"].astype(int)
    mode = np.array([fd.eps[e]["mode"] for e in fd.ep_of_win[idx]])
    m = (mode == "walk") & (y >= 0)
    return float(f1_score(y[m], pred_terrain[m], labels=sorted(set(y[m].tolist())), average="macro"))


# ----------------------------------------------------------------------------------------------- real difference
def real_diff(a, b, metric):
    """D14 rule. a, b: per-seed values of condition A and the reference condition B. Returns ('A better' |
    'A worse' | 'no difference' | 'n/a', details)."""
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    b = np.asarray([x for x in b if x is not None and np.isfinite(x)], float)
    if len(a) == 0 or len(b) == 0:
        return "n/a", {}
    ma, mb, sa, sb = a.mean(), b.mean(), a.std(), b.std()
    d = ma - mb
    higher_better = metric in F1_KEYS or metric.startswith("M1") or metric.startswith("M2") or metric == "G1_1"
    min_abs = 0.02 if higher_better else 0.10 * abs(mb)
    real = abs(d) > 2 * max(sa, sb) and abs(d) >= min_abs
    det = {"mean_a": ma, "mean_b": mb, "std_a": sa, "std_b": sb, "diff": d, "min_abs": min_abs}
    if not real:
        return "no difference", det
    better = d > 0 if higher_better else d < 0
    return ("A better" if better else "A worse"), det


# ----------------------------------------------------------------------------------------------- course
def course_score(fd, idx, preds_by_ep: dict, max_lat=5.0) -> dict:
    """Question 5 on course runs. preds_by_ep: {episode_id: {'t_end', 'terrain', 'sup_L', 'sup_R'}} (any detector;
    windows are matched to the standard window grid by t_end). Per run and pooled per source: accuracy, macro-F1,
    recall per chunk type, support macro-F1 over the labelled windows; latency at every boundary where the class
    changes: t0 = first touchdown on the new chunk; detection = first window end >= t0 with the new class predicted in
    two consecutive windows; miss if not detected within 5 s or before the next chunk's first touchdown / run end."""
    t_end = fd.lab["t"][fd.last[idx]]
    by_ep = {}
    for i, e in enumerate(fd.ep_of_win[idx]):
        by_ep.setdefault(int(e), []).append(i)
    rows, lat = [], []
    for e, ii in by_ep.items():
        em = fd.eps[e]
        pr = preds_by_ep.get(em["episode"])
        if pr is None:
            continue
        ii = np.array(sorted(ii, key=lambda i: t_end[i]))
        te = t_end[ii]
        key = np.round(np.asarray(pr["t_end"], float), 3)
        pos = {k: j for j, k in enumerate(key)}
        sel = np.array([pos.get(round(float(x), 3), -1) for x in te])
        ok = sel >= 0
        y = fd.win_labels(idx[ii])
        p_t = np.where(ok, np.asarray(pr["terrain"])[np.maximum(sel, 0)], -99)
        p_l = np.where(ok, np.asarray(pr["sup_L"])[np.maximum(sel, 0)], -99)
        p_r = np.where(ok, np.asarray(pr["sup_R"])[np.maximum(sel, 0)], -99)
        rows.append({"episode": em["episode"], "source": em["source"], "y": y["terrain"].astype(int), "p": p_t,
                     "sL": y["sup_L"].astype(int), "sR": y["sup_R"].astype(int), "pL": p_l, "pR": p_r,
                     "matched": float(ok.mean()), "fell": em["fall_time"] is not None})
        # boundaries
        tds = em["touchdowns"]
        chunks = em.get("chunks") or []
        regions = em["regions"]
        t_run_end = float(te[-1]) if len(te) else 0.0
        t0s = []
        for c in chunks:
            r = c["region"]
            ts = [t for t, _, rr in tds if r in rr]
            t0s.append(min(ts) if ts else None)
        prev_cls = TERRAIN_LABELS_ID("hard_grip")  # start pad
        for k, c in enumerate(chunks):
            new = int(c["label"])
            if t0s[k] is None:
                break                                   # never reached (fell or ran out of time)
            if new == prev_cls:
                prev_cls = new
                continue
            t0 = t0s[k]
            t_stop = min([x for x in t0s[k + 1:] if x is not None] + [t_run_end, t0 + max_lat])
            det = None
            for j in range(len(te) - 1):
                if te[j] >= t0 - 1e-6 and te[j] <= t_stop + 1e-6 and p_t[j] == new and p_t[j + 1] == new:
                    det = float(te[j])
                    break
            d = (det - t0) if det is not None else min(max_lat, t_stop - t0)
            steps = sum(1 for t, _, rr in tds if c["region"] in rr and t0 - 1e-6 <= t <= t0 + d + 1e-6)
            lat.append({"episode": em["episode"], "source": em["source"], "from": prev_cls, "to": new, "latency_s": d,
                        "steps": steps, "miss": det is None})
            prev_cls = new
    return _course_tables(rows, lat)


def TERRAIN_LABELS_ID(name):  # noqa: N802
    return {v: k for k, v in TERRAIN_LABELS.items()}[name]


def _course_tables(rows, lat):
    out = {}
    for src in sorted({r["source"] for r in rows}) + ["pooled"]:
        R = [r for r in rows if src == "pooled" or r["source"] == src]
        L = [x for x in lat if src == "pooled" or x["source"] == src]
        if not R:
            continue
        y = np.concatenate([r["y"] for r in R])
        p = np.concatenate([r["p"] for r in R])
        m = y >= 0
        present = sorted(set(y[m].tolist()))
        sup_y = np.concatenate([np.concatenate([r["sL"], r["sR"]]) for r in R])
        sup_p = np.concatenate([np.concatenate([r["pL"], r["pR"]]) for r in R])
        per_run_acc = [float((r["p"][r["y"] >= 0] == r["y"][r["y"] >= 0]).mean()) for r in R if (r["y"] >= 0).any()]
        out[src] = {"runs": len(R), "fell": int(sum(r["fell"] for r in R)), "windows": int(m.sum()),
                    "matched_frac": float(np.mean([r["matched"] for r in R])),
                    "acc": float((p[m] == y[m]).mean()),
                    "macro_f1": float(f1_score(y[m], p[m], labels=present, average="macro", zero_division=0)),
                    "per_run_acc_mean": float(np.mean(per_run_acc)) if per_run_acc else float("nan"),
                    "recall": {TERRAIN_LABELS[c]: float((p[m][y[m] == c] == c).mean()) for c in present},
                    "support_f1": float(f1_score(sup_y, sup_p, labels=sorted(set(sup_y.tolist())), average="macro",
                                                 zero_division=0)),
                    "boundaries": len(L),
                    "latency_median_s": float(np.median([x["latency_s"] for x in L])) if L else float("nan"),
                    "latency_median_steps": float(np.median([x["steps"] for x in L])) if L else float("nan"),
                    "miss_rate": float(np.mean([x["miss"] for x in L])) if L else float("nan")}
    return out
