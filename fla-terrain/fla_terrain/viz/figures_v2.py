"""Figures for the FLA-0 v2 page (results/v2/<run>). Every number is read at run time from the committed job JSONs and
course.json through the same accessors as fla_terrain.models.v2report (``load``, ``vals``); bar = mean of the
per-seed values (as ``v2report.ms``), markers = seeds. Each fig function returns (figure, [(key, value), ...]) with
keys that make_figures.py --check-tables matches against tables.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..models.v2report import CK, INV, load, vals
from .figures import DPI, GRID, INK, MUTED, H, W, plt  # noqa: F401  (rcParams are set on import)

# condition identity: sole-only (light) vs with proprioception (dark); P alone = reference line
C_SOLE, C_PSOLE, C_P, C_RULE = "#56B4E9", "#0072B2", "#E69F00", "#009E73"
CKEYS = ["C1", "C2", "C3", "C4", "C5"]
CDESC = {"C1": "N_real+S_real", "C2": "N_real+S_blk", "C3": "N_real", "C4": "N_blk+S_blk", "C5": "N_blk"}


def _finite(v):
    return np.array([x for x in v if x is not None and np.isfinite(x)], float)


def _bar(ax, x, v, color, width, values, key, hatch="", fmt="{:.3f}", label=None):
    v = _finite(v)
    m = v.mean()
    ax.bar(x, m, width, color=color, hatch=hatch, edgecolor="white", linewidth=0, zorder=2, label=label)
    jit = np.linspace(-width * 0.22, width * 0.22, len(v)) if len(v) > 1 else [0]
    ax.scatter(x + np.asarray(jit), v, s=55, facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3)
    if fmt:
        top = max(m, v.max())
        span = np.diff(ax.get_ylim())[0] if ax.get_ylim()[1] > 1.0 or ax.get_ylim()[0] > 0 else 1.0
        ax.text(x, top + 0.012 * span, fmt.format(m), ha="center", va="bottom", fontsize=12, color=INK, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
    values.append((key, m))
    for i, x_ in enumerate(v):
        values.append((f"{key} [seed {i}]", x_))


def fig_combos(out):
    """Five sensor combinations with and without P: M1 (terrain macro-F1) and M2 (support-state macro-F1)."""
    M = load(out, "main")
    vals_ = []
    fig, axs = plt.subplots(1, 2, figsize=(W, H), sharey=False)
    for ax, (m, title) in zip(axs, (("M1", "M1 terrain macro-F1"), ("M2", "M2 support-state macro-F1"))):
        w = 0.38
        for i, c in enumerate(CKEYS):
            _bar(ax, i - w / 2, vals(M, CK[c], m), C_SOLE, w * 0.95, vals_, f"{m} | {c}", label="sole only (Ck)" if i == 0 else None)
            _bar(ax, i + w / 2, vals(M, "P+" + CK[c], m), C_PSOLE, w * 0.95, vals_, f"{m} | P+{c}", hatch="//",
                 label="sole + P (P+Ck)" if i == 0 else None)
        pm = _finite(vals(M, "P", m)).mean()
        vals_.append((f"{m} | P", pm))
        ax.axhline(pm, color=C_P, lw=3, ls="--", label="P alone (dashed line, per panel)", zorder=1)
        ax.set_xticks(range(5), [f"{c}\n{CDESC[c]}" for c in CKEYS], fontsize=13)
        ax.set_ylim(0, 1.0)  # F1 axes start at 0 (as in the v1 figures)
        ax.set_title(title, fontsize=19)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("v2: the five sensor combinations with and without proprioception (walk, test; bar = mean of 3 seeds)",
                 fontsize=19)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, vals_


def fig_partial(out):
    """Partial support: partial-class F1 and long edge-event miss rate (M5, >= 0.3 s) per input set."""
    M = load(out, "main")
    vals_ = []
    sets = ["P"] + [f"P+{CK[c]}" for c in CKEYS] + [CK[c] for c in CKEYS]
    fig, axs = plt.subplots(1, 2, figsize=(W, H))
    for ax, (m, title, fmt) in zip(axs, (("M2_partial", "partial-support F1 (higher is better)", "{:.2f}"),
                                         ("M5_long_miss", "long edge events (>= 0.3 s): miss rate\n(lower is better)", "{:.2f}"))):
        for i, s in enumerate(sets):
            col = C_P if s == "P" else (C_PSOLE if s.startswith("P+") else C_SOLE)
            _bar(ax, i, vals(M, s, m), col, 0.75, vals_, f"{m} | {INV[s]}", hatch="//" if s.startswith("P+") else "", fmt=fmt)
        ax.set_xticks(range(len(sets)), [INV[s] for s in sets], fontsize=13, rotation=45)
        ax.set_title(title, fontsize=17)
        ax.set_ylim(0, 1.0)
    fig.suptitle("v2: partial support (walk, test; bar = mean of 3 seeds, markers = seeds)", fontsize=19)
    fig.tight_layout()
    return fig, vals_


def fig_logo(out):
    """Leave-one-gait-out: M1 on the held-out source (mean over held-out sources per seed) vs in-distribution."""
    import glob
    LG = {os.path.basename(d)[5:]: load(out, os.path.basename(d)) for d in sorted(glob.glob(f"{out}/jobs/logo_*"))}
    vals_ = []
    fig, ax = plt.subplots(figsize=(W, H))
    w = 0.38
    for i, c in enumerate(CKEYS):
        for j, (view, colr, hatch) in enumerate(((CK[c], C_SOLE, ""), ("P+" + CK[c], C_PSOLE, "//"))):
            per_seed = {}
            ind = {}
            for h, R in LG.items():
                for s, r in R.get(view, {}).items():
                    per_seed.setdefault(s, []).append(r["test"]["M1"])
                    ind.setdefault(s, []).append(r["test"].get("in_dist_M1", np.nan))
            un = [float(np.mean(v)) for v in per_seed.values() if len(v) == len(LG)]
            x = i + (j - 0.5) * w
            _bar(ax, x, un, colr, w * 0.95, vals_, f"LOGO mean held-out M1 | {INV[view]}", hatch=hatch,
                 label=("sole only, unseen gait" if j == 0 else "sole + P, unseen gait") if i == 0 else None)
            indm = np.nanmean([np.nanmean(v) for v in ind.values()])
            ax.plot([x - w * 0.45, x + w * 0.45], [indm] * 2, color=INK, lw=3, zorder=5,
                    label="same model class, seen gait (in-distribution)" if (i == 0 and j == 0) else None)
            vals_.append((f"LOGO in-dist mean | {INV[view]}", indm))
    ax.set_xticks(range(5), [f"{c}\n{CDESC[c]}" for c in CKEYS], fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("M1 terrain macro-F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=14)
    ax.set_title(f"v2 leave-one-gait-out ({', '.join(sorted(LG))}): mean over held-out sources", fontsize=19)
    fig.tight_layout()
    return fig, vals_


def fig_course(out):
    """Mixed course (pooled over sources): selected sole model, P, rule-based detector."""
    C = {k: v for k, v in json.load(open(f"{out}/course.json")).items() if not k.startswith("_")}
    sel = json.load(open(f"{out}/course_selection.json"))["selected"]
    dets = [(sel, f"sole model ({INV[sel]})", C_PSOLE), ("P", "P", C_P)] + ([("rule", "rule-based", C_RULE)] if "rule" in C else [])
    vals_ = []
    fig, axs = plt.subplots(1, 3, figsize=(W, H))
    for ax, (k, title, fmt) in zip(axs, (("acc", "accuracy", "{:.3f}"), ("macro_f1", "macro-F1", "{:.3f}"),
                                         ("latency_median_s", "boundary latency, median s\n(lower is better)", "{:.2f}"))):
        for i, (d, name, colr) in enumerate(dets):
            v = [C[d][s]["pooled"][k] for s in sorted(C[d])]
            _bar(ax, i, v, colr, 0.7, vals_, f"course {k} | {INV.get(d, d)}", fmt=fmt)
        ax.set_xticks(range(len(dets)), [n for _, n, _ in dets], fontsize=13, rotation=15)
        ax.set_title(title, fontsize=17)
    runs = C[sel][sorted(C[sel])[0]]["pooled"]
    fig.suptitle(f"v2 mixed course, test only, pooled over sources ({runs['runs']} runs, {runs['fell']} fell; scored up to "
                 "the fall)", fontsize=18)
    fig.tight_layout()
    return fig, vals_


def fig_sens(out):
    """Sensitivity: M1 of the nominal-trained models on each test setting, and the trained-without-saturation arm."""
    S, SN = load(out, "sens"), load(out, "sens_nosat")
    sets = [x for x in ("nominal", "shifted", "pads", "g1", "nosat") if all(x in r["tests"] for v in S.values() for r in v.values())]
    views = [f"P+{CK[c]}" for c in CKEYS if f"P+{CK[c]}" in S]
    cols = ["#0072B2", "#56B4E9", "#D55E00", "#CC79A7", "#999999"]
    vals_ = []
    fig, ax = plt.subplots(figsize=(W, H))
    w = 0.8 / (len(sets) + 1)
    for i, v in enumerate(views):
        for j, st in enumerate(sets):
            x = i + (j - len(sets) / 2) * w
            _bar(ax, x, [S[v][s]["tests"][st]["M1"] for s in sorted(S[v])], cols[j], w * 0.95, vals_,
                 f"sens M1 | {INV[v]} | {st}", fmt=None, label=(f"trained nominal, tested {st}" if i == 0 else None))
        if v in SN:
            x = i + (len(sets) - len(sets) / 2) * w
            _bar(ax, x, [SN[v][s]["tests"]["nosat"]["M1"] for s in sorted(SN[v])], "#F0E442", w * 0.95, vals_,
                 f"sens M1 | {INV[v]} | trained+tested nosat", fmt=None, hatch="xx",
                 label="trained and tested without saturation" if i == 0 else None)
    ax.set_xticks(range(len(views)), [INV[v] for v in views], fontsize=16)
    ax.set_ylim(0, 1.0)  # from 0: the map models fall far on pads / g1
    ax.set_ylabel("M1 terrain macro-F1 (walk, test)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=13)
    ax.set_title("v2 sensitivity: does simulation detail matter? (bar = mean of 3 seeds)", fontsize=19)
    fig.tight_layout()
    return fig, vals_


FIGURES_V2 = [("v2_fig1_combinations.png", fig_combos), ("v2_fig2_partial_support.png", fig_partial),
              ("v2_fig3_unseen_gait.png", fig_logo), ("v2_fig4_course.png", fig_course),
              ("v2_fig5_sensitivity.png", fig_sens)]
