"""Render results/<v>/tables.md from metrics.json (all seeds), including the G1 gate per D7."""
from __future__ import annotations

import numpy as np

from ..schema import TERRAIN_LABELS

MAIN = ("P", "P+N", "P+N+S")
DIAG = ("P+N_lp", "P+Ntot", "P+Ntot+Stot")


def _get(d, path):
    for p in path:
        if isinstance(d, list) and isinstance(p, int) and p < len(d):
            d = d[p]
        elif isinstance(d, dict) and p in d:
            d = d[p]
        else:
            return float("nan")
    return d


def _ms(vals, scale=1.0, fmt="{:.3f}"):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float) * scale
    if not len(v):
        return "n/a"
    return (fmt + " ± " + fmt).format(v.mean(), v.std()) + " (" + ", ".join(fmt.format(x) for x in v) + ")"


def seeds_of(metrics, s):
    return sorted(metrics["runs"].get(s, {}).items())


def col(metrics, s, path):
    return [_get(r, path) for _, r in seeds_of(metrics, s)]


EXPL = ("P+N+S+U",)
REPRODUCE = """Reproduce (the compute server, env_fla; every step refuses a dirty git tree):
- from the existing dataset: `scripts/run_ablation.sh /path/to/fla-terrain/data/v1 results/v1`
- from scratch: `PY=python3; $PY scripts/gen_dataset.py --plan configs/dataset_v1.yaml --out /path/to/fla-terrain/data/v1 --workers 40 && scripts/run_ablation.sh /path/to/fla-terrain/data/v1 results/v1`"""


def render(metrics, baseline=None, extra=None) -> str:
    extra = extra or {}
    sets = [s for s in MAIN + DIAG if s in metrics["runs"]]
    dirty = metrics.get("git_dirty", True) or any(
        (v or {}).get("git_dirty") or ((v or {}).get("_provenance") or {}).get("git_dirty")
        for v in list(extra.values()) + [baseline] if isinstance(v, dict))
    L = [f"# FLA-0 results ({metrics.get('dataset', '')})" + (" — DIRTY TREE, NOT FOR THE REPORT" if dirty else ""), "",
         REPRODUCE, "",
         f"commit `{metrics.get('git_commit', '?')}` (clean tree: {not metrics.get('git_dirty', True)}), command "
         f"`{metrics.get('command', '')}`; test split only, mean ± std over seeds (per-seed values in brackets). "
         f"Metric definitions: DECISIONS.md D7.", ""]
    if extra.get("repro"):
        L += [f"Reproducibility: {extra['repro']}", ""]
    L += [f"windows: {metrics.get('windows', {})}", "", "## Headline: mode walk, test (test_in + test_out)", "",
          "| input set | terrain macro-F1 | accuracy | friction MAE | log10 K MAE | slope MAE (deg) | support F1 |",
          "|---|---|---|---|---|---|---|"]
    for s in sets:
        h = ("headline_walk",)
        L.append(f"| {s} | {_ms(col(metrics, s, h + ('terrain_f1',)))} | {_ms(col(metrics, s, h + ('terrain_acc',)))} | "
                 f"{_ms(col(metrics, s, h + ('mu_mae',)))} | {_ms(col(metrics, s, h + ('logk_mae',)))} | "
                 f"{_ms(col(metrics, s, h + ('slope_mae',)), fmt='{:.2f}')} | {_ms(col(metrics, s, h + ('support_f1',)))} |")
    if baseline:
        L += ["", "LightGBM baseline (hand features): a single run (seed 0), not mean ± std; same subset:", "",
              "| input set | terrain macro-F1 | friction MAE | log10 K MAE | support F1 |", "|---|---|---|---|---|"]
        for s, r in baseline.items():
            if s.startswith("_"):
                continue
            h = r.get("headline_walk", {})
            L.append(f"| {s} | {h.get('terrain_f1', float('nan')):.3f} | {h.get('mu_mae', float('nan')):.3f} | "
                     f"{h.get('logk_mae', float('nan')):.3f} | {h.get('support_f1', float('nan')):.3f} |")
    # G1
    L += ["", "## Gate G1 (D7): P+N+S versus P+N", ""]
    f1n, f1s = col(metrics, "P+N", ("g1_grip_slip_f1",)), col(metrics, "P+N+S", ("g1_grip_slip_f1",))
    d1 = 100 * (np.nanmean(f1s) - np.nanmean(f1n))
    ln, ls = col(metrics, "P+N", ("strip", "median_latency_steps")), col(metrics, "P+N+S", ("strip", "median_latency_steps"))
    d2 = np.nanmean(ln) - np.nanmean(ls)
    mn, ms_ = col(metrics, "P+N", ("headline_walk", "mu_mae")), col(metrics, "P+N+S", ("headline_walk", "mu_mae"))
    r3 = np.nanmean(ms_) / np.nanmean(mn)
    c1, c2, c3 = d1 >= 5.0, d2 >= 1.0, r3 <= 0.8
    L += ["| criterion | P+N | P+N+S | difference | threshold | met |", "|---|---|---|---|---|---|",
          f"| 1. grip/slip F1 (walk, flat) | {_ms(f1n)} | {_ms(f1s)} | {d1:+.1f} points | >= +5.0 | {'YES' if c1 else 'no'} |",
          f"| 2. strip latency, median steps | {_ms(ln, fmt='{:.2f}')} | {_ms(ls, fmt='{:.2f}')} | {d2:+.2f} steps earlier | >= 1.0 | {'YES' if c2 else 'no'} |",
          f"| 3. friction MAE (walk, contact) | {_ms(mn)} | {_ms(ms_)} | ratio {r3:.3f} | <= 0.80 | {'YES' if c3 else 'no'} |",
          "", f"**G1: shear {'IS' if (c1 or c2 or c3) else 'is NOT'} useful in sim by the pre-registered criteria** "
          f"({sum([c1, c2, c3])}/3 criteria met).", ""]
    L += ["Secondary for criterion 3: friction MAE on hard-grip + hard-slip only: " + "; ".join(
        f"{s} {_ms(col(metrics, s, ('headline_walk', 'mu_mae_grip_slip')))}" for s in sets), ""]
    # bands and modes
    L += ["## By band (walk) and by mode", "",
          "Support F1 here is the macro-F1 over the support classes PRESENT in each subset (stand/rig/lifted lack some "
          "classes; the D7 walk number counts all three).", "",
          "| input set | subset | terrain F1 | friction MAE | log10 K MAE | support F1 (present classes) |",
          "|---|---|---|---|---|---|"]
    for s in sets:
        for b in ("test_in", "test_out"):
            p = ("by_band_walk", b)
            L.append(f"| {s} | walk {b} | {_ms(col(metrics, s, p + ('terrain_f1',)))} | {_ms(col(metrics, s, p + ('mu_mae',)))} | "
                     f"{_ms(col(metrics, s, p + ('logk_mae',)))} | {_ms(col(metrics, s, p + ('support_f1_present',)))} |")
        for md in ("stand", "rig", "lifted"):
            p = ("by_mode", md)
            L.append(f"| {s} | {md} | {_ms(col(metrics, s, p + ('terrain_f1',)))} | {_ms(col(metrics, s, p + ('mu_mae',)))} | "
                     f"{_ms(col(metrics, s, p + ('logk_mae',)))} | {_ms(col(metrics, s, p + ('support_f1_present',)))} |")
    # per-class support F1
    L += ["", "## Support state per class (walk, test; F1 per class, mean of L and R feet)", "",
          "| input set | airborne | partial | full |", "|---|---|---|---|"]
    for s in sets + [x for x in EXPL if x in metrics["runs"]]:
        L.append(f"| {s} | " + " | ".join(_ms(col(metrics, s, ("headline_walk", "support_f1_per_class", c)))
                                          for c in range(3)) + " |")
    if extra.get("support_note"):
        L += ["", extra["support_note"]]
    # latency / delays
    L += ["", "## Surface-change latency (strips, test), edge and lifted detection delay", "",
          "| input set | strip latency s (median) | steps (median) | miss rate | edge delay s (walk) | edge miss | lifted delay s | lifted miss |",
          "|---|---|---|---|---|---|---|---|"]
    for s in sets:
        L.append(f"| {s} | {_ms(col(metrics, s, ('strip', 'median_latency_s')), fmt='{:.2f}')} | "
                 f"{_ms(col(metrics, s, ('strip', 'median_latency_steps')), fmt='{:.2f}')} | "
                 f"{_ms(col(metrics, s, ('strip', 'miss_rate')), fmt='{:.2f}')} | "
                 f"{_ms(col(metrics, s, ('edge_delay', 'walk', 'median_delay_s')), fmt='{:.2f}')} | "
                 f"{_ms(col(metrics, s, ('edge_delay', 'walk', 'miss_rate')), fmt='{:.2f}')} | "
                 f"{_ms(col(metrics, s, ('lifted_delay', 'median_delay_s')), fmt='{:.2f}')} | "
                 f"{_ms(col(metrics, s, ('lifted_delay', 'miss_rate')), fmt='{:.2f}')} |")
    types = sorted({t for s in sets for _, r in seeds_of(metrics, s) for t in r.get("strip", {}).get("per_type", {})})
    if types:
        L += ["", "Strip latency per type (median steps, mean ± std over seeds):", "",
              "| input set | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
        for s in sets:
            L.append(f"| {s} | " + " | ".join(_ms(col(metrics, s, ('strip', 'per_type', t, 'median_latency_steps')),
                                                  fmt='{:.2f}') for t in types) + " |")
    # confusion
    L += ["", "## Confusion matrices (walk, test, summed over seeds; rows true, columns predicted)", ""]
    names = [TERRAIN_LABELS[i] for i in range(6)]
    for s in MAIN:
        if s not in metrics["runs"]:
            continue
        cms = [np.array(_get(r, ("headline_walk", "confusion"))) for _, r in seeds_of(metrics, s)]
        cms = [c for c in cms if c.ndim == 2]
        if not cms:
            continue
        cm = sum(cms)
        L += [f"**{s}**", "", "| true \\ pred | " + " | ".join(names) + " |", "|---|" + "---|" * 6]
        for i in range(6):
            L.append(f"| {names[i]} | " + " | ".join(str(int(x)) for x in cm[i]) + " |")
        L.append("")
    L += _data_section(extra.get("data_summary"), metrics.get("dataset"))
    L += _exploratory(metrics, extra)
    if extra.get("shortcut"):
        L += ["## Shortcut tests (post hoc)", ""] + [f"- {x}" for x in extra["shortcut"]["lines"]] + [""]
    return "\n".join(L)


def data_losses(data_dir) -> list[str]:
    """Data-loss lines computed from the episode metadata (and labels for edge partial support)."""
    import glob
    import json
    from collections import Counter
    eps = []
    for f in sorted(glob.glob(f"{data_dir}/episode_*.json")):
        m = json.load(open(f))
        eps.append((f, m, m["terrain_params"]))
    out = []

    def falls(cls, src, variant=None):
        sel = [(m, tp) for _, m, tp in eps if m["mode"] == "walk" and tp.get("class") == cls and tp.get("source") == src
               and (variant is None or tp.get("variant") == variant)]
        fell = [(m, tp) for m, tp in sel if m.get("fell")]
        return sel, fell
    sel, fell = falls("edge", "walk_sonic", "steps")
    out.append(f"SONIC falls on edge-steps in {len(fell)}/{len(sel)} walk episodes (kept up to the fall)")
    sel, fell = falls("hard_slip", "walk_g80")
    bands = Counter(tp.get("mu_band", "?") for _, tp in fell)
    out.append(f"G80 falls on hard_slip in {len(fell)}/{len(sel)} walk episodes; by friction band: "
               + ", ".join(f"{b} {n}" for b, n in sorted(bands.items(), key=lambda kv: -kv[1])))
    st = [(m, tp) for _, m, tp in eps if str(tp.get("class", "")).startswith("strip")]
    never = [1 for m, _ in st if not any(1 in r for _, _, r in m.get("touchdowns", []))]
    out.append(f"{len(never)}/{len(st)} strip episodes never touch the new surface (excluded from latency, counted)")
    sel, fell = falls("slope", "walk_sonic")
    early = [tp for m, tp in fell if m["fall_time"] < 1.0]
    if early:
        rel = [abs(np.degrees(tp["uphill_rel_heading"])) for tp in early]
        out.append(f"SONIC falls within 1 s of spawn on slopes: {len(early)}/{len(sel)} walk episodes (of {len(fell)} SONIC "
                   f"slope falls), uphill direction {min(rel):.0f}-{max(rel):.0f} deg from heading (cross-slope)")
    ed = [f for f, m, tp in eps if tp.get("class") == "edge"]
    nop = Counter()
    for f in ed:
        z = np.load(f[:-5] + ".npz")
        if not ((z["label_support_L"] == 1).any() or (z["label_support_R"] == 1).any()):
            m = json.load(open(f))
            nop[(m["terrain_params"].get("variant"), m["mode"])] += 1
    out.append(f"{sum(nop.values())}/{len(ed)} edge episodes without any partial-support frame: "
               + ", ".join(f"{v}/{md} {n}" for (v, md), n in sorted(nop.items(), key=lambda kv: -kv[1])))
    return out


def _data_section(ds, data_dir=None):
    if not ds:
        return []
    L = ["## Dataset v1 (class x mode x source)", "",
         f"{ds['episodes']} episodes, validator bad {ds['bad']}, falls {ds['falls']} (frames after a fall labelled -1: "
         f"{'PASS' if ds['fall_labels_ok'] else 'FAIL'}), split_keys {ds['split_keys']}.", "",
         "| class | mode | source | episodes | hours | falls |", "|---|---|---|---|---|---|"]
    for c in ds["cells"]:
        L.append(f"| {c['class']} | {c['mode']} | {c['source']} | {c['episodes']} | {c['hours']:.3f} | {c['falls']} |")
    if data_dir:
        L += ["", "Data losses (computed from the episode metadata):"] + [f"- {x}" for x in data_losses(data_dir)] + [""]
    return L


def _exploratory(metrics, extra):
    L = ["## EXPLORATORY (post hoc; not part of the G1 gate)", ""]
    ex = [x for x in EXPL if x in metrics["runs"]]
    if ex:
        L += ["### R4: explicit friction-utilisation channels (P+N+S+U)", "",
              "u = |shear| / Fz per foot and per zone, 0 where Fz < 2 % body weight, clipped to [0, 2]. Same recipe, "
              "selection on val, 3 seeds, test once.", "",
              "| input set | val loss (selected) | walk terrain F1 | friction MAE | grip/slip F1 | support F1 |",
              "|---|---|---|---|---|---|"]
        for s in ("P+N", "P+N+S") + tuple(ex):
            L.append(f"| {s} | {_ms(col(metrics, s, ('train_info', 'val_loss')), fmt='{:.4f}')} | "
                     f"{_ms(col(metrics, s, ('headline_walk', 'terrain_f1')))} | {_ms(col(metrics, s, ('headline_walk', 'mu_mae')))} | "
                     f"{_ms(col(metrics, s, ('g1_grip_slip_f1',)))} | {_ms(col(metrics, s, ('headline_walk', 'support_f1')))} |")
        L.append("")
    an = extra.get("analysis")
    if an:
        L += ["### R3: where is friction observable? (friction MAE, walk test windows with a foot in contact)", "",
              "Reference = predicting the train-set mean mu of the model's predicted class (P+N+S seed 0 classes).", "",
              "| subset | n | P | P+N | P+N+S | reference |", "|---|---|---|---|---|---|"]
        have = [s for s in ("P", "P+N", "P+N+S") if an["sets"].get(s)]
        ref_set = "P+N+S" if "P+N+S" in have else have[-1]
        for k in an["sets"][have[0]][0]:
            vals = [_ms([q[k]["mae"] for q in an["sets"][s]]) if s in have else "n/a" for s in ("P", "P+N", "P+N+S")]
            ref = an["sets"][ref_set][0][k]["ref_pred_class_mean"]
            L.append(f"| {k} | {an['sets'][have[0]][0][k]['n']} | " + " | ".join(vals) + f" | {ref:.3f} |")
        L.append("")
    lo = extra.get("lofo")
    if lo:
        L += ["### R5: transfer to an unseen gait (leave-one-source-out)", "",
              "Trained/selected without any episode of the held-out walker; tested on its walk test windows. "
              "'seen' = main ablation model on the same windows.", "",
              "| held out | input set | terrain F1 unseen | seen | grip/slip F1 unseen | seen | friction MAE unseen | seen |",
              "|---|---|---|---|---|---|---|---|"]
        for held, d in lo["runs"].items():
            for s, runs in d.items():
                r = list(runs.values())
                g = lambda a, b: [x.get(a, {}).get(b, np.nan) for x in r]  # noqa: E731
                L.append(f"| {held} | {s} | {_ms(g('unseen', 'terrain_f1'))} | {_ms(g('seen', 'terrain_f1'))} | "
                         f"{_ms(g('unseen', 'grip_slip_f1'))} | {_ms(g('seen', 'grip_slip_f1'))} | "
                         f"{_ms(g('unseen', 'mu_mae'))} | {_ms(g('seen', 'mu_mae'))} |")
        L.append("")
    return L
