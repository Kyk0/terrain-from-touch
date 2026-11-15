"""Tables for FLA-0 v2 (D14): main table, coded verdicts of questions 1-4, 6, 7, course table, v1 against v2.
Every verdict comes from v2eval.real_diff (D14 rule); nothing is judged by eye."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .v2eval import real_diff

CK = {"C1": "Nreal+Sreal", "C2": "Nreal+Sblk", "C3": "Nreal", "C4": "Nblk+Sblk", "C5": "Nblk"}
INV = {**{v: k for k, v in CK.items()}, **{f"P+{v}": f"P+{k}" for k, v in CK.items()}, "P": "P", "Pcmd": "P (cmd)",
       "Sblk": "S_blk", "Sreal": "S_real"}
ORDER = ["P", "P+Nreal+Sreal", "P+Nreal+Sblk", "P+Nreal", "P+Nblk+Sblk", "P+Nblk", "Nreal+Sreal", "Nreal+Sblk", "Nreal",
         "Nblk+Sblk", "Nblk", "Sblk", "Sreal", "Pcmd"]
MAIN_COLS = [("M1", "M1 F1"), ("acc", "acc"), ("M2", "M2 F1"), ("M2_partial", "partial F1"), ("M3", "M3 mu MAE"),
             ("G1_1", "G1-1"), ("M4_s", "M4 s"), ("M4_steps", "M4 steps"), ("M4_miss", "M4 miss"),
             ("M5_long_miss", "M5 long miss"), ("M5_long_delay_s", "M5 long delay s"), ("M5_short_miss", "M5 short miss"),
             ("M6_s", "M6 s"), ("M6_miss", "M6 miss")]


def load(out, sub):
    R = {}
    for p in sorted(glob.glob(f"{out}/jobs/{sub}/*.json")):
        r = json.load(open(p))
        R.setdefault(r["job"]["view"], {})[str(r["job"]["seed"])] = r
    return R


def vals(R, view, key, sub="test"):
    return [R[view][s][sub][key] for s in sorted(R.get(view, {}))] if view in R else []


def ms(v, f="{:.3f}"):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "n/a"
    return (f + " ± " + f).format(v.mean(), v.std()) + " [" + ", ".join(f.format(x) for x in v) + "]"


def verdict_rows(pairs, metrics, getter):
    rows = []
    for a, b in pairs:
        for m in metrics:
            v, d = real_diff(getter(a, m), getter(b, m), m)
            rows.append((a, b, m, v, d))
    return rows


def fmt_rows(rows):
    L = ["| A | reference B | metric | mean A | mean B | diff | 2 x max std | min | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for a, b, m, v, d in rows:
        if d:
            L.append(f"| {INV.get(a, a)} | {INV.get(b, b)} | {m} | {d['mean_a']:.4f} | {d['mean_b']:.4f} | {d['diff']:+.4f} | "
                     f"{2 * max(d['std_a'], d['std_b']):.4f} | {d['min_abs']:.4f} | **{v}** |")
        else:
            L.append(f"| {INV.get(a, a)} | {INV.get(b, b)} | {m} | | | | | | {v} |")
    return L


def build_tables(out, sources, git):
    M = load(out, "main")
    S = load(out, "sens")
    SN = load(out, "sens_nosat")
    LG = {os.path.basename(d)[5:]: load(out, os.path.basename(d)) for d in sorted(glob.glob(f"{out}/jobs/logo_*"))}
    g = lambda view, m: vals(M, view, m)  # noqa: E731
    T = {"sources": sources, "git": git["commit"], "git_dirty": git["dirty"]}
    L = [f"# FLA-0 v2 results, sources {', '.join(sources)}", "",
         f"commit `{git['commit']}` (clean tree: {not git['dirty']}). Test split, mean ± population std over seeds, every "
         "seed in brackets. Definitions: DECISIONS.md D14 (v1 D7 code reused). Command: `scripts/run_v2.sh "
         f"{','.join(sources)}`.", "",
         "Training recipe note (A9): every sole view is trained with a fresh random gain (std 0.1) and offset (std 0.3 N) "
         "per window, foot, point and axis; without it the point-map models memorised the fixed per-episode sensor "
         "pattern and overfit after one epoch (val F1 0.75 vs train 0.98). Selection on validation only.", "",
         "## Main table (mode walk, test)", "", "| input set | " + " | ".join(c[1] for c in MAIN_COLS) + " |",
         "|---|" + "---|" * len(MAIN_COLS)]
    for v in ORDER:
        if v in M:
            L.append(f"| {INV[v]} | " + " | ".join(ms(g(v, k), "{:.3f}" if "M4_steps" != k else "{:.2f}") for k, _ in MAIN_COLS) + " |")
    T["main"] = {v: {k: g(v, k) for k, _ in MAIN_COLS} for v in M}
    # hard_mid and ground analysis
    L += ["", "## hard-mid (test only, no terrain label): friction MAE and assigned class (share of windows, mean of seeds)", "",
          "| input set | mu MAE | most assigned classes |", "|---|---|---|"]
    for v in ORDER:
        if v in M:
            hm = [M[v][s]["test"]["hard_mid"] for s in sorted(M[v])]
            mae = [h["mu_mae"] for h in hm]
            cls = {}
            for h in hm:
                for c, x in h["assigned"].items():
                    cls[c] = cls.get(c, 0) + x / len(hm)
            top = ", ".join(f"{c} {100 * x:.0f} %" for c, x in sorted(cls.items(), key=lambda kv: -kv[1])[:3])
            L.append(f"| {INV[v]} | {ms(mae)} | {top} |")
    L += ["", "## Per-foot ground labels: recall of rough / edge windows by whether a foot touched a bump / an edge "
          "(analysis only, D14)", "", "| input set | rough with bump | rough without | edge with partial | edge without |",
          "|---|---|---|---|---|"]
    for v in ORDER:
        if v in M:
            ga = [M[v][s]["test"]["ground_analysis"] for s in sorted(M[v])]
            cell = lambda c, w: ms([x[c][w]["recall"] for x in ga])  # noqa: E731
            L.append(f"| {INV[v]} | {cell('rough', 'with')} | {cell('rough', 'without')} | {cell('edge', 'with')} | "
                     f"{cell('edge', 'without')} |")
    # verdicts
    q = {}
    L += ["", "## Coded verdicts (D14 'real difference' rule; A vs reference B)", ""]
    rows1 = verdict_rows([("P+Nreal", "P+Nblk"), ("Nreal", "Nblk")], ["M1", "M2", "M2_partial", "M3", "M5_long_miss"], g)
    q["Q1"] = rows1
    L += ["### Q1 realistic pressure map vs four blocks (C3 vs C5)", ""] + fmt_rows(rows1)
    for a, b in (("P+Nreal", "P+Nblk"), ("Nreal", "Nblk")):
        r = {m: v for aa, bb, m, v, _ in rows1 if aa == a}
        better = r["M1"] == "A better" or r["M2"] == "A better" or r["M2_partial"] == "A better"
        worse = any(x == "A worse" for x in r.values())
        L.append(f"- {INV[a]} vs {INV[b]}: " + ("**the map beats blocks**" if better and not worse else
                                                "the map does not beat blocks by the D14 rule") + "; per metric: "
                 + ", ".join(f"{m} {v}" for m, v in r.items()))
    for qn, pairs, title, note in (
            ("Q2", [("P+Nreal+Sblk", "P+Nreal"), ("Nreal+Sblk", "Nreal")], "Q2 block shear on top of the realistic map (C2 vs C3)", ""),
            ("Q3", [("P+Nreal+Sreal", "P+Nreal+Sblk"), ("Nreal+Sreal", "Nreal+Sblk")], "Q3 per-point shear over block shear (C1 vs C2)",
             "Label (D13): the simulator gives per-point shear no independent content (it is the foot's total shear "
             "shared by pressure plus a twist term), so this tests noise handling and encoding, not physics.")):
        rows = verdict_rows(pairs, ["M1", "M2", "M3", "M4_s", "G1_1"], g)
        q[qn] = rows
        L += ["", f"### {title}", ""] + ([note, ""] if note else []) + fmt_rows(rows)
    # Q6
    pairs6 = [("P+Nreal", "P+Nblk"), ("Nreal", "Nblk"), ("P+Nreal+Sblk", "P+Nreal"), ("Nreal+Sblk", "Nreal"),
              ("P+Nreal+Sreal", "P+Nreal+Sblk"), ("Nreal+Sreal", "Nreal+Sblk")] + \
             [(f"P+{c}", "P") for c in CK.values()]
    rows6 = verdict_rows(pairs6, ["M5_long_miss", "M5_long_delay_s", "M2_partial"], g)
    q["Q6"] = rows6
    L += ["", "### Q6 partial support (M5 long events, partial-class F1)", ""] + fmt_rows(rows6)
    # Q4
    sel_path = f"{out}/course_selection.json"
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else None
    if LG:
        L += ["", "### Q4 transfer to an unseen gait (leave-one-gait-out; M1 on the held-out source's walk test windows)", "",
              f"Held-out sources: {', '.join(sorted(LG))}. In-distribution = the main model of the same input set and "
              "seed on the same windows.", "", "| input set | " + " | ".join(f"{h} held out | in-dist" for h in sorted(LG))
              + " | mean over held-out | in-dist mean |", "|---|" + "---|---|" * len(LG) + "---|---|"]
        from .v2eval import walk_m1  # noqa: F401
        meanM1 = {}
        for v in [f"P+{c}" for c in CK.values()] + list(CK.values()):
            cells, per_seed, ind_seed = [], {}, {}
            for h in sorted(LG):
                R = LG[h]
                un = vals(R, v, "M1")
                ind = [R[v][s]["test"].get("in_dist_M1", np.nan) for s in sorted(R.get(v, {}))]
                for s_, x_ in zip(sorted(R.get(v, {})), ind):
                    ind_seed.setdefault(s_, []).append(x_)
                cells += [ms(un), ms(ind)]
                for s in sorted(R.get(v, {})):
                    per_seed.setdefault(s, []).append(R[v][s]["test"]["M1"])
            mean_seed = [float(np.mean(x)) for x in per_seed.values() if len(x) == len(LG)]
            meanM1[v] = mean_seed
            ind_mean = [float(np.nanmean(x)) for x in ind_seed.values() if len(x) == len(LG)]
            L.append(f"| {INV[v]} | " + " | ".join(cells) + f" | {ms(mean_seed)} | {ms(ind_mean)} |")
        rows4 = []
        for c in CK.values():
            v, d = real_diff(meanM1.get(c, []), meanM1.get(f"P+{c}", []), "M1")
            rows4.append((c, f"P+{c}", "M1 mean over held-out", v, d))
        q["Q4"] = rows4
        head = sel["selected"] if sel else None
        hk = head.replace("P+", "") if head else None
        L += ["", "Ck vs P+Ck on the mean over held-out sources (headline k = the k of the selected sole model: "
              f"{INV.get(hk, '?') if hk else 'not selected yet'}):", ""] + fmt_rows(rows4)
    # Q7
    if S:
        L += ["", "### Q7 does simulation detail pay off (trained on nominal; tested on four sensor settings)", ""]
        sets = tuple(x for x in ("nominal", "shifted", "pads", "g1", "nosat")
                     if all(x in r["tests"] for v in S.values() for r in v.values()))
        gs = lambda st: (lambda view, m: [S[view][s]["tests"][st][m] for s in sorted(S.get(view, {}))] if view in S else [])  # noqa: E731
        L += [f"Test sets: {', '.join(sets)} (nosat = nominal without per-point saturation, D19).", ""]
        L += ["| input set | metric | " + " | ".join(sets) + " |", "|---|---|" + "---|" * len(sets)]
        for v in ["P+Nreal+Sreal", "P+Nreal+Sblk", "P+Nreal", "P+Nblk+Sblk", "P+Nblk"]:
            if v not in S:
                continue
            for m in ("M1", "M2", "M3"):
                L.append(f"| {INV[v]} | {m} | " + " | ".join(ms(gs(st)(v, m)) for st in sets) + " |")
        L += ["", "Verdicts recomputed on each test set (survives = same verdict on all four):", "",
              "| question | pair | metric | " + " | ".join(sets) + " | survives |", "|---|---|---|" + "---|" * len(sets) + "---|"]
        q7 = []
        for qn, (a, b) in (("Q1", ("P+Nreal", "P+Nblk")), ("Q2", ("P+Nreal+Sblk", "P+Nreal")),
                           ("Q3", ("P+Nreal+Sreal", "P+Nreal+Sblk"))):
            for m in ("M1", "M2", "M3"):
                vs = [real_diff(gs(st)(a, m), gs(st)(b, m), m)[0] for st in sets]
                surv = len(set(vs)) == 1
                q7.append((qn, a, b, m, vs, surv))
                L.append(f"| {qn} | {INV[a]} vs {INV[b]} | {m} | " + " | ".join(vs) + f" | {'yes' if surv else '**NO**'} |")
        q["Q7"] = q7
        L += ["", "Absolute change from the nominal test set (mean of seeds):", "",
              "| input set | metric | " + " | ".join(sets[1:]) + " |", "|---|---|" + "---|" * len(sets[1:])]
        for v in ["P+Nreal+Sreal", "P+Nreal+Sblk", "P+Nreal", "P+Nblk+Sblk", "P+Nblk"]:
            if v not in S:
                continue
            for m in ("M1", "M2", "M3"):
                base = np.nanmean(gs("nominal")(v, m))
                L.append(f"| {INV[v]} | {m} | " + " | ".join(f"{np.nanmean(gs(st)(v, m)) - base:+.3f}" for st in sets[1:]) + " |")
    if SN:
        gn = lambda view, m: [SN[view][s]["tests"]["nosat"][m] for s in sorted(SN.get(view, {}))] if view in SN else []  # noqa: E731
        gnom = lambda view, m: [S[view][s]["tests"]["nominal"][m] for s in sorted(S.get(view, {}))] if view in S else []  # noqa: E731
        L += ["", "### Q7, fifth arm (D19): trained AND tested without per-point saturation", "",
              "| input set | M1 | M2 | M3 |", "|---|---|---|---|"]
        for v in ("P+Nreal+Sblk", "P+Nreal", "P+Nblk"):
            if v in SN:
                L.append(f"| {INV[v]} | " + " | ".join(ms(gn(v, m)) for m in ("M1", "M2", "M3")) + " |")
        L += ["", "| question | pair | metric | nominal (trained/tested nominal) | no saturation (trained/tested) | same verdict |",
              "|---|---|---|---|---|---|"]
        qn7 = []
        for qn, (a, b) in (("Q1", ("P+Nreal", "P+Nblk")), ("Q2", ("P+Nreal+Sblk", "P+Nreal"))):
            for m in ("M1", "M2", "M3"):
                v_nom = real_diff(gnom(a, m), gnom(b, m), m)[0]
                v_ns = real_diff(gn(a, m), gn(b, m), m)[0]
                qn7.append((qn, a, b, m, v_nom, v_ns))
                L.append(f"| {qn} | {INV[a]} vs {INV[b]} | {m} | {v_nom} | {v_ns} | {'yes' if v_nom == v_ns else '**NO**'} |")
        L.append("Q3 needs P+C1, which is not in the unsaturated arm (D19 trains P+C2, P+C3, P+C5).")
        q["Q7_nosat"] = qn7
    # course
    cp = f"{out}/course.json"
    if os.path.exists(cp) and sel:
        C = {k: v for k, v in json.load(open(cp)).items() if not k.startswith("_")}
        L += ["", "## Q5 mixed course (test only; all runs scored up to the fall)", "",
              f"Selected sole model (D14 rule, written to course_selection.json before any course prediction): "
              f"**{sel['selected_name']}** (val M1 {sel['val_M1_mean'][sel['selected']]:.4f}); runner-up "
              + ", ".join(f"{INV[k]} {x:.4f}" for k, x in sorted(sel["val_M1_mean"].items(), key=lambda kv: -kv[1])[1:4]), ""]
        srcs = sorted({s for d in C.values() for r in d.values() for s in r})
        keys = [("acc", "acc"), ("macro_f1", "macro-F1"), ("support_f1", "support F1"), ("latency_median_s", "latency s"),
                ("latency_median_steps", "latency steps"), ("miss_rate", "miss rate")]
        L += ["| detector | source | runs (fell) | " + " | ".join(k[1] for k in keys) + " | recall per chunk type |",
              "|---|---|---|" + "---|" * len(keys) + "---|"]
        for det in [sel["selected"], "P"] + (["rule"] if "rule" in C else []):
            if det not in C:
                continue
            for src in srcs:
                rr = [C[det][s][src] for s in sorted(C[det]) if src in C[det][s]]
                if not rr:
                    continue
                rec = {}
                for r in rr:
                    for c, x in r["recall"].items():
                        rec.setdefault(c, []).append(x)
                L.append(f"| {INV.get(det, det)} | {src} | {rr[0]['runs']} ({rr[0]['fell']}) | "
                         + " | ".join(ms([r[k] for r in rr], "{:.3f}") for k, _ in keys) + " | "
                         + ", ".join(f"{c} {np.mean(x):.2f}" for c, x in rec.items()) + " |")
        rowsq5 = []
        for a, b in ((sel["selected"], "P"),) + (((sel["selected"], "rule"),) if "rule" in C else ()):
            for k in ("acc", "macro_f1", "latency_median_s"):
                va = [C[a][s]["pooled"][k] for s in sorted(C[a])]
                vb = [C[b][s]["pooled"][k] for s in sorted(C[b])]
                v, d = real_diff(va, vb, "M1" if k != "latency_median_s" else "delay")
                rowsq5.append((a, b, f"pooled {k}", v, d))
        q["Q5"] = rowsq5
        L += ["", "Comparisons (pooled over sources; the rule-based detector has one deterministic run, std 0):", ""] + fmt_rows(rowsq5)
    # v1 vs v2
    v1p = "results/v1/metrics.json"
    if os.path.exists(v1p):
        V1 = json.load(open(v1p))["runs"]
        L += ["", "## v1 against v2 on the shared metrics (mode walk, test; NOT a like-for-like ablation)", "",
              "Both runs: three walking sources (G113, G80, SONIC) plus stand / rig / lifted, the same six terrain "
              "classes and friction / stiffness bands, test split by band, 3 seeds, the same metric code (D7 / D14). "
              "Differences are listed per row.", "",
              "| run | input | what it is | terrain macro-F1 | support F1 | partial F1 | friction MAE | strip latency s | "
              "strip latency steps | lifted delay s |", "|---|---|---|---|---|---|---|---|---|---|"]
        V1DESC = {"P": "joints, IMU AND command; v1 data 24.7 h, v1 CNN (no augmentation)",
                  "P+N": "+ Fz of 2x4 rigid zones per foot (MuJoCo per-zone contact forces, fixed sensor model)",
                  "P+N+S": "+ Fx, Fy of the 2x4 zones"}
        for key in ("P", "P+N", "P+N+S"):
            if key in V1:
                rr = [V1[key][s] for s in sorted(V1[key])]
                pc = [r["headline_walk"].get("support_f1_per_class", [np.nan] * 3)[1] for r in rr]
                L.append(f"| v1 | {key + (' (with cmd)' if key == 'P' else '')} | {V1DESC[key]} | "
                         f"{ms([r['headline_walk']['terrain_f1'] for r in rr])} | {ms([r['headline_walk']['support_f1'] for r in rr])} | "
                         f"{ms(pc)} | {ms([r['headline_walk']['mu_mae'] for r in rr])} | "
                         f"{ms([r['strip'].get('median_latency_s', np.nan) for r in rr], '{:.2f}')} | "
                         f"{ms([r['strip'].get('median_latency_steps', np.nan) for r in rr], '{:.2f}')} | "
                         f"{ms([r['lifted_delay'].get('median_delay_s', np.nan) for r in rr], '{:.2f}')} |")
        V2DESC = {"P": "joints, IMU, NO command; v2 data 33.2 h, v2 CNN",
                  "Pcmd": "as v2 P but WITH command (diagnostic, closest to v1 P)",
                  "P+Nblk": "+ 4 block Fz per foot DERIVED from the reconstructed point map, per-episode random sensor "
                            "parameters, 60 N per-point saturation, training augmentation; nearest to v1 P+N",
                  "P+Nblk+Sblk": "+ 4 block shear vectors (same construction); nearest to v1 P+N+S"}
        for v in ("P", "Pcmd", "P+Nblk", "P+Nblk+Sblk"):
            if v in M:
                L.append(f"| v2 | {INV[v]} | {V2DESC[v]} | {ms(g(v, 'M1'))} | {ms(g(v, 'M2'))} | {ms(g(v, 'M2_partial'))} | "
                         f"{ms(g(v, 'M3'))} | {ms(g(v, 'M4_s'), '{:.2f}')} | {ms(g(v, 'M4_steps'), '{:.2f}')} | "
                         f"{ms(g(v, 'M6_s'), '{:.2f}')} |")
    T["verdicts"] = {k: [(r[0], r[1], r[2], r[3]) if len(r) == 5 else r for r in v] for k, v in q.items()}
    json.dump(T, open(f"{out}/metrics.json", "w"), indent=1, default=float)
    open(f"{out}/tables.md", "w").write("\n".join(L) + "\n")
