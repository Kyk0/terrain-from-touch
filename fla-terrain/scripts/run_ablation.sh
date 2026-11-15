#!/bin/bash
# Full results pipeline from an existing dataset (every step refuses a dirty git tree):
# leakage test + LightGBM baseline; TerrainNet 3 seeds x {P, P+N, P+N+S, P+N_lp, P+Ntot, P+Ntot+Stot} + exploratory
# P+N+S+U; D7 evaluation on test; support diagnostics; friction analysis (R3); shortcut tests; leave-one-gait-out (R5);
# dataset summary; reproducibility vs results/v1_run1 if present; tables.md.
#   scripts/run_ablation.sh [DATA] [OUT]
set -e
cd "$(dirname "$0")/.."
PY=python3
D=${1:-/path/to/fla-terrain/data/v1}
OUT=${2:-results/v1}
$PY scripts/train_baseline.py --data "$D" --out "$OUT"
$PY scripts/run_ablation.py --data "$D" --out "$OUT" --seeds 0,1,2 --force
$PY scripts/support_diagnostics.py --data "$D" --out "$OUT"
$PY scripts/analyze_friction.py --data "$D" --out "$OUT"
$PY scripts/shortcut_tests.py --data "$D" --out "$OUT"
$PY scripts/transfer_lofo.py --data "$D" --out "$OUT" --seeds 0,1,2
$PY scripts/data_summary.py --data "$D" --json "$OUT/data_summary.json" > /dev/null
if [ -d "$OUT"_run1 ]; then $PY scripts/compare_runs.py --a "$OUT"_run1 --b "$OUT"; fi
$PY scripts/render_tables.py --out "$OUT"
echo PIPELINE_DONE
