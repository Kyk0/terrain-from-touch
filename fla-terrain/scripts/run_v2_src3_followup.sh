#!/bin/bash
# Queued after the src3 main job (PID in logs/wA_run_v2_src3.pid): D19 nosat arm (test nominal-trained sensitivity
# models on the unsaturated test set; train P+C2/C3/C5 x 3 on the unsaturated set), course scoring rerun WITH the
# rule-based detector files (selection file is reused, not recomputed), tables.
cd "$(dirname "$0")/.."
P=$(cat /path/to/fla-terrain/logs/wA_run_v2_src3.pid)
while kill -0 "$P" 2>/dev/null; do sleep 60; done
PY=python3
$PY scripts/build_v2_caches.py
$PY scripts/run_v2.py --sources walk_g113,walk_g80,walk_sonic --out results/v2/src3 \
    --models-root /path/to/fla-terrain/models/v2_src3 --workers 3 --stages sens,sens_nosat,select,course,tables \
    --rule-dir results/v2/rule_detector/course
