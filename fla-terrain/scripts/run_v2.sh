#!/bin/bash
# FLA-0 v2: all models and every D14 table for a list of walking sources (DECISIONS.md D17).
#   scripts/run_v2.sh walk_g113,walk_g80,walk_sonic results/v2/src3 /path/to/fla-terrain/models/v2_src3 [RULE_DIR]
# (four sources: append the new policy, e.g. walk_g113,walk_g80,walk_sonic,walk_k1v2 results/v2/src4 .../models/v2_src4)
set -e
cd "$(dirname "$0")/.."
PY=python3
SRC=${1:-walk_g113,walk_g80,walk_sonic}
OUT=${2:-results/v2/src3}
MOD=${3:-/path/to/fla-terrain/models/v2_src3}
RULE=${4:-}
$PY scripts/build_v2_caches.py
$PY scripts/run_v2.py --sources "$SRC" --out "$OUT" --models-root "$MOD" --workers 3 ${RULE:+--rule-dir "$RULE"}
# rule-based detector predictions (the independent evaluation): results/v2/rule_detector/course
