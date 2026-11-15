#!/bin/bash
# T4 smoke: 2 G113 walking episodes per terrain class (edge: each variant) and split, plus lifted modes; summary.
cd "$(dirname "$0")/.."
PY=python3
D=${1:-/path/to/fla-terrain/data/t4_smoke}
rm -rf "$D"
run() { $PY scripts/gen_data.py "$@" --episodes 2 --seconds 20 --seed 4 --out "$D" --workers 2 > /dev/null 2>&1 || echo "FAILED: $*" & }
for sp in train val test; do
  for c in hard_grip hard_slip soft rough slope strip_grip_slip strip_slip_grip strip_grip_soft strip_grip_rough; do
    run --terrain $c --source walk_g113 --split $sp
  done
  for v in planks steps; do run --terrain edge --variant $v --source walk_g113 --split $sp; done
  run --terrain hard_grip --source lifted --split $sp
  run --terrain hard_grip --source lift_mid --split $sp
done
wait
$PY scripts/data_summary.py --data "$D"
