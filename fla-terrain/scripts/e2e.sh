#!/bin/bash
# Tiny end-to-end pass: generate ~5 min (G113, flat grip vs slip), validate, baseline, CNN, diagnostics.
set -e
cd "$(dirname "$0")/.."
PY=python3
D=${1:-/path/to/fla-terrain/data/e2e}
rm -rf "$D"
for t in hard_grip hard_slip; do
  $PY scripts/gen_data.py --terrain $t --source walk_g113 --episodes 8 --seconds 20 --seed 1 --out "$D" --workers 8 | grep -v "^  out"
done
$PY -c "import glob,sys; sys.path.insert(0,'.'); from fla_terrain.schema import validate_episode as v; ps=sorted(glob.glob('$D/*.npz')); b=[p for p in ps if v(p,raise_on_error=False)]; print('validator:', len(ps), 'episodes,', len(b), 'bad', 'PASS' if not b else 'FAIL')"
$PY scripts/train_baseline.py --data "$D"
$PY scripts/train_model.py --data "$D"
$PY scripts/e2e_diagnostics.py --data "$D"
