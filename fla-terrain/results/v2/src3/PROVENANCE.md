# Provenance of results/v2/src3 (A9-A11)

- **Training run** (`scripts/run_v2.sh walk_g113,walk_g80,walk_sonic results/v2/src3 /path/to/fla-terrain/models/v2_src3`)
  launched from commit `0b2624d0c5e6d3f586ade14dce958a9f1425c858`, clean tree (run_v2 refuses a dirty tree), 147 models,
  224 min, 0 failures. Its three GPU worker processes imported the code once at start, so every model of the run was
  trained and evaluated with the code of `0b2624d`.
- **The checkout moved while the run was going** (A10 work and two merges): `0114652` (clean and dirty states),
  `15f207f` (merge, dirty states), `c9cda26`. Each model's `config.json` records the checkout's commit and dirty flag at
  the moment it was SAVED, not the code it ran with: 6 models say `0b2624d` clean, 7 `0114652` clean, 3 `0114652`
  dirty, 2 `15f207f` dirty, 129 `c9cda26` clean. The training and evaluation code (`fla_terrain/models/{v2,nets,
  v2eval,evaluate}.py`) is identical between `0b2624d` and `c9cda26` (`git diff 0b2624d c9cda26 -- fla_terrain/models`
  changes only `dataset_v2.py` (float16 cache clip at 60 kN, reached by no saturated value) and `v2report.py`).
- The course selection (`course_selection.json`) was written by the end stage of that run before any course
  prediction existed (13:04 UTC); code `0b2624d` in memory, checkout `c9cda26` clean (recorded).
- **Follow-up** (`scripts/run_v2_src3_followup.sh`, D19): from `c9cda26cf...` clean: nominal-trained sensitivity models
  tested on `nosat`, P+C2/C3/C5 x 3 trained on `nosat` (9 models), course scoring rerun with the rule-based detector
  files (selection reused), tables.
- **Sensitivity readouts** shifted / g1 / nosat were patched before any sensitivity model was trained or tested
  (`scripts/v2_readout_share_proprio.py`, A10): proprioception copied from the nominal episode, so the arms differ only
  in the foot maps.
- **M5 recomputed** (A11, `scripts/recompute_m5.py`, commit `38b230f32e...` clean): whole-frame split (>= 15 frames =
  long) from stored predictions (sensitivity arms without stored predictions re-predicted from the saved weights; no
  retraining). Old values kept in each job JSON under `M5_float_old`. Tables re-rendered from `38b230f` (metrics.json
  records it).
- Not in git: per-window predictions `results/v2/src3/course_preds/**.npz` (gitignored `*.npz`, kept on the compute server) and the
  model directories `/path/to/fla-terrain/models/v2_src3/` (weights, configs, test/val predictions); see
  results/v2/WEIGHTS.md.
