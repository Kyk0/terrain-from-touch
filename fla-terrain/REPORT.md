# FLA-0 "Terrain from Touch" — Report (simulation, dataset v1)

Run of 2026-11-14 (started 01:25 EET). All numbers are from `results/v1/` (`metrics.json`, `tables.md`), produced by one clean run at commit `b19d1ef`; definitions of every metric and of the gate were fixed in the pre-registered protocol D7 (commit `d6b476a`) before the dataset existed.

## 1. Verdict

**Gate G1: not met (0 of 3 criteria). In this simulation, adding shear to proprioception plus zone-resolved normal pressure did not reach any of the thresholds fixed in advance.** Shear did lower the friction error in all three seeds (−10 % pooled, −18 % on hard-slip, −23 % on the foot rig); it did not improve terrain classification, support-state detection or detection speed.

| G1 criterion (`P+N+S` versus `P+N`, walking, test split, 3 seeds) | `P+N` | `P+N+S` | Result | Needed |
|---|---|---|---|---|
| 1. hard-grip vs hard-slip F1 | 0.975 ± 0.004 | 0.972 ± 0.008 | −0.3 points | ≥ +5 points |
| 2. Surface change detected earlier (median steps) | 2.00 | 2.00 | 0 steps (0.52 s vs 0.55 s) | ≥ 1 step |
| 3. Friction MAE | 0.166 ± 0.006 | 0.150 ± 0.003 | −10 % | ≥ −20 % |

Three things qualify that verdict, and section 6 gives the evidence for each:

- **Walking data was available, so this is the primary test the brief asked for.** Three different gaits were used (two velocity policies and SONIC-K1).
- **Proprioception alone does not solve the task, but it nearly solves the one comparison shear was expected to win.** `P` reaches 0.948 F1 on grip versus slip and `P+N` 0.975, so the +5 points of criterion 1 were arithmetically out of reach: the most `P+N+S` could have gained is +2.5.
- **The simulator is the weakest link for this question.** A rigid-body solver has no elastomer, no partial slip inside the contact patch, and distributes force across zones by a numerical regularizer. The result says shear did not pass the gate *in this sim*; it does not say it will not help on the real sole.

## 2. What was built

| Piece | Where | Check |
|---|---|---|
| Schema v0.2, validator, writer, windowing loader, splits | `fla_terrain/schema`, `fla_terrain/data` | `pytest` (68 tests pass on `main`) |
| Virtual sole on the K1 (2 x 4 zones per foot, sole-frame forces, 150 Hz sensor model, 50 Hz output) | `fla_terrain/sim` | `scripts/check_sole.py`: 6/6 PASS |
| Terrain suite with disjoint train/val/test parameter bands | `fla_terrain/sim/terrains.py` | `scripts/terrain_table.py`: OVERLAP TEST PASS |
| Three walking sources as `MotionSource` | `walking/` | fall rates in section 3 |
| Dataset v1 | `/path/to/fla-terrain/data/v1` (1.2 GB) | validator: 0 bad of 4572 |
| LightGBM baseline, causal 1D-CNN with three heads, ablation runner | `fla_terrain/models`, `scripts/run_ablation.sh` | one command, deterministic |
| Optional: virtual-sole probe; taxel-grid converter stub for the real sole | `scripts/probe_virtual_sole.py`, `fla_terrain/real/convert.py` | tests pass; converter geometry is a placeholder |

**Server and simulator.** Everything ran on **the compute server**. The simulator is **MuJoCo 4.0.1** on CPU (D3): trained K1 policies already replay in MuJoCo, the compute server has no renderer and cannot convert URDF to USD, and MuJoCo exposes the full 3D force of each contact. The Isaac/PhysX spike of ticket T2 was therefore not run; the T2 check itself was run unchanged in MuJoCo.

## 3. Setup

**Robot and sole.** K1 22-DoF MJCF from `k1_play_kit` (19.7 kg, weight 192.9 N). Each foot's collision box is replaced by 2 x 4 boxes tiling the same footprint with the same contact parameters; mass, joints and actuators are untouched. Zone force = sum of MuJoCo contact forces on that box, rotated into the sole frame (x toe, y left, z up; Fz ≥ 0 is load). Sensor model: mean over physics steps per 1/150 s, per-episode gain (3 %) and offset (0.1 N), noise 0.15 N shear / 0.25 N normal, 0.05 N quantization, then mean of 3 samples to 50 Hz. The noise values are placeholders: the PX-FOOTRIX noise is not known.

**Sanity checks (ticket T2 and T3, all mandatory, all pass; re-run by independent verification).**

| Check | Result |
|---|---|
| Sliding box, |Ft|/Fn vs set friction 0.3 / 0.6 / 0.9 (0.2 m/s) | 0.300 / 0.600 / 0.900 (tolerance 10 %) |
| Standing: sum of Fz over both feet | 192.92 N = m g, error +0.000 % (tolerance 2 %) |
| Single-leg stance | stance foot −0.02 % of m g, swing foot 0 N, CoP (+18, +3) mm inside the ±90 x ±35 mm sole |
| Foot dragged at mu 0.3 / 0.6 / 0.9 (foot-only rig) | shear/Fz 0.300 / 0.600 / 0.900 |
| Standing on a 5° / 10° slope | shear/Fz within 0.1 % of tan θ |
| Swing / lifted | exactly 0 N |

**Terrains.** hard-grip (mu 0.7–1.3), hard-slip (mu 0.08–0.36), soft (8–108 kN/m), rough (rigid bumps: maximum height per episode 0.5–2 cm, individual bumps 0.25–2 cm), slope (5–10°, random direction), edge (plank fields with gaps, and low platforms), plus strips where one surface changes to another (grip→slip, slip→grip, grip→soft, grip→rough) and a lifted mode (whole episode, and lifted mid-walk). Friction and stiffness are drawn from separated bands per split; test has one band inside the training span (`test_in`) and one outside it (`test_out`).

**Motion sources.** No policy was trained tonight; working policies were already available.

| Source | What it is | Flat hard-grip | Gait |
|---|---|---|---|
| `walk_g113` | velocity policy G113 m2200 | 0/64 falls | 2.07 steps/s |
| `walk_g80` | velocity policy G80 m4200 | 0/64 falls | 2.37 steps/s |
| `walk_sonic` | SONIC-K1 motion tracking, 291 walking clips | 0/32 falls | 1.4 steps/s |
| `stand` | scripted weight shifts, squats, ankle rolls under PD | | |
| `rig` | foot-only rig: random load, tilt, sliding speed, overhang | | |

The flat-ground fall counts for G113 and G80 are from the independent evaluation's evaluation on the plain K1 model, before the sole was added; on the sole model both again had 0/16. In dataset v1 itself, hard-grip walking episodes ended in a fall in 0/180 (G113), 0/180 (G80) and 5/180 (SONIC) cases.

**Dataset v1.** 4572 episodes of 20 s, 24.7 h, 50 Hz: walking 17.4 h + strips 2.7 h, stand 1.8 h, rig 1.5 h, lifted 1.3 h. Splits by `split_key` (one per terrain instance): 3100 / 452 / 1020 episodes. 235 episodes end in a fall (5.1 %); frames after a fall are labelled −1. Full class x mode x source table: `results/v1/tables.md`.

**Models.** Windows of 1.0 s, stride 0.2 s (290 k / 43 k / 93 k windows). Causal dilated 1D-CNN (177 k parameters for `P`, slightly more with sole channels), heads for terrain class, properties (mu, log stiffness, slope) and support state per foot; label = last frame of the window; epoch chosen on val; 3 seeds. LightGBM on hand features as the classical baseline.

## 4. Results (mode walk, test split, mean ± std over 3 seeds; LightGBM is a single run)

| Input set | Terrain macro-F1 | Friction MAE | log10 stiffness MAE | Support-state F1 |
|---|---|---|---|---|
| `P` (joints, IMU, command) | 0.802 ± 0.001 | 0.186 ± 0.008 | 0.187 ± 0.004 | 0.755 ± 0.003 |
| `P+N` (+ Fz per zone) | 0.861 ± 0.007 | 0.166 ± 0.006 | 0.145 ± 0.015 | 0.866 ± 0.010 |
| `P+N+S` (+ Fx, Fy per zone) | 0.860 ± 0.006 | 0.150 ± 0.003 | 0.141 ± 0.011 | 0.853 ± 0.010 |
| LightGBM `P` / `P+N` / `P+N+S` | 0.730 / 0.826 / 0.841 | 0.232 / 0.212 / 0.162 | 0.358 / 0.110 / 0.104 | 0.674 / 0.758 / 0.761 |

Per-seed values, accuracy, slope error and the split by `test_in` / `test_out` are in `results/v1/tables.md`.

**Confusions.** With every input set the largest errors are rough predicted as hard-grip (about 30 % of rough windows) and edge predicted as hard-grip (34 % with `P`, 22 % with `P+N`, 19 % with `P+N+S`); then edge predicted as rough (10–14 %) and hard-grip predicted as rough (5–7 %). Soft, slope and hard-slip are recognised in more than 95 % of windows by `P+N`. A likely contributor to the rough and edge errors, which nothing here measures, is the labelling: a foot that lands between bumps, or fully on a plank, is physically on flat grippy ground but carries the class label.

**Support state per foot (F1 per class).**

| Input set | Airborne | Partial | Full |
|---|---|---|---|
| `P` | 0.968 | 0.353 | 0.942 |
| `P+N` | 0.992 | 0.633 | 0.972 |
| `P+N+S` | 0.992 | 0.599 | 0.968 |

Partial support is the hard class and the one that matters for the "steps off the stage" case. Zone-resolved normal force nearly doubles its F1; adding shear does not improve it (0.599 versus 0.633, a drop of a little over one standard deviation across seeds).

**Detection delays.**

| Input set | Surface change: s / steps (median) | Edge: delay s / missed | Lifted: delay s / missed |
|---|---|---|---|
| `P` | 0.67 / 2 | 0.12 / 52 % | 0.29 / 0 % |
| `P+N` | 0.52 / 2 | 0.12 / 29 % | 0.07 / 0 % |
| `P+N+S` | 0.55 / 2 | 0.10 / 28 % | 0.08 / 0 % |

Being lifted is detected in every episode, four times faster with normal force. A partial-support touchdown is still missed in 28–29 % of cases with the sole. By strip type, grip→slip is detected on the first step by `P+N` (median 1.0) and slip→grip takes about 4 steps for every input set.

**By mode.**

| Mode | Input | Terrain F1 | Friction MAE | Support F1 (classes present) |
|---|---|---|---|---|
| stand | `P` / `P+N` / `P+N+S` | 0.534 / 0.714 / 0.701 | 0.330 / 0.209 / 0.211 | 0.487 / 0.729 / 0.727 |
| rig | `P` / `P+N` / `P+N+S` | 0.084 / 0.466 / 0.508 | 0.359 / 0.302 / 0.232 | 0.303 / 0.736 / 0.700 |
| lifted | `P` / `P+N` / `P+N+S` | 0.710 / 0.821 / 0.821 | 0.212 / 0.182 / 0.168 | 0.714 / 0.882 / 0.870 |

The foot rig is the only mode where shear clearly helps a property: friction MAE drops 23 % (0.302 → 0.232). That is the case where the foot is deliberately slid.

## 5. Does proprioception alone solve the task?

No, but it does a large part of it. `P` alone gives 0.80 terrain macro-F1, 0.948 on grip versus slip, and detects lifting in 0.29 s. It fails on partial support (F1 0.35) and misses half of the edge touchdowns. The probe that predicts the sole from proprioception (`results/v1/probe.json`, 2 seeds) explains why: on hard-grip, a GRU reconstructs foot total Fz, total shear and CoP from joints and IMU with stance R² 0.93–0.95 (CoP error 4 mm) and per-zone shear with 0.85; on hard-slip that drops to 0.63–0.80, and per-zone shear is the least predictable signal (R² 0.70). Shear that proprioception cannot explain is about twice as large on the slippery floor (RMS 15.1 N versus 7.2 N).

For terrain classification the gain from normal force comes from the zone pattern, not the load: per-foot totals (`P+Ntot`) score the same as `P` (0.800 versus 0.802), while zones give 0.861. For the airborne state totals are enough: they take lifted detection from 0.29 s to 0.06 s and airborne F1 from 0.968 to 0.990.

## 6. Why shear did not pass, as far as the evidence goes

Items marked *exploratory* were run after the gate result was known. They are for explanation only and do not enter the gate.

1. **Ceiling on criterion 1.** `P` already gets 0.948 and `P+N` 0.975 on grip versus slip. When a foot slides, the joints and IMU show it. With `P+N` at 0.975 the largest possible gain was +2.5 points, so the +5 threshold could not have been met by any added sensor.
2. **Criterion 2 is at its floor, and too coarse.** With a 0.2 s stride and a 2-window confirmation, the minimum is one step by definition (the first touchdown on the new surface is itself counted) and two is typical; `P+N` already detects grip→slip on the first step. Every run's median is exactly 2.0. In seconds shear is not faster either (0.55 s versus 0.52 s). This is a limitation of the criterion under the predefined protocol; it is retained.
3. **Friction is observable only at the friction limit.** Checked against true foot slip in a separate diagnostic set (done before the gate result was known): at mu ≤ 0.2 at least 92 % of steps reach the limit; at mu 0.4, 65–76 %; at mu 0.9, 1–12 %. Criterion 3 pools all classes, and most windows are on grippy ground (mu 0.7–1.3), where this 50 Hz zone-averaged sensor rarely shows the limit: real touchdown skids of 9–26 mm per step exist even at mu 0.9, but the saturation test on the sensor signal catches only 30–47 % of real slides. On hard-grip every model is close to the trivial "predict the class mean" reference (0.156–0.209 versus 0.214; *exploratory* breakdown).
4. **Where friction is observable, shear helps more, but unevenly.** The pre-registered secondary number, hard-grip + hard-slip only: MAE 0.141 → 0.121 (−14 %). *Exploratory* breakdown: hard-slip only 0.076 → 0.062 (−18 %); `test_in` slip band (mu 0.285–0.295) 0.076 → 0.052 (−32 %), although there the trivial class-mean reference (0.044) beats every model; `test_out` slip band (mu 0.08–0.12) 0.077 → 0.075 (−3 %). So friction *within* a class is barely recovered and most of the gain is from getting the class right. The gain is also not robust: with the command channels removed (one seed, section 7) the order reverses.
5. **The CNN may be using shear poorly for friction.** Friction needs a ratio of two inputs. The LightGBM baseline (planned in advance, single run), which has shear/normal as a hand feature, shows −24 % (0.212 → 0.162). Taken alone that would pass the 20 % threshold; the gate is defined on the 3-seed CNN, whose `P+N` (0.166) is already about as good as LightGBM's `P+N+S`. *Exploratory:* giving the CNN explicit ratio channels (`P+N+S+U`, selected on val, 3 seeds) gives 0.138, i.e. −17 % versus `P+N`. Closer, still under the threshold, and post-hoc.
6. **The simulated contact is not what a shear-sensing sole measures.** Each zone is a rigid box with four corner contacts; how load and shear are split between zones is set by the solver's soft-constraint regularization, not by elastomer mechanics. Only the per-foot totals are physically trustworthy, and totals are what proprioception already predicts. There is no partial slip at the edge of the contact patch (the signal real shear sensors use to detect slip before it happens), no stick-slip, no surface texture. This may understate shear; that is an interpretation, not a measurement.
7. **Friction model artifacts.** MuJoCo's friction is regularized: below about 0.05 m/s slip it under-reports (mu 0.9 reads 0.835), and sliding makes the normal force chatter at about 60 Hz. A control with low-passed Fz (`P+N_lp`) scores the same as `P+N` (0.864 versus 0.861; grip/slip 0.974 versus 0.975), so the chatter is not needed for the result. This does not prove the unfiltered model ignores it.
8. **No friction between 0.36 and 0.7 was sampled.** The class definition leaves a gap. In that range the foot would slide less often and proprioception might lose its cue; this run cannot say.

**An exploratory result** (*exploratory*, 3 seeds, large spread): transfer to a gait that was not in training.

| Held-out gait | `P` | `P+N` | `P+N+S` | Same gait seen in training |
|---|---|---|---|---|
| G113 | 0.31 | 0.54 ± 0.03 | 0.72 ± 0.04 | 0.85–0.89 |
| G80 | 0.30 | 0.55 ± 0.01 | 0.58 ± 0.04 | 0.83–0.88 |
| SONIC | 0.11 | 0.11 | 0.07 | 0.72–0.82 |

Two readings. First, the models are strongly gait-specific: nothing transfers to SONIC (0.07–0.11, the same as the label-shuffle control's 0.09, i.e. no information), and on a held-out velocity policy the loss is 53–54 points with `P`, 33–34 with `P+N`, and 17 (G113) or 29 (G80) with `P+N+S`. For the real robot this matters more than the shear question: the training data must contain the controller that will be running. Second, on one of the three hold-outs shear is worth +17 points, far more than its in-distribution effect. That is one case with a seed spread of ±0.04 and it was not pre-registered, so it is a hypothesis: shear may buy robustness to a change of gait rather than accuracy on a known gait.

## 7. Integrity

| Check | Result | Command |
|---|---|---|
| Leakage: `split_key`, bands and actual mu / stiffness values per split | none shared (two independent implementations) | `scripts/train_baseline.py`, `scripts/qa_leakage_b.py` |
| Independent recomputation of macro-F1, grip/slip F1, friction MAE, strip latency from raw episodes and saved predictions, without the evaluation code (done on the first run's predictions; the clean run equals the first to 0.0000) | difference 0.0000 on all six runs; latency in steps identical | `scripts/qa_recompute_metrics.py` |
| Label-shuffle control (`P+N+S`, labels permuted across training episodes) | test macro-F1 0.091, at chance | `scripts/qa_label_shuffle.py` |
| Trivial shortcuts (episode length, commands, start pose → class) | macro-F1 0.15 against 0.10 chance | `scripts/walk_qa_dataset.py` |
| Command channels removed (seed 0 only) | terrain F1 changes by at most 0.003, grip/slip F1 by up to 0.015. Friction MAE moves 0.015–0.020 in opposite directions for the two sets, which on this one seed reverses criterion 3 (`P+N` 0.155, `P+N+S` 0.166) | `scripts/shortcut_tests.py` |
| Reproducibility: second full run from a clean tree | max difference on headline numbers 0.0000 (training is deterministic) | `scripts/compare_runs.py` |
| Manager's own full re-run in a separate clean clone | see section 9 | `scripts/run_ablation.sh` |

No threshold or tolerance was changed. Every seed is reported. The first complete run recorded a commit older than the code that produced it; it is kept as `results/v1_run1/` and the reported numbers come from the clean re-run.

## 8. Approximations, losses and what was not done

- Zone-level force and shear distribution is a solver artifact (section 6, item 6). Sensor noise values are placeholders. Per-zone sensor saturation (300 N normal, 150 N shear) is modelled and affects 0.04 % of walking frames.
- Standing protocol uses ankle gains x4 and an ankle balance feedback, because the walking policy's PD gains cannot hold the K1 upright under plain PD. Rig episodes use a foot-only rig; the whole-robot rig tipped the feet 20–30° and was dropped.
- In the sole frame, shear/Fz exceeds mu at heel strike and toe-off (p99 0.36 at mu 0.2) because the foot is tilted about 10°; shear/Fz ≈ mu only holds for a flat foot.
- Data losses: SONIC falls on low platforms in 55 of 90 episodes; G80 falls on hard-slip in 54 of 180 (23 in the `test_out` band mu 0.08–0.12, 26 in the lowest training band 0.14–0.19, 5 elsewhere); 21 of 504 strip episodes never reach the new surface; 5 SONIC episodes fall at spawn on cross-slopes; 38 of 678 edge episodes have no partial-support frame. All are kept up to the fall and flagged.
- Not done: the Isaac/PhysX shear spike (D3); sand, gravel, grass, mud (excluded by the brief); a compliant-sole contact model; friction between 0.36 and 0.7; a second model architecture; anything on real hardware. No video exists: the compute server has no renderer.
- Scope cuts from the brief's list: none were needed.

## 9. Reproduce

On the compute server, in a clean checkout of this repo, with `env_fla`:

```
scripts/run_ablation.sh /path/to/fla-terrain/data/v1 results/v1
```

From scratch (dataset generation takes about 12 minutes on 40 cores; the full results pipeline about 46 minutes on the H100, of which CNN training is 19):

```
PY=python3
$PY scripts/gen_dataset.py --plan configs/dataset_v1.yaml --out /path/to/fla-terrain/data/v1 --workers 40
scripts/run_ablation.sh /path/to/fla-terrain/data/v1 results/v1
```

Sanity checks: `$PY scripts/check_sole.py`, `$PY scripts/spike_shear_mujoco.py`, `$PY scripts/terrain_table.py`, `$PY -m pytest -q`.

Independent verification reproduced all headline numbers with a maximum difference of 0.0000.

## 10. Next steps

1. **Decide what the real-sole test should measure before collecting data.** Grip versus slip classification is close to solved without the sole in this sim; partial support and early slip are where a sole has room. Re-state the gate around partial-support detection, slip onset before the foot has moved, and transfer to an unseen gait.
2. **Record the real K1 with the controller that will actually run**, and with more than one: the gait dependence in section 6 is the largest effect in this report.
3. **If sim is to be used further for the shear question, replace the rigid sole** with a compliant one (many small soft pads or an FEM/soft-contact model) so that zone distribution and partial slip mean something. Without that, more sim data will keep answering "no" for a reason unrelated to the real sensor.
4. Fill the friction gap (0.36–0.7) and test the unseen-gait hypothesis properly (pre-registered, more seeds, more gaits).
