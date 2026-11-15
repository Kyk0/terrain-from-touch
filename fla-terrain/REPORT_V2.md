# FLA-0 "Terrain from Touch" — v2 report

Run of 2026-11-14 under `V2_BRIEF.md`. Booster K1 in MuJoCo, three walking sources, a point-map sole read on top of rigid contact. All numbers are from `results/v2/src3/tables.md` (commit recorded there; provenance in `results/v2/src3/PROVENANCE.md`): test split, walking, mean of three seeds unless stated. Metric definitions and the rule for calling a difference real were fixed in the pre-registered protocol D14 before the dataset existed. v1 results in `results/v1` are untouched.

## 1. Answers

**The one thing to read first: the advantage of the point map over four pressure blocks is large in this simulation and does not survive a change in how the map is produced.** Models trained on one sole setting and tested on another lose most of their accuracy if they read the point map, and much less if they read blocks (up to 22 points on the pad sole, 2 to 9 elsewhere). So the simulation supports "a pressure map carries useful geometry about what is under the foot", and it cannot yet tell how much of that will hold on the real sole.

| # | Question | Answer by the D14 rule | Key numbers |
|---|---|---|---|
| 1 | Does the realistic pressure map beat four pressure blocks? | **Yes on the main dataset, on every metric; not robust (question 7).** | `P+C3` against `P+C5`: terrain F1 0.877 against 0.829, support F1 0.929 against 0.810, partial-support F1 0.802 against 0.471, friction MAE 0.156 against 0.174, long edge events missed 2.0 % against 26.6 % |
| 2 | Does block shear add anything on top of the map? | **No difference with proprioception. Without it, terrain F1 is better and nothing else.** | `P+C2` against `P+C3`: 0.889 against 0.877 terrain F1, inside the rule's 2-point floor. `C2` against `C3`: 0.881 against 0.856 |
| 3 | Does per-point shear add anything over block shear? | **No, with two exceptions with proprioception (friction MAE better, partial-support F1 worse), and the simulator cannot test this question.** | `P+C1` against `P+C2`: friction MAE 0.141 against 0.158 (better), partial-support F1 0.801 against 0.827 (worse); all else no difference. Per-point shear here is the foot's total shear shared out by pressure (section 3) |
| 4 | Do sole-only models transfer to an unseen gait better than models with proprioception? | **Yes, for all five combinations.** | Mean terrain F1 on the held-out gait, `C2` against `P+C2`: 0.629 against 0.532. On held-out SONIC: 0.526 against 0.071 |
| 5 | On the mixed course, how much better is the sole model than proprioception and than rules? | **Better than both on accuracy, F1 and latency.** | Selected model `P+C2`: accuracy 0.842, boundary latency 0.62 s. `P`: 0.785 and 0.71 s. Rule-based: 0.311 and 3.57 s, half the boundaries missed |
| 6 | Partial support on edge events of 0.3 s or longer? | **The point map detects them; blocks and proprioception miss a quarter to a third.** | Missed long events: `P` 37.3 %, block inputs 26.6-28.0 %, point-map inputs 2.0-2.6 %. Shear does not change the detection of long events; the one real shear effect is the lower partial-support F1 of `P+C1` (question 3) |
| 7 | Do the answers survive a change of simulation detail? | **Answers 2 and 3 survive on the three metrics tested (terrain F1, support F1, friction MAE). Answer 1 does not.** | Nominal-trained point-map models fall from terrain F1 0.86-0.88 to 0.53-0.60 with shifted sensor parameters and to 0.20-0.27 on the pad sole and on the foot-outline mask; block models fall to 0.59-0.76 |

Three further points that qualify these answers:

- **Three walking sources, not four.** The new velocity policy was parked by the brief's rule after three fixes without measurable progress (section 6). Dataset, leave-one-gait-out and courses use G113, G80 and SONIC.
- **While walking, the simulated foot still loads its corners.** The rigid foot is a box that rocks on its edges, so at heel strike and toe-off two to four corner points carry the load and reach the sensor model's 60 N limit (section 7). Standing and the hand-press rig clips show an evenly loaded front and back, as described for the real sole; walking does not. This is the most visible difference between the renders and a real sole.
- **Brief touches of an edge are mostly missed by everything.** Edge events shorter than 0.3 s are missed 74-76 % of the time with the point map and 82-85 % without it. v1's page count pointed this way; it is now measured.

## 2. What changed since v1

| | v1 | v2 |
|---|---|---|
| Sole | 2 x 4 rigid zones per foot, forces taken per zone from the contact solver | Contact: four rigid blocks (front and back, each left and right half) and a middle raised 1 mm. Reading: 19 x 8 points at 10 mm per foot (152), rebuilt from the foot's total wrench |
| Sensor views | zones: `N`, `S` | `N_real`, `S_real` (per point), `N_blk`, `S_blk` (four blocks per foot), all from one recording |
| Sensor parameters | one fixed setting | random per episode (boundaries, clearance, layer compliance, neighbour spread, crosstalk, gain, offset, noise, quantization), plus fixed settings for the sensitivity test |
| Proprioception `P` | joints, IMU, command | joints, IMU; no command (one diagnostic run with it) |
| Models | temporal CNN | same temporal CNN; point maps first pass a small masked 2D encoder per foot and frame. Weights of all 156 models are saved |
| Dataset | 24.7 h, 4572 episodes | 33.2 h, 6139 episodes (train 21.9 h, val 3.2 h, test 8.1 h), plus `hard-mid` friction 0.36-0.7 as a test-only set, per-foot ground labels, sensitivity sets and 120 course runs |
| New tests | | leave-one-gait-out with and without `P`; mixed-terrain course against `P` and a rule-based detector; sensitivity of the conclusions to the sole model |
| Gate | G1 with three thresholds | no gate; seven questions with one pre-registered rule for "real difference" |

## 3. Setup, with what is approximate

**Contact geometry.** Rigid, same outer box as v1 (180 x 70 mm), so the walking policies see the heel and toe edges they were trained on. Front and back parts carry load; the middle is 1 mm higher. Boundaries default to thirds and are randomized per episode. All three sources were checked on the new foot on flat hard-grip: 0 falls in 64 episodes each, forward speed within 0.1 % of v1.

**The K1 foot mesh** in the assets (the model wears shoes) shows a flat heel, then a 16 mm long groove 4-15 mm deep, then a flat front with toe spring. That "middle" is a groove, not a third of the length. Thirds are kept as the brief says; the random range of the boundaries covers the mesh value (QUESTIONS Q9).

**Why the map is reconstructed from the foot wrench, not from region totals.** The brief assumed the rigid simulator gets each region's total force right. Measured on G113 walking on flat ground: one block carries over 95 % of a foot's load in 33-38 % of loaded physics steps, and the whole load can move to another block within one 2 ms step. For coplanar parts of one rigid body the split is statically indeterminate and the solver's choice is an artifact, the same one that put the load in one corner zone in v1. What the dynamics do determine is the foot's total force, its centre of pressure and the twist about it. The map is rebuilt from those at every physics step: a thin elastic layer on a rigid plane, pressure at a point proportional to how far the plane presses into the layer given the raycast gap to the ground under that point, with the plane's height and two tilts solved so that total force and centre of pressure match the simulator exactly. Then the sensor model: neighbour spread, crosstalk, per-point gain and offset, noise, a 60 N per-point limit, quantization, 150 Hz to 50 Hz.

**Consequences, stated as limits of this simulator:**
- `N_blk` and `S_blk` are sums over the reconstructed map, not solver outputs.
- **Per-point shear has no independent content.** It is the foot's total shear shared in proportion to pressure, plus a twist term. All shear in v2 comes from three numbers per foot. Question 3 therefore tests noise handling and encoding, not physics. `S_real` alone even reaches support F1 0.879, because it carries the pressure pattern.
- On flat ground the map is a function of total force and centre of pressure. What it adds over blocks is the ground geometry under the foot: bumps, voids, edges.

**Point mask.** A 19 x 8 rectangle (152 points per foot) on the contact box, not the foot-shaped mask of about 100 points the brief asks for: the rigid foot touches at its box corners and a foot-shaped mask has no points there (QUESTIONS Q8). The foot outline traced from screenshot is one arm of the sensitivity test and is drawn in the renders.

**Layer compliance** was lowered from 0.5 to 0.2 mm per 100 N after the pilot, because with 0.5 mm the middle carried 5-6 % of the load while walking on flat ground. With 0.2 mm it carries 3.4-4.1 % on average (median 0; 0 when standing). "The middle does not touch flat ground" holds when standing and for most of the stance, not all of it.

**Sole checks** (`scripts/check_sole_v2.py`, 17 of 17): the six v1 checks on the new geometry; standing, no point above 1.0 % of the foot's load (limit 15 %) and the middle at 0.00 % (limit 5 %); centre of pressure from the map within 0.006 mm of the simulator's (limit 5 mm); a 5 mm bump under the middle raises its share to 80 %; half over an edge, points over the void read exactly zero before noise and the support fraction matches ground truth; sum over points equals the foot force within 0.0009 % (limit 1 %). "Load shared by many points" has no number in the brief; the training implementation's rule is "at least half the points of the front and of the back carry at least half the uniform share" (measured 100 %), which replaced its own first rule that no even spread could meet (D13).

**Dataset v2.** Same six classes and disjoint friction and stiffness bands as v1. Walking with G113, G80 and SONIC, plus standing, foot rig, lifted and surface-change strips. 310 falls in 5869 walking, standing and lifted episodes, kept up to the fall. Validator passes on every episode; no `split_key`, band or raw array is shared between splits (checked by the training implementation, an independent check and the independent evaluation separately).

**Training.** Point-map models at first memorised each episode's fixed per-point gain and offset (train F1 0.98, validation 0.75 after one epoch). Every sole view is therefore trained with a fresh random gain (std 0.1) and offset (std 0.3 N) per window; validation and test data are not perturbed (D19).

**Rule-based detector.** Hand-written rules on kinematic foot slip in double support, joint tracking error, stance-foot tilt and its scatter, foothold height difference and a contact estimate, thresholds fitted on the training split only (`results/v2/rule_detector/thresholds.json`, rules in `NOTES_B.md`). It never outputs "soft". Terrain F1 0.285 on the test split.

## 4. Results

### Main table (walking, test)

| Input | Terrain F1 | Support F1 | Partial F1 | Friction MAE | Grip/slip F1 | Strip latency, s | Long edge events missed | Short edge events missed | Lifted delay, s |
|---|---|---|---|---|---|---|---|---|---|
| `P` | 0.817 | 0.760 | 0.358 | 0.181 | 0.948 | 0.64 | 0.373 | 0.845 | 0.30 |
| `P+C1` (`N_real`+`S_real`) | 0.872 | 0.929 | 0.801 | 0.141 | 0.949 | 0.35 | 0.022 | 0.743 | 0.11 |
| `P+C2` (`N_real`+`S_blk`) | 0.889 | 0.938 | 0.827 | 0.158 | 0.978 | 0.36 | 0.026 | 0.755 | 0.11 |
| `P+C3` (`N_real`) | 0.877 | 0.929 | 0.802 | 0.156 | 0.957 | 0.38 | 0.020 | 0.744 | 0.10 |
| `P+C4` (`N_blk`+`S_blk`) | 0.836 | 0.813 | 0.477 | 0.176 | 0.969 | 0.49 | 0.280 | 0.816 | 0.11 |
| `P+C5` (`N_blk`) | 0.829 | 0.810 | 0.471 | 0.174 | 0.945 | 0.51 | 0.266 | 0.820 | 0.12 |
| `C1` | 0.877 | 0.929 | 0.802 | 0.141 | 0.967 | 0.33 | 0.027 | 0.757 | 0.10 |
| `C2` | 0.881 | 0.934 | 0.816 | 0.156 | 0.977 | 0.39 | 0.022 | 0.751 | 0.10 |
| `C3` | 0.856 | 0.923 | 0.786 | 0.170 | 0.957 | 0.47 | 0.026 | 0.747 | 0.10 |
| `C4` | 0.817 | 0.795 | 0.428 | 0.175 | 0.963 | 0.50 | 0.278 | 0.826 | 0.11 |
| `C5` | 0.780 | 0.792 | 0.419 | 0.193 | 0.919 | 0.57 | 0.286 | 0.854 | 0.10 |
| `S_blk` alone | 0.793 | 0.746 | 0.313 | 0.172 | 0.947 | 0.58 | 0.366 | 0.854 | 0.35 |
| `S_real` alone | 0.857 | 0.879 | 0.674 | 0.165 | 0.961 | 0.41 | 0.050 | 0.758 | 0.20 |

Strip latency in steps is 2.00 for every input without the point map and 1.33-2.00 with it; with three seeds of integer medians that is not a real difference by the rule. Per-seed values and spreads are in `tables.md`.

### Question 1: map against blocks

Real difference on all five metrics, with and without proprioception. The size is in support state, not terrain class: partial-support F1 goes from 0.47 to 0.80 and long edge events missed from 27 % to 2 %, while terrain F1 gains 5 points. The per-foot labels show where: on edge terrain, windows in which a foot was actually on an edge are recognised 94 % of the time with the map and 59-65 % without; windows of edge terrain where no foot touched an edge stay at 28-43 % for every input. A window of "edge" or "rough" terrain in which no foot touched an edge or a bump is recognised much less often by every input (edge 28-43 %, rough 8-18 %), which supports what v1 could only assert: part of the class error is in the label, not in the sensing. The per-foot labels are analysis only (D14).

### Question 2: block shear on top of the map

With proprioception: no difference on terrain F1 (+1.3 points, under the 2-point floor), support F1, friction MAE, strip latency or grip/slip F1. Without proprioception, terrain F1 is really better (+2.6 points) and nothing else. Shear's v1 effect on friction is not there: 0.158 against 0.156.

### Question 3: per-point shear over block shear

No difference, except friction MAE with proprioception (0.141 against 0.158, a real difference by the rule) and partial-support F1, which is really worse (0.801 against 0.827). Without proprioception the friction difference is the same size (0.141 against 0.156); it passes the spread test and misses the 10 % floor by a hair (0.0152 against 0.0156), so by the rule it is no difference. Given how per-point shear is built here, this is an effect of feeding the same information in a different encoding. It says nothing about the real sensor's per-point shear.

### Question 4: an unseen gait

| Input | G113 held out | G80 held out | SONIC held out | Mean | Same input, gait seen |
|---|---|---|---|---|---|
| `P+C1` | 0.798 | 0.673 | 0.120 | 0.530 | 0.872 |
| `P+C2` | 0.841 | 0.686 | 0.071 | 0.532 | 0.889 |
| `P+C3` | 0.788 | 0.633 | 0.098 | 0.506 | 0.876 |
| `P+C4` | 0.488 | 0.554 | 0.079 | 0.374 | 0.834 |
| `P+C5` | 0.498 | 0.481 | 0.066 | 0.348 | 0.828 |
| `C1` | 0.779 | 0.715 | 0.516 | 0.670 | 0.876 |
| `C2` | 0.717 | 0.644 | 0.526 | 0.629 | 0.881 |
| `C3` | 0.653 | 0.629 | 0.418 | 0.567 | 0.855 |
| `C4` | 0.623 | 0.578 | 0.363 | 0.521 | 0.816 |
| `C5` | 0.466 | 0.490 | 0.240 | 0.399 | 0.780 |

Sole-only models are really better on the held-out mean for every combination. Most of the gain comes from SONIC: with proprioception in the input, a model that never saw SONIC scores 0.07-0.12 on it, the level of the shuffled-label control; without proprioception it scores 0.24-0.53. Between the two velocity policies, which move alike, the picture is mixed: with the point map, proprioception helps on held-out G113 (`P+C2` 0.841 against `C2` 0.717); with block pressure and block shear it hurts there (`P+C4` 0.488 against `C4` 0.623), and with block pressure alone it makes little difference (`P+C5` 0.498 against `C5` 0.466). Every model still loses 21 to 48 points against a gait it has seen. v1's conclusion stands and is sharper: the controller that will run on the real robot must be in the training data, and proprioception is the part of the input that ties the model to a gait.

### Question 5: the mixed course

120 recorded runs (40 courses per source; planned length 60-69 s, completed runs last 70-80 s; six chunk types in random order), test only. No source reaches the brief's 80 %: every chunk is crossed in 19 of 40 runs by G113, 22 by G80 and 6 by SONIC (SONIC stays up in 12, but in six of those it does not reach the last chunk: its clips turn and wander as well as walk slowly). All runs are scored up to the fall, as written in D16; the training implementation generated the runs shortly before D16 was pushed, with the same design (D17). The sole model was selected on validation data before any course result: `P+C2` (validation terrain F1 0.884; runner-up `P+C3` 0.880).

| Detector | Accuracy | Macro-F1 | Support F1 | Boundary latency, s | steps | Boundaries missed |
|---|---|---|---|---|---|---|
| Sole model `P+C2` | 0.842 | 0.841 | 0.928 | 0.62 | 2 | 0.7 % |
| `P` only | 0.785 | 0.783 | 0.776 | 0.71 | 2 | 1.3 % |
| Rule-based | 0.311 | 0.272 | 0.633 | 3.57 | 8 | 49.9 % |

The sole model is really better than `P` on accuracy (+5.7 points), macro-F1 (+5.8) and median latency (0.09 s earlier); in steps both detect the new ground on the second step in the pooled data (`P` needs three steps on SONIC). The largest gap is on SONIC (accuracy 0.812 against 0.699, latency 0.62 s against 0.89 s). The rule-based detector is far behind both: its tracking-error rule fires on SONIC's low-gain controller for the whole run, and it has no rule that separates soft ground. A learned model on signals the controller already has (`P`) is a much stronger baseline than hand rules.

### Question 6: partial support

| Input | Long events missed | Median delay, s | Partial-support F1 | Short events missed |
|---|---|---|---|---|
| `P` | 37.3 % | 0.12 | 0.358 | 84.5 % |
| `P` + blocks (`C4`, `C5`) | 26.6-28.0 % | 0.12-0.13 | 0.47-0.48 | 81.6-82.0 % |
| `P` + point map (`C1`-`C3`) | 2.0-2.6 % | 0.10 | 0.80-0.83 | 74.3-75.5 % |

Shear makes no difference to any of these.

### `hard-mid`, friction 0.36-0.7, never seen in training

Friction MAE: `P` 0.140, `P+C5` 0.159, `P+C3` 0.128, `P+C2` 0.109, `C1` 0.107. Models assign these windows to hard-grip in 53-64 % of cases and to hard-slip in 27-39 %. The friction estimate interpolates into the unseen range about as well as it works inside the trained ranges; shear inputs give the lowest errors here, which is the one place in v2 where shear shows a consistent benefit. It was not one of the pre-registered comparisons, so it is reported as an observation.

### Question 7: does simulation detail pay off

Models trained on the fixed nominal sole, tested on the same test simulations read in five ways (terrain F1; support F1 behaves alike):

| Input | Nominal | Shifted parameters | Pad sole | Foot-outline mask | No saturation |
|---|---|---|---|---|---|
| `P+C1` | 0.865 | 0.579 | 0.273 | 0.204 | 0.786 |
| `P+C2` | 0.876 | 0.597 | 0.226 | 0.209 | 0.788 |
| `P+C3` | 0.860 | 0.530 | 0.209 | 0.241 | 0.779 |
| `P+C4` | 0.805 | 0.737 | 0.588 | 0.743 | 0.670 |
| `P+C5` | 0.778 | 0.688 | 0.590 | 0.755 | 0.650 |

- **Answer 1 flips.** On shifted parameters, the pad sole and the outline mask the block models are really better than the point-map models on terrain F1; on the pad sole and the outline mask also on support F1.
- **Answers 2 and 3 hold** on all five test sets: no difference.
- **The 60 N limit does not decide answers 1 and 2.** Trained and tested without saturation, their verdicts are the same as nominal (`P+C3` 0.864 against `P+C5` 0.776). Questions 3 to 6 were not rerun without the limit. A model trained with saturation and tested without loses 8-14 points, so the per-point range of the real sensor must be known before training for it (QUESTIONS Q12).

By the brief's own criterion: on the point "a fine pressure map beats four blocks", the simulation cannot be trusted until real recordings exist. A point-map model learns the reading method along with the ground. Block sums are much less sensitive to it, and that robustness is itself an argument for them, or for training on real data.

### Exploratory, after the verdict: why the point-map models fail there

Not part of the seven answers (`results/v2/src3/exploratory_robustness.md`). The main models, trained on per-episode random sensor parameters, were tested on the same five test sets (terrain F1):

| Model | Training | Nominal | Shifted | Pad sole | Outline mask | No saturation |
|---|---|---|---|---|---|---|
| `P+C2` | randomized | 0.895 | 0.884 | 0.266 | 0.127 | 0.827 |
| `P+C2` | nominal only | 0.876 | 0.597 | 0.226 | 0.209 | 0.788 |
| `P+C3` | randomized | 0.884 | 0.869 | 0.212 | 0.133 | 0.797 |
| `P+C3` | nominal only | 0.860 | 0.530 | 0.209 | 0.241 | 0.779 |
| `P+C5` | randomized | 0.838 | 0.820 | 0.643 | 0.824 | 0.797 |
| `P+C5` | nominal only | 0.778 | 0.688 | 0.590 | 0.755 | 0.650 |

- **Randomizing the sensor parameters buys robustness to those parameters:** on shifted parameters the point-map models recover from 0.53-0.60 to 0.87-0.88.
- **It does not buy robustness to how the map is formed.** On the pad sole and the outline mask the randomized models fail as badly as the nominal ones.
- What differs in the inputs there (walking frames with load): the same share of load in the corner points on the pad sole, but the load sits on fewer points (90 % of it on 34 points for pads and 25 for the outline mask, against 42) and neighbouring points differ more. The outline mask also removes the corner and edge points where the box foot puts its load (25 % of the load in the outer ring against 63 %). For the outline mask, which reads the same simulations, the block sums stay close to nominal (correlation 0.96, median difference 16 %); for the pad sole, a different simulation, only the distributions can be compared (overlap 0.93-0.94).
- Reading: the point-map models use the fine spatial shape that the elastic-layer reconstruction produces. Whether a real sole produces that shape is what the simulation cannot say. A real recording would settle it with four numbers: how many taxels carry 90 % of the load at heel strike, mid-stance and toe-off; the share of load at the edges; how much neighbouring taxels differ; the per-taxel range.

## 5. v1 against v2 on the shared metrics

Not a like-for-like ablation: the sole, the sensor model, the command channels, the dataset size and the network all differ. Both runs use the same three walking sources, classes, bands and metric code.

| Run | Input | Terrain F1 | Support F1 | Partial F1 | Friction MAE | Strip latency, s | Lifted delay, s |
|---|---|---|---|---|---|---|---|
| v1 | `P` with command | 0.802 | 0.755 | 0.353 | 0.186 | 0.67 | 0.29 |
| v1 | `P+N` (2 x 4 zone Fz from the solver) | 0.861 | 0.866 | 0.633 | 0.166 | 0.52 | 0.07 |
| v1 | `P+N+S` | 0.860 | 0.853 | 0.599 | 0.150 | 0.55 | 0.08 |
| v2 | `P`, no command | 0.817 | 0.760 | 0.358 | 0.181 | 0.64 | 0.30 |
| v2 | `P` with command (diagnostic) | 0.818 | 0.765 | 0.369 | 0.180 | 0.62 | 0.27 |
| v2 | `P+C5` (four block Fz from the map) | 0.829 | 0.810 | 0.471 | 0.174 | 0.51 | 0.12 |
| v2 | `P+C4` (plus block shear) | 0.836 | 0.813 | 0.477 | 0.176 | 0.49 | 0.11 |
| v2 | `P+C2` (point map plus block shear) | 0.889 | 0.938 | 0.827 | 0.158 | 0.36 | 0.11 |

- `P` reproduces across the two runs, with or without the command.
- v1's eight zones per foot scored higher than v2's four blocks. The two differ in resolution and in where the numbers come from (v1 read the solver's per-zone forces, which D13 showed to be unreliable while walking), so the gap cannot be assigned to one cause from these runs.
- v1's friction gain from zone shear (0.166 to 0.150) is not reproduced by block shear in v2 (0.174 against 0.176 for blocks, 0.156 against 0.158 for the map); per-point shear does lower the friction error (0.141 against 0.158, question 3). With proprioception in the input, neither version found a shear effect on terrain class or support state; without proprioception, v2 found one on terrain class (question 2).

## 6. The new walking policy (parked)

Trained in `k1_gym` (Isaac Lab) on adder, warm-started from G113, with generated terrain (rough, slopes to 10 degrees, stepping stones, 3-6 cm platforms) and friction 0.1-1.0. Acceptance rule fixed before the evaluation that mattered (D20): every chunk crossed in at least 16 of 20 baseline courses, then at least 80 % of 40 fresh courses.

| Stage | Run | What changed | Best checkpoint | Crossed, of 20 | No fall, of 20 |
|---|---|---|---|---|---|
| baseline | G113 / G80 | existing policies, same courses | | 7 / 14 | 7 / 14 |
| start | run3 | G113 recipe plus generated terrain, warm start | 200 | walked at 0.13 m/s and lost speed tracking | |
| fix 1 | run4 | command curriculum off, terrain promotion by distance walked | 500 / 1000 | 14 / 11 | 13 / 15 |
| fix 2 | run5 | more stepping stones and gaps, stricter promotion | 300 | 0 | 2 |
| fix 3 | run6 | from run4-1000: run4's terrain mix with a slightly larger plank share (0.20 against 0.15; run5 had 0.24), no gaps, smaller policy step, more weight on forward speed | 500 | 10 | 16 |

Three different fixes, none inside reach of 16 of 20 and the last within chance of the first: parked. Remaining failures are falls on plank chunks and slowing down on terrain. The checkpoints stay on adder (`/path/to/fla-terrain/k1v2/`); the evaluated checkpoints are exported in the format of the other sources (`/path/to/fla-terrain/b_runs/k1v2/policies/`) but are not used anywhere in v2.

## 7. Integrity, approximations, failures

**Verified independently.** the independent evaluation recomputed terrain F1, support F1, partial F1 and friction MAE for eight seed-0 models from the raw test episodes with its own code: equal to four decimals. Saved weights reproduce the stored predictions on 139,410 test windows (two models by the independent evaluation; all 156 models by the training implementation's script, 0 mismatches). A model trained on shuffled terrain labels scores 0.062 (real model 0.887). A shortcut check on the large partial-support gain found none: unloaded points read offset plus noise (about 0.25 N in partial-support frames; 0.06 ± 0.24 N over the void in the rig edge clip), and exact zeros are as frequent in full support as in partial. One caveat stays: the map and the support label are both computed from the same simulated geometry, so this result is as good as that geometry. Tables re-render from the committed metrics in a clean checkout with an empty diff. Independent verification re-ran the sole checks (17 of 17) and the test suite.

**Corrections made after the first tables.** Edge events lasting exactly 0.30 s (21 of 2908 in the main test set) were first split by comparing float32 times, which put 13 of them on the short side. The split is now on the integer frame difference (last frame of the stance minus first partial frame of 15 or more, that is 0.30 s or longer), which puts all 21 on the long side: 828 short and 2080 long events. Recomputed from the stored predictions: long-event miss rates moved by at most 0.003 and short-event miss rates by up to 0.042 (one leave-one-gait-out model); no verdict changed. the independent evaluation's independent recomputation compares float64 times, which rounds 7 of those 21 events to the short side; its long-event miss rates agree with the tables and its short-event miss rates are about 0.005 lower.

**Approximations and placeholders.**
- Per-point shear and block shear as described in section 3.
- **The 60 N per-point limit** is a placeholder carried over from v1. On the noise-free map 31 % of loaded walking samples have a point above 60 N; in the recorded data, after neighbour spread and averaging, 8.4 % of loaded foot frames have a point at the limit. Per stance on hard-grip with G113, the recorded foot force is 9 % (normal) and 11 % (shear) below the true force in the median, and 18-24 % below at the 5th percentile (D19). Section 4 shows answers 1 and 2 do not depend on the limit; absolute forces in the dataset are biased low.
- Sole size, point count and pitch, part boundaries, middle clearance, layer compliance, noise and resolution are defaults or guesses (brief section 10), randomized per episode within the ranges listed in `NOTES_A.md`.
- The rectangular mask and the corner loading while walking (section 1).
- The pilot numbers were produced with the earlier compliance and are not comparable with the final tables.

**Failures and things not done.**
- No fourth walking source (section 6).
- No source completes the course in 80 % of runs; courses were kept as designed and scored up to the fall. Completed runs last 70-80 s, not 60.
- Leave-one-gait-out has three folds, not four.
- The sensitivity test trains on one fixed setting by design; the exploratory table in section 4 adds the randomization-trained models and is labelled as post hoc.
- Provenance: the training run started from a clean tree at `0b2624d`; the checkout moved during the run, so five model configs record a dirty tree although the training code was identical (`PROVENANCE.md`).
- Process: the Mac lost its VPN from 09:05 to 14:54 EET; a deputy manager on the compute server ran that part of the work (D15-D20). One of our own training processes hung on adder for about six hours holding 5 GB of the shared GPU before it was found and killed.
- Renders show the raw map with saturated points marked. Corner loading differs from observations of the real sole.

## 8. Deliverables and how to reproduce

- Weights of all 156 models: `/path/to/fla-terrain/models/v2_src3/`, listed in `results/v2/WEIGHTS.md`. Check: `$PY scripts/verify_v2_weights.py --root /path/to/fla-terrain/models/v2_src3 --all`.
- Tables: `results/v2/src3/tables.md`; metrics `results/v2/src3/metrics.json`; course scores `course.json`.
- Datasets on  `data/v2` (8.0 GB), `data/v2_sens`, `data/v2_course`.
- Renders: `~/fla/results/v2/videos/` (three course runs, three sole close-ups at 0.25x, five foot-rig clips), figures in `~/fla/results/v2/figures/`, page `~/fla/results/v2/index.html`.
- Reproduce on the compute server, in a clean checkout (`PY=python3`):
  - sole checks: `$PY scripts/check_sole_v2.py`
  - all models and tables from the existing datasets: `scripts/run_v2.sh walk_g113,walk_g80,walk_sonic results/v2/src3 <model dir>`, then `scripts/run_v2_src3_followup.sh`
  - tables only, from the committed metrics: see `PROVENANCE.md`.

## 9. Next steps

1. **Record the real sole before any more simulation of it.** Ten minutes of standing, hand presses and walking on two surfaces would answer what section 4 cannot: how load is spread while walking, the per-point range, and whether per-point shear has content of its own.
2. **Decide the foot.** If the real K1 sole really carries load on front and back with a groove between, the simulated foot should get rounded edges so that it stops rocking on box corners; that changes what the walking policies see and needs decision (Q8, Q9).
3. **Train for robustness to the reading method**, not on one setting, and keep block sums as the fallback input: they lose 5 points of terrain F1 and most of the partial-support detection, and they survive a change of sensor.
4. **Put the deployed controller in the training data** and consider a sole-only model where the gait is not known in advance.
