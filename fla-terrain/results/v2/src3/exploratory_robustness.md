# EXPLORATORY (A12, post hoc): why the point-map models fail on the pad sole and the outline mask

Not part of the seven D14 answers; the pre-registered question-7 verdict stands. No retraining: saved weights evaluated on the sensitivity test sets (mode walk, test; mean ± population std over 3 seeds). Commit `9b11e2300d881d5b4e01161ce9be491ae62c670c` (clean tree: True). Command: `scripts/exploratory_robustness.py`.

## 1. Trained with per-episode randomized sole/sensor parameters (main models) vs trained on the fixed nominal setting (sensitivity models)

| model | training | metric | nominal | shifted | pads | g1 | nosat |
|---|---|---|---|---|---|---|---|
| P+C2 | randomized (main) | M1 | 0.895 ± 0.002 | 0.884 ± 0.003 | 0.266 ± 0.023 | 0.127 ± 0.017 | 0.827 ± 0.022 |
| P+C2 | randomized (main) | M2 | 0.941 ± 0.002 | 0.933 ± 0.002 | 0.587 ± 0.014 | 0.500 ± 0.077 | 0.927 ± 0.005 |
| P+C2 | randomized (main) | M3 | 0.162 ± 0.006 | 0.166 ± 0.005 | 0.231 ± 0.015 | 0.247 ± 0.026 | 0.159 ± 0.003 |
| P+C2 | nominal only (sens) | M1 | 0.876 ± 0.003 | 0.597 ± 0.038 | 0.226 ± 0.009 | 0.209 ± 0.060 | 0.788 ± 0.007 |
| P+C2 | nominal only (sens) | M2 | 0.935 ± 0.002 | 0.836 ± 0.023 | 0.571 ± 0.005 | 0.549 ± 0.029 | 0.915 ± 0.001 |
| P+C2 | nominal only (sens) | M3 | 0.175 ± 0.007 | 0.215 ± 0.015 | 0.231 ± 0.011 | 0.240 ± 0.011 | 0.163 ± 0.007 |
| P+C3 | randomized (main) | M1 | 0.884 ± 0.003 | 0.869 ± 0.006 | 0.212 ± 0.002 | 0.133 ± 0.038 | 0.797 ± 0.020 |
| P+C3 | randomized (main) | M2 | 0.929 ± 0.013 | 0.926 ± 0.007 | 0.545 ± 0.028 | 0.526 ± 0.120 | 0.912 ± 0.016 |
| P+C3 | randomized (main) | M3 | 0.158 ± 0.006 | 0.169 ± 0.003 | 0.234 ± 0.009 | 0.243 ± 0.014 | 0.162 ± 0.002 |
| P+C3 | nominal only (sens) | M1 | 0.860 ± 0.002 | 0.530 ± 0.018 | 0.209 ± 0.014 | 0.241 ± 0.009 | 0.779 ± 0.012 |
| P+C3 | nominal only (sens) | M2 | 0.936 ± 0.003 | 0.792 ± 0.011 | 0.584 ± 0.007 | 0.512 ± 0.053 | 0.927 ± 0.007 |
| P+C3 | nominal only (sens) | M3 | 0.177 ± 0.006 | 0.252 ± 0.019 | 0.247 ± 0.009 | 0.236 ± 0.008 | 0.168 ± 0.003 |
| P+C5 | randomized (main) | M1 | 0.838 ± 0.001 | 0.820 ± 0.002 | 0.643 ± 0.008 | 0.824 ± 0.003 | 0.797 ± 0.010 |
| P+C5 | randomized (main) | M2 | 0.832 ± 0.001 | 0.811 ± 0.005 | 0.703 ± 0.004 | 0.813 ± 0.005 | 0.802 ± 0.013 |
| P+C5 | randomized (main) | M3 | 0.174 ± 0.001 | 0.179 ± 0.001 | 0.202 ± 0.001 | 0.175 ± 0.002 | 0.166 ± 0.001 |
| P+C5 | nominal only (sens) | M1 | 0.778 ± 0.010 | 0.688 ± 0.003 | 0.590 ± 0.007 | 0.755 ± 0.008 | 0.650 ± 0.011 |
| P+C5 | nominal only (sens) | M2 | 0.785 ± 0.011 | 0.729 ± 0.003 | 0.704 ± 0.002 | 0.763 ± 0.010 | 0.746 ± 0.009 |
| P+C5 | nominal only (sens) | M3 | 0.212 ± 0.014 | 0.217 ± 0.008 | 0.238 ± 0.013 | 0.217 ± 0.015 | 0.214 ± 0.008 |

Note: the main models were trained on data/v2 (randomized), so their nominal column is a sensor setting inside their training distribution but other episodes (the sensitivity test episodes); shifted lies at the edges of the randomization ranges (spread 10 mm, compliance 0.4 mm, crosstalk 0.05 are range ends); pads, outline mask and no saturation are outside it.

## 2. What differs in the inputs (walk test episodes, foot frames with recorded load > 20 N)

| test set | loaded foot-frames | load share in 4 corner points | load share in outer ring | frames with a point at 60 N | points for 90 % of load, median [p10, p90] | spatial roughness | block sums vs nominal |
|---|---|---|---|---|---|---|---|
| nominal | 331240 | 18.4 % | 63.0 % | 8.4 % | 42 [11, 90] | 0.22 | reference |
| pads | 330857 | 18.3 % | 61.1 % | 10.6 % | 34 [10, 81] | 0.29 | different simulation (pad sole): share overlap per block 0.93, 0.94, 0.93, 0.93 |
| g1 | 331335 | 0.0 % | 25.5 % | 15.3 % | 25 [6, 67] | 0.42 | same simulation: r = 0.957, median rel. diff 16.3 %; share overlap 0.92, 0.93, 0.95, 0.96 |

## 3. Diagnostic: outline mask with the 'point exists' input channel forced to all ones

The map encoder has a mask channel (1 = point exists); in training it was always all ones. Here the outline test set is re-predicted with the channel set to ones (the absent points still read 0 N), M1:

| model | M1 outline, mask channel as recorded | M1 outline, mask channel all ones |
|---|---|---|
| P+C2 randomized (main) | 0.127 ± 0.017 | 0.110 ± 0.023 |
| P+C2 nominal only (sens) | 0.209 ± 0.060 | 0.152 ± 0.074 |
| P+C3 randomized (main) | 0.133 ± 0.038 | 0.109 ± 0.026 |
| P+C3 nominal only (sens) | 0.241 ± 0.009 | 0.195 ± 0.024 |

Block share overlap = histogram intersection (0..1) of each block's share of the foot load (front_l, front_r, back_l, back_r) against nominal; 1 = identical distributions.
