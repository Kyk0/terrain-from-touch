# B19: fact-check of REPORT_V2.md (main at 989098c)

Sources: results/v2/src3/{tables.md, exploratory_robustness.md, PROVENANCE.md, jobs/**/*.json}, DECISIONS.md D12-D23,
NOTES_B.md, results/v2/qa_b/*, data/v2_course/*.json. REPORT_V2.md not edited.

## WRONG (contradicted by a source)

1. §1 Q3 cell "No, with one exception (friction MAE with proprioception)": tables.md Q3 coded verdicts also give
   P+C1 vs P+C2 M2_partial **A worse** (0.801 vs 0.827). §4 Q3 text has it; §1 does not.
2. §1 Q6 cell "Shear changes nothing here" and §4 Q6 "Shear makes no difference to any of these": same partial-F1
   verdict (P+C1 vs P+C2, A worse) contradicts it.
3. §4 Q3 "Without proprioception the friction difference ... does not pass the spread test": C1 vs C2 M3 diff -0.0152,
   2 x max std 0.0121 (passes spread), minimum 0.0156 -> fails the 10 % floor, not the spread test (tables.md Q3).
4. §4 Q4 "Every model still loses 20 to 40 points against a gait it has seen": held-out mean vs seen: P+C4 0.374 vs
   0.834 (46), P+C5 0.348 vs 0.828 (48), C1 21, C5 38 -> 21-48 (tables.md Q4).
5. §5 "Both versions agree that shear did not improve terrain class": v2 C2 vs C3 M1 is A better (+2.6, real,
   tables.md Q2). Same bullet "v1's friction gain from shear ... is not reproduced" omits the real friction gain of
   per-point shear (P+C1 0.141 vs P+C2 0.158).
6. §4 Q5 "All runs are scored up to the fall, as fixed in D16 before any run existed": D17: the training implementation generated the
   course runs before D16 was pushed (design matched D16).
7. §6 baseline "No fall, of 20: 7 / 15": G80 no-fall on the 20 auto-T courses was 14 (B11 ref run). The 15 is my own
   error in the NOTES_B B16 table (corrected there now).
8. §6 fix 3 "fewer planks": run6 started from run4-1000 and raised plank share 0.15 -> 0.20 relative to run4; fewer only
   than run5 (0.24). (walking/k1v2_train configs, NOTES_B B16.)
9. §7 float32 correction: "6 of 21 ... on the wrong side. Largest effect: 0.003 on a long-event miss rate". Job JSONs
   (M5 vs M5_float_old): long-miss max change 0.0029 (true), but short-miss changes up to 0.042
   (logo_walk_g80/P+Nreal_s0: 0.673 -> 0.714); main P_s0 0.822 -> 0.830. Event counts moved 841/2067 -> 828/2080,
   i.e. 13 events, not 6; my float64 recompute (B17) gives 835/2073 (6 moved). The whole-frame split (">= 15 frames =
   long", PROVENANCE) puts 7 more events on the long side than a 0.30 s duration split; looks like a one-frame
   definition difference (frames counted vs intervals). Long-miss values agree with mine; short-miss values are
   ~0.005 above mine for all 8 seed-0 main models. No verdict uses short miss, so no verdict changes.

## IMPRECISE

10. §1 bullet "brief touches ... 82-85 % without it": P+C4 0.816 -> 81.6-85.4 %.
11. §1 top "block models lose little": on the pad sole blocks lose 19-22 points (0.805 -> 0.588, 0.778 -> 0.590).
12. §4 Q4 "with blocks it hurts there (P+C4 0.488 vs C4 0.623)": for C5 P helps (P+C5 0.498 vs C5 0.466).
    "The gain comes almost entirely from SONIC": C4 vs P+C4 also +13.5 on held-out G113.
13. §4 Q5 "SONIC ... walks too slowly to reach the end in six of them": SONIC clips also turn and wander; the six
    no-fall runs end before the last chunk (B18), not only from slowness. "In steps both detect ... on the second
    step": pooled yes; P on SONIC needs 3 steps (tables.md Q5).
14. §4 Q5 / §7 "Runs last 64-78 s" (D17 says 64-75): recorded no-fall runs end at 70.3-79.5 s; planned length
    60.0-69.2 s (data/v2_course, terrain_params.planned_s).
15. §3 rule-based detector inputs "foot slip speed in contact, joint tracking error, base attitude and contact timing":
    rules use double-support kinematic slip, tracking error (T_TRACK 0.08), stance-foot tilt (not base attitude),
    foothold height difference, tilt scatter, contact estimate; torque gate off (thresholds.json).
16. §3 sensor model order lists quantization before the 60 N limit; code clips, then quantizes.
17. §7 "recorded foot force per stance 9-11 % below the true force on average (D19)": D19 gives medians on hard-grip
    G113 (Fz 0.91, shear 0.89); 5th percentile 0.82 / 0.76, hard-slip Fz p5 0.52. "On average" -> median, one terrain.
18. §7 "about a third of loaded samples have a point at the limit": D19's 31 % is "above 60 N on the noise-free map".
    In the recorded nominal data 8.4 % of loaded foot-frames have a point at 60 N (exploratory_robustness.md §2).
19. §7 shortcut check "points over a void read offset plus noise (about 0.25 N)": 0.25 N is the mean |Fz| of low points
    in partial frames (B17); the void-specific number is the rig edge, 0.06 +- 0.24 N (B18). Omits that map and label
    share simulator geometry (B17 caveat).
20. §4 Q7 "loses 8-13 points": nosat changes 7.9-13.5.
21. Exploratory bullet "Block sums barely change (correlation 0.96 with nominal ...)": r = 0.957 is outline mask only,
    with median relative difference 16.3 %; for pads only share overlap (0.93-0.94) exists (different simulation).

## STATED MORE STRONGLY THAN THE EVIDENCE

22. §4 Q1 "'edge' or 'rough' on a stride that landed on flat ground is not recoverable by any input": edge recall
    without contact is 0.28-0.43 for every input, not near zero; per-foot labels are analysis-only (D14).
23. §4 Q7 / §7 "The 60 N limit does not decide any answer" / "Section 4 shows no answer depends on it": D19 planned
    Q1-Q3 in the nosat-trained arm, but only P+C2/C3/C5 were trained, so only Q1 and Q2 were recomputed; Q3 and Q4-Q6
    were not tested without saturation.
24. §1 Q7 "Answers 2 and 3 survive": survives table covers M1-M3; partial F1 (where Q3 has a real difference) not in it.

## CHECKS OUT

§4 main table, every cell; §1 Q1/Q2/Q4 numbers and verdicts; Q3 friction 0.141 vs 0.158 real; Q4 table cells, C2 0.629
vs P+C2 0.532, SONIC 0.526 vs 0.071; Q6 table; Q7 table, "Answer 1 flips", fifth arm 0.864 vs 0.776; hard-mid MAEs and
shares; Q5 course table (0.842/0.785/0.311, 0.62/0.71/3.57 s, 0.7/1.3/49.9 %), "really better" verdicts (+5.7, +5.8,
0.09 s), SONIC 0.812 vs 0.699 and 0.62 vs 0.89 s, rule SONIC tracking-error firing (B11 fit); selection P+C2 0.884,
runner-up 0.880; crossings 19/22/6, SONIC 12 no-fall; exploratory table and 0.53-0.60 -> 0.87-0.88, corner share,
34/25/42 points, outer ring 25 vs 63 %; 156 models (147 main run + 9 nosat, D19; 30+30+30+42+15+9 directories);
provenance (0b2624d clean, five dirty configs: 3 + 2, code identical); groove 16 mm, 4-15 mm (D13); compliance and
middle share (D16, D17); dataset 6139 eps, 33.2 h, 310 falls; data/v2 8.0 GB; aug gain/offset, 0.98 vs 0.75; rule
F1 0.285; v1 and v2 rows of §5; §6 rows except items 7-8; recompute to 4 decimals, 139,410 windows, shuffle 0.062 vs
0.887, VPN gap, hung process.

## NOT CHECKED

Sole-check numbers (1.0 %, 0.00 %, 0.006 mm, 80 %, 0.0009 %; not re-run); "tables re-render with an empty diff"
(not re-run); renders and page on the Mac; manager's 17/17 and pytest re-run; v1 numbers beyond tables.md.
