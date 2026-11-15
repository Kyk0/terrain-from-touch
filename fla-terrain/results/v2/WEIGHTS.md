# FLA-0 v2 model weights (the compute server, not in git)

Every model directory holds `model.pt` (state dict), `config.json` (view, seed, data dir, architecture, input
normalization, augmentation, training info, git state at save time), `test_pred.npz` (test window indices and
predictions) and `val_pred.npz`. Seeds 0, 1, 2 everywhere. Views: P = joints + IMU (no command), Pcmd = with
command (diagnostic), Nreal / Sreal = per-point maps, Nblk / Sblk = four block sums; C1 Nreal+Sreal, C2 Nreal+Sblk,
C3 Nreal, C4 Nblk+Sblk, C5 Nblk.

| group | path | models | views | trained on | size |
|---|---|---|---|---|---|
| main | /path/to/fla-terrain/models/v2_src3/main/ | 42 | P, Pcmd, P+C1..C5, C1..C5, Sblk, Sreal (14 x 3 seeds) | data/v2 (G113, G80, SONIC + stand, rig, lifted) | 212 MB |
| leave-one-gait-out, G113 held out | .../v2_src3/logo_walk_g113/ | 30 | C1..C5, P+C1..C5 | data/v2 without any G113 episode | 82 MB |
| leave-one-gait-out, G80 held out | .../v2_src3/logo_walk_g80/ | 30 | as above | data/v2 without G80 | 83 MB |
| leave-one-gait-out, SONIC held out | .../v2_src3/logo_walk_sonic/ | 30 | as above | data/v2 without SONIC | 82 MB |
| sensitivity, nominal-trained | .../v2_src3/sens/ | 15 | P+C1..C5 | data/v2_sens/nominal | 29 MB |
| sensitivity, no saturation (D19) | .../v2_src3/sens_nosat/ | 9 | P+C2, P+C3, P+C5 | data/v2_sens/nosat | 18 MB |
| pilot (A7, compliance 0.5 mm) | /path/to/fla-terrain/models/pilot_h{1.6,5}/ | 6 + 6 | P, P+C1..C5 (seed 0) | data/v2_pilot/h{1.6,5} | 15 MB |

Total v2_src3: 156 models, 504 MB.

Reproduce stored test predictions from the weights:

    PY=python3
    $PY scripts/verify_v2_weights.py --root /path/to/fla-terrain/models/v2_src3 --all
    # one model in Python:
    #   from fla_terrain.models.dataset_v2 import FrameDataV2, cache_path_v2
    #   from fla_terrain.models.v2 import predict_saved
    #   fd = FrameDataV2.build(D, cache=cache_path_v2(D)); pred = predict_saved(model_dir, fd)   # stored test windows

Unit test of the loader: `pytest tests/test_models_v2.py`.
