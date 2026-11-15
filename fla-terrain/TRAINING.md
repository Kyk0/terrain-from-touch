# Training and configuration

All training source and experiment configurations are included in this repository. Commands below run from `fla-terrain/`.

| Purpose | Files |
|---|---|
| Main dataset and train/validation/test splits | `configs/dataset_v2.yaml` |
| Sensor sensitivity experiments | `configs/sens_v2.yaml`, `configs/sens_v2_nosat.yaml` |
| Mixed-course evaluation | `configs/course_v2.yaml` |
| Pilot configurations | `configs/pilot_v2.yaml`, `configs/pilot_v2_5h.yaml` |
| First experiment | `configs/dataset_v1.yaml` |
| Dataset generation | `scripts/gen_dataset.py` |
| Training and model architecture | `scripts/train_v2.py`, `fla_terrain/models/v2.py`, `fla_terrain/models/nets.py` |
| Full experiment and evaluation | `scripts/run_v2.py`, `scripts/run_v2.sh`, `scripts/run_v2_src3_followup.sh` |
| Flat-foot walking checkpoints | `walking/checkpoints/G113_m2200.npz`, `walking/checkpoints/G80_m4200.npz` |

## Reproduction

Set the `/path/to/fla-terrain/` placeholders to your local dataset, model and robot asset locations before running. These paths occur in the scripts and model modules. Robot assets and the SONIC policy are external dependencies; see `walking/README.md` for their interfaces and setup.

Generate the main dataset:

```sh
python3 scripts/gen_dataset.py --plan configs/dataset_v2.yaml --out /path/to/fla-terrain/data/v2 --workers 40
```

Train selected input combinations:

```sh
python3 scripts/train_v2.py --data /path/to/fla-terrain/data/v2 --tag local_run --views P,P+Nreal,P+Nreal+Sblk,Nreal+Sblk --seeds 0,1,2
```

The default training duration is 25 epochs. Architecture, normalization, optimizer and augmentation settings are in `fla_terrain/models/v2.py` and `nets.py`; the YAML files configure data generation, not model training. Training records each model's configuration and source commit.

The full experiment also requires the sensitivity and course datasets. See `REPORT_V2.md` for the experiment protocol and `results/v2/WEIGHTS.md` for the trained-model inventory.

## Included and external artifacts

The repository includes training code, dataset-generation configurations, evaluation code, metrics, sample episodes and two flat-foot walking checkpoints. The full v2 dataset (about 8 GB), 156 trained terrain models (about 504 MB including prediction artifacts), external robot assets and SONIC checkpoint are not included in this publication. Configuration files and code alone do not reproduce the results without those inputs and the required Python environment.
