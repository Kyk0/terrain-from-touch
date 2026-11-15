#!/bin/bash
cd /path/to/fla-terrain/a
PY=python3
D=/path/to/fla-terrain/data
L=/path/to/fla-terrain/logs
$PY scripts/gen_dataset.py --plan configs/dataset_v2.yaml --out $D/v2 --workers 40 > $L/wA_gen_v2_main.log 2>&1
$PY scripts/gen_dataset.py --plan configs/sens_v2.yaml --out $D/v2_sens --workers 40 > $L/wA_gen_v2_sens.log 2>&1
$PY scripts/gen_dataset.py --plan configs/course_v2.yaml --out $D/v2_course --workers 40 > $L/wA_gen_v2_course.log 2>&1
echo ALL_DONE
