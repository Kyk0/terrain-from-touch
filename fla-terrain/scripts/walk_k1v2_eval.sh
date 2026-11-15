#!/bin/bash
# Full MuJoCo evaluation of one imported K1V2 checkpoint (B11): speed tracking, flat check (v2 foot), 20 courses
# (v2 foot, auto duration as the recorded runs) with the plain command and with the command-side heading hold.
# usage: scripts/walk_k1v2_eval.sh K1V2_r4_mN
set -e
cd "$(dirname "$0")/.."
N=$1; PY=python3; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY -c "
import numpy as np, mujoco
from walking import K1VelocityPolicy, load_k1_model
from walking.k1_velocity import quat_rotate_inverse
m=load_k1_model(); out=[]
for vx in (0.2,0.35,0.5,0.8):
    d=mujoco.MjData(m); p=K1VelocityPolicy('$N'); p.reset(m,d,np.random.default_rng(0)); v=[]
    for i in range(7500):
        p.pre_step(m,d,(vx,0,0) if i>500 else (0,0,0)); mujoco.mj_step(m,d)
        if i>2500: v.append(quat_rotate_inverse(d.qpos[3:7],d.qvel[:3])[0])
    out.append('%.2f->%.3f%s'%(vx,np.mean(v),' FELL' if p.fallen(m,d) else ''))
print('[tracking] $N cmd->vx: '+', '.join(out))"
$PY scripts/walk_check_v2_foot.py --sources $N | tail -1
$PY scripts/walk_eval_course.py --sources $N --sole v2 --courses 20 --auto-T --tag ${N}_autoT | tail -3
$PY scripts/walk_eval_course.py --sources $N --sole v2 --courses 20 --auto-T --heading-hold --tag ${N}_autoT_hold | tail -3
