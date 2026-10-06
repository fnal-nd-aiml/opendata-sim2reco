#!/usr/bin/env bash
# Zero-flag models: full test evaluation into the model directories (the 2p2h-only evaluation had overwritten them),
# 2p2h and control evaluations into separate directories, holdout comparison and its bootstrap uncertainties.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_zero; mkdir -p $L; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1"; shift; "$@" || { echo "$(stamp) FAILED"; exit 1; }; echo "$(stamp) done"; }
for M in m3_1A_z m3_1A_z_no2p2h; do mkdir -p reports/${M}_on2p2h && ln -sfn ../$M/model.pt reports/${M}_on2p2h/model.pt; done
mkdir -p reports/m3_1A_z_no2p2h_control && ln -sfn ../m3_1A_z_no2p2h/model.pt reports/m3_1A_z_no2p2h_control/model.pt
run "eval A full"   $PY -u scripts/run_m3.py reports/m3_1A_z --eval-only --steps 64 > $L/eval_A.log 2>&1
run "eval B full"   $PY -u scripts/run_m3.py reports/m3_1A_z_no2p2h --eval-only --steps 64 > $L/eval_B.log 2>&1
run "A on 2p2h"     $PY -u scripts/run_m3.py reports/m3_1A_z_on2p2h --eval-only --steps 64 --only-inttype 8 > $L/eval_A_on2p2h.log 2>&1
run "B on 2p2h"     $PY -u scripts/run_m3.py reports/m3_1A_z_no2p2h_on2p2h --eval-only --steps 64 --only-inttype 8 > $L/eval_B_on2p2h.log 2>&1
run "B control"     $PY -u scripts/run_m3.py reports/m3_1A_z_no2p2h_control --eval-only --steps 64 --only-inttype 1 2 3 4 > $L/eval_B_control.log 2>&1
run "holdout cmp"   $PY scripts/holdout_compare.py reports/holdout_2p2h_1A_z --a reports/m3_1A_z_on2p2h --b reports/m3_1A_z_no2p2h_on2p2h --b-control reports/m3_1A_z_no2p2h_control --a-full reports/m3_1A_z > $L/holdout.log 2>&1
run "holdout unc"   $PY scripts/holdout_uncertainty.py reports/holdout_2p2h_1A_z --a reports/m3_1A_z --b reports/m3_1A_z_no2p2h > $L/holdout_unc.log 2>&1
echo "$(stamp) PIPELINE ZERO HOLDOUT DONE"
