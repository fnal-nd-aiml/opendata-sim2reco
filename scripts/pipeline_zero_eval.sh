#!/usr/bin/env bash
# Post-training stages of the zero-flag models (rerun after the band-fill fix of the sampler): full test evaluation,
# 2p2h holdout evaluation, calibration (test and training controls), figures, floor transfer, positive control, NuWro inference.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_zero; mkdir -p $L; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1"; shift; "$@" || { echo "$(stamp) FAILED"; exit 1; }; echo "$(stamp) done"; }
run "eval A"      $PY -u scripts/run_m3.py reports/m3_1A_z --eval-only --steps 64 > $L/eval_A.log 2>&1
run "eval B"      $PY -u scripts/run_m3.py reports/m3_1A_z_no2p2h --eval-only --steps 64 > $L/eval_B.log 2>&1
run "B on 2p2h"   $PY -u scripts/run_m3.py reports/m3_1A_z_no2p2h --eval-only --steps 64 --only-inttype 8 > $L/eval_B_on2p2h.log 2>&1
run "A on 2p2h"   $PY -u scripts/run_m3.py reports/m3_1A_z --eval-only --steps 64 --only-inttype 8 > $L/eval_A_on2p2h.log 2>&1
run "cal B test"  $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z --model reports/m3_1A_z_no2p2h --tag _B --draws 8 --n-control 300000 > $L/cal_B.log 2>&1
run "cal A test"  $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z --model reports/m3_1A_z --tag _A --draws 8 --n-control 300000 > $L/cal_A.log 2>&1
run "cal B train" $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z_trainctl --model reports/m3_1A_z_no2p2h --tag _B --draws 8 --n-control 300000 --control-split 0 > $L/cal_B_train.log 2>&1
run "cal A train" $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z_trainctl --model reports/m3_1A_z --tag _A --draws 8 --n-control 300000 --control-split 0 > $L/cal_A_train.log 2>&1
run "figures"     $PY scripts/bayes_binned_figures.py reports/bayes_1A_z --draws 8 > $L/figures.log 2>&1
run "floor"       $PY scripts/bayes_floor_transfer.py reports/bayes_1A_z reports/bayes_1A_z_trainctl > $L/floor.log 2>&1
run "pos control" $PY scripts/bayes_positive_control.py reports/bayes_1A_z --model reports/m3_1A_z_no2p2h --tag _B > $L/pc.log 2>&1
run "NuWro A"     $PY scripts/surrogate_to_ntuple.py reports/m3_1A_z data/nuwro_2026-10/nuwro_me_fhc_tracker.truth.parquet data/nuwro_2026-10/surrogate_Az_nuwro_me_fhc_tracker.root --epi-draws 8 > $L/nuwro_A.log 2>&1
echo "$(stamp) PIPELINE ZERO EVAL DONE"
