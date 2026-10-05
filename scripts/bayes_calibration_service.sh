#!/usr/bin/env bash
# Calibration evals (B then A) on the test-split control, then on the training-split control (attribution test), then figures.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A --model reports/m3_1A_no2p2h --tag _B --draws 8 --n-control 300000 > reports/bayes_1A_B.log 2>&1
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A --model reports/m3_1A --tag _A --draws 8 --n-control 300000 > reports/bayes_1A_A.log 2>&1
.venv/bin/python scripts/bayes_binned_figures.py reports/bayes_1A --draws 8 > reports/bayes_1A/figure.log 2>&1
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A_trainctl --model reports/m3_1A_no2p2h --tag _B --draws 8 --n-control 300000 --control-split 0 > reports/bayes_1A_trainctl_B.log 2>&1
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A_trainctl --model reports/m3_1A --tag _A --draws 8 --n-control 300000 --control-split 0 > reports/bayes_1A_trainctl_A.log 2>&1
echo "BAYES EVAL DONE"
