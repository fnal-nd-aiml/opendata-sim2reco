#!/usr/bin/env bash
# Attribution test: the marginal calibration with the control sample drawn from the TRAINING split.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A_trainctl --model reports/m3_1A_no2p2h --tag _B --draws 8 --n-control 300000 --control-split 0 > reports/bayes_1A_trainctl_B.log 2>&1
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A_trainctl --model reports/m3_1A --tag _A --draws 8 --n-control 300000 --control-split 0 > reports/bayes_1A_trainctl_A.log 2>&1
echo "TRAINCTL DONE"
