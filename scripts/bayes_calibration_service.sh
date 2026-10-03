#!/usr/bin/env bash
cd "$(dirname "$0")/.."
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A --model reports/m3_1A_no2p2h --tag _B --draws 8 --n-control 40000 > reports/bayes_1A_B.log 2>&1
.venv/bin/python scripts/bayes_uncertainty_eval.py reports/bayes_1A --model reports/m3_1A --tag _A --draws 8 --n-control 40000 > reports/bayes_1A_A.log 2>&1
echo "BAYES EVAL DONE"
