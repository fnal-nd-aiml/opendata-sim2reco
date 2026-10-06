#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_zero; stamp() { date +%H:%M:%S; }
mkdir -p reports/m3_1A_z_paper && ln -sfn ../m3_1A_z/model.pt reports/m3_1A_z_paper/model.pt
SIM2RECO_REAL_LABEL="open dataset" $PY -u scripts/run_m3.py reports/m3_1A_z_paper --eval-only --steps 64 > $L/eval_A_paper.log 2>&1 && echo "$(stamp) paper figures done"
$PY scripts/confusion_symmetry.py reports/m3_1A_z > $L/symmetry.log 2>&1 && echo "$(stamp) symmetry done"
echo "$(stamp) PAPER DONE"
