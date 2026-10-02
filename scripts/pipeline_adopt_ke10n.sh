#!/usr/bin/env bash
# After adopting 10 MeV + neutron tokens: retrain the 2p2h-blind model B with the new definition, regenerate all
# downstream products of model A (reports/m3_1A), and the comparisons. Run as a user service:
#   systemd-run --user --unit sim2reco-adopt --working-directory=$PWD --collect bash scripts/pipeline_adopt_ke10n.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/pipeline_adopt; mkdir -p $L
stamp() { date "+%F %T"; }
while pgrep -f "scripts/run_m[23].py" >/dev/null; do sleep 60; done
FIRST=$(ls data/slim_1A/*.truth.parquet | head -1 | sed 's/.truth.parquet//')
$PY scripts/run_m1.py $FIRST reports/m1_1A > $L/m1.log 2>&1; echo "$(stamp) M1 (new selection) done"
# model B with the adopted definition
$PY -u scripts/run_m2.py reports/m2_1A_no2p2h --epochs 30 --bs 1024 --lr 3e-4 --exclude-inttype 8 > $L/m2_B.log 2>&1; echo "$(stamp) M2 B done: $(grep -E '^epoch 29' $L/m2_B.log | cut -c1-110)"
$PY -u scripts/run_m3.py reports/m3_1A_no2p2h --init reports/m2_1A_no2p2h/model.pt --epochs 16 --lr 2e-4 --steps 64 --exclude-inttype 8 > $L/m3_B.log 2>&1; echo "$(stamp) M3 B done"
mkdir -p reports/m3_1A_no2p2h_on2p2h reports/m3_1A_no2p2h_control
ln -sfn ../m3_1A_no2p2h/model.pt reports/m3_1A_no2p2h_on2p2h/model.pt; ln -sfn ../m3_1A_no2p2h/model.pt reports/m3_1A_no2p2h_control/model.pt
$PY -u scripts/run_m3.py reports/m3_1A_no2p2h_on2p2h --eval-only --steps 64 --only-inttype 8 > $L/eval_B_on2p2h.log 2>&1; echo "$(stamp) eval B on 2p2h done"
$PY -u scripts/run_m3.py reports/m3_1A_no2p2h_control --eval-only --steps 64 --only-inttype 1 2 3 4 > $L/eval_B_control.log 2>&1; echo "$(stamp) eval B control done"
# model A downstream products with the adopted definition
mkdir -p reports/m3_1A_paper && ln -sfn ../m3_1A/model.pt reports/m3_1A_paper/model.pt
SIM2RECO_REAL_LABEL="open dataset" $PY -u scripts/run_m3.py reports/m3_1A_paper --eval-only --steps 64 > $L/eval_A_paper.log 2>&1; echo "$(stamp) paper figures done"
$PY scripts/confusion_symmetry.py reports/m3_1A > $L/symmetry.log 2>&1; echo "$(stamp) symmetry figure done"
$PY scripts/holdout_compare.py reports/holdout_2p2h_1A --a reports/m3_1A_on2p2h --b reports/m3_1A_no2p2h_on2p2h --b-control reports/m3_1A_no2p2h_control --a-full reports/m3_1A > $L/holdout.log 2>&1; echo "$(stamp) holdout compare done"
$PY scripts/variants_compare.py reports/variants_1A --variants ke50:reports/m3_1A_ke50:reports/m3_1A_ke50_on2p2h ke10:reports/m3_1A_ke10:reports/m3_1A_ke10_on2p2h ke10n:reports/m3_1A:reports/m3_1A_on2p2h > $L/variants.log 2>&1; echo "$(stamp) variants compare done"
$PY scripts/technote_numbers.py reports/m3_1A reports/m3_1A/numbers.tex --m1 reports/m1_1A/metrics.json --holdout reports/holdout_2p2h_1A > $L/numbers.log 2>&1; echo "$(stamp) numbers done"
echo "$(stamp) ADOPT DONE"
