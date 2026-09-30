#!/usr/bin/env bash
# Full retraining chain on playlist 1A, then the 2p2h holdout (model B). Runs unattended; logs in reports/pipeline_1A/.
# Usage: setsid nohup bash scripts/pipeline_1A.sh > reports/pipeline_1A/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/pipeline_1A; mkdir -p $L
stamp() { date "+%F %T"; }
# 0) wait for the detached slimming to finish
while pgrep -f "slim_remote.py configs/MediumEnergy_FHC_StandardMC_Playlist1A" >/dev/null; do sleep 60; done
echo "$(stamp) slimming done: $(ls data/slim_1A/*.truth.parquet | wc -l) files, $(du -sh data/slim_1A | cut -f1)"
# 1) caches
$PY -c "
import sys, glob, time; sys.path.insert(0,'.')
from sim2reco.data.compact import load_compact
stems=sorted(p[:-len('.truth.parquet')] for p in glob.glob('data/slim_1A/*.truth.parquet'))
t=time.time(); d=load_compact(stems); print(len(stems),'files:',len(d['subrun']),'events',d['p_offsets'][-1],'prongs; reco frac %.3f'%d['reco_exists'].mean(),'%.0f s'%(time.time()-t))
" > $L/caches.log 2>&1; echo "$(stamp) caches: $(tail -1 $L/caches.log)"
# 2) M1 baselines on the first file (tree-baseline reference columns)
FIRST=$(ls data/slim_1A/*.truth.parquet | head -1 | sed 's/.truth.parquet//')
$PY scripts/run_m1.py $FIRST reports/m1_1A > $L/m1.log 2>&1; echo "$(stamp) M1 done"
# 3) model A: M2 stage then M3 stage
$PY -u scripts/run_m2.py reports/m2_1A --epochs 30 --bs 1024 --lr 3e-4 > $L/m2_A.log 2>&1; echo "$(stamp) M2 A done: $(grep -E '^epoch 29' $L/m2_A.log | cut -c1-120)"
$PY -u scripts/run_m3.py reports/m3_1A --init reports/m2_1A/model.pt --epochs 16 --lr 2e-4 --steps 64 > $L/m3_A.log 2>&1; echo "$(stamp) M3 A done"
# 4) model B: 2p2h (intType 8) held out of train and validation, both stages from scratch
$PY -u scripts/run_m2.py reports/m2_1A_no2p2h --epochs 30 --bs 1024 --lr 3e-4 --exclude-inttype 8 > $L/m2_B.log 2>&1; echo "$(stamp) M2 B done"
$PY -u scripts/run_m3.py reports/m3_1A_no2p2h --init reports/m2_1A_no2p2h/model.pt --epochs 16 --lr 2e-4 --steps 64 --exclude-inttype 8 > $L/m3_B.log 2>&1; echo "$(stamp) M3 B done"
# 5) evaluations on the 2p2h test events (A and B) and on the non-2p2h control (B)
for M in m3_1A m3_1A_no2p2h; do
  mkdir -p reports/${M}_on2p2h && ln -sf ../$M/model.pt reports/${M}_on2p2h/model.pt
  $PY -u scripts/run_m3.py reports/${M}_on2p2h --eval-only --steps 64 --only-inttype 8 > $L/eval_${M}_on2p2h.log 2>&1; echo "$(stamp) eval $M on 2p2h done"
done
mkdir -p reports/m3_1A_no2p2h_control && ln -sf ../m3_1A_no2p2h/model.pt reports/m3_1A_no2p2h_control/model.pt
$PY -u scripts/run_m3.py reports/m3_1A_no2p2h_control --eval-only --steps 64 --only-inttype 1 2 3 4 > $L/eval_B_control.log 2>&1; echo "$(stamp) eval B control done"
echo "$(stamp) PIPELINE DONE"
