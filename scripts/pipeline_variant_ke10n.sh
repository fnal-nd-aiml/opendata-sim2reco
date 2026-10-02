#!/usr/bin/env bash
# Remaining part of the variant study: the neutron variant (10 MeV, neutrons as tokens) and the final comparison.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/pipeline_variants; mkdir -p $L
stamp() { date "+%F %T"; }
while pgrep -f "scripts/run_m[23].py" >/dev/null; do sleep 60; done
TAG=ke10n; OPTS="--ke-cut 10 --neutrons"
echo "$(stamp) === variant $TAG ($OPTS) restart ==="
$PY -u scripts/run_m2.py reports/m2_1A_$TAG --epochs 30 --bs 1024 --lr 3e-4 $OPTS > $L/m2_$TAG.log 2>&1; echo "$(stamp) M2 $TAG done: $(grep -E '^epoch 29' $L/m2_$TAG.log | cut -c1-110)"
$PY -u scripts/run_m3.py reports/m3_1A_$TAG --init reports/m2_1A_$TAG/model.pt --epochs 16 --lr 2e-4 --steps 64 > $L/m3_$TAG.log 2>&1; echo "$(stamp) M3 $TAG done"
mkdir -p reports/m3_1A_${TAG}_on2p2h && ln -sf ../m3_1A_$TAG/model.pt reports/m3_1A_${TAG}_on2p2h/model.pt
$PY -u scripts/run_m3.py reports/m3_1A_${TAG}_on2p2h --eval-only --steps 64 --only-inttype 8 > $L/eval_${TAG}_on2p2h.log 2>&1; echo "$(stamp) eval $TAG on 2p2h done"
$PY scripts/variants_compare.py reports/variants_1A --variants baseline:reports/m3_1A:reports/m3_1A_on2p2h ke10:reports/m3_1A_ke10:reports/m3_1A_ke10_on2p2h ke10n:reports/m3_1A_ke10n:reports/m3_1A_ke10n_on2p2h > $L/compare.log 2>&1; echo "$(stamp) compare done"
echo "$(stamp) VARIANTS DONE"
