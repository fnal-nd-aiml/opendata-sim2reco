#!/usr/bin/env bash
# NuWro -> truth parquet -> surrogate ntuple (model A, K=8 epistemic draws). Waits for the file copy and for the GPU
# calibration job (unit sim2reco-bayes6) before using the GPU.
cd "$(dirname "$0")/.."
D=data/nuwro_2026-10
until [ "$(ls $D/nuwro_*.root 2>/dev/null | wc -l)" -ge 46 ] && ! pgrep -u "$USER" -x scp >/dev/null; do sleep 30; done
.venv/bin/python scripts/nuwro_to_truth.py $D $D/nuwro_me_fhc_tracker.truth.parquet > $D/convert.log 2>&1 || { echo "CONVERT FAILED"; exit 1; }
echo "CONVERT DONE"
while systemctl --user is-active --quiet sim2reco-bayes6; do sleep 60; done
.venv/bin/python scripts/surrogate_to_ntuple.py reports/m3_1A $D/nuwro_me_fhc_tracker.truth.parquet $D/surrogate_A_nuwro_me_fhc_tracker.root --epi-draws 8 > $D/surrogate_A.log 2>&1 || { echo "SURROGATE FAILED"; exit 1; }
echo "NUWRO SURROGATE DONE"
