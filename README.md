# opendata-sim2reco

An AI/ML surrogate for the MINERvA detector: per-event mapping from GENIE final-state truth to MasterAnaDev
reconstructed variables, trained on the [MINERvA open data](https://minerva.fnal.gov/opendata/) MC, built to
extrapolate to final states the released MC does not cover (alternative generators, higher multiplicity, new
kinematic regions).

Status: M0 (data pipeline), M1 (baselines) and M2 (set encoder + flow-matching surrogate, closure AUC 0.57) done; M3 (prong set model + vertex-plane head) done; 2p2h holdout done on playlist 1A; input definition adopted: 10 MeV threshold with neutron tokens (variant study in `reports/variants_1A`); Bayesian last layers give calibrated epistemic uncertainty (`reports/bayes_1A`). Out-of-support holdout next. Read [`docs/PROJECT.md`](docs/PROJECT.md) first; performance numbers are in `reports/performance/main.tex` (compiled with tectonic; describes the current model, not the milestone history).

## Model schematic

![model schematic](docs/figures/model_schematic.svg)

Architecture of the current surrogate (encoder, classification heads, vertex-plane head, event flow, prong set flow, decoder). Source: `docs/figures/model_schematic.tex` (TikZ); regenerate with `bash scripts/render_schematic.sh`.

## Quick start

```
bash scripts/setup_env.sh                                  # Python >= 3.9 venv with torch (CUDA), uproot, ...
source .venv/bin/activate
python scripts/slim_remote.py configs/MediumEnergy_FHC_StandardMC_Playlist1A.txt data/slim_1A --budget-gb 9.5
                                                           # stream from xrootd, 20 GB ROOT -> 300 MB Parquet per file
python scripts/slim.py data/slim some_local_file.root      # same for a local ROOT file
python scripts/dataset_summary.py data/slim/MasterAnaDev_mc_AnaTuple_run00113069_Playlist
python scripts/run_m1.py data/slim/MasterAnaDev_mc_AnaTuple_run00113069_Playlist reports/m1   # baselines, ~1 min on a GPU
python scripts/run_m2.py reports/m2_1A --epochs 30                                         # surrogate, ~2 h on an RTX 3090, all files in data/slim
python scripts/run_m3.py reports/m3_1A --epochs 16 --init reports/m2_1A/model.pt                 # prong model, warm start from M2
python scripts/surrogate_to_ntuple.py reports/m3_1A data/slim/<stem>.truth.parquet out.root  # truth -> pruned MasterAnaDev ntuple (80 branches)
python scripts/conform_ntuple.py out.root MasterAnaDev_data_AnaTuple_run00010255_Playlist.root out_full.root --strict
                                                           # optional: full 3,687-branch schema + Meta tree, unmodelled branches at their sentinel defaults
bash scripts/pipeline_1A.sh                                 # unattended: slim-wait, caches, M1, model A, 2p2h-blind model B, evaluations
python scripts/holdout_compare.py reports/holdout_2p2h_1A --a reports/m3_1A_on2p2h --b reports/m3_1A_no2p2h_on2p2h --b-control reports/m3_1A_no2p2h_control --a-full reports/m3_1A
python scripts/fit_bayes_last.py reports/m3_1A                  # Bayesian last layers on the frozen model (~1 min)
python scripts/bayes_uncertainty_eval.py reports/bayes_1A --model reports/m3_1A_no2p2h --tag _B   # epistemic flag + calibration
pytest -q                                                  # round-trip tests (ROOT file or its slimmed Parquet)
(cd reports/performance && tectonic -X compile main.tex)   # performance report PDF
```

## Layout

- `sim2reco/io/` — branch lists, ROOT -> Parquet slimming, pruned-ntuple writer.
- `sim2reco/prep/` — input pipeline (particle selection, context), canonical prong table encode/decode, muon decode, beam/detector frames.
- `sim2reco/data/` — torch dataset (padded particle sets, context, Tier 0/1 targets) and subrun-level splits.
- `sim2reco/models/` — set encoder, flow matching, surrogate (M2), MDN baseline (M1); `sim2reco/train/` — `baselines.py` (M1), `m2.py` (training + closure evaluation); `sim2reco/eval/` — metrics and plots.
- `sim2reco/data/compact.py` — ragged multi-file dataset and the Tier 1 transform (model space <-> tuple units).
- `reports/m1/` — M1 metrics, LaTeX tables, figures; `reports/performance/main.tex` — the performance report.
- `configs/data.yaml` — selection and split settings.
- `tests/` — pipeline and exact round-trip tests on the example ntuple.
- `docs/PROJECT.md` — project design: goal, inputs, conditioning, outputs, variable-length handling, extrapolation strategy, milestones, open questions.
- `docs/tuple_notes.md` — what is actually in the open-data MC AnaTuple, with measured numbers from one ME FHC file.
- `docs/branches/` — full branch lists of the `MasterAnaDev`, `Truth`, `Meta` trees, and a copy of the upstream `MAD_tuple_MainDoc.csv`.

## Data

Open-data AnaTuples are streamed from Fermilab's xrootd door and slimmed to Parquet under `data/slim/`
(not committed). No ROOT file needs to be stored locally; reading needs only `uproot`, `awkward` and the
`xrootd` Python bindings. Source: https://minerva.fnal.gov/getdata.

## Acknowledgment

The authors thank the MINERvA Collaboration for making their data, simulated data, and analysis tools available
to the community. Data release: https://doi.org/10.15484/3022562 (CC0).
