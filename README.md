# EWGB-TAD: Reproducible Release

This repository contains the EWGB-TAD implementation and the predefined
experiment protocol used for this release. EWGB-TAD is an unsupervised
trajectory-level anomaly detector with three complementary views:

- `spatial_path`: 16 dimensions combining spatial-shape and path-deviation features;
- `kinematic`: 8 speed, acceleration, stopping, and turning features;
- `trajectory_shape`: 10 PCA dimensions learned from flattened, length-32 trajectories.

Each view is standardized, partitioned into adaptive granular balls, and scored
with locally adapted feature weights. This release uses a
sample-size-adaptive histogram entropy estimator. For a terminal ball with
`n_ball` samples, its local histogram uses
`max(2, ceil(log2(n_ball) + 1))` bins over the observed range of each active
feature. Empty bins are omitted from the Shannon sum. Globally constant
features are excluded from the active set; a feature that is constant only
within a terminal ball has zero local entropy and is treated as locally
reliable. If all active reliabilities vanish, uniform active-feature weights
are used; if no active features remain, a finite uniform all-feature fallback
is used. The three normalized view scores are fused with the fixed 20-bin
full-pool score-entropy rule.

## Repository contents

`src/` contains the detector, feature extraction, data construction, baselines,
evaluation, deep-model tuning, phase timing, and experiment runners. `configs/`
contains the predefined protocol, selected dataset-specific LSTM-AE/USAD settings,
and anomaly-injection manifest. `results/` contains the latest summary tables,
statistical analyses, runtime records, protocol cards, and smoke-test artifacts.
The official third-party implementations included for recent trajectory
baselines are in `external_baselines_phase3/LMTAD` and
`external_baselines_phase3/MST-OATD`.

`results/phase4/protocol_cards/BASELINE_PROTOCOL_INDEX.md` maps all 12
comparison methods to complete protocol records. The records distinguish
verified official repositories from local reproducible implementations and
paper-guided local adaptations. They specify source hashes, inputs,
normalization, objectives or scoring rules, initialization, fixed parameters or
search spaces, selection and stopping rules, seeds, hardware records, and
failure handling.
Exact label-free training/validation index arrays and their SHA-256 values are
stored in `results/phase4/protocol/unlabeled_inner_splits.npz` and
`unlabeled_inner_split_manifest.json`.

To keep the public package compact and free of private/raw trajectory data,
the 30 generated dataset caches and the main/analysis per-job raw result
directories are not included. They are rebuilt by the commands in
`REPRODUCE.md`; the included summary files and representative deep-baseline
phase-timing records are the latest result snapshot.

`results/phase4/metadata/CURRENT_RESULT_SET.json` is the sole result-set lock.
It records the selected attempt, result hash, and score hash for every included
job, together with hashes of all dataset caches and manuscript summaries. Do
not combine this snapshot with files from another result directory.

The cached `trajectory_shape` matrix is used only for the shared
34-dimensional input used by generic baselines. Every EWGB-TAD main,
component, sensitivity, stability, timing, and figure path fits the
trajectory-shape PCA again from the fixed length-32 `trajectories` array via
`canonical_shape_view` or the fitted `ThreeViewEWGBDetector`.

The released benchmark includes 30 matched dataset-seed blocks:

| Dataset | Normal | Injected anomalies | Seeds |
|---|---:|---:|---:|
| Synthetic | 5,000 | 552 | 10 |
| Grid-Network | 5,000 | 552 | 10 |
| Porto-derived | 5,000 | 552 | 5 |
| GeoLife | 3,000 | 332 | 5 |

The four anomaly types are detour, loop, speed, and route deviation. Labels
are used only by the evaluation code; no labels are used for fitting or model
selection.

The main-table baselines are IForest, ECOD, iBoost-ODE, CoMadOut, Shape-KNN,
SegmentOD, TADS, Profile-TAD, LSTM-AE, USAD, LM-TAD, and MST-OATD. Baselines
outside this list are outside the release protocol.

## Quick start

Use Python 3.10 or newer in a clean virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the deterministic unit tests and compile check:

```powershell
python -m compileall src tests
python -m unittest discover -s tests -v
```

Run the included official SOTA adapter smoke tests:

```powershell
python src/run_phase3_sota_smoke.py --baseline all
```

The smoke tests use a small synthetic pool and are not the manuscript
benchmark. For the complete protocol, follow [REPRODUCE.md](REPRODUCE.md).

## Result snapshot

The included summary uses the sample-adaptive-histogram configuration. Across
the 30 matched dataset-seed blocks, the reported EWGB-TAD
macro means are AUC `0.8972`, AUPRC `0.6780`, and F1 `0.6283`; its AUC average
rank is `3.300`. The exact seed-level values and uncertainty estimates are in
`results/phase4/summary/main_seed_level.csv` and
`results/phase4/summary/main_aggregate.json`. These numbers are provided as a
traceable snapshot, not as a replacement for rerunning the protocol.

The release contains 390/390 main jobs, 570/570 component/view/local-
sensitivity jobs, 210/210 extended-sensitivity jobs, 120/120 fixed local
entropy-bin diagnostic jobs, 16/16 deep-baseline phase-
timing jobs, and 30/30 EWGB-TAD phase-timing jobs. All status files report zero
missing jobs, and the phase-timing score checks match the main benchmark.

## Data and scope

Synthetic and Grid-Network data are generated locally. Porto-derived and
GeoLife experiments require external raw trajectory files that are not
redistributed in this repository. See [DATA.md](DATA.md) for the expected
files and environment variables. The released Porto and GeoLife benchmark
labels are controlled anomaly injections, not natural-anomaly ground truth.

See [VERSION.md](VERSION.md) for the release boundary and [NOTICE.md](NOTICE.md)
for third-party attribution and license notes. Exact CPU/GPU software and
hardware records are documented in [ENVIRONMENT.md](ENVIRONMENT.md).
