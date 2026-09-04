# MST-OATD protocol card

Benchmark role: included in the 13-method benchmark with disclosed temporal-input adaptation.

## Identity and provenance

- Paper: *Multi-Scale Detection of Anomalous Spatio-Temporal Trajectories in
  Evolving Trajectory Datasets*
- Venue: ACM KDD 2024
- DOI: `10.1145/3637528.3671874`
- Official repository: <https://github.com/chwang0721/MST-OATD>
- Verified commit: `db94b41c1b6fd333d6776ef2dcbbbeec39d16c02`
- License: MIT
- Local vendor tree: `external_baselines_phase3/MST-OATD`
- Vendor policy: repository remains unmodified; adaptation is in
  `src/phase3_mstoatd_adapter.py`.

## Unsupervised protocol

- Fit population: the complete unlabeled pool for each dataset and seed.
- Spatial input: `18 x 18` global-grid tokens with a four-neighbour grid
  adjacency matrix and a reserved padding token.
- Temporal input: deterministic 15-second position-derived time vectors shared
  across trajectories.
- Temporal limitation: the four predefined coordinate pools do not expose
  comparable observed timestamps. The temporal branch therefore receives a
  sequence-position clock, not event time. This limitation must be disclosed
  when the full results are interpreted.
- Labels: unavailable to preprocessing, training, prior fitting, and scoring.
- API compatibility: the official trainer requires a label-shaped argument for
  optional evaluation logging. The adapter passes an all-zero placeholder,
  never the benchmark labels; this placeholder is not used by the
  reconstruction or KL training losses.
- Objective: official spatial and temporal reconstruction losses plus
  categorical and Gaussian KL terms.
- Score: official `1 - max spatial likelihood * max temporal likelihood`, with
  higher values indicating greater anomaly likelihood.

## Full-run configuration

- Architecture: embedding dimension 128, hidden dimension 512, 20 clusters,
  scale sizes 2 and 4.
- Pretraining: 8 epochs, spatial and temporal learning rates `2e-3`.
- Main training: 10 epochs, spatial and temporal learning rates `3e-4`.
- Optimizers: official AdamW for model parameters and Adam for mixture-prior
  parameters.
- Schedule: official StepLR, step size 2, gamma 0.9.
- Batch size: 1,600; reduce only the micro-batch size if required by memory and
  record the deviation.
- Early stopping: disabled; official fixed epoch counts are used.
- Tuning budget: no label-based search; official defaults and the common fixed
  grid are used for all datasets.
- Seeds: 10 for Synthetic and Grid-Network; 5 for Porto-derived and GeoLife.
- Hardware and timing evidence: formal runs use a single GPU. Per-dataset and
  per-seed end-to-end runtime and peak GPU memory are recorded in
  `results/phase4/summary/runtime_by_dataset_method.csv` and the corresponding
  JSON summary. The separate phase-timing summary with
  preparation/training/scoring decomposition is provided in the same
  representative phase-timing summary as the other three deep baselines.
- Temporal checkpoint SHA-256:
  `0CB1A9DAF3C0DAC2331381384B9782B7A8458E6952F05745756268CDB8471B8F`.

## Failure handling

- Do not remove either spatial or temporal branch or replace the mixture model.
- A method-level incompatibility is reported as a failed official baseline, not
  repaired with a custom architecture.
- Keep command, seed, traceback, runtime, and partial output for every failure.

## Smoke-test verification

- The included adapter smoke test is diagnostic and is not part of the formal result table.
- Test pool: 128 normal plus 16 injected trajectories, length 32.
- Reduced smoke model: hidden dimension 32, 3 clusters, 1 pretraining epoch,
  and 1 training epoch. The official 128-dimensional temporal checkpoint,
  losses, dual branches, and score are preserved.
- Gate checks: training completed; 144/144 finite scores returned; score
  direction verified; no labels consumed during fitting.
- Diagnostic AUC/AUPRC are smoke-only and cannot be used as manuscript results
  or hyperparameter evidence.
