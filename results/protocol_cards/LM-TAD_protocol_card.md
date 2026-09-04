# LM-TAD protocol card

Benchmark role: included in the 13-method benchmark.

## Identity and provenance

- Paper: *Trajectory Anomaly Detection with Language Models*
- Venue: ACM SIGSPATIAL 2024
- DOI: `10.1145/3678717.3691257`
- Official repository: <https://github.com/jonathankabala/LMTAD>
- Verified commit: `80bb89a8ea108db8f13cb9959826424e9c45f41c`
- License: MIT
- Local vendor tree: `external_baselines_phase3/LMTAD`
- Vendor policy: repository remains unmodified; adaptation is in
  `src/phase3_lmtad_adapter.py`.

## Unsupervised protocol

- Fit population: the complete unlabeled pool for each dataset and seed.
- Input: trajectory coordinates are mapped through one globally fitted
  `18 x 18` grid. Each sequence is `SOT + 32 cell tokens + EOT`.
- Labels: unavailable to tokenization, training, validation, checkpoint
  selection, and scoring.
- Objective: official autoregressive next-token cross-entropy.
- Score: official mean negative log next-token probability, reported with
  higher values indicating greater anomaly likelihood.

## Full-run configuration

- Architecture: 8 layers, 12 heads, embedding dimension 768, dropout 0.2.
- Optimizer: official AdamW, learning rate `3e-4`, weight decay `0.1`, betas
  `(0.9, 0.99)`, gradient clipping `1.0`.
- Schedule: 5,000-step linear warmup followed by official cosine decay to
  `3e-5` at 60,000 steps.
- Batch size: 32.
- Epochs: fixed 50.
- Validation: deterministic unlabeled 90/10 split by seed; retain the
  checkpoint with minimum validation cross-entropy.
- Early stopping: disabled to match the official fixed-epoch script.
- Tuning budget: no label-based search; official architecture and optimizer
  settings plus the common fixed grid are used for every dataset.
- Seeds: 10 for Synthetic and Grid-Network; 5 for Porto-derived and GeoLife.
- Hardware and timing evidence: formal runs use a single GPU. Per-dataset and
  per-seed end-to-end runtime and peak GPU memory are recorded in
  `results/phase4/summary/runtime_by_dataset_method.csv` and the corresponding
  JSON summary. The separate phase-timing summary with
  preparation/training/scoring decomposition is provided in the same
  representative phase-timing summary as the other three deep baselines.

## Failure handling

- Reduce micro-batch size only if memory requires it; preserve the effective
  optimization configuration and record the change.
- Do not alter the model, loss, token order, or score definition.
- A failed run is stored with its traceback and is not replaced by a custom
  language model.

## Smoke-test verification

- The included adapter smoke test is diagnostic and is not part of the formal result table.
- Test pool: 128 normal plus 16 injected trajectories, length 32.
- Reduced smoke model: 2 layers, 4 heads, embedding dimension 64, 3 epochs.
- Gate checks: training completed; 144/144 finite scores returned; score
  direction verified; no labels consumed during fitting.
- Diagnostic AUC/AUPRC are explicitly smoke-only and cannot be reported in the
  manuscript.
