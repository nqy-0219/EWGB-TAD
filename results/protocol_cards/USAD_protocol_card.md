# USAD protocol card

## Implementation

- Code source: local reproducible implementation in `src/baselines_dl.py`; the code SHA-256 is recorded in `protocol/source_hashes.json` and matches the canonical result set.
- Input and normalization: USAD receives the 34-dimensional canonical trajectory feature vector: 16 spatial-path, 8 kinematic, and 10 trajectory-shape dimensions. A `StandardScaler` is fitted on the 90% inner training partition for the current dataset and seed, then applied to the complete scoring population.
- Objective and score: the two-autoencoder USAD reconstruction objective uses both reconstruction paths. The resulting reconstruction score is oriented so that larger values indicate greater anomaly likelihood.
- Initialization: PyTorch default Linear initialization after the predefined per-job random seed; no pretrained weights are used.
- Model selection: one tuning run with seed 42 is performed for each dataset without loading anomaly labels. The candidate with minimum inner unlabeled validation reconstruction score is selected and then fixed for every formal seed of that dataset.
- Early stopping: maximum 80 epochs, patience 8, minimum 10 epochs, minimum improvement `1e-5`, and restoration of the best validation checkpoint. The current selected runs reach the 80-epoch cap; the checkpoint rule remains part of the protocol.
- Formal evaluation: 10 seeds for Synthetic and Grid-Network, and 5 seeds for Porto-derived and GeoLife. Test labels are read only after fitting and scoring for AUC, AUPRC, and oracle-top-k F1.
- Hyperparameter search space: 12 explicitly enumerated candidates, covering hidden dimensions 32/64/128, latent dimensions 8/16/32, learning rates `3e-4`/`1e-3`, and batch size 128. These are the tested candidate tuples, not an assertion that every Cartesian combination was evaluated.
- Selected configurations: see `deep_tuning/selected_configurations.json`; candidate-level validation records are available under `deep_tuning/*/usad/candidate_results.csv`.
- Timing evidence: the separate phase-timing summary records preparation, training, scoring, end-to-end time, peak GPU memory, and exact score agreement with the main result for all four datasets at seed 42.
