# LSTM-AE protocol card

## Implementation

- Code source: local reproducible implementation in `src/baselines_dl.py`; the code SHA-256 is recorded in `protocol/source_hashes.json` and matches the canonical result set.
- Input and normalization: LSTM-AE receives coordinate sequences with shape `(N, 32, 2)`. A `StandardScaler` is fitted on the 90% inner training partition for the current dataset and seed, then applied to the complete scoring population.
- Objective and score: the model minimizes sequence reconstruction mean squared error. The reconstruction error is used as the anomaly score, with larger values indicating greater anomaly likelihood.
- Initialization: PyTorch default LSTM and Linear initialization after the predefined per-job random seed; no pretrained weights are used.
- Model selection: one tuning run with seed 42 is performed for each dataset without loading anomaly labels. The candidate with minimum inner unlabeled validation reconstruction MSE is selected and then fixed for every formal seed of that dataset.
- Early stopping: maximum 80 epochs, patience 8, minimum 10 epochs, minimum improvement `1e-5`, and restoration of the best validation checkpoint. A run may reach the 80-epoch cap while still restoring an earlier best checkpoint.
- Formal evaluation: 10 seeds for Synthetic and Grid-Network, and 5 seeds for Porto-derived and GeoLife. Test labels are read only after fitting and scoring for AUC, AUPRC, and oracle-top-k F1.
- Hyperparameter search space: 12 explicitly enumerated candidates, covering hidden dimensions 32/64/128, latent dimensions 16/32, 1/2 LSTM layers, dropout 0/0.1, learning rates `3e-4`/`1e-3`, and batch size 128. These are the tested candidate tuples, not an assertion that every Cartesian combination was evaluated.
- Selected configurations: see `deep_tuning/selected_configurations.json`; candidate-level validation records are available under `deep_tuning/*/lstm_ae/candidate_results.csv`.
- Timing evidence: the separate phase-timing summary records preparation, training, scoring, end-to-end time, peak GPU memory, and exact score agreement with the main result for all four datasets at seed 42.
