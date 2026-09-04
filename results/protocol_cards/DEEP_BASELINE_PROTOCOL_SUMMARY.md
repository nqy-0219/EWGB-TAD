# Deep baseline protocol summary

This summary records the deep-baseline protocol used in the reproducibility
materials.

| Item | LSTM-AE | USAD |
|---|---|---|
| Paper reference | Malhotra et al., 2016, LSTM-based encoder-decoder for multi-sensor anomaly detection | Audibert et al., KDD 2020, USAD |
| Code source/version | Local reproducible implementation in `src/baselines_dl.py`, SHA-256 in `protocol/source_hashes.json` | Local reproducible implementation in `src/baselines_dl.py`, SHA-256 in `protocol/source_hashes.json` |
| Input | Coordinate sequence `(N, 32, 2)` | Canonical feature matrix `(N, 34)` |
| Normalization | StandardScaler fit on inner unlabeled training partition, then applied to all samples | Same rule, applied to feature vectors |
| Objective | Sequence reconstruction mean squared error | USAD two-autoencoder reconstruction objective |
| Initialization | PyTorch default LSTM/Linear initialization after deterministic seed | PyTorch default Linear initialization after deterministic seed |
| Search space | 12 candidates: hidden 32/64/128, latent 16/32, layers 1/2, dropout 0/0.1, learning rate 3e-4/1e-3, batch 128 | 12 candidates: hidden 32/64/128, latent 8/16/32, learning rate 3e-4/1e-3, batch 128 |
| Selection | Minimum inner unlabeled validation reconstruction MSE, seed 42 per dataset | Minimum inner unlabeled validation reconstruction score, seed 42 per dataset |
| Early stopping | Max 80 epochs, patience 8, min 10 epochs, min delta 1e-5, restore best weights | Same rule |
| Formal seeds | Synthetic/Grid: 10; Porto-derived/GeoLife: 5 | Same rule |
| Test-label use | Metrics only after fit and score | Metrics only after fit and score |
| Score direction | Higher is more anomalous | Higher is more anomalous |

Candidate-level records are stored under
`deep_tuning/*/{lstm_ae,usad}/candidate_results.csv`. Selected configurations
are stored in `deep_tuning/selected_configurations.json`, and per-seed formal
outputs are stored under `raw/`.

The final directory also contains separate protocol cards for LM-TAD and
MST-OATD. Source hashes and verified external-repository commits are recorded in
`protocol/source_hashes.json`.

Representative phase-resolved timing for all four deep baselines, across all
four datasets at seed 42, is recorded in
`summary/deep_phase_timing_seed42.csv` and summarized in Supplementary Table
S3. The timing record includes preparation, optimization, scoring,
end-to-end time, peak GPU memory, and exact agreement with the corresponding
formal score vector.
