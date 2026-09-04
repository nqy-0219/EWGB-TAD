# Baseline protocol index

This directory records the complete protocol for all 12 comparison methods in
the EWGB-TAD main benchmark. The cards distinguish verified official code from
local reproducible implementations and paper-guided local adaptations.

| Baseline group | Methods | Protocol record |
|---|---|---|
| General CPU tabular | IForest, ECOD, iBoost-ODE, CoMadOut | `NON_NEURAL_BASELINE_PROTOCOLS.md` |
| Trajectory-structured CPU | Shape-KNN, SegmentOD, TADS, Profile-TAD | `NON_NEURAL_BASELINE_PROTOCOLS.md` |
| Tuned neural | LSTM-AE, USAD | `LSTM-AE_protocol_card.md`, `USAD_protocol_card.md`, and `DEEP_BASELINE_PROTOCOL_SUMMARY.md` |
| Official trajectory neural | LM-TAD, MST-OATD | `LM-TAD_protocol_card.md` and `MST-OATD_protocol_card.md` |

For every method, the records specify the source and version, input contract,
normalization, scoring or training objective, initialization, parameter or
search space, model-selection rule, stopping rule, seed policy, hardware
record, and failure policy. Machine-readable source hashes are stored in
`../protocol/source_hashes.json`. Dataset-specific LSTM-AE and USAD selections
and candidate-level searches are stored under `../deep_tuning/`.
Exact label-free inner-split arrays and per-split hashes are stored in
`../protocol/unlabeled_inner_splits.npz` and
`../protocol/unlabeled_inner_split_manifest.json`.

All methods are fitted to the complete unlabeled dataset--seed pool or to a
label-free inner split explicitly described in their card. Labels are read only
after scoring to compute AUC, AUPRC, and oracle-top-k F1. No failed method is
replaced by an undisclosed alternative.
