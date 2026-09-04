# Deep-baseline phase timing (representative seed 42)

Data preparation, training, and scoring are measured separately for all four deep baselines under their formal dataset-specific configurations.

| Dataset | Method | Prep (s) | Pretrain (s) | Train (s) | Validation (s) | Optimization (s) | Score (s) | End-to-end (s) | Peak GPU (GiB) | Main score exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic | LSTM-AE | 1.489 | - | 20.650 | - | 20.650 | 0.029 | 22.346 | 1.207 | True |
| Synthetic | USAD | 1.594 | - | 27.719 | - | 27.719 | 0.006 | 29.561 | 0.073 | True |
| Synthetic | LM-TAD | 2.500 | - | 571.184 | 16.366 | 589.240 | 0.953 | 592.962 | 1.163 | True |
| Synthetic | MST-OATD | 0.819 | 10.423 | 20.078 | - | 30.501 | 3.355 | 36.464 | 6.487 | True |
| Grid-Network | LSTM-AE | 1.679 | - | 12.086 | - | 12.086 | 0.030 | 14.017 | 2.318 | True |
| Grid-Network | USAD | 1.414 | - | 28.986 | - | 28.986 | 0.006 | 30.568 | 0.073 | True |
| Grid-Network | LM-TAD | 2.326 | - | 856.656 | 19.838 | 879.827 | 0.881 | 883.232 | 1.163 | True |
| Grid-Network | MST-OATD | 1.063 | 10.294 | 19.344 | - | 29.637 | 3.406 | 35.646 | 6.487 | True |
| Porto-derived | LSTM-AE | 1.594 | - | 15.150 | - | 15.150 | 0.030 | 17.034 | 1.195 | True |
| Porto-derived | USAD | 1.430 | - | 28.908 | - | 28.908 | 0.007 | 30.647 | 0.073 | True |
| Porto-derived | LM-TAD | 2.887 | - | 883.881 | 19.842 | 905.744 | 0.881 | 909.858 | 1.163 | True |
| Porto-derived | MST-OATD | 0.852 | 10.783 | 19.939 | - | 30.722 | 3.568 | 36.824 | 6.487 | True |
| GeoLife | LSTM-AE | 1.346 | - | 10.794 | - | 10.794 | 0.031 | 12.341 | 1.421 | True |
| GeoLife | USAD | 1.361 | - | 17.781 | - | 17.781 | 0.007 | 19.432 | 0.069 | True |
| GeoLife | LM-TAD | 2.549 | - | 510.356 | 12.909 | 526.085 | 0.578 | 529.479 | 1.165 | True |
| GeoLife | MST-OATD | 0.628 | 7.594 | 14.312 | - | 21.906 | 2.351 | 26.592 | 6.487 | True |
