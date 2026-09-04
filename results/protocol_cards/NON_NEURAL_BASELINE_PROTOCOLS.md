# Non-neural baseline protocols

This card covers the eight non-neural comparison methods in the formal
13-method benchmark. Their configurations are fixed across datasets; no
anomaly labels are used for preprocessing, fitting, parameter selection, or
score calibration.

## Shared execution contract

- Formal runner: `src/phase4_run_job.py`, SHA-256
  `D107A82E20683F954FAA7D572575513512D2C7D65E24BBCF3CA5EC74327F343F`.
- General implementation file: `src/baselines.py`, SHA-256
  `ACEE6C31E95658AEA5E9CC764E1CF561D7412822011BA9BEB4D8D6AFEDA71533`.
- Trajectory implementation file: `src/baselines_v2.py`, SHA-256
  `8277C2C75B7931E898EBDB868581BB8D17707E0A0F248AF836447938EBBDD609`.
- Fit population: the complete unlabeled pool for the current dataset and
  seed.
- Trajectory length: 32 coordinate points.
- Canonical feature input: 34 dimensions comprising 16 spatial-path, 8
  kinematic, and 10 PCA-based trajectory-shape features.
- Formal seeds: 10 for Synthetic and Grid-Network; 5 for Porto-derived and
  GeoLife.
- Score direction: every returned score is oriented so that larger values
  indicate greater anomaly likelihood.
- Model selection and stopping: the fixed settings below are used for all
  datasets. These methods have no label-based selection, validation
  checkpoint, or early stopping.
- Hardware and software: each raw job records the operating system, core
  runtime versions, requested device, elapsed time, and score hash. The full
  public environment specification is in `ENVIRONMENT.md` at the repository
  root.
- Failure policy: preserve the command, seed, environment, traceback, and
  partial artifacts; do not substitute another detector or modify the score
  definition after observing labels.

## IForest

- Source status: scikit-learn `IsolationForest` wrapped by the released local
  implementation.
- Input and normalization: the 34-dimensional canonical feature matrix;
  `StandardScaler` is fitted to the full unlabeled pool.
- Objective and score: isolation-tree fitting; the negative
  `score_samples` output is the anomaly score.
- Initialization: dataset--seed random state.
- Fixed parameters: 200 trees, contamination 0.10, all CPU cores.
- Search space: none.

## ECOD

- Source status: local reproducible implementation of marginal empirical-tail
  scoring.
- Input and normalization: the unscaled 34-dimensional canonical feature
  matrix. No scale-dependent distance is used.
- Objective and score: for each feature, compute left- and right-tail
  empirical probabilities and sum the larger negative log-tail score across
  dimensions.
- Initialization: deterministic; no random state.
- Fixed parameters and search space: no tuned parameter.

## iBoost-ODE

- Source status: local reproducible implementation of the iterative
  random-subspace outlier-ensemble mechanism.
- Input and normalization: standardized 34-dimensional canonical features;
  each round samples 65% of the feature dimensions.
- Objective and score: five rounds update sample influence from rank-aggregated
  isolation, local-neighbor, empirical-tail, and, when computationally
  applicable, one-class evidence. Formal pools contain more than 2,500 samples,
  so the optional one-class term is zero under the released computational
  guard. The final score aggregates the fitted isolation and empirical-tail
  evidence over rounds.
- Initialization: dataset--seed random state; round-specific isolation seeds.
- Fixed parameters: five rounds, contamination 0.10, 80 isolation trees per
  round, maximum isolation subsample 512, LOF neighborhood at most 25.
- Search space: none.

## CoMadOut

- Source status: local reproducible CoMAD-based robust-scatter implementation.
- Input and normalization: the 34-dimensional canonical feature matrix,
  centered by the median and scaled by the median absolute deviation. The
  maximum dimension is 40, so PCA is not activated for the formal 34-feature
  input.
- Objective and score: a shrinkage-stabilized co-median absolute dependency
  matrix defines a robust Mahalanobis term, combined with the maximum robust
  coordinate deviation using rank weights 0.75 and 0.25.
- Initialization: deterministic; PCA random state 0 if the dimension guard is
  activated.
- Fixed parameters: maximum dimension 40, shrinkage 0.08.
- Search space: none.

## Shape-KNN

- Source status: local reproducible coordinate-neighborhood baseline.
- Input and normalization: each length-32 coordinate trajectory is flattened
  to 64 coordinates; no feature standardization is applied.
- Objective and score: mean Euclidean distance to the five nearest other
  trajectories.
- Initialization: deterministic.
- Fixed parameters: `k=5`; search space: none.

## SegmentOD

- Source status: local reproducible segment-based baseline inspired by
  trajectory segment outlier detection.
- Input and normalization: each trajectory is split into four consecutive
  segments. Segment direction in two coordinates, path length, and endpoint
  displacement form 16 features, which are standardized on the full unlabeled
  pool.
- Objective and score: distance to the 10th nearest trajectory in the
  standardized segment-feature space.
- Initialization: deterministic.
- Fixed parameters: four segments and 10 neighbors; search space: none.

## TADS

- Source status: paper-guided local adaptation to the unified length-32
  coordinate protocol; it is not represented as an official repository.
- Input and normalization: trajectories are mapped to a global `18 x 18` grid.
  Symbolic route mismatch, cell and transition rarity, repetition, stay, path,
  and turning statistics form 11 features, which are standardized on the full
  unlabeled pool.
- Objective and score: 18 MiniBatchKMeans route prototypes and a 100-tree
  isolation model are combined with fixed route, rarity, stay, and turning
  evidence weights.
- Initialization: dataset--seed random state; MiniBatchKMeans uses three
  initializations.
- Fixed parameters: grid side 18, 18 prototypes, contamination 0.10; search
  space: none.

## Profile-TAD

- Source status: paper-guided local profile-monitoring adaptation; it is not
  represented as an official repository.
- Input and normalization: start/end route features are standardized before
  route clustering. Each trajectory is then represented by route-normalized
  longitudinal, lateral, speed, relative-heading, and heading-change profiles.
- Objective and score: at most 20 KMeans route clusters define robust
  within-route median/MAD profiles. The anomaly score combines the largest
  standardized deviations and their exceedance rate.
- Initialization: dataset--seed random state; KMeans uses 10 initializations.
- Fixed parameters: maximum 20 route clusters, global robust-profile fallback
  for clusters with fewer than eight members; search space: none.
