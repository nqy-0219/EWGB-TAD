# Current EWGB-TAD Result Set

This directory is the full result source for the Neurocomputing submission.
The machine-readable lock is `metadata/CURRENT_RESULT_SET.json`.

The lock identifies every `COMPLETE.json`, selected attempt, result
file, score file, dataset cache, summary artifact, and relevant source hash.
Files outside this result tree must not be used to update the manuscript,
response letter, tables, figures, or public release.

The `dataset_cache` directory is an immutable experiment input and must be
retained. Its cached `trajectory_shape` array belongs only to the shared
34-dimensional generic-baseline representation. EWGB-TAD fits the
trajectory-shape PCA from the cached `trajectories` array through
`canonical_shape_view` or the fitted `ThreeViewEWGBDetector` in all main,
component, sensitivity, stability, timing, and figure paths.

Directories named `attempt_*` are included only when their parent
`COMPLETE.json` selects them. Run `src/prune_obsolete_result_attempts.py`
without `--apply` to audit this invariant and with `--apply` to remove only
unreferenced attempts.
