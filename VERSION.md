# Release Boundary

Version: `EWGB-TAD-Neurocomputing-2026-09-04`

This is the current release for the revised EWGB-TAD experiments. Its defining
algorithmic choice is `sample-adaptive-histogram` local entropy with
`min_samples=8`, `purity_threshold=0.85`, 10 PCA trajectory-shape dimensions,
20-bin score entropy for view fusion, and a base fusion weight of `0.1`.

The result snapshot corresponds to the predefined protocol:

- 390/390 main jobs;
- 570/570 factorial component, view, and local-sensitivity jobs;
- 210/210 extended-sensitivity jobs;
- 120/120 fixed local entropy-bin diagnostic jobs;
- 16/16 representative deep-baseline phase-timing jobs;
- 30/30 EWGB-TAD phase-timing jobs, with exact score agreement;
- all included summary status files report zero missing jobs.

The public package contains the summary snapshot and the sixteen representative
deep-baseline phase-timing job records. Each job's `COMPLETE.json` identifies
the selected successful attempt; unreferenced attempts are excluded. It does not
redistribute generated dataset caches, main-experiment per-job raw outputs, or
private Porto/GeoLife source data. The package-local execution manifest records
this intentionally empty cache boundary; a fresh reproduction populates those
paths before validation.

Experiment branches, manuscript source files, cover letters, and previous
submission directories are outside this release boundary and are not included.
