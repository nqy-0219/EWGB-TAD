# Reproduction Guide

This guide describes the current reproducible release. Run commands from the
repository root. The scripts use top-level imports from `src`, so invoke them
as shown below rather than changing module names.

## 1. Install and validate

```powershell
python -m pip install -r requirements.txt
python -m compileall src tests
python -m unittest discover -s tests -v
python src/run_phase3_sota_smoke.py --baseline all
```

The exact CPU and GPU environments used for the current result snapshot are
listed in `ENVIRONMENT.md`.

Before rerunning a baseline, consult
`results/phase4/protocol_cards/BASELINE_PROTOCOL_INDEX.md`. It links the
complete source, input, normalization, objective, parameter-selection,
stopping, seed, hardware, and failure-handling record for every comparison
method.

The exact label-free training/validation indices used by LSTM-AE, USAD, and
LM-TAD are included in
`results/phase4/protocol/unlabeled_inner_splits.npz`; the accompanying JSON
manifest records the rule, sample counts, applicable methods, and SHA-256 of
every split. Regenerate and verify this record with:

```powershell
python src/build_unlabeled_split_manifest.py --result-root results/phase4
```

The smoke test writes small diagnostic artifacts under
`results/phase3_sota_smoke/`.

## 2. Prepare the predefined dataset caches

The complete benchmark first creates one immutable cache per dataset and seed.
Synthetic and Grid-Network need no external files. Porto-derived and GeoLife
need the files described in `DATA.md`.

```powershell
python src/phase4_prepare_data.py --datasets Synthetic Grid-Network
python src/phase4_prepare_data.py --datasets Porto-derived GeoLife
```

The cache is written under `results/phase4/dataset_cache/`. A cache contains
length-32 trajectories, labels for evaluation, feature arrays, and the
34-dimensional concatenated representation used by generic baselines. The
cached `trajectory_shape` array is not an EWGB-TAD input. EWGB-TAD rebuilds
its trajectory-shape PCA from `trajectories` through `canonical_shape_view`
or its fitted detector on every execution path. The cache metadata stores a
SHA-256 digest.

## 3. Build and run the main benchmark

Build deterministic manifests first:

```powershell
python src/phase4_build_manifest.py --cpu-workers 8 --gpu-workers 1
```

Run a preflight on the predefined Synthetic seed 42 pool:

```powershell
python src/phase4_worker.py --manifest results/phase4/manifests/preflight.json --device cpu --profile preflight
```

Then run the complete CPU and GPU manifests. Use a CUDA device for the GPU
manifest when available; the scripts preserve the requested device and record
the environment in each result.

```powershell
python src/phase4_worker.py --manifest results/phase4/manifests/main_cpu_all.json --device cpu --profile full --keep-going
python src/phase4_worker.py --manifest results/phase4/manifests/main_gpu_all.json --device cuda:0 --profile full --keep-going
```

Aggregate and validate the main results:

```powershell
python src/phase4_summarize_main.py
python src/phase4_summarize_runtime.py
python src/phase4_validate.py --scope main
```

The main protocol has 390 jobs: 13 methods across 30 dataset-seed blocks.

## 4. Run ablation and sensitivity analyses

```powershell
python src/phase4_build_analysis_manifests.py --workers 8
python src/phase4_analysis_worker.py --manifest results/phase4/manifests/analysis_worker_0.json --keep-going
```

Repeat the analysis worker command for every generated
`analysis_worker_*.json`, then summarize and validate:

```powershell
python src/phase4_summarize_analysis.py
python src/phase4_validate.py --scope analysis
```

The current core analysis protocol has 570 jobs: 240 factorial component jobs,
210 view-complementarity jobs, and 120 minimum-ball-size sensitivity jobs. The
separate extended-sensitivity runner adds 210 jobs for the granular-ball
quality threshold and the fixed bin count used only by full-pool score fusion.
The selected local sample-adaptive histogram estimator does not use a kernel
bandwidth.

The local entropy-bin diagnostic is a separate 120-job analysis. It freezes all
other EWGB-TAD settings and evaluates fixed local histogram counts of 4, 8, 16,
and 32, without changing the main protocol:

```powershell
python src/run_local_entropy_bin_sensitivity.py --workers 8
```

The summary is written to
`results/phase4/sensitivity_stability/local_entropy_bins/` when
`EWGB_PHASE4_ROOT` points to the local full-results root.

## 5. Timing and reporting

Deep-baseline phase timing is run separately so data preparation, optimization,
and scoring can be reported consistently for all four deep methods:

```powershell
python src/phase4_run_deep_timing.py --dataset Synthetic --method LSTM-AE --seed 42 --device cuda:0
python src/phase4_run_deep_timing.py --dataset Synthetic --method USAD --seed 42 --device cuda:0
python src/phase4_run_deep_timing.py --dataset Synthetic --method LM-TAD --seed 42 --device cuda:0
python src/phase4_run_deep_timing.py --dataset Synthetic --method MST-OATD --seed 42 --device cuda:0
```

Repeat for `Grid-Network`, `Porto-derived`, and `GeoLife`, then run:

```powershell
python src/phase4_summarize_deep_timing.py
python src/phase4_validate.py --scope all
```

The included snapshot records seed-42 timing for all four deep baselines on all
four datasets. It also records whether rerun scores and metrics match the main
benchmark.

Run the EWGB-TAD fit/score timing audit across all 30 predefined caches:

```powershell
python src/run_ewgb_phase_timing.py --workers 8
```

This audit checks that every regenerated EWGB-TAD score vector is elementwise
identical to the corresponding main-benchmark vector.

The repository snapshot intentionally omits generated caches and the main and
analysis per-job raw directories. It includes the 16 representative deep
phase-timing records described above: four deep baselines across four datasets.
Therefore, run
`phase4_validate.py --scope all` only after the cache and benchmark steps above
have recreated the omitted artifacts.

Before using any included numbers, inspect
`results/phase4/metadata/CURRENT_RESULT_SET.json`. This is the only canonical
result lock; its selected-attempt and SHA-256 records must not be mixed with
another result tree.

## 6. Rebuild the manuscript assets

After all main and analysis results have been regenerated and validated, set
`EWGB_OVERLEAF_ROOT` to the local `overleaf_neurocomputing` directory and run:

```powershell
$env:EWGB_OVERLEAF_ROOT = 'E:\path\to\overleaf_neurocomputing'
python src/build_phase5_manuscript_assets.py
```

This regenerates the LaTeX tables and the released runtime/rank figures from
the current Phase 4 summaries. Compile the manuscript from that directory with
the journal template, for example:

```powershell
cd $env:EWGB_OVERLEAF_ROOT
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

The repository does not include private data, generated caches, or the full
per-job main-result tree. A clean reproduction must therefore complete Steps
2--4 before Step 6 can produce a self-contained manuscript build.

## Evaluation rules

AUC and AUPRC use the injected labels only after fitting and scoring. F1 is an
oracle-top-k diagnostic: the number of predicted anomalies equals the realized
number of injected anomalies for that dataset and seed, and scores are sorted
descending with stable tie handling. This threshold is not used during
training, feature construction, or hyperparameter selection.
