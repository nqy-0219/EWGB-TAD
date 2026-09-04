# Execution Environments

The release records two environments because CPU EWGB-TAD jobs and neural GPU
jobs were executed on different hardware. The algorithm, data caches, seeds,
and evaluation code are shared; timing values must be interpreted with the
reported hardware.

## Linux GPU environment

- Python 3.10.8
- Ubuntu/CentOS-compatible x86_64 server environment
- NVIDIA RTX 3090, 24 GiB per timing job
- CUDA 11.8 runtime
- PyTorch 2.6.0+cu118
- NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2
- pandas 2.3.2, matplotlib 3.10.9

`requirements.txt` pins the corresponding public package versions. Install the
CUDA wheel following the PyTorch 2.6.0 CUDA 11.8 instructions when GPU execution
is required. Every raw neural result also stores its requested device, visible
GPU name, CUDA availability, package versions, score hash, and peak memory.

## Windows CPU verification environment

- Python 3.13.11
- Windows 10/11 x86_64
- PyTorch 2.10.0+cpu
- NumPy 2.4.3, SciPy 1.17.1, scikit-learn 1.8.0
- pandas 3.0.1, matplotlib 3.10.9

The 30 EWGB-TAD fit/score timing checks were rerun in this environment because
it is the environment that generated the reference CPU score vectors. All 30
regenerated vectors were elementwise identical to the main results. Cross-OS
floating-point reruns may differ at approximately machine precision and should
be checked with a declared tolerance rather than interpreted as method changes.

## Determinism controls

All runners seed Python, NumPy, and PyTorch; disable cuDNN benchmarking; request
deterministic algorithms; and record the dataset-cache SHA-256. LSTM-AE and
USAD timing reruns use one GPU per job and exactly reproduce their corresponding
main score vectors and metrics.
