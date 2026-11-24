# fft_bench

Numerical tests and timing comparisons for CPU FFT libraries through their Python and C++ APIs.

Python: upstream DUCC and DUCC SIMD fork (x86-64 v1/v3/v4), NumPy with the synchronized DUCC FFT port (v2/v3/v4), upstream NumPy PocketFFT, SciPy. C++: upstream DUCC, DUCC SIMD fork and C++23 FFT (each v1/v3/v4), FFTW, Admiral. Workload groups are defined in `cases.py`.

## Run

Linux x86-64, Python 3.13, Git, GCC, CMake, Ninja and the FFTW development libraries are required. CI uses Ubuntu 26.04 and GCC 14.

```sh
python run.py --suite fast --output results/fast
python run.py --suite full --output results/full
python run.py --only ducc-python,numpy --group batched --output results/batched
python run.py --only ducc-cpp,ducc-cpp23 --output results/ducc-cpp
python summarize.py results/fast
python -m pip install matplotlib
python plot.py results/fast
```

Use `--check-only` to skip timings. Use separate output directories for independent runs.

Edit `sources.json` or pass `--config path/to/sources.json` to choose repositories, branches, tags or commit hashes. Add multiple entries under `ducc` or `numpy` to compare forks. DUCC entries default to `ducc-python` and `ducc-cpp` (C++17); set `"candidates": ["ducc-cpp23"]` to select the C++23 build. The SIMD fork uses [mwyau/ducc:fft-simd-improvements](https://github.com/mwyau/ducc/tree/fft-simd-improvements) at `f80b987`. NumPy v2/v3/v4 use the [synchronized NumPy port](https://github.com/mwyau/numpy/tree/enh/duccfft-simd-improvements) at `b7053b3`. These two revisions are pinned together so the tested FFT kernels remain identical. Upstream [numpy/numpy:main](https://github.com/numpy/numpy/tree/main) supplies `numpy-pocketfft` using default build flags. The default C++23 entry tracks [`phiadaarr/ducc:cpp23_fft`](https://github.com/phiadaarr/ducc/tree/cpp23_fft); each run records the resolved commit SHA. An entry can set `"dispatch": true` to request a multiarch DUCC or dispatch-enabled NumPy build. The default builds use fixed ISA levels. Unsupported host ISA levels are skipped.

## GitHub Actions

Pushes to `main` run the fast suite on one benchmark runner. Runs do not cancel earlier runs. For manual runs select fast/full, the number of benchmark runners (1–40), and optionally a source configuration or correctness-only mode. One build job resolves source revisions, compiles all fixed ISA targets and caches the wheels and portable executables using resolved sources, toolchain and build code. Workers download the same build artifact, install wheels into fresh environments and skip unsupported ISA levels. Admiral uses runtime dispatch in shared builds rather than builder-specific native flags. AVX-512 availability depends on each benchmark runner.

The build job attaches compilation logs and binaries in `fft-build`. Each benchmark runner attaches raw results, runtime logs, metadata, `normalized.csv`, `grouped.csv` and `summary.md`. Multiple runners also produce an `fft-summary` artifact and a combined Actions summary. Tables show test groups by candidate and ISA, per CPU model and across runners. The reference is the resolved upstream `mreineck/ducc:ducc0` fixed-ISA v1 build (`ducc-py-v1` and `ducc-cpp-v1` by default). The regular v1 candidate serves as the reference; custom source configurations retain an upstream v1 reference. The SHA is recorded and reports never pool different reference revisions. Ratios are candidate/reference time (1.000 = reference; lower is faster), computed on matched passing cases within each runner, then geometric-averaged with equal runner weight. Cells show matched case and runner counts. Python allocation and explicit buffers are separate; FFTW and Admiral reusable-plan execution excludes plan creation. Missing or unsupported results are not imputed. See `v4.md` and `v4.csv` for per-length, factorization and worst-case comparisons. Cells marked † include individually passing cases from a failed candidate or reference run; inspect the correctness counts before interpreting them.

The `plots/` directory in each artifact contains absolute median runtime (µs/call) versus FFT length for every timed group, with separate Python and C++ panels. Colors identify packages, line styles identify ISA levels, and panels hold workload attributes fixed. Multidimensional length is the product of transformed dimensions. Cross-runner artifacts retain plots per runner; absolute runtimes are not pooled across hosts. `plots/points.csv` records exact plotted cases.

The first version targets Linux x86-64. See [AGENTS.md](AGENTS.md) for development conventions.
