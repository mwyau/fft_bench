# fft_bench

Numerical tests and timing comparisons for CPU FFT libraries through their Python and C++ APIs.

Python: DUCC (x86-64 v1/v3/v4), NumPy (v2/v3/v4), SciPy. C++: DUCC (C++17 and C++23, each v1/v3/v4), FFTW, Admiral. Workload groups are defined in `cases.py`.

## Run

Linux x86-64, Python 3.13, Git, GCC, CMake, Ninja and the FFTW development libraries are required. CI uses Ubuntu 26.04 and GCC 14.

```sh
python run.py --suite fast --output results/fast
python run.py --suite full --output results/full
python run.py --only ducc-python,numpy --group batched --output results/batched
python run.py --only ducc-cpp,ducc-cpp23 --output results/ducc-cpp
python summarize.py results/fast
```

Use `--check-only` to skip timings. Use separate output directories for independent runs.

Edit `sources.json` or pass `--config path/to/sources.json` to choose repositories, branches, tags or commit hashes. Add multiple entries under `ducc` or `numpy` to compare forks. An entry can set `"dispatch": true` to request a multiarch DUCC or dispatch-enabled NumPy build. The default builds use fixed ISA levels. Unsupported host ISA levels are skipped.

## GitHub Actions

Pushes to `main` run the fast suite on one runner. For manual runs, select fast or full and set the replica array (e.g. `[1,2,3,4]`). Each runner builds and tests all selected candidates independently. AVX-512 availability depends on the assigned runner.

The results artifact contains per-case JSONL files, build logs, CPU and build information, and `grouped.csv` with matched case comparisons. Compare timings on the same runner. Python allocation and explicit output-buffer timings are separate. FFTW and Admiral plan execution excludes plan creation; DUCC native calls use the FFT API directly. Unsupported cases have explicit skip reasons.

The first version targets Linux x86-64. See [AGENTS.md](AGENTS.md) for development conventions.
