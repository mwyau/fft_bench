"""Run FFT correctness tests and benchmarks against one installed candidate."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict

# Set before importing numerical libraries.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from cases import cases
from reference import Unsupported, check, fft_call, prepare, types


def capability(case, candidate):
    kind = case.kind
    ordinary = kind in ("fft", "ifft", "rfft", "irfft")
    if candidate == "numpy":
        if not ordinary:
            raise Unsupported("NumPy has no corresponding FFT API")
        if case.threads != 1:
            raise Unsupported("NumPy FFT has no thread-count argument")
    if candidate == "scipy":
        if not ordinary and kind not in ("dct", "dst"):
            raise Unsupported("SciPy FFT has no corresponding transform API")
        if case.mode in ("out", "inplace"):
            raise Unsupported("SciPy FFT has no explicit output-buffer API")
    if candidate == "ducc0" and case.lengths:
        raise Unsupported("DUCC requires explicit resizing before the FFT")
    if case.mode == "inplace" and kind not in ("fft", "ifft"):
        raise Unsupported("in-place layout not defined for this transform")


def python_call(case, candidate, x, ref, dtype, kernel, axes, lengths):
    capability(case, candidate)
    kind = case.kind
    out = x if case.mode == "inplace" else np.empty(ref.shape, dtype=dtype)
    explicit = case.mode in ("out", "inplace")

    def bind(fn):
        fn.expected_output = out if explicit or kind.startswith("conv_") else None
        return fn

    if candidate in ("numpy", "scipy"):
        if candidate == "numpy":
            module = np.fft
            kwargs = {"out": out} if explicit else {}
        else:
            import scipy.fft as module
            kwargs = {"workers": case.threads}
        if kind in ("dct", "dst"):
            fn = getattr(module, kind + "n")
            return bind(lambda: fn(x, axes=axes, type=case.transform_type, **kwargs))
        return bind(lambda: fft_call(module, kind, x, axes, lengths, case.norm, **kwargs))

    import ducc0
    fft = ducc0.fft
    inverse = kind in ("ifft", "irfft")
    inorm = {"backward": 2 if inverse else 0, "forward": 0 if inverse else 2,
             "ortho": 1}[case.norm]
    kwargs = dict(axes=axes, nthreads=case.threads, inorm=inorm)
    if explicit:
        kwargs["out"] = out
    if kind in ("fft", "ifft"):
        return bind(lambda: fft.c2c(x, forward=not inverse, **kwargs))
    if kind == "rfft":
        return bind(lambda: fft.r2c(x, forward=True, **kwargs))
    if kind == "irfft":
        return bind(lambda: fft.c2r(x, lastsize=lengths[-1], forward=False, **kwargs))
    if kind in ("dct", "dst"):
        return bind(lambda: getattr(fft, kind)(x, type=case.transform_type, **kwargs))
    if kind == "r2r_fftpack":
        return bind(lambda: fft.r2r_fftpack(x, real2hermitian=True, forward=True, **kwargs))
    if kind == "r2r_fftw":
        return bind(lambda: fft.r2r_fftw(x, forward=True, **kwargs))
    if kind.startswith("conv_"):
        return bind(lambda: fft.convolve_axis(x, out, axes[0], kernel, nthreads=case.threads))
    return bind(lambda: getattr(fft, kind)(x, **kwargs))


def timings(fn, reset, samples, target_ms, check_only):
    if check_only:
        return []
    for _ in range(2):
        if reset:
            reset()
        fn()
    if reset:
        count = 1
    else:
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        count = min(512, max(1, round(target_ms * 1e6 / max(elapsed, 1))))
    result = []
    for _ in range(samples):
        if reset:
            reset()
        start = time.perf_counter_ns()
        for _ in range(count):
            fn()
        result.append((time.perf_counter_ns() - start) / count)
    return result


def native_call(case, executable, x, storage, ref, dtype, kernel, axes, lengths,
                samples, target_ms, check_only, planning):
    if case.lengths:
        raise Unsupported("native API requires explicit resizing")
    if case.mode in ("out", "alloc"):
        raise Unsupported("Python output allocation comparison")
    if case.kind.startswith("conv_") or case.kind == "r2r_fftpack":
        raise Unsupported("native adapter does not yet implement this transform")
    def csv(values):
        return ",".join(map(str, values))
    inverse = case.kind in ("ifft", "irfft")
    scale = {"backward": "inverse" if inverse else "one",
             "forward": "one" if inverse else "inverse",
             "ortho": "ortho"}[case.norm]
    with tempfile.TemporaryDirectory() as work:
        infile, outfile = Path(work) / "input.bin", Path(work) / "output.bin"
        storage.ravel(order="K").tofile(infile)
        offset = (x.ctypes.data - storage.ctypes.data) // x.itemsize
        cmd = [str(Path(executable).resolve()), case.kind, case.precision,
               csv(x.shape), csv(ref.shape), csv(axes),
               csv(s // x.itemsize for s in x.strides), str(offset),
               str(infile), str(outfile), str(case.threads), case.mode,
               scale, str(case.transform_type), str(0 if check_only else samples),
               str(target_ms), planning]
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
        if completed.returncode == 77:
            raise Unsupported(completed.stderr.strip() or "native API capability")
        if completed.returncode:
            raise RuntimeError(f"native exit {completed.returncode}: {completed.stderr}")
        report = json.loads(completed.stdout)
        got = np.fromfile(outfile, dtype=dtype).reshape(ref.shape)
    return got, report


def invalid_case(case, candidate):
    if candidate == "native":
        raise Unsupported("native invalid-input contract not implemented")
    real, complex_ = types(case)
    axes = case.axes
    kind = case.kind
    shape = list(case.shape)
    if kind == "irfft":
        shape[axes[-1]] = shape[axes[-1]] // 2 + 1
    x = np.empty(shape, dtype=real if kind == "rfft" else complex_)
    lengths = tuple(case.shape[a] for a in axes)
    try:
        if candidate == "ducc0":
            import ducc0
            f = ducc0.fft
            if kind in ("fft", "ifft"):
                f.c2c(x, axes=axes, forward=kind == "fft")
            elif kind == "rfft":
                f.r2c(x, axes=axes)
            else:
                f.c2r(x, axes=axes, lastsize=lengths[-1], forward=False)
        else:
            if candidate == "numpy":
                module = np.fft
            else:
                import scipy.fft as module
            fft_call(module, kind, x, axes, lengths)
    except (ValueError, RuntimeError) as exc:
        return dict(expected_exception=type(exc).__name__, message=str(exc))
    raise AssertionError("accepted a zero-length transform axis")


def inplace_real(case, candidate):
    if candidate != "ducc0":
        raise Unsupported("DUCC Python padded real in-place API")
    import ducc0
    import scipy.fft as sp_fft
    real, complex_ = types(case)
    n = case.shape[-1]
    buf = np.empty((*case.shape[:-1], n // 2 + 1), dtype=complex_)
    view = buf.view(real)[..., :n]
    source = np.random.default_rng(1234).standard_normal(case.shape).astype(real)
    view[:] = source
    ducc0.fft.r2c(view, axes=case.axes, out=buf, nthreads=1)
    e1 = check(buf, sp_fft.rfftn(source.astype(np.float64), axes=case.axes), complex_, case)
    ducc0.fft.c2r(buf, axes=case.axes, out=view, lastsize=n, forward=False, inorm=2, nthreads=1)
    e2 = check(view, source, real, case)
    return dict(max_error=max(e1[0], e2[0]), l2_error=max(e1[1], e2[1]))


def run_case(case, args):
    if case.group == "invalid":
        return invalid_case(case, args.candidate)
    if case.group == "inplace_real":
        return inplace_real(case, args.candidate)
    if case.threads > len(os.sched_getaffinity(0)):
        raise Unsupported("requested threads exceed available logical CPUs")
    if args.candidate != "native":
        capability(case, args.candidate)
    x, storage, ref, dtype, kernel, axes, lengths = prepare(case, args.candidate)
    check_only = args.check_only or case.check_only
    if args.candidate == "native":
        got, report = native_call(case, args.executable, x, storage, ref, dtype,
                                  kernel, axes, lengths, args.samples, args.target_ms,
                                  check_only, args.planning)
        errors = check(got, ref, dtype, case)
        return dict(report, max_error=errors[0], l2_error=errors[1], output_dtype=str(dtype))

    fn = python_call(case, args.candidate, x, ref, dtype, kernel, axes, lengths)
    original = x.copy()
    got = fn()
    if fn.expected_output is not None and got is not fn.expected_output:
        raise AssertionError("explicit output buffer was not returned")
    errors = check(got, ref, dtype, case)
    if case.mode == "inplace":
        if not np.shares_memory(x, got):
            raise AssertionError("expected in-place output")
    elif not np.array_equal(x, original):
        raise AssertionError("input modified")
    reset = (lambda: np.copyto(x, original)) if case.mode == "inplace" else None
    samples = timings(fn, reset, args.samples, args.target_ms, check_only)
    mode = case.mode if case.mode != "default" else ("out" if case.kind.startswith("conv_") else "alloc")
    return dict(max_error=errors[0], l2_error=errors[1], output_dtype=str(dtype),
                samples_ns=samples, timing_mode=mode)


def environment(candidate):
    data = dict(cpu=platform.processor(), architecture=platform.machine(),
                python=sys.version, numpy=np.__version__, numpy_file=np.__file__,
                affinity=sorted(os.sched_getaffinity(0)),
                longdouble_bits=np.finfo(np.longdouble).nmant + 1)
    if Path("/proc/cpuinfo").exists():
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        for key in ("model name", "flags", "microcode"):
            data[key] = next((l.split(":", 1)[1].strip() for l in lines if l.startswith(key)), None)
    if candidate == "numpy":
        try:
            from numpy.fft import _duccfft_umath as m
            data["fft_backend"] = "duccfft"
            data["fft_target"] = getattr(m, "__duccfft_cpu_dispatch_target__", None)
            data["fft_targets"] = getattr(m, "__duccfft_cpu_dispatch_targets__", None)
        except ImportError:
            from numpy.fft import _pocketfft_umath  # verify the actual extension exists
            data["fft_backend"] = "pocketfft"
            data["fft_target"] = "PocketFFT"
        from numpy._core import _multiarray_umath as m
        data["numpy_cpu_baseline"] = list(m.__cpu_baseline__)
        data["numpy_cpu_dispatch"] = list(m.__cpu_dispatch__)
    if candidate == "ducc0":
        import ducc0
        data.update(ducc_version=ducc0.__version__, ducc_file=ducc0.__file__,
                    ducc_wrapper=getattr(ducc0, "__wrapper__", None))
        if hasattr(ducc0.misc, "psabi_level"):
            data["psabi_level"] = ducc0.misc.psabi_level()
    if candidate == "scipy":
        import scipy
        data.update(scipy_version=scipy.__version__, scipy_file=scipy.__file__)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("fast", "full"), default="fast")
    parser.add_argument("--candidate", choices=("numpy", "scipy", "ducc0", "native"), required=True)
    parser.add_argument("--executable")
    parser.add_argument("--planning", choices=("estimate", "measure"), default="estimate")
    parser.add_argument("--label")
    parser.add_argument("--group", action="append", help="repeat to select several workload groups")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=3)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if args.candidate == "native" and not args.executable:
        parser.error("--executable required for native candidate")
    if args.samples < 1 or args.target_ms <= 0:
        parser.error("positive samples and target-ms required")
    rows = [c for c in cases(args.suite) if not args.group or c.group in args.group]
    if not rows:
        parser.error("no selected workloads")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.metadata.read_text()) if args.metadata else {}
    meta.update(environment(args.candidate))
    meta.update(candidate=args.candidate, label=args.label or args.candidate, suite=args.suite)
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    expected_backend = meta.get("expected_fft_backend")
    if expected_backend and meta.get("fft_backend") != expected_backend:
        raise RuntimeError(f"Expected FFT backend {expected_backend}, got {meta.get('fft_backend')}")
    if meta.get("build_mode") == "dispatch":
        level = meta["requested_level"]
        if args.candidate == "ducc0" and meta.get("psabi_level") != level:
            raise RuntimeError(f"DUCC dispatch selected {meta.get('psabi_level')}, expected {level}")
        if args.candidate == "numpy":
            target = f"X86_V{level}"
            if meta.get("fft_target") not in (target, f"baseline({target})"):
                raise RuntimeError(f"NumPy FFT selected {meta.get('fft_target')}, expected {target}")
    counts = {"pass": 0, "skip": 0, "fail": 0}
    with args.output.open("w") as stream:
        for case in rows:
            row = dict(asdict(case), case_id=case.id, label=meta["label"])
            try:
                row.update(run_case(case, args), status="pass")
                samples = row.get("samples_ns", [])
                if samples:
                    row["median_ns"] = statistics.median(samples)
            except Unsupported as exc:
                row.update(status="skip", reason=str(exc))
            except Exception as exc:
                row.update(status="fail", reason=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            counts[row["status"]] += 1
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    print(meta["label"], counts)
    return int(counts["fail"] > 0 or counts["pass"] == 0)


if __name__ == "__main__":
    sys.exit(main())
