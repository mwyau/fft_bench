"""Reference calculations, input layouts, and error checks."""

import math

import numpy as np
import scipy.fft as sp_fft


class Unsupported(Exception):
    pass


def types(case):
    real = np.dtype({"f32": "float32", "f64": "float64", "ld": np.longdouble}[case.precision])
    if case.precision == "ld" and np.finfo(real).nmant == np.finfo("float64").nmant:
        raise Unsupported("longdouble has no additional precision")
    complex_ = np.dtype({"f32": "complex64", "f64": "complex128", "ld": np.clongdouble}[case.precision])
    return real, complex_


def fft_call(module, kind, x, axes, lengths=(), norm="backward", **kwargs):
    if len(axes) == 1:
        if lengths:
            kwargs["n"] = lengths[0]
        return getattr(module, kind)(x, axis=axes[0], norm=norm, **kwargs)
    if lengths:
        kwargs["s"] = lengths
    return getattr(module, kind + "n")(x, axes=axes, norm=norm, **kwargs)


def direct_fft(x, axis, inverse=False):
    n = x.shape[axis]
    j = np.arange(n, dtype=np.longdouble)
    phase = np.outer(j, j) * (2 * np.arccos(np.longdouble(-1)) / n)
    matrix = np.exp((1j if inverse else -1j) * phase)
    result = np.moveaxis(x, axis, -1) @ matrix
    if inverse:
        result /= n
    return np.moveaxis(result, -1, axis)


def direct_r2r(x, axis, kind, ty):
    n = x.shape[axis]
    j = np.arange(n, dtype=np.longdouble)[None, :]
    pi = np.arccos(np.longdouble(-1))
    data = np.moveaxis(x, axis, -1)
    result = np.empty(data.shape, dtype=np.float64)
    for start in range(0, n, 128):
        k = np.arange(start, min(n, start + 128), dtype=np.longdouble)[:, None]
        if kind == "dct":
            if ty == 1:
                m = 2 * np.cos(pi * k * j / (n - 1))
                m[:, (0, -1)] *= 0.5
            elif ty == 2:
                m = 2 * np.cos(pi * k * (j + 0.5) / n)
            elif ty == 3:
                m = 2 * np.cos(pi * (k + 0.5) * j / n)
                m[:, 0] *= 0.5
            else:
                m = 2 * np.cos(pi * (k + 0.5) * (j + 0.5) / n)
        else:
            if ty == 1:
                m = 2 * np.sin(pi * (k + 1) * (j + 1) / (n + 1))
            elif ty == 2:
                m = 2 * np.sin(pi * (k + 1) * (j + 0.5) / n)
            elif ty == 3:
                m = 2 * np.sin(pi * (k + 0.5) * (j + 1) / n)
                m[:, -1] *= 0.5
            else:
                m = 2 * np.sin(pi * (k + 0.5) * (j + 0.5) / n)
        result[..., start:start + len(k)] = data @ m.astype(np.float64).T
    return np.moveaxis(result, -1, axis)


def make_layout(data, layout):
    if layout in ("C", "F"):
        storage = np.array(data, order=layout, copy=True)
        return storage, storage
    if layout == "step2":
        storage = np.empty((*data.shape[:-1], 2 * data.shape[-1]), dtype=data.dtype)
        storage.fill(0)
        view = storage[..., ::2]
    elif layout == "reverse":
        storage = np.empty(data.shape, dtype=data.dtype)
        view = storage[..., ::-1]
    else:
        raise ValueError(layout)
    view[...] = data
    return view, storage


def prepare(case, candidate):
    real, complex_ = types(case)
    rng = np.random.default_rng(1234)
    kind = case.kind
    axes = tuple(a % len(case.shape) for a in case.axes)
    # Use the same stored values across candidates; compute references at higher
    # precision when the input is single precision.
    x = rng.standard_normal(case.shape).astype(real)
    if kind in ("fft", "ifft", "conv_complex"):
        x = (x + 1j * rng.standard_normal(case.shape)).astype(complex_)
    reference_fft = np.fft if candidate == "scipy" else sp_fft
    high = np.clongdouble if case.precision == "ld" else np.complex128
    high_real = np.longdouble if case.precision == "ld" else np.float64
    lengths = case.lengths or tuple(case.shape[a] for a in axes)
    kernel = None

    if kind == "irfft":
        if case.group == "edge":
            z = direct_fft(x, axes[0])
            x = z[:case.shape[axes[0]] // 2 + 1].astype(complex_)
        else:
            x = fft_call(reference_fft, "rfft", x.astype(high_real), axes).astype(complex_)
    source = x.astype(high if x.dtype.kind == "c" else high_real)
    if kind in ("fft", "ifft", "rfft", "irfft"):
        if case.group == "edge":
            if kind == "irfft":
                n = lengths[0]
                tail = source[1:(n + 1) // 2].conj()[::-1]
                ref = direct_fft(np.concatenate((source, tail)), 0, inverse=True).real
            else:
                ref = direct_fft(source, 0, inverse=kind == "ifft")
                if kind == "rfft":
                    ref = ref[:lengths[0] // 2 + 1]
        else:
            ref = fft_call(reference_fft, kind, source, axes, lengths, case.norm)
        output_dtype = real if kind == "irfft" else complex_
    elif kind in ("dct", "dst"):
        if candidate == "scipy":
            ref = source
            for a in axes:
                ref = direct_r2r(ref, a, kind, case.transform_type)
        else:
            ref = getattr(sp_fft, kind + "n")(source, axes=axes, type=case.transform_type)
        output_dtype = real
    elif kind == "r2r_fftpack":
        from scipy.fftpack import rfft
        ref = rfft(source, axis=axes[0])
        output_dtype = real
    elif kind == "r2r_fftw":
        z = sp_fft.rfft(source)
        n = source.shape[-1]
        ref = np.concatenate((z.real, z.imag[1:(n + 1) // 2][::-1]))
        output_dtype = real
    elif kind == "separable_fht":
        ref = source
        for a in axes:
            z = sp_fft.fft(ref, axis=a)
            ref = z.real - z.imag
        output_dtype = real
    elif kind == "genuine_fht":
        z = sp_fft.fftn(source, axes=axes)
        ref = z.real - z.imag
        output_dtype = real
    elif kind.startswith("conv_"):
        a = axes[0]
        kernel = rng.standard_normal(case.shape[a]).astype(x.dtype)
        if x.dtype.kind == "c":
            kernel += 1j * rng.standard_normal(case.shape[a]).astype(real)
        shape = [1] * x.ndim
        shape[a] = len(kernel)
        z = sp_fft.fft(source, axis=a) * sp_fft.fft(kernel.astype(source.dtype)).reshape(shape)
        ref = sp_fft.ifft(z, axis=a)
        if x.dtype.kind != "c":
            ref = ref.real
        output_dtype = x.dtype
    else:
        raise ValueError(kind)

    view, storage = make_layout(x, case.layout)
    return view, storage, ref, output_dtype, kernel, axes, lengths


def check(got, ref, dtype, case):
    got = np.asarray(got)
    if got.shape != ref.shape or got.dtype != dtype:
        raise AssertionError(f"shape/dtype {got.shape}/{got.dtype}; expected {ref.shape}/{dtype}")
    if not np.all(np.isfinite(got)):
        raise AssertionError("nonfinite output")
    if got.size == 0:
        return 0.0, 0.0
    difference = np.abs(got - ref)
    magnitude = np.abs(ref)
    denom = max(float(np.max(magnitude)), np.finfo(float).tiny)
    max_error = float(np.max(difference)) / denom
    l2_error = float(np.sqrt(np.sum((difference / denom)**2) /
                             np.sum((magnitude / denom)**2))) if np.any(magnitude) else max_error
    eps = np.finfo(types(case)[0]).eps
    # Bounds allow accumulated error but still detect small SIMD-tail failures.
    limit = float(100 * eps * max(1, math.log2(max(2, math.prod(case.shape)))))
    if max(max_error, l2_error) > limit:
        raise AssertionError(f"max={max_error:.3g}, l2={l2_error:.3g}, limit={limit:.3g}")
    return max_error, l2_error
