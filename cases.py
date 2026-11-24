"""FFT workloads shared by the Python and C++ drivers."""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Case:
    group: str
    kind: str
    shape: tuple
    axes: tuple
    precision: str
    layout: str = "C"
    threads: int = 1
    mode: str = "default"
    norm: str = "backward"
    transform_type: int = 2
    lengths: tuple = ()
    check_only: bool = False

    @property
    def id(self):
        fields = (self.group, self.kind, "x".join(map(str, self.shape)),
                  "a" + "_".join(map(str, self.axes)), self.precision,
                  self.layout, str(self.threads), self.mode, self.norm,
                  str(self.transform_type), "x".join(map(str, self.lengths)))
        return "-".join(fields).rstrip("-")


def cases(suite):
    if suite not in ("fast", "full"):
        raise ValueError(suite)
    full = suite == "full"
    rows = []

    def add(group, shapes, kinds, *, precisions=("f32", "f64"),
            axes=None, layouts=("C",), threads=(1,), modes=("default",),
            norm="backward", types=(2,), lengths=(), check_only=False):
        for shape, kind, p, layout, t, mode, ty in product(
                dict.fromkeys(shapes), kinds, precisions, layouts, threads, modes, types):
            ax = tuple(range(len(shape))) if axes is None else axes
            rows.append(Case(group, kind, shape, ax, p, layout, t, mode,
                             norm, ty, lengths, check_only))

    complex_ops = ("fft", "ifft")
    real_ops = ("rfft", "irfft")
    # Small transforms test special cases, not batched throughput.
    tiny = (1, 2, 3, 7, 8, 16, 31, 32) + ((4, 5, 11, 23, 29, 33, 63, 64, 65) if full else ())
    add("edge", [(n,) for n in tiny], complex_ops + real_ops, check_only=True)
    add("empty", [(0, 128)], complex_ops + real_ops + ("dct", "dst"), axes=(1,), check_only=True)
    add("empty", [(128, 0)], complex_ops + real_ops, axes=(0,), check_only=True)
    add("empty", [(0, 4, 8)], complex_ops + real_ops + ("dct", "dst"), axes=(1, 2), check_only=True)
    add("empty", [(4, 0, 8)], complex_ops + real_ops, axes=(0, 2), check_only=True)
    add("invalid", [(0,)], ("fft", "ifft", "rfft"), check_only=True)
    add("invalid", [(4, 0, 8)], complex_ops + real_ops, axes=(1, 2), check_only=True)
    add("inplace_real", [(8, 127), (8, 256), (16, 17, 31)], ("rfft",), check_only=True)
    add("overhead", [(1,), (8,), (32,), (128,)], ("fft", "rfft"), modes=("default", "out"))

    n1 = (128, 257, 1024, 4095, 4096, 4097)
    if full:
        n1 += (100, 127, 129, 255, 256, 288, 299, 300, 301, 303, 304, 313,
               512, 1000, 1023, 2047, 2048, 4093, 4094, 6144, 8192, 16384)
    add("1d", [(n,) for n in n1], complex_ops + real_ops)

    batches = list(product((8, 64), (1024, 4095, 4096, 8192))) + [(16, 4096)]
    if full:
        batches += list(product((1, 2, 4, 7, 9, 15, 17, 32, 33), (1024, 4096)))
        batches += [(64, n) for n in (256, 257, 1023, 2047, 2048, 4093, 4097, 6144)]
        batches += [(128, 2048), (128, 256), (16, 301)]
    add("batched", batches, ("fft", "rfft", "irfft"), axes=(1,))

    strides = [(64, n) for n in (256, 1024, 4095, 4096)] + [(128, 256)]
    if full:
        strides += [(8, 4096), (16, 4096), (32, 2048), (64, 1023),
                    (64, 2048), (128, 1024), (256, 512), (8, 65536)]
    add("strided", strides, ("fft", "rfft", "irfft"), axes=(1,),
        layouts=("F", "step2", "reverse"))
    add("strided", [(1024, 64)] + [(n, 8) for n in (256, 1024, 4095, 4096)],
        ("fft", "rfft"), axes=(0,))
    if full:
        add("strided", [(128, 256), (256, 512)], ("fft", "rfft"),
            axes=(0,), layouts=("F", "step2", "reverse"))

    nd = [(64, 64), (127, 129), (128, 256), (256, 256),
          (16, 17, 31), (16, 32, 64), (32, 32, 64), (32, 64, 128)]
    if full:
        nd += [(73, 144), (181, 360), (721, 1440), (313, 100), (256, 256),
               (256, 512), (16, 32, 64), (32, 64, 128), (64, 128, 128)]
    add("nd", nd, complex_ops + real_ops)
    add("nd_axes", [(32, 64), (64, 128), (128, 256), (256, 512)],
        complex_ops + real_ops, axes=(1, 0), layouts=("C", "F"))
    add("nd_axes", [(16, 32, 64)], complex_ops + real_ops, axes=(2, 0))

    long = (32768, 65536, 131072, 262144, 1048576)
    if full:
        long += (32768, 65528, 65544, 81920, 92160, 98304, 100000, 100352,
                 131064, 131080, 262144, 262152, 524288)
    add("long", [(n,) for n in long], complex_ops + real_ops)
    add("long_batch", [(8, n) for n in (32768, 65536, 131072, 262144)]
        + [(16, 32768)], ("fft", "rfft"), axes=(1,))
    if full:
        add("long_batch", [(2, 65528), (2, 65536), (7, 65536), (9, 65536),
                           (16, 65536), (32, 65536), (16, 131072),
                           (32, 131072), (16, 262144)], ("fft", "rfft"), axes=(1,))
    add("threads", [(64, 4096)] + [(8, n) for n in (4096, 16384, 65536, 131072)],
        ("fft", "rfft"), axes=(1,), threads=(4,))
    add("threads", [(64, 128, 128), (262144,)], complex_ops + real_ops, threads=(4,))

    add("buffers", [(64, n) for n in (1024, 4095, 4096, 8192)],
        ("fft", "rfft", "irfft"), axes=(1,), modes=("out",))
    add("buffers", [(n,) for n in (128, 1024, 4096, 8192)] + [(64, 4096)],
        complex_ops, axes=(-1,), modes=("inplace",))
    add("buffers", [(128, 256)], ("fft", "rfft"), modes=("out",))
    for norm in ("forward", "ortho"):
        add("normalization", [(257,), (16, 17)], complex_ops + real_ops, norm=norm, check_only=True)
    for n in (512, 1000, 1031, 2048):
        add("resize", [(1024,)], complex_ops + real_ops, lengths=(n,))

    r2r_sizes = (128, 257, 1024, 4096) + ((129, 256, 512, 2048) if full else ())
    add("r2r", [(n,) for n in r2r_sizes], ("dct", "dst"), types=(1, 2, 3, 4))
    add("r2r", [(128, 256)], ("dct", "dst", "separable_fht", "genuine_fht"))
    add("r2r", [(n,) for n in r2r_sizes],
        ("r2r_fftpack", "r2r_fftw", "separable_fht", "genuine_fht"))
    add("convolution", [(256,), (1024,), (4096,), (8192,)], ("conv_real", "conv_complex"))
    add("convolution", [(128, 1024)], ("conv_real", "conv_complex"), axes=(1,))
    # Distinct extended precision on platforms and APIs that support it.
    ld = (256, 1024, 4096, 65536) + ((16384, 262144, 1048576) if full else ())
    add("extended", [(n,) for n in ld], complex_ops + real_ops, precisions=("ld",))

    ids = [c.id for c in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate case IDs")
    return rows
