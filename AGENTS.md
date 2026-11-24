# Repository instructions

Follow [write-like-a-scientist](https://github.com/mwyau/write-like-a-scientist), including its research-software profile. Keep the README short; put test definitions in code.

- Use one set of case IDs for Python and C++ comparisons. Keep the test driver and build scripts readable. Prefer functions and plain data over a benchmark framework.
- Keep correctness and timing distinct. Check shape, dtype, sign, normalization, numerical error and input preservation where the API promises it. Use direct DFTs for small cases and independent reference calculations or analytic inputs for larger cases. A round trip alone is insufficient.
- Use deterministic inputs. Keep allocation, planning, input restoration and validation outside execution-only timing. Label allocating, preallocated, in-place, one-shot and reusable-plan measurements separately.
- Record repository URL, resolved SHA, compiler and flags, CPU model and features, actual selected FFT target when exposed, thread count, dtype, layout, axes, samples and errors.
- Normalize timings to DUCC 0.42.0 v1 within each runner and API. Lower ratios are faster. Aggregate normalized matched-case ratios, not raw timings, across runners; report case and runner counts and separate source revisions and timing modes. Do not treat missing AVX-512 support as a successful v4 test. Build and runtime dispatch settings are different; record both. Use separate processes for dispatch overrides.
- Use x86-64-v1/v3/v4 for DUCC and v2/v3/v4 for the NumPy DUCC FFT port. Keep upstream NumPy PocketFFT separate with its default build settings; verify the FFT backend before timing. Check host and OS ISA support before loading a target-specific binary. Never silently fall back and label the result as a higher ISA.
- Allow independent DUCC source revisions for C++17 and C++23; the latter currently tracks `phiadaarr/ducc:cpp23_fft`. Use identical compiler and ISA flags apart from the language standard. Record the source SHA and standard in each build's metadata. Do not attribute differences between source revisions solely to the C++ standard.
- Fast is a subset of full. Test tiny and empty transforms for correctness; time small transforms separately as API overhead. Use realistic batched lengths and bounded memory for throughput. Cover odd/even, prime, mixed radix, SIMD tails, cache boundaries, strides, axes and multidimensional transforms.
- Unsupported API capabilities get explicit skip reasons. Unexpected exceptions, incorrect results and wrong ISA selections fail the run. Save results even when some cases fail.
- Plot absolute runtime against transformed length on log-log axes per runner. Use package colors and ISA line styles. Keep batch shape, transform, precision, threads and layout fixed within a panel; do not mix raw runtimes across CPUs.
- Keep source checkouts and generated builds/results out of Git. Do not modify candidate repositories. Do not push to candidate forks or upstream repositories.
- Never automatically cancel previous workflow runs. Attach results from each runner and an aggregate report for multiple runners.
- Compile once and distribute portable fixed-ISA wheels and executables; never distribute builder-specific native code to other CPUs. Cache only exact resolved source/toolchain/build inputs. Use a single benchmark runner by default on push. Optional replicated runs should repeat the same comparison on each host rather than distribute competing implementations across hosts.

