"""Build pinned candidates and compare them on one Linux x86-64 host."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
KINDS = ("ducc-python", "ducc-cpp", "ducc-cpp23", "numpy", "numpy-pocketfft", "scipy", "fftw", "admiral")
DUCC_CPP_STANDARDS = {"ducc-cpp": 17, "ducc-cpp23": 23}
# ducc0_0_42_0, pinned independently of the candidate configuration.
REFERENCE = dict(name="ducc-0-42", repo="https://github.com/mreineck/ducc.git",
                 ref="42ce8ab5c5d79e0e2fd9af4a09c2ea4693520e1b")
PY_REFERENCE = "ducc-0-42-v1"
CPP_REFERENCE = "ducc-0-42-cpp-v1"


def command(args, log, *, env=None, cwd=None):
    args = [str(a) for a in args]
    print(shlex.join(args), flush=True)
    with log.open("a") as stream:
        stream.write("\n$ " + shlex.join(args) + "\n")
        stream.flush()
        subprocess.run(args, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=3600)


def capture(*args):
    return subprocess.check_output(args, text=True).strip()


def cpu_levels():
    if platform.machine() != "x86_64":
        raise RuntimeError("fixed x86-64 profiles require a Linux x86-64 host")
    # glibc's hwcaps probe includes OS support for the extended register state.
    text = capture("/lib64/ld-linux-x86-64.so.2", "--help")
    return [1] + [int(n) for n in re.findall(r"x86-64-v([234]) \(supported", text)]


def arch(level):
    return "x86-64" if level == 1 else f"x86-64-v{level}"


def checkout(spec, dest, log):
    dest.mkdir()
    command(["git", "init", dest], log)
    command(["git", "-C", dest, "remote", "add", "origin", spec["repo"]], log)
    command(["git", "-C", dest, "fetch", "--depth=1", "origin", spec["ref"]], log)
    command(["git", "-C", dest, "checkout", "--detach", "FETCH_HEAD"], log)
    command(["git", "-C", dest, "submodule", "update", "--init", "--recursive", "--depth=1"], log)
    return capture("git", "-C", str(dest), "rev-parse", "HEAD")


def venv(path, config, log):
    command([sys.executable, "-m", "venv", path], log)
    py = path / "bin/python"
    command([py, "-m", "pip", "install", config["reference_numpy"], config["scipy"]], log)
    return py


def build_python(kind, spec, src, level, work, config, log, jobs):
    py = venv(work / "venv", config, log)
    wheel_dir = work / "wheel"
    wheel_dir.mkdir()
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(jobs))
    env.pop("DUCC0_MAX_PSABI_LEVEL", None)
    env.pop("NPY_DISABLE_CPU_FEATURES", None)
    env.pop("DUCC0_USE_NANOBIND", None)
    env.pop("DUCC0_CFLAGS", None)
    env.pop("DUCC0_FLAGS", None)
    env.pop("DUCC0_LFLAGS", None)
    args = []
    if kind == "ducc":
        env["DUCC0_OPTIMIZATION"] = "multiarch-strip" if spec.get("dispatch") else "native-strip"
        env.pop("CFLAGS", None)
        env.pop("CXXFLAGS", None)
        args = [f"-Ccmake.define.DUCC0_ARCH_FLAGS=-march={arch(level)}",
                "-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-Cbuild-dir={work / 'build'}"]
    elif level is None:
        # Upstream PocketFFT control: default NumPy compiler/dispatch settings.
        env.pop("CFLAGS", None)
        env.pop("CXXFLAGS", None)
        args = ["-Csetup-args=-Dallow-noblas=true", f"-Ccompile-args=-j{jobs}",
                f"-Cbuild-dir={work / 'build'}"]
    else:
        env.update(CFLAGS=f"-march={arch(level)}", CXXFLAGS=f"-march={arch(level)}")
        dispatch = "max" if spec.get("dispatch") else "none"
        args = [f"-Csetup-args=-Dcpu-baseline=X86_V{level}",
                f"-Csetup-args=-Dcpu-dispatch={dispatch}",
                "-Csetup-args=-Dallow-noblas=true", f"-Ccompile-args=-j{jobs}",
                f"-Cbuild-dir={work / 'build'}"]
    command([py, "-m", "pip", "wheel", "-v", "--no-cache-dir", "--no-deps",
             "-w", wheel_dir, *args, src], log, env=env)
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected one source-built wheel")
    command([py, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheels[0]], log)
    for i, file in enumerate((work / "build").rglob("compile_commands.json")):
        shutil.copyfile(file, log.with_suffix(f".compile-{i}.json"))
    return py, wheels[0].name


def build_native(kind, src, target, work, log, jobs, *, cxx_standard=17):
    args = [f"-DBENCH_BACKEND={kind}", f"-DBENCH_ARCH={target}"]
    if kind == "ducc":
        args += [f"-DDUCC_SOURCE={src}", f"-DDUCC_CXX_STANDARD={cxx_standard}"]
    elif kind == "admiral":
        args.append(f"-DADMIRAL_SOURCE={src}")
    command(["cmake", "-S", ROOT / "cpp", "-B", work, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", *args], log)
    command(["cmake", "--build", work, "--parallel", jobs, "--verbose"], log)
    shutil.copyfile(work / "compile_commands.json", log.with_suffix(".compile.json"))
    return work / "fft_native"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("fast", "full"), default="fast")
    p.add_argument("--config", type=Path, default=ROOT / "sources.json")
    p.add_argument("--only", default=",".join(KINDS), help="comma-separated candidate kinds")
    p.add_argument("--group", action="append")
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--work", type=Path, default=ROOT / "work")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--planning", choices=("estimate", "measure"), default="estimate")
    args = p.parse_args()
    selected = set(args.only.split(","))
    if selected - set(KINDS) or args.jobs < 1:
        p.error("invalid --only or --jobs")
    config = json.loads(args.config.read_text())
    names = [s["name"] for k in ("ducc", "numpy", "admiral") for s in config.get(k, [])]
    if len(names) != len(set(names)) or set(names) & {"scipy", "fftw", "reference", REFERENCE["name"]}:
        p.error("source names must be distinct and not reserved")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        p.error("--output must be an empty directory")
    out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work.resolve()))
    levels = cpu_levels()
    compiler = os.environ.get("CXX", "c++")
    host = dict(usable_levels=sorted(levels), compiler=capture(compiler, "--version"),
                cpuinfo=Path("/proc/cpuinfo").read_text().split("\n\n")[0],
                kernel=platform.platform(), runner=os.environ.get("RUNNER_NAME", "local"))
    (out / "host.json").write_text(json.dumps(host, indent=2) + "\n")
    (out / "sources.json").write_text(json.dumps(config, indent=2) + "\n")
    runs = []
    (out / "runs.json").write_text("[]\n")

    def record(label, status, reason=""):
        runs.append(dict(label=label, status=status, reason=reason))
        (out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(label, status, reason, flush=True)

    def execute(label, py, candidate, meta, *, executable=None, env=None):
        log = out / (label + ".log")
        meta_path = out / (label + ".build.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        cmd = [py, ROOT / "bench.py", "--candidate", candidate, "--suite", args.suite,
               "--label", label, "--output", out / (label + ".jsonl"),
               "--metadata", meta_path, "--samples", 5 if args.suite == "fast" else 9]
        if args.check_only:
            cmd.append("--check-only")
        for group in args.group or []:
            cmd += ["--group", group]
        if executable:
            cmd += ["--executable", executable, "--planning",
                    "estimate" if candidate == "native" and meta.get("backend") == "ducc"
                    else args.planning]
        command(cmd, log, env=env, cwd=work)

    def attempt(label, fn):
        try:
            fn()
            record(label, "pass")
        except Exception as exc:
            record(label, "fail", str(exc))

    base = venv(work / "reference", config, out / "reference.log")
    # References are mandatory for the selected APIs, even with --only.
    need_python = bool(selected & {"ducc-python", "numpy", "numpy-pocketfft", "scipy"})
    need_cpp = bool(selected & {"ducc-cpp", "ducc-cpp23", "fftw", "admiral"})
    ref_src = work / "ducc-0-42"
    ref_sha = None
    if need_python or need_cpp:
        try:
            ref_sha = checkout(REFERENCE, ref_src, out / "ducc-0-42-checkout.log")
        except Exception as exc:
            record("ducc-0-42-checkout", "fail", str(exc))

    ref_meta = dict(repo=REFERENCE["repo"], ref=REFERENCE["ref"], sha=ref_sha,
                    backend="ducc", architecture="x86-64", requested_level=1,
                    build_mode="fixed", reference=True, release="0.42.0")
    if need_python and ref_sha:
        def python_reference():
            folder = work / PY_REFERENCE
            folder.mkdir()
            py, wheel = build_python("ducc", REFERENCE, ref_src, 1, folder, config,
                                     out / (PY_REFERENCE + ".log"), args.jobs)
            execute(PY_REFERENCE, py, "ducc0", dict(ref_meta, wheel=wheel))
        attempt(PY_REFERENCE, python_reference)
    if need_cpp and ref_sha:
        def cpp_reference():
            exe = build_native("ducc", ref_src, "x86-64", work / CPP_REFERENCE,
                               out / (CPP_REFERENCE + ".log"), args.jobs)
            execute(CPP_REFERENCE, base, "native", dict(ref_meta, cxx_standard=17),
                    executable=exe)
        attempt(CPP_REFERENCE, cpp_reference)

    if "scipy" in selected:
        attempt("scipy", lambda: execute("scipy", base, "scipy",
                                        dict(backend="scipy", package=config["scipy"])))

    if "fftw" in selected:
        def fftw():
            exe = build_native("fftw", None, "x86-64", work / "fftw", out / "fftw.log", args.jobs)
            version = capture("dpkg-query", "-W", "libfftw3-dev")
            execute("fftw", base, "native", dict(backend="fftw", package=version,
                    architecture="system-library"), executable=exe)
        attempt("fftw", fftw)

    for kind in ("ducc", "numpy", "admiral"):
        relevant = {"ducc": {"ducc-python", *DUCC_CPP_STANDARDS}, "numpy": {"numpy", "numpy-pocketfft"},
                    "admiral": {"admiral"}}[kind]
        if not relevant & selected:
            continue
        for spec in config.get(kind, []):
            defaults = ("ducc-python", "ducc-cpp") if kind == "ducc" else (kind,)
            candidates = set(spec.get("candidates", defaults))
            if not candidates or candidates - relevant:
                p.error(f"invalid candidates for {kind}: {sorted(candidates)}")
            enabled = candidates & selected
            if not enabled:
                continue
            name = spec["name"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("source name must contain only letters, numbers, _ or -")
            src = work / (kind + "-" + name)
            try:
                sha = checkout(spec, src, out / (name + "-checkout.log"))
            except Exception as exc:
                record(name + "-checkout", "fail", str(exc))
                continue
            origin = dict(repo=spec["repo"], ref=spec["ref"], sha=sha, backend=kind)
            if kind == "admiral":
                def admiral():
                    exe = build_native(kind, src, "native", work / (name + "-cpp"),
                                       out / (name + ".log"), args.jobs)
                    execute(name, base, "native", dict(origin, architecture="native"), executable=exe)
                attempt(name, admiral)
                continue
            if kind == "numpy" and "numpy-pocketfft" in enabled:
                def pocketfft():
                    label = name if "numpy" not in candidates else name + "-pocketfft"
                    folder = work / label
                    folder.mkdir()
                    py, wheel = build_python(kind, spec, src, None, folder, config,
                                             out / (label + ".log"), args.jobs)
                    execute(label, py, "numpy", dict(origin, architecture="upstream-default",
                            build_mode="upstream-default", wheel=wheel,
                            expected_fft_backend="pocketfft"))
                label = name if "numpy" not in candidates else name + "-pocketfft"
                attempt(label, pocketfft)
                if "numpy" not in enabled:
                    continue
            targets = (1, 3, 4) if kind == "ducc" else (2, 3, 4)
            py_selected = "ducc-python" in enabled if kind == "ducc" else "numpy" in enabled
            dispatch_build = spec.get("dispatch", False)
            dispatch_py = None
            dispatch_wheel = None
            if py_selected and dispatch_build:
                label = name + "-dispatch-build"
                try:
                    folder = work / label
                    folder.mkdir()
                    dispatch_py, dispatch_wheel = build_python(kind, spec, src, targets[0], folder,
                        config, out / (label + ".log"), args.jobs)
                except Exception as exc:
                    record(label, "fail", str(exc))
            for level in targets:
                prefix = name + ("-py" if kind == "ducc" else "") + f"-v{level}"
                if py_selected:
                    if level not in levels:
                        record(prefix, "skip", "host/OS does not support requested ISA")
                    elif not dispatch_build or dispatch_py:
                        def python_profile():
                            label = prefix
                            log = out / (label + ".log")
                            if dispatch_build:
                                py, wheel = dispatch_py, dispatch_wheel
                            else:
                                folder = work / label
                                folder.mkdir()
                                py, wheel = build_python(kind, spec, src, level, folder, config, log, args.jobs)
                            meta = dict(origin, architecture=arch(level), requested_level=level,
                                        build_mode="dispatch" if dispatch_build else "fixed", wheel=wheel)
                            if kind == "numpy":
                                meta["expected_fft_backend"] = spec.get("fft_backend", "duccfft")
                            env = dict(os.environ)
                            env.pop("DUCC0_MAX_PSABI_LEVEL", None)
                            env.pop("NPY_DISABLE_CPU_FEATURES", None)
                            if dispatch_build:
                                if kind == "ducc":
                                    env["DUCC0_MAX_PSABI_LEVEL"] = str(level)
                                else:
                                    env["NPY_DISABLE_CPU_FEATURES"] = ",".join(
                                        f"X86_V{v}" for v in targets if v > level)
                            execute(label, py, "ducc0" if kind == "ducc" else "numpy", meta, env=env)
                        attempt(prefix, python_profile)
                if kind == "ducc":
                    for candidate, standard in DUCC_CPP_STANDARDS.items():
                        if candidate not in enabled:
                            continue
                        suffix = "cpp" if standard == 17 else f"cpp{standard}"
                        label = name + f"-{suffix}-v{level}"
                        if level not in levels:
                            record(label, "skip", "host/OS does not support requested ISA")
                            continue

                        def ducc_cpp():
                            exe = build_native(kind, src, arch(level), work / label,
                                               out / (label + ".log"), args.jobs,
                                               cxx_standard=standard)
                            execute(label, base, "native", dict(origin, architecture=arch(level),
                                    requested_level=level, build_mode="fixed",
                                    cxx_standard=standard), executable=exe)

                        attempt(label, ducc_cpp)

    return int(not runs or any(r["status"] == "fail" for r in runs)
               or not any(r["status"] == "pass" for r in runs))


if __name__ == "__main__":
    sys.exit(main())
