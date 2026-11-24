"""Plot absolute FFT runtimes from one runner's results."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from summarize import cpu_name, read_json, mode

STYLES = {1: "-", 2: "--", 3: "-.", 4: ":", None: (0, (8, 3))}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", None: "x"}
# Keep package colors stable across runners, including missing ISA profiles.
COLORS = {"ducc-upstream": "#222222", "ducc": "#0072B2", "ducc-simd": "#332288", "phiadaarr": "#009E73",
          "numpy": "#D55E00", "numpy-pocketfft": "#882255", "scipy": "#CC79A7", "fftw": "#E69F00",
          "admiral": "#56B4E9"}
EXTRA_COLORS = ("#882255", "#332288", "#44AA99", "#999933", "#AA4499",
                "#661100", "#6699CC", "#117733", "#AA4466")


def workload(row):
    """Return transformed length and the fixed attributes of a size sweep."""
    shape = tuple(row["shape"])
    axes = tuple(a % len(shape) for a in row["axes"])
    lengths = tuple(row["lengths"]) or tuple(shape[a] for a in axes)
    batch = tuple((i, n) for i, n in enumerate(shape) if i not in axes)
    # Keep batch shape, axes, precision, layout, thread count and API mode fixed.
    panel = (row["kind"], row["precision"], row["layout"], row["threads"],
             axes, batch, row["mode"], row["norm"], row["transform_type"],
             bool(row["lengths"]))
    return math.prod(lengths), panel, lengths


def read_points(root):
    host = read_json(root / "host.json", {})
    points = []
    for run in read_json(root / "runs.json", []):
        label = run["label"]
        path = root / (label + ".jsonl")
        meta = read_json(path.with_suffix(".meta.json"), {})
        if not path.is_file() or not meta or run["status"] not in ("pass", "fail"):
            continue
        api = "C++" if meta["candidate"] == "native" else "Python"
        level = meta.get("requested_level")
        source = label.removesuffix(f"-v{level}") if level else label
        package = source
        if meta.get("backend") == "ducc":
            suffix = "-py" if api == "Python" else (
                "-cpp23" if meta.get("cxx_standard") == 23 else "-cpp")
            package = source.removesuffix(suffix)
        revision = meta.get("sha") or meta.get("package") or meta.get("scipy_version", "")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ns = row.get("median_ns", 0)
            if row["status"] != "pass" or not ns or ns <= 0 or not math.isfinite(ns):
                continue
            length, panel, lengths = workload(row)
            if length <= 0:
                continue
            points.append(dict(api=api, group=row["group"], panel=panel,
                               package=package, candidate=label, level=level,
                               revision=revision, timing_mode=mode(row),
                               run_status=run["status"], case_id=row["case_id"],
                               length=length, shape=row["shape"], lengths=lengths,
                               runtime_ns=ns))
    return cpu_name(host), points


def title(panel):
    kind, precision, layout, threads, axes, batch, buffer, norm, ty, resize = panel
    pieces = [kind, precision, layout, f"{threads} thread(s)"]
    if kind in ("dct", "dst"):
        pieces.append(f"type {ty}")
    if buffer != "default":
        pieces.append(buffer)
    if norm != "backward":
        pieces.append(norm)
    if resize:
        pieces.append("resize")
    dims = ",".join(map(str, axes))
    batches = " × ".join(str(n) for _, n in batch) or "1"
    return " / ".join(pieces) + "\n" + f"axes ({dims}); batch {batches}"


def draw(root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cpu, points = read_points(root)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["api", "group", "panel", "package", "candidate", "level", "revision",
              "timing_mode", "run_status", "case_id", "length", "shape", "lengths",
              "runtime_ns"]
    with (out / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)

    colors = dict(COLORS)
    for i, package in enumerate(sorted({p["package"] for p in points} - colors.keys())):
        colors[package] = EXTRA_COLORS[i % len(EXTRA_COLORS)]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in points:
        # Native one-shot and reusable-plan execution are distinct legend entries.
        series = (p["candidate"], p["package"], p["level"],
                  p["timing_mode"], p["run_status"])
        groups[(p["api"], p["group"])][p["panel"]][series].append(p)

    index = [f"# FFT runtimes: {cpu}", "",
             "Absolute median runtime per call in microseconds; log-log axes.",
             "Length = product of transformed dimensions (not batch size).",
             "Each panel fixes transform, precision, layout, threads, axes, batch shape, "
             "buffer mode and normalization. A single measured length is a marker.",
             "† Candidate run failed elsewhere; only passing cases are plotted.",
             "Reusable-plan execution excludes plan creation. See points.csv for exact shapes.",
             ""]

    for (api, group), panels in sorted(groups.items()):
        names = []
        # At most six panels per page, rather than one enormous group figure.
        ordered = sorted(panels)
        for start in range(0, len(ordered), 6):
            page = ordered[start:start + 6]
            nrows = math.ceil(len(page) / 2)
            fig, axs = plt.subplots(nrows, 2, figsize=(13, 3.4 * nrows + 2.4),
                                    squeeze=False)
            handles = {}
            for ax, panel in zip(axs.flat, page):
                for key, rows in sorted(panels[panel].items()):
                    label, package, level, timing, status = key
                    legend = f"{label} / {timing}" + (" †" if status == "fail" else "")
                    # Multiple different shapes at a single length remain separate
                    # points, not an arbitrary vertical line or an averaged runtime.
                    by_length = defaultdict(list)
                    for p in rows:
                        by_length[p["length"]].append(p["runtime_ns"] / 1000)
                    xs = sorted(by_length)
                    color, style, marker = colors[package], STYLES[level], MARKERS[level]
                    ys = [by_length[x][0] if len(by_length[x]) == 1 else math.nan for x in xs]
                    ax.plot(xs, ys, color=color, linestyle=style, marker=marker,
                            markersize=3.5, linewidth=1.2)
                    for x in xs:
                        if len(by_length[x]) > 1:
                            ax.scatter([x] * len(by_length[x]), by_length[x],
                                       color=color, marker=marker, s=13)
                    handles[legend] = Line2D([0], [0], color=color, linestyle=style,
                                             marker=marker, markersize=4, label=legend)
                ax.set_title(title(panel), fontsize=9)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("FFT length (transformed elements)")
                ax.set_ylabel("Runtime (µs/call)")
                ax.grid(True, which="major", alpha=0.25)
            for ax in list(axs.flat)[len(page):]:
                ax.set_visible(False)
            fig.suptitle(f"{cpu}\n{api} / {group}", fontsize=12)
            fig.legend(handles=list(handles.values()), loc="lower center",
                       ncol=3, fontsize=8, frameon=False)
            fig.tight_layout(rect=(0, min(0.35, 0.07 + 0.022 * math.ceil(len(handles) / 3)),
                                   1, 0.92))
            name = f"{'cpp' if api == 'C++' else 'python'}-{group}-{start // 6 + 1:02d}.png"
            fig.savefig(out / name, dpi=140)
            plt.close(fig)
            names.append(name)
        index += [f"## {api} / {group}", ""]
        index += [f"- [{name}]({name})" for name in names]
        index.append("")
    if not points:
        index.append("No passing timed cases. Correctness-only groups have no runtime plot.")
    (out / "README.md").write_text("\n".join(index) + "\n")
    return len(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    count = draw(args.root, args.output or args.root / "plots")
    print(f"Plotted {count} API/workload groups.")


if __name__ == "__main__":
    main()
