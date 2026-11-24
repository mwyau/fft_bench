"""Detailed v4 tables; use actual transform dimensions, not batch size."""
from collections import defaultdict
import csv
import math


def transform_lengths(row):
    shape = row["shape"]
    return tuple(row["lengths"]) or tuple(shape[a] for a in row["axes"])


def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def length_class(lengths):
    def classify(n):
        if n <= 1:
            return "unit"
        factors = prime_factors(n)
        if len(factors) == 1:
            return "prime"
        if len(set(factors)) == 1:
            return f"power of {factors[0]}"
        parity = "odd" if n % 2 else "even"
        return f"{parity} mixed radix (largest prime {'<=7' if max(factors) <= 7 else '>7'})"
    return " × ".join(classify(n) for n in lengths)


def geomean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def write_v4_report(out, records):
    rows = [r for r in records if r["level"] == "v4"]
    fields = ["cpu", "api", "group", "reference_mode", "candidate", "revision",
              "timing_mode", "category", "value", "ratio", "cases", "runners",
              "worst_ratio", "worst_case", "worst_runner", "flagged_runners"]
    groups = defaultdict(list)
    for r in rows:
        prefix = tuple(r[k] for k in fields[:7])
        for category, value in (("length", r["lengths"]), ("factorization", r["length_class"]),
                                ("group", "all")):
            groups[prefix + (category, value)].append(r)
    results = []
    for key, cases in sorted(groups.items()):
        per_runner = defaultdict(list)
        for r in cases:
            per_runner[r["runner"]].append(r["ratio"])
        worst = max(cases, key=lambda r: r["ratio"])
        flagged = {r["runner"] for r in cases if r["run_status"] == "fail"
                   or r["reference_status"] == "fail"}
        results.append(dict(zip(fields, (*key, geomean([geomean(v) for v in per_runner.values()]),
                          len(cases), len(per_runner), worst["ratio"], worst["case_id"],
                          worst["runner"], len(flagged)))))
    with (out / "v4.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    sections = defaultdict(list)
    for r in results:
        sections[tuple(r[k] for k in ("cpu", "api", "group", "reference_mode", "category"))].append(r)
    lines = ["# v4 FFT detail", "", "Ratios to DUCC 0.42.0 v1 (lower is faster). "
             "Lengths are transformed dimensions; batches are excluded. "
             "Cells are geometric means of matched passing cases with equal runner weight; "
             "max is the largest individual matched ratio. † denotes unexpected failures elsewhere. "
             "Coverage may differ. See v4.csv for counts, revisions and worst case IDs.", ""]
    for (cpu, api, group, mode, cat), entries in sorted(sections.items()):
        lines += [f"## {cpu} / {api} / {group} / {mode} / {cat}", ""]
        cols = sorted({(r["candidate"], r["revision"], r["timing_mode"]) for r in entries})
        def label(c):
            name, rev, timing = c
            if sum(x[0] == name for x in cols) > 1:
                name += "@" + rev[:8]
            return name + (f" ({timing})" if timing != mode else "")
        lines += ["| Length or class | " + " | ".join(map(label, cols)) + " |",
                  "|---|" + "|".join("---:" for _ in cols) + "|"]
        cells = {(r["value"], (r["candidate"], r["revision"], r["timing_mode"])): r for r in entries}
        values = sorted({r["value"] for r in entries},
                        key=lambda v: tuple(map(int, v.split("x"))) if cat == "length" else v)
        for value in values:
            cells_text = []
            for c in cols:
                r = cells.get((value, c))
                cells_text.append(f"{r['ratio']:.3f} (max {r['worst_ratio']:.3f})"
                                  + (" †" if r["flagged_runners"] else "") if r else "—")
            lines.append("| " + value + " | " + " | ".join(cells_text) + " |")
        lines.append("")
    if not rows:
        lines.append("No v4 timings available.")
    (out / "v4.md").write_text("\n".join(lines) + "\n")
