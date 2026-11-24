"""Normalize FFT timings to DUCC 0.42.0 v1 within each runner."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from run import CPP_REFERENCE, PY_REFERENCE, REFERENCE


def read_json(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def geomean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def mode(row):
    return row["timing_mode"] + ("/" + row["planning"] if row.get("planning") else "")


def cpu_name(host):
    for line in host.get("cpuinfo", "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return "Unknown CPU"


def read_runner(root):
    host = read_json(root / "host.json", {})
    runs = read_json(root / "runs.json", [])
    cpu = cpu_name(host)
    runner = root.name
    print(f"### {runner}: {cpu}\n")
    print("Usable ISA:", ", ".join(f"v{v}" for v in sorted(host.get("usable_levels", []))), "\n")
    print("| Candidate | Run | Passed | Skipped | Failed |")
    print("|---|---|---:|---:|---:|")
    timed, metas, statuses = {}, {}, {}
    for run in runs:
        label = run["label"]
        path = root / (label + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.is_file() else []
        counts = [sum(row["status"] == s for row in rows) for s in ("pass", "skip", "fail")]
        print(f"| {label} | {run['status']} | {counts[0]} | {counts[1]} | {counts[2]} |")
        if run["status"] not in ("pass", "fail"):
            continue
        meta = read_json(path.with_suffix(".meta.json"), {})
        if not meta:
            continue
        metas[label] = meta
        statuses[label] = run["status"]
        timed[label] = {row["case_id"]: row for row in rows
                        if row["status"] == "pass" and row.get("median_ns", 0) > 0
                        and math.isfinite(row["median_ns"])}
    print()
    records = []
    for label, rows in timed.items():
        meta = metas[label]
        api = "C++" if meta["candidate"] == "native" else "Python"
        baseline = CPP_REFERENCE if api == "C++" else PY_REFERENCE
        ref = metas.get(baseline, {})
        if ref.get("sha") != REFERENCE["ref"] or ref.get("requested_level") != 1:
            continue
        level = f"v{meta['requested_level']}" if meta.get("requested_level") else "native/wheel"
        package = label.removesuffix("-" + level)
        revision = meta.get("sha") or meta.get("package") or meta.get("scipy_version", "")
        for case_id in sorted(rows.keys() & timed[baseline].keys()):
            row, refrow = rows[case_id], timed[baseline][case_id]
            left, right = mode(refrow), mode(row)
            if api == "Python" and left != right:
                continue
            records.append(dict(runner=runner, cpu=cpu, api=api, level=level,
                                package=package, candidate=label, revision=revision,
                                group=row["group"], case_id=case_id,
                                reference_mode=left, timing_mode=right,
                                run_status=statuses[label], reference_status=statuses[baseline],
                                ratio=row["median_ns"] / refrow["median_ns"]))
    if not records:
        print("No comparable timings (reference failed, unavailable, or correctness-only run).\n")
    return records


def aggregate(records, view, scope, by_isa=False):
    # Equal weight per runner after matching case IDs within that runner.
    groups = defaultdict(lambda: defaultdict(list))
    flagged = defaultdict(set)
    for r in records:
        isa = r["level"] if by_isa else "all"
        column = r["package"] if by_isa else r["candidate"]
        key = (r["api"], isa, r["group"], r["reference_mode"], column,
               r["revision"], r["timing_mode"])
        groups[key][r["runner"]].append(r["ratio"])
        if r.get("run_status") == "fail" or r.get("reference_status") == "fail":
            flagged[key].add(r["runner"])
    result = []
    for key, runners in sorted(groups.items()):
        api, isa, group, refmode, column, revision, timing = key
        result.append(dict(view=view, scope=scope, api=api, isa=isa, group=group,
                           reference_mode=refmode, column=column, revision=revision,
                           timing_mode=timing, ratio=geomean([geomean(v) for v in runners.values()]),
                           cases=sum(map(len, runners.values())), runners=len(runners),
                           flagged_runners=len(flagged[key])))
    return result


def tables(rows):
    sections = defaultdict(list)
    for row in rows:
        sections[(row["api"], row["isa"], row["reference_mode"])].append(row)
    for (api, isa, refmode), section in sorted(sections.items()):
        print(f"#### {api} / {isa} / reference: {refmode}\n")
        revisions = defaultdict(set)
        for r in section:
            revisions[r["column"]].add(r["revision"])

        def column(r):
            name = r["column"]
            if len(revisions[name]) > 1:
                name += "@" + r["revision"][:12]
            if r["timing_mode"] != refmode:
                name += " (" + r["timing_mode"] + ")"
            return name

        cols = sorted({column(r) for r in section},
                      key=lambda s: (not s.startswith("ducc-0-42"), s))
        groups = sorted({r["group"] for r in section})
        cells = {(r["group"], column(r)): r for r in section}
        print("| Test group | " + " | ".join(cols) + " |")
        print("|---|" + "|".join("---:" for _ in cols) + "|")
        for group in groups:
            values = []
            for col in cols:
                r = cells.get((group, col))
                if r:
                    flag = " †" if r["flagged_runners"] else ""
                    values.append(f"{r['ratio']:.3f} ({r['cases']}/{r['runners']}){flag}")
                else:
                    values.append("—")
            print("| " + group + " | " + " | ".join(values) + " |")
        print()


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.roots[0]
    if len(args.roots) > 1 and not args.output:
        parser.error("--output required for multiple runners")
    out.mkdir(parents=True, exist_ok=True)
    print("## FFT results\n")
    print("Reference: DUCC 0.42.0 v1, separately for Python and C++. "
          "Candidate/reference median time; **1.000 = reference, lower is faster**. "
          "Each cell is a geometric mean of matched passing cases, followed by "
          "(case-runner pairs / contributing runners). Missing cells are not imputed. "
          "Candidate case coverage can differ. † At least one contributing runner had a "
          "failed candidate or reference run; only individually passing cases contribute. "
          "Native reusable-plan timings exclude planning.\n")
    records = []
    grouped = []
    for root in args.roots:
        rows = read_runner(root)
        records.extend(rows)
        if rows:
            result = aggregate(rows, "runner", root.name)
            # Individual job summaries already show per-runner timing tables.
            if len(args.roots) == 1:
                print("#### Per-runner comparison\n")
                tables(result)
            grouped.extend(result)
    if records:
        print("## Per CPU model\n")
        cpus = sorted({r["cpu"] for r in records})
        for cpu in cpus:
            print(f"### {cpu}\n")
            result = aggregate([r for r in records if r["cpu"] == cpu], "cpu", cpu)
            tables(result)
            grouped.extend(result)
        print("## Per ISA level\n")
        print("Only fixed or requested ISA profiles are assigned v1–v4. "
              "System libraries, wheels and native builds appear separately.\n")
        for cpu in cpus + (["All CPUs"] if len(cpus) > 1 else []):
            print(f"### {cpu}\n")
            rows = records if cpu == "All CPUs" else [r for r in records if r["cpu"] == cpu]
            result = aggregate(rows, "isa", cpu, by_isa=True)
            tables(result)
            grouped.extend(result)
    write_csv(out / "normalized.csv", records,
              ["runner", "cpu", "api", "level", "package", "candidate", "revision",
               "group", "case_id", "reference_mode", "timing_mode", "run_status",
               "reference_status", "ratio"])
    write_csv(out / "grouped.csv", grouped,
              ["view", "scope", "api", "isa", "group", "reference_mode", "column",
               "revision", "timing_mode", "ratio", "cases", "runners", "flagged_runners"])


if __name__ == "__main__":
    main()
