"""Summarize matched FFT cases on one runner."""

import csv
from itertools import combinations
import json
import math
from pathlib import Path
import sys


def main(root):
    runs = json.loads((root / "runs.json").read_text())
    host = json.loads((root / "host.json").read_text())
    print("## FFT results\n")
    print("Usable x86-64 levels:", ", ".join(map(str, host["usable_levels"])), "\n")
    print("| Candidate | Run | Passed | Skipped | Failed |")
    print("|---|---|---:|---:|---:|")
    timed = {}
    api = {}
    for run in runs:
        label = run["label"]
        path = root / (label + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
        counts = [sum(row["status"] == s for row in rows) for s in ("pass", "skip", "fail")]
        print(f"| {label} | {run['status']} | {counts[0]} | {counts[1]} | {counts[2]} |")
        if not path.exists():
            continue
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        api[label] = "C++" if meta["candidate"] == "native" else "Python"
        timed[label] = {row["case_id"]: row for row in rows
                        if row["status"] == "pass" and row.get("median_ns", 0) > 0}
    with (root / "grouped.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("api", "baseline", "candidate", "group", "baseline_mode",
                         "candidate_mode", "matched_cases", "geomean_speedup"))
        for a, b in combinations(timed, 2):
            if api[a] != api[b]:
                continue
            groups = {}
            for case_id in sorted(timed[a].keys() & timed[b].keys()):
                left, right = timed[a][case_id], timed[b][case_id]
                ma, mb = left["timing_mode"], right["timing_mode"]
                if api[a] == "Python" and ma != mb:
                    continue
                if left.get("planning"):
                    ma += "/" + left["planning"]
                if right.get("planning"):
                    mb += "/" + right["planning"]
                key = (left["group"], ma, mb)
                groups.setdefault(key, []).append(math.log(left["median_ns"] / right["median_ns"]))
            for (group, ma, mb), ratios in sorted(groups.items()):
                writer.writerow((api[a], a, b, group, ma, mb, len(ratios),
                                 math.exp(sum(ratios) / len(ratios))))
    print("\nMatched case comparisons: grouped.csv. A speedup above 1 means the candidate was faster than the baseline. Native plan execution and one-shot timings are labeled separately.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
