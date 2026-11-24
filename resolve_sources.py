"""Resolve moving Git refs before generating a cache key."""
import json
from pathlib import Path
import re
import subprocess
import sys


def resolve(config):
    for kind in ("ducc", "numpy", "admiral"):
        for spec in config.get(kind, []):
            ref = spec["ref"]
            if re.fullmatch(r"[0-9a-f]{40}", ref):
                continue
            refs = [ref] if ref.startswith("refs/") else ["refs/heads/" + ref, "refs/tags/" + ref]
            patterns = refs + [r + "^{}" for r in refs if r.startswith("refs/tags/")]
            output = subprocess.check_output(["git", "ls-remote", spec["repo"], *patterns], text=True)
            found = {name: sha for sha, name in (line.split() for line in output.splitlines())}
            for name in refs:
                sha = found.get(name + "^{}") or found.get(name)
                if sha:
                    spec["ref"] = sha
                    spec["source_ref"] = ref
                    break
            else:
                raise RuntimeError(f"Could not resolve {spec['repo']}:{ref}; use a full SHA")
    return config


if __name__ == "__main__":
    config = resolve(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + "\n")
