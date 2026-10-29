"""Check release file hashes without importing models or starting an analysis."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(package, code_only=False):
    name = "CODE_FILE_MANIFEST.csv" if code_only else "FILE_MANIFEST.csv"
    with (package / name).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    missing, changed = [], []
    for row in rows:
        path = (package / row["Path"]).resolve()
        if not path.is_relative_to(package) or not path.is_file():
            missing.append(row["Path"])
        elif digest(path) != row["SHA256"]:
            changed.append(row["Path"])
    return {
        "scope": "code" if code_only else "code_and_companion_data",
        "files_checked": len(rows),
        "missing_files": missing,
        "changed_files": changed,
        "passed": not (missing or changed),
        "training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.code_only)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
