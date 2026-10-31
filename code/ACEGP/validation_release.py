"""Verify the registered reconstructed validation input version."""

import hashlib
import json
from pathlib import Path

DATA_VERSION = "validation_reconstructed_20261029"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_validation_release(package, include_raw=False):
    """Check versioned files; historical hashes remain provenance only."""
    package = Path(package).resolve()
    registration_path = package / "documentation/VALIDATION_DATA_VERSION.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration["data_version"] != DATA_VERSION:
        raise ValueError("The companion data version does not match this code release.")
    for entry in registration["registered_files"]:
        relative = entry["path"]
        if not include_raw and "/genotypes/" in relative:
            continue
        path = (package / relative).resolve()
        if not path.is_relative_to(package) or not path.is_file():
            raise ValueError(f"Missing registered validation file: {relative}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Changed registered validation file: {relative}")
    return registration
