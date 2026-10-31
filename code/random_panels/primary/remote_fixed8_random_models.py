"""Fit fixed primary random panels on the first saved fivefold partition."""

from pathlib import Path
import os
import sys
import json
import time
import hashlib
import traceback
import fcntl
import subprocess
import shutil
import argparse
import numpy as np
import pandas as pd

PACKAGE = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "documentation/PACKAGE_STATUS.json").is_file()
)
CODE = PACKAGE / "code/ACEGP"
sys.path[:0] = [str(CODE), str(CODE / "src")]
from public_paths import (
    DATA_ROOT,
    RESULT_ROOT,
    RANDOM_ROOT,
    output_root,
    read_saved_fold,
    read_validation_real_fold,
)

ROOT = None

from acegp_uniform import fit_holdout, stable_seed, CODE_HASH, MODELS
from fixed9_ld_tools import BedReader
import statistical_reporting as st
import remote_fixed_model_v2 as source

OUT = None
META = RANDOM_ROOT / "primary"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = Path(str(p) + ".partial." + str(os.getpid()))
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


def fold(number, fi):
    folder = OUT / f"panel_{number:03d}" / f"r01_f{fi}"
    folder.mkdir(parents=True, exist_ok=True)
    held = (folder / "run.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / "done.json").exists():
        old = json.loads((folder / "done.json").read_text())
        assert sha(folder / "predictions.csv") == old["predictions_sha256"]
        return
    start = time.monotonic()
    accept = json.loads((META / "metadata/acceptance.json").read_text())
    assert (
        accept["state"] == "PASS_100_FIXED8_PANELS_ALL_FROZEN_METADATA_GATES"
        and accept["panels"] == 100
    )
    p = META / "panels" / f"panel_{number:03d}.json"
    npfile = META / "panels" / f"panel_{number:03d}_SNP_rows.npy"
    for file in [p, npfile]:
        assert (
            sha(file)
            == accept["files"]["random_fixed8_metadata/" + file.relative_to(META).as_posix()][
                "sha256"
            ]
        )
    reg = json.loads((META / "metadata/fixed8_random_model_registration.json").read_text())
    assert reg["fitter_CODE_HASH"] == CODE_HASH and len(MODELS) == 16
    panel = json.loads(p.read_text())
    rows = np.load(npfile, allow_pickle=False)
    source.COHORT = "development"
    ids, g, _, _ = source.arrays()
    fam = pd.read_csv(
        META / "genotype_source/PMB2026_full_241.fam", sep=r"\s+", header=None, dtype=str
    )
    reorder = np.array([fam[1].tolist().index(s) for s in ids.sample_id])
    q = BedReader(META / "genotype_source/PMB2026_full_241.bed", 241, 4817964)(rows)[
        reorder
    ].astype(float)
    q[q < 0] = np.nan
    marker_source = META / "genotype_source/raw_metadata_compressed.npz"
    with np.load(marker_source, allow_pickle=False) as f:
        chrom = f["chromosome"][rows]
        pos = f["position_bp"][rows]
    groups = []
    coverage = np.zeros(len(rows), int)
    for r in panel["regions"]:
        ix = np.flatnonzero((chrom == r["chr"]) & (pos >= r["start_bp"]) & (pos <= r["end_bp"]))
        assert len(ix) == r["n"]
        groups.append((r["qid"], ix))
        coverage[ix] += 1
    assert len(groups) == 8 and np.all(coverage == 1) and len(rows) == panel["panel"]["n"]
    splits = json.loads((DATA_ROOT / "inputs/development_20x5.json").read_text())
    item = splits["repeats"][0]["folds"][fi - 1]
    tr = np.array(item["train_rows"])
    te = np.array(item["test_rows"])
    assert ids.sample_id.iloc[te].tolist() == item["test_ids"]
    config = json.loads((DATA_ROOT / "inputs/model_config.json").read_text())
    config.update(
        blas_threads=2,
        new_cache_directory=str(folder / "cache"),
        cache_max_GiB=4,
        minimum_free_disk_GiB=25,
    )
    seed = stable_seed(20261013, "development", 1, fi, "outer_refit")
    result = fit_holdout(
        g[tr], g[te], ids.loss_SF.to_numpy(float)[tr], q[tr], q[te], groups, config, seed
    )
    core = read_saved_fold("repeated_cv", "development", 1, fi, item["test_ids"])
    assert core.sample_id.tolist() == item["test_ids"]
    error = float(np.max(np.abs(core.Genome_wide.to_numpy(float) - result["Genome_wide"])))
    assert error <= 1e-8, error
    frame = pd.DataFrame(
        {
            "sample_id": item["test_ids"],
            "row_index": te,
            "repeat": 1,
            "fold": fi,
            "panel": number,
            "Genome_wide": result["Genome_wide"],
            "random_QTL": result["QTL_informed"],
        }
    )
    assert np.isfinite(frame[["Genome_wide", "random_QTL"]].to_numpy()).all()
    frame.to_csv(folder / "predictions.csv", index=False)
    atomic(folder / "training_choices.json", result["audit"])
    atomic(
        folder / "done.json",
        {
            "state": "PASS_FROZEN_RANDOM8_FULL16_FOLD",
            "panel": number,
            "repeat": 1,
            "fold": fi,
            "context_seed": seed,
            "SNP_columns": len(rows),
            "region_count": 8,
            "GW_max_absolute_error": error,
            "y_test_passed_to_fitter": False,
            "fitter_CODE_HASH": CODE_HASH,
            "source_random_panel_json_sha256": sha(p),
            "source_SNP_rows_file_sha256": sha(npfile),
            "predictions_sha256": sha(folder / "predictions.csv"),
            "choices_sha256": sha(folder / "training_choices.json"),
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    cache = (folder / "cache").resolve()
    assert cache.is_relative_to(OUT.resolve()) and not cache.is_symlink()
    if cache.exists():
        shutil.rmtree(cache)


def configure_output(path=None):
    """Set output paths for an explicit execution request."""
    global ROOT, OUT
    ROOT = output_root(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT = ROOT / "random_fixed8_models"
    OUT.mkdir(parents=True, exist_ok=True)
