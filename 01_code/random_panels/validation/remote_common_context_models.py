"""Fit the real and random validation panels in the shared imputation context."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import numpy as np
import pandas as pd

PACKAGE = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "04_documentation/PACKAGE_STATUS.json").is_file()
)
CODE = PACKAGE / "01_code/ACEGP"
sys.path[:0] = [str(CODE), str(CODE / "src")]
from public_paths import (
    DATA_ROOT,
    RESULT_ROOT,
    RANDOM_ROOT,
    output_root,
    read_saved_fold,
    read_validation_real_fold,
)


VALIDATION_DATA_ROOT = RANDOM_ROOT / "validation"
PREDICTION_INPUT_ROOT = DATA_ROOT
METADATA_ROOT = VALIDATION_DATA_ROOT / "metadata"
SAVED_RESULTS_ROOT = VALIDATION_DATA_ROOT / "results"
PENDING_INPUT_ROOT = VALIDATION_DATA_ROOT / "PENDING_common_genotypes"
from acegp_uniform import fit_holdout, stable_seed, CODE_HASH, MODELS
import statistical_reporting as st

REG = json.loads((METADATA_ROOT / "COMMON_CONTEXT_MODEL_REGISTRATION.json").read_text())
OUTPUT_ROOT = None
CATALOG_PATH = PENDING_INPUT_ROOT / "observed_catalog/raw_metadata_validation151.npz"
MATRIX_ROOT = PENDING_INPUT_ROOT / "common_final_matrix"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".partial.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def verified_sources():
    sources = {
        "src/acegp_uniform.py": CODE / "src/acegp_uniform.py",
        "inputs/validation_20x5.json": PREDICTION_INPUT_ROOT / "inputs/validation_20x5.json",
        "inputs/model_config.json": PREDICTION_INPUT_ROOT / "inputs/model_config.json",
        "validation_inputs/background_markers.csv": PREDICTION_INPUT_ROOT
        / "validation_inputs/background_markers.csv",
        "inputs/validation_regions.csv": PREDICTION_INPUT_ROOT / "inputs/validation_regions.csv",
        "common_final_matrix/COMPLETE.json": PENDING_INPUT_ROOT / "historical_COMPLETE.json",
        "common_context_model_metadata/MAPPING_ACCEPTANCE.json": METADATA_ROOT
        / "MAPPING_ACCEPTANCE.json",
        "random_validation7_constrained_joint_v2/output/acceptance.json": METADATA_ROOT
        / "panel_acceptance.json",
    }
    for label, path in sources.items():
        assert sha(path) == REG["input_file_SHA256"][label], label
    assert CODE_HASH == REG["fitter_CODE_HASH"] and len(MODELS) == 16
    mapping = read(METADATA_ROOT / "MAPPING_ACCEPTANCE.json")
    assert mapping["state"] == "PASS_FROZEN_REAL7_AND_BACKGROUND_METADATA_MAPPING_TO_COMMON_CATALOG"
    assert mapping["real_region_unique_raw_SNPs"] == 67104
    assert mapping["marker_lists"]["background"]["catalog_positions_found"] == 32666
    order = read(METADATA_ROOT / "SAMPLE_ORDER_ACCEPTANCE.json")
    assert order["state"] == "PASS_151_SAMPLE_ORDER_IDENTICAL_COMMON_GT_AND_OLD_VALIDATION_INPUT"
    matrix = read(PENDING_INPUT_ROOT / "historical_COMPLETE.json")
    assert matrix["state"] == "PASS_12_CHROMOSOME_COMMON_GT_MATRIX_ALL_KNOWN_OBSERVED_PRESERVED"
    assert matrix["sites"] == 4817083 and matrix["samples"] == 151
    for chrom, entry in enumerate(matrix["records"], 1):
        assert sha(MATRIX_ROOT / f"chr{chrom}/A1_GT.npy") == entry["matrix_sha256"]
    random = read(METADATA_ROOT / "panel_acceptance.json")
    assert random["state"] == "PASS_100_VALIDATION7_CONSTRAINED_PANELS_ALL_FROZEN_METADATA_GATES"
    assert random["panels"] == 100 and random["regions_per_panel"] == 7
    quality_path = METADATA_ROOT / "FIVE_SEED_ACCEPTANCE.json"
    assert quality_path.exists(), "All five independent quality audits are required"
    quality = read(quality_path)
    assert quality["state"] == "PASS_ALL_FIVE_COMMON_CONTEXT_QUALITY_GATES_INDEPENDENTLY_AUDITED"
    assert len(quality["seeds"]) == 5 and all(row["gate_passed"] for row in quality["seeds"])
    return {
        "source_SHA256": {label: REG["input_file_SHA256"][label] for label in sources},
        "quality_acceptance_sha256": sha(quality_path),
        "sample_order_acceptance_sha256": sha(METADATA_ROOT / "SAMPLE_ORDER_ACCEPTANCE.json"),
        "fitter_CODE_HASH": CODE_HASH,
        "all_12_matrix_SHA256": {
            str(i): x["matrix_sha256"] for i, x in enumerate(matrix["records"], 1)
        },
    }


def load_ids_and_split(fold):
    ids = pd.read_csv(
        PREDICTION_INPUT_ROOT / "inputs/validation_samples.csv", dtype={"sample_id": str}
    )
    order = pd.read_csv(
        PREDICTION_INPUT_ROOT / "validation_inputs/sample_order.csv", dtype={"sample_id": str}
    )
    assert ids.sample_id.tolist() == order.sample_id.tolist()
    assert len(ids) == 151
    splits = read(PREDICTION_INPUT_ROOT / "inputs/validation_20x5.json")
    item = splits["repeats"][0]["folds"][fold - 1]
    tr = np.asarray(item["train_rows"], dtype=np.int64)
    te = np.asarray(item["test_rows"], dtype=np.int64)
    assert ids.sample_id.iloc[te].tolist() == item["test_ids"]
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 151
    return ids, item, tr, te


def catalog():
    with np.load(CATALOG_PATH, allow_pickle=False) as z:
        chrom, pos = z["chromosome"], z["position_bp"]
    assert len(chrom) == len(pos) == 4817083
    key = chrom.astype(np.int64) * 1000000000 + pos.astype(np.int64)
    assert np.all(key[1:] > key[:-1])
    return chrom, pos, key


def marker_rows(key):
    with (PREDICTION_INPUT_ROOT / "validation_inputs/background_markers.csv").open(
        newline=""
    ) as stream:
        records = list(csv.DictReader(stream))
    requested = np.asarray(
        [int(x["chr"]) * 1000000000 + int(x["pos"]) for x in records], dtype=np.int64
    )
    rows = np.searchsorted(key, requested)
    assert len(rows) == 32666 and np.all(rows < len(key))
    assert np.array_equal(key[rows], requested)
    return rows


def real_panel(chrom, pos):
    with (PREDICTION_INPUT_ROOT / "inputs/validation_regions.csv").open(newline="") as stream:
        regions = list(csv.DictReader(stream))
    assert len(regions) == 7
    coverage = np.zeros(len(chrom), dtype=np.uint8)
    for region in regions:
        mask = (
            (chrom == int(region["chr"]))
            & (pos >= int(region["start_bp"]))
            & (pos <= int(region["end_bp"]))
        )
        assert np.any(mask) and not np.any(coverage[mask])
        coverage[mask] = 1
    rows = np.flatnonzero(coverage)
    assert len(rows) == 67104
    return rows, regions


def random_panel(number):
    assert 1 <= number <= 100
    acceptance = read(METADATA_ROOT / "panel_acceptance.json")
    path = VALIDATION_DATA_ROOT / "panels" / f"panel_{number:03d}.json"
    npy = VALIDATION_DATA_ROOT / "panels" / f"panel_{number:03d}_SNP_rows.npy"
    for file in (path, npy):
        name = "output/" + str(file.relative_to(VALIDATION_DATA_ROOT)).replace("\\", "/")
        assert sha(file) == acceptance["files"][name]["sha256"], name
    panel = read(path)
    rows = np.load(npy, allow_pickle=False)
    assert len(panel["regions"]) == 7
    assert rows.ndim == 1 and np.all(rows[1:] > rows[:-1])
    assert len(rows) == panel["panel"]["n"]
    return rows, panel["regions"], sha(path), sha(npy)


def load_gt_rows(rows, chrom):
    assert rows.ndim == 1 and len(rows) > 0
    assert np.all(rows[1:] > rows[:-1])
    assert rows.min() >= 0 and rows.max() < len(chrom)
    out = np.empty((151, len(rows)), dtype=np.float64)
    for c in np.unique(chrom[rows]):
        col = np.flatnonzero(chrom[rows] == c)
        first = int(np.searchsorted(chrom, c, side="left"))
        matrix = np.load(MATRIX_ROOT / f"chr{c}/A1_GT.npy", mmap_mode="r", allow_pickle=False)
        block = np.asarray(matrix[rows[col] - first], dtype=np.float64).T
        block[block == 255] = np.nan
        out[:, col] = block
    assert out.shape == (151, len(rows))
    assert np.all(np.isfinite(out) | np.isnan(out))
    return out


def build_groups(rows, regions, chrom, pos):
    c, p = chrom[rows], pos[rows]
    groups = []
    coverage = np.zeros(len(rows), dtype=np.uint8)
    for region in regions:
        match = np.flatnonzero(
            (c == int(region["chr"]))
            & (p >= int(region["start_bp"]))
            & (p <= int(region["end_bp"]))
        )
        assert len(match) > 0 and not np.any(coverage[match])
        if "n" in region:
            assert len(match) == int(region["n"])
        coverage[match] = 1
        groups.append((region["qid"], match))
    assert len(groups) == 7 and np.all(coverage == 1)
    return groups


def fold(kind, number, fi):
    assert kind in ("real", "random") and 1 <= fi <= 5
    assert (kind == "real" and number == 0) or (kind == "random" and 1 <= number <= 100)
    folder = OUTPUT_ROOT / ("real7" if kind == "real" else f"random_{number:03d}") / f"r01_f{fi}"
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / "done.json").exists():
        done = read(folder / "done.json")
        assert sha(folder / "predictions.csv") == done["predictions_sha256"]
        assert sha(folder / "training_choices.json") == done["choices_sha256"]
        return
    start = time.monotonic()
    accepted = read(SAVED_RESULTS_ROOT / "INPUT_ACCEPTANCE.json")
    assert accepted["state"] == "PASS_SAME_CONTEXT_REAL_RANDOM_AND_FIVE_SEED_QUALITY"
    quality = METADATA_ROOT / "FIVE_SEED_ACCEPTANCE.json"
    assert sha(quality) == accepted["quality_acceptance_sha256"]
    chrom, pos, key = catalog()
    bg_rows = marker_rows(key)
    g = load_gt_rows(bg_rows, chrom)
    if kind == "real":
        rows, regions = real_panel(chrom, pos)
        panel_sha = None
        snp_rows_sha = None
    else:
        rows, regions, panel_sha, snp_rows_sha = random_panel(number)
    q = load_gt_rows(rows, chrom)
    groups = build_groups(rows, regions, chrom, pos)
    ids, item, tr, te = load_ids_and_split(fi)
    config = read(PREDICTION_INPUT_ROOT / "inputs/model_config.json")
    config.update(
        blas_threads=2,
        new_cache_directory=str(folder / "cache"),
        cache_max_GiB=4,
        minimum_free_disk_GiB=25,
    )
    seed = stable_seed(20261013, "validation", 1, fi, "outer_refit")
    y_train = ids.loss_SF.to_numpy(float)[tr]
    result = fit_holdout(g[tr], g[te], y_train, q[tr], q[te], groups, config, seed)
    gw = result["Genome_wide"]
    qtl = result["QTL_informed"]
    if kind == "random":
        real = read_validation_real_fold(fi, item["test_ids"])
        assert real.sample_id.tolist() == item["test_ids"]
        error = float(np.max(np.abs(real.Genome_wide.to_numpy(float) - gw)))
        assert error <= 1e-8, ("Same-context Genome-wide mismatch", number, fi, error)
    else:
        error = None
    assert np.isfinite(gw).all() and np.isfinite(qtl).all()
    frame = pd.DataFrame(
        {
            "sample_id": item["test_ids"],
            "row_index": te,
            "repeat": 1,
            "fold": fi,
            "panel": kind if kind == "real" else number,
            "Genome_wide": gw,
            "candidate_QTL": qtl,
        }
    )
    frame.to_csv(folder / "predictions.csv", index=False)
    save(folder / "training_choices.json", result["audit"])
    save(
        folder / "done.json",
        {
            "state": "PASS_SAME_CONTEXT_FULL16_ACEGP_OUTER_FOLD",
            "kind": kind,
            "panel": number,
            "repeat": 1,
            "fold": fi,
            "test_ids_exact_frozen_split": True,
            "context_seed": seed,
            "SNP_columns": len(rows),
            "background_columns": len(bg_rows),
            "region_count": len(groups),
            "same_context_GW_max_absolute_error": error,
            "y_test_passed_to_fitter": False,
            "fitter_CODE_HASH": CODE_HASH,
            "source_random_panel_json_sha256": panel_sha,
            "source_random_SNP_rows_sha256": snp_rows_sha,
            "predictions_sha256": sha(folder / "predictions.csv"),
            "choices_sha256": sha(folder / "training_choices.json"),
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    cache = (folder / "cache").resolve()
    assert cache.is_relative_to(OUTPUT_ROOT.resolve()) and not cache.is_symlink()
    if cache.exists():
        shutil.rmtree(cache)


def configure_output(path=None):
    global OUTPUT_ROOT
    OUTPUT_ROOT = output_root(path) / "common_context_models"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
