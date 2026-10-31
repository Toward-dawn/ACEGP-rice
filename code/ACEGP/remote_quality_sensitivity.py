"""Fit validation quality subsets using retained rows from the saved folds."""

from pathlib import Path
import json
import sys
import time
import argparse
import hashlib
import fcntl
import subprocess
import shutil
import traceback
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "src"))
from public_paths import DATA_ROOT, RESULT_ROOT, output_root, read_saved_fold

ROOT = None
from acegp_uniform import fit_holdout, stable_seed
import statistical_reporting as st
import remote_fixed_model_v2 as loader

OUT = None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = Path(str(p) + ".partial." + str(__import__("os").getpid()))
    t.write_text(json.dumps(d, indent=2, allow_nan=False))
    t.replace(p)


def fold(n, ri, fi):
    folder = OUT / f"subset{n}" / f"r{ri:02d}_f{fi}"
    folder.mkdir(parents=True, exist_ok=True)
    held = (folder / "run.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / "done.json").exists():
        assert (
            sha(folder / "predictions.csv")
            == json.loads((folder / "done.json").read_text())["predictions_sha256"]
        )
        return
    reg = json.loads((DATA_ROOT / "quality_sensitivity_registration.json").read_text())
    keep = reg["subsets"][str(n)]["retained_ids"]
    loader.COHORT = "validation"
    ids, g, q, groups = loader.arrays()
    selected = np.flatnonzero(ids.sample_id.isin(keep).to_numpy())
    assert len(selected) == n
    split = json.loads((DATA_ROOT / "inputs/validation_20x5.json").read_text())["repeats"][ri - 1][
        "folds"
    ][fi - 1]
    allowed = set(map(int, selected))
    tr = np.array([i for i in split["train_rows"] if i in allowed], int)
    te = np.array([i for i in split["test_rows"] if i in allowed], int)
    assert len(tr) + len(te) == n and not set(tr) & set(te)
    config = json.loads((DATA_ROOT / "inputs/model_config.json").read_text())
    config.update(
        blas_threads=2,
        new_cache_directory=str(folder / "cache"),
        cache_max_GiB=4,
        minimum_free_disk_GiB=25,
    )
    context = stable_seed(20261013, "validation", ri, fi, "outer_refit")
    result = fit_holdout(
        g[tr], g[te], ids.loss_SF.to_numpy(float)[tr], q[tr], q[te], groups, config, context
    )
    frame = pd.DataFrame(
        {
            "sample_id": ids.sample_id.iloc[te],
            "original_row": te,
            "repeat": ri,
            "fold": fi,
            "Genome_wide": result["Genome_wide"],
            "QTL_informed": result["QTL_informed"],
        }
    )
    assert np.isfinite(frame[["Genome_wide", "QTL_informed"]].to_numpy()).all()
    frame.to_csv(folder / "predictions.csv", index=False)
    atomic(folder / "training_choices.json", result["audit"])
    atomic(
        folder / "done.json",
        {
            "state": "PASS_HISTORICAL_QC_SUBSET_FULL16_REFIT",
            "subset": n,
            "training_n": len(tr),
            "test_n": len(te),
            "same_original151_partition_restricted": True,
            "heldout_phenotype_passed_to_fitter": False,
            "predictions_sha256": sha(folder / "predictions.csv"),
            "choices_sha256": sha(folder / "training_choices.json"),
            "registration_sha256": sha(DATA_ROOT / "quality_sensitivity_registration.json"),
        },
    )
    cache = (folder / "cache").resolve()
    assert cache.is_relative_to(OUT.resolve()) and not cache.is_symlink()
    if cache.exists():
        shutil.rmtree(cache)


def summarize(n):
    reg = json.loads((DATA_ROOT / "quality_sensitivity_registration.json").read_text())
    keep = set(reg["subsets"][str(n)]["retained_ids"])
    truth = (
        pd.read_csv(DATA_ROOT / "inputs/validation_samples.csv", dtype={"sample_id": str})
        .set_index("sample_id")
        .loss_SF
    )
    frames = []
    for r in range(1, 21):
        for f in range(1, 6):
            folder = OUT / f"subset{n}" / f"r{r:02d}_f{f}"
            done = json.loads((folder / "done.json").read_text())
            assert sha(folder / "predictions.csv") == done["predictions_sha256"]
            frames.append(pd.read_csv(folder / "predictions.csv"))
    allpred = pd.concat(frames)
    allpred.to_csv(OUT / f"subset{n}" / "all_oof_predictions.csv", index=False)
    records = []
    for r in range(1, 21):
        d = allpred[allpred["repeat"] == r]
        assert len(d) == n and set(d.sample_id) == keep and d.sample_id.nunique() == n
        y = truth.loc[d.sample_id].to_numpy(float)
        a = st.metrics(y, d.QTL_informed.to_numpy(float))
        b = st.metrics(y, d.Genome_wide.to_numpy(float))
        records.append(
            {
                "repeat": r,
                "QTL_r": a["pearson_r"],
                "GW_r": b["pearson_r"],
                "delta_r": (
                    None
                    if a["pearson_r"] is None or b["pearson_r"] is None
                    else a["pearson_r"] - b["pearson_r"]
                ),
                "QTL_MAE": a["mae"],
                "GW_MAE": b["mae"],
            }
        )
    pd.DataFrame(records).to_csv(OUT / f"subset{n}" / "paired_repeat_results.csv", index=False)
    valid = all(r["delta_r"] is not None for r in records)
    atomic(
        OUT / f"subset{n}" / "COMPLETE.json",
        {
            "state": "COMPLETE20x5_HISTORICAL_QC_SUBSET_REFITS",
            "subset": n,
            "repeats": 20,
            "mean_QTL_r": float(np.mean([r["QTL_r"] for r in records])) if valid else None,
            "mean_GW_r": float(np.mean([r["GW_r"] for r in records])) if valid else None,
            "mean_delta_r": float(np.mean([r["delta_r"] for r in records])) if valid else None,
            "positive_repeats": sum(r["delta_r"] > 0 for r in records) if valid else None,
            "current_quality_classification_recomputed": False,
            "main151_result_or_panel_changed": False,
            "added_P_tests": False,
        },
    )


def configure_output(path=None):
    """Set output paths for an explicit execution request."""
    global ROOT, OUT
    ROOT = output_root(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT = ROOT / "quality_sensitivity"
    OUT.mkdir(parents=True, exist_ok=True)
