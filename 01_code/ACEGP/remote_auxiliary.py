"""Cross-validation by genetic group using the frozen genotype-derived labels."""

from pathlib import Path
import os
import sys
import json
import time
import csv
import hashlib
import argparse
import subprocess
import traceback
import fcntl
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "src"))
from public_paths import DATA_ROOT, RESULT_ROOT, output_root, read_saved_fold

ROOT = None
from acegp_uniform import fit_holdout, stable_seed, qc, pcs, threadpool_limits
import remote_fixed_model_v2 as loader
import statistical_reporting as st

OUT = None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = Path(str(p) + ".partial")
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


def config(folder):
    c = json.loads((DATA_ROOT / "inputs/model_config.json").read_text())
    c.update(
        blas_threads=2,
        new_cache_directory=str(folder / "cache"),
        cache_max_GiB=4,
        minimum_free_disk_GiB=25,
    )
    return c


def arrays(cohort):
    loader.COHORT = cohort
    return loader.arrays()


def groups(cohort):
    folder = RESULT_ROOT / "genetic_group_cv" / cohort
    ids, _, _, _ = arrays(cohort)
    p = folder / "genetic_groups.csv"
    reg = json.loads((folder / "genetic_groups_registration.json").read_text())
    assert sha(p) == reg["labels_sha256"]
    d = pd.read_csv(p, dtype={"sample_id": str})
    assert d.sample_id.tolist() == ids.sample_id.tolist()
    return d.group_id.to_numpy(int)


def group_task(cohort, k):
    folder = OUT / cohort / f"group{k}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "done.json").exists():
        return
    held = (folder / "run.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ids, g, q, blocks = arrays(cohort)
    labels = groups(cohort)
    tr = np.flatnonzero(labels != k)
    te = np.flatnonzero(labels == k)
    assert len(tr) >= 20 and len(te) > 0
    result = fit_holdout(
        g[tr],
        g[te],
        ids.loss_SF.to_numpy(float)[tr],
        q[tr],
        q[te],
        blocks,
        config(folder),
        stable_seed(20261013, cohort, "leave_genetic_group", k, "outer_refit"),
    )
    frame = pd.DataFrame(
        {
            "sample_id": ids.sample_id.iloc[te],
            "row_index": te,
            "group_id": k,
            "Genome_wide": result["Genome_wide"],
            "QTL_informed": result["QTL_informed"],
        }
    )
    frame.to_csv(folder / "predictions.csv", index=False)
    atomic(folder / "training_choices.json", result["audit"])
    atomic(
        folder / "done.json",
        {
            "state": "PASS_LEAVE_GROUP_FOLD",
            "n_train": len(tr),
            "n_test": len(te),
            "predictions_sha256": sha(folder / "predictions.csv"),
            "training_choices_sha256": sha(folder / "training_choices.json"),
            "heldout_group_phenotype_passed_to_fitter": False,
        },
    )


def group_summary(cohort):
    ids, _, _, _ = arrays(cohort)
    frame = pd.concat(
        [pd.read_csv(OUT / cohort / f"group{k}/predictions.csv") for k in range(1, 6)]
    ).sort_values("row_index")
    assert (
        frame.row_index.tolist() == list(range(len(ids)))
        and frame.sample_id.tolist() == ids.sample_id.tolist()
    )
    frame.to_csv(OUT / cohort / "leave_group_oof_predictions.csv", index=False)
    atomic(
        OUT / cohort / "COMPLETE.json",
        {
            "state": "COMPLETE_FIVE_GROUP_STRESS_TEST",
            "population": cohort,
            "pooled_OOF_metrics": {
                c: st.metrics(ids.loss_SF.to_numpy(float), frame[c].to_numpy(float))
                for c in ["Genome_wide", "QTL_informed"]
            },
            "independent_environments_or_biological_repeats_added": False,
            "selection_bias_removed": False,
        },
    )


def configure_output(path=None):
    """Set output paths for an explicit execution request."""
    global ROOT, OUT
    ROOT = output_root(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT = ROOT / "auxiliary"
    OUT.mkdir(parents=True, exist_ok=True)
