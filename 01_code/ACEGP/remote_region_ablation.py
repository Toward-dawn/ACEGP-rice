"""Evaluate each region conditional on the other fixed QTL regions."""

from pathlib import Path
import json
import sys
import time
import argparse
import hashlib
import fcntl
import subprocess
import traceback
import shutil
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "src"))
from public_paths import DATA_ROOT, RESULT_ROOT, output_root, read_saved_fold

ROOT = None
from acegp_uniform import fit_holdout, stable_seed
import remote_fixed_model_v2 as loader
import statistical_reporting as st

OUT = None


def atomic(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = Path(str(p) + ".partial")
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arrays(c):
    loader.COHORT = c
    return loader.arrays()


def work(c, ri, fi):
    folder = OUT / c / f"r{ri:02d}_f{fi}"
    folder.mkdir(parents=True, exist_ok=True)
    held = (folder / "run.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ids, g, q, groups = arrays(c)
    split = json.loads((DATA_ROOT / "inputs" / f"{c}_20x5.json").read_text())["repeats"][ri - 1][
        "folds"
    ][fi - 1]
    tr = np.array(split["train_rows"])
    te = np.array(split["test_rows"])
    assert ids.sample_id.iloc[te].tolist() == split["test_ids"]
    core = read_saved_fold("repeated_cv", c, ri, fi, split["test_ids"])
    assert core.sample_id.tolist() == split["test_ids"]
    context = stable_seed(20261013, c, ri, fi, "outer_refit")
    config = json.loads((DATA_ROOT / "inputs/model_config.json").read_text())
    config.update(
        blas_threads=2,
        new_cache_directory=str(folder / "cache"),
        cache_max_GiB=4,
        minimum_free_disk_GiB=25,
    )
    for excluded, columns in groups:
        dest = folder / excluded
        dest.mkdir(exist_ok=True)
        if (dest / "done.json").exists():
            old = json.loads((dest / "done.json").read_text())
            assert sha(dest / "predictions.csv") == old["predictions_sha256"]
            continue
        keep = np.setdiff1d(np.arange(q.shape[1]), columns)
        mapping = np.full(q.shape[1], -1, int)
        mapping[keep] = np.arange(len(keep))
        newgroups = [(name, mapping[ix]) for name, ix in groups if name != excluded]
        assert sum(len(ix) for _, ix in newgroups) == len(keep) and all(
            (ix >= 0).all() for _, ix in newgroups
        )
        qm = q[:, keep]
        result = fit_holdout(
            g[tr],
            g[te],
            ids.loss_SF.to_numpy(float)[tr],
            qm[tr],
            qm[te],
            newgroups,
            config,
            context,
        )
        error = float(np.max(np.abs(result["Genome_wide"] - core.Genome_wide.to_numpy(float))))
        assert error <= 1e-8, error
        pd.DataFrame(
            {
                "sample_id": split["test_ids"],
                "row_index": te,
                "repeat": ri,
                "fold": fi,
                "excluded_region": excluded,
                "Genome_wide": result["Genome_wide"],
                "QTL_without_region": result["QTL_informed"],
                "full_QTL": core.QTL_informed.to_numpy(float),
            }
        ).to_csv(dest / "predictions.csv", index=False)
        atomic(dest / "training_choices.json", result["audit"])
        atomic(
            dest / "done.json",
            {
                "state": "PASS_CONDITIONAL_REGION_PRIOR_ABLATION",
                "excluded_region": excluded,
                "raw_unique_QTL_SNPs_removed": len(columns),
                "remaining_QTL_SNPs": len(keep),
                "other_stable_group_ids": [n for n, _ in newgroups],
                "GW_max_absolute_error": error,
                "background_unchanged_including_any_markers_in_excluded_region": True,
                "frozen_main_panel_changed": False,
                "scores_used_to_reselect_main_panel": False,
                "predictions_sha256": sha(dest / "predictions.csv"),
                "choices_sha256": sha(dest / "training_choices.json"),
                "full_same_training_only16_learner_pipeline": True,
                "deterministic_train_only_cache_reused_within_same_fold": True,
            },
        )
    atomic(
        folder / "done.json",
        {
            "state": "COMPLETE_ALL_REGION_PRIOR_ABLATIONS_FOR_FOLD",
            "case_count": len(groups),
            "cases": [n for n, _ in groups],
        },
    )
    cache = (folder / "cache").resolve()
    assert cache.is_relative_to(OUT.resolve()) and not cache.is_symlink()
    if cache.exists():
        shutil.rmtree(cache)


def summarize(c):
    ids, _, _, groups = arrays(c)
    truth = ids.set_index("sample_id").loss_SF
    all_rows = []
    summary = []
    for name, cols in groups:
        frame = pd.concat(
            [
                pd.read_csv(OUT / c / f"r{r:02d}_f{f}" / name / "predictions.csv")
                for r in range(1, 21)
                for f in range(1, 6)
            ]
        )
        case = OUT / c / name
        case.mkdir(exist_ok=True)
        frame.to_csv(case / "all_oof_predictions.csv", index=False)
        records = []
        for ri in range(1, 21):
            d = frame[frame["repeat"] == ri].sort_values("row_index")
            assert d.row_index.tolist() == list(range(len(ids)))
            y = truth.loc[d.sample_id].to_numpy(float)
            a = st.metrics(y, d.full_QTL.to_numpy(float))
            b = st.metrics(y, d.QTL_without_region.to_numpy(float))
            g = st.metrics(y, d.Genome_wide.to_numpy(float))
            difference = (
                None
                if a["pearson_r"] is None or b["pearson_r"] is None
                else a["pearson_r"] - b["pearson_r"]
            )
            records.append(
                {
                    "repeat": ri,
                    "full_r": a["pearson_r"],
                    "without_region_r": b["pearson_r"],
                    "Genome_wide_r": g["pearson_r"],
                    "conditional_contribution_delta_r": difference,
                    "full_MAE": a["mae"],
                    "without_MAE": b["mae"],
                }
            )
        pd.DataFrame(records).to_csv(case / "paired_repeat_results.csv", index=False)
        d = [r["conditional_contribution_delta_r"] for r in records]
        row = {
            "region": name,
            "raw_unique_QTL_SNPs_removed": len(cols),
            "mean_conditional_delta_r": (
                float(np.mean(d)) if all(v is not None for v in d) else None
            ),
            "positive_contribution_repeats": (
                int(np.sum(np.array(d) > 0)) if all(v is not None for v in d) else None
            ),
            "mean_without_r": (
                float(np.mean([r["without_region_r"] for r in records]))
                if all(r["without_region_r"] is not None for r in records)
                else None
            ),
            "main_panel_changed": False,
            "genetic_causal_effect_estimated": False,
        }
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / c / "conditional_region_contributions.csv", index=False)
    atomic(
        OUT / c / "COMPLETE.json",
        {
            "state": "COMPLETE_REGISTERED_CONDITIONAL_REGION_ABLATIONS",
            "region_count": len(groups),
            "repeats": 20,
            "folds": 5,
            "selection_or_main_panel_replacement_permitted": False,
            "new_significance_claims_added": False,
        },
    )


def configure_output(path=None):
    """Set output paths for an explicit execution request."""
    global ROOT, OUT
    ROOT = output_root(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT = ROOT / "region_ablation"
    OUT.mkdir(parents=True, exist_ok=True)
