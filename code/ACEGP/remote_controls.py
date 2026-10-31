from pathlib import Path
import os
import sys
import argparse
import json
import time
import traceback
import hashlib
import shutil
import fcntl
import subprocess
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "src"))
from public_paths import DATA_ROOT, RESULT_ROOT, output_root, read_saved_fold

ROOT = None
from acegp_uniform import base, stable_seed, inner_results, prepare, CODE_HASH, threadpool_limits
from acegp_without_block import fit_without_block
import remote_fixed_model_v2 as inputs

MODELS = ["GBLUP_GW", "RKHS_GW", "GBLUP_GW_QTL", "RKHS_GW_QTL", "ACEGP_withoutBlock"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = Path(str(p) + ".partial")
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


def task(cohort, ri, fi):
    inputs.COHORT = cohort
    folder = ROOT / "controls" / cohort / f"r{ri:02d}_f{fi}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "done.json").exists():
        return
    held = (folder / "run.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    ids, g, q, groups = inputs.arrays()
    config = json.loads((DATA_ROOT / "inputs/model_config.json").read_text())
    config.update(
        blas_threads=2,
        new_cache_directory=str(folder / "cache"),
        cache_max_GiB=4,
        minimum_free_disk_GiB=25,
    )
    split = json.loads((DATA_ROOT / "inputs" / f"{cohort}_20x5.json").read_text())["repeats"][
        ri - 1
    ]["folds"][fi - 1]
    tr = np.array(split["train_rows"])
    te = np.array(split["test_rows"])
    assert ids.sample_id.iloc[te].tolist() == split["test_ids"]
    seed = stable_seed(20261013, cohort, ri, fi, "outer_refit")
    y = ids.loss_SF.to_numpy(float)[tr]
    with threadpool_limits(limits=2):
        result = fit_without_block(g[tr], g[te], y, q[tr], q[te], groups, config, seed)
        core = read_saved_fold("repeated_cv", cohort, ri, fi, split["test_ids"])
        assert core.sample_id.tolist() == split["test_ids"]
        err = float(np.max(np.abs(result["Genome_wide"] - core.Genome_wide.to_numpy(float))))
        assert err <= 1e-8, err
        folds = base.folds_for(len(tr), seed + 1001, int(config["inner_folds"]))
        audit = {
            "withoutBlock": result["audit"],
            "GW_max_absolute_recovery_error": err,
            "learner_controls": {},
            "inner_valid_relative_rows": [list(map(int, f)) for f in folds],
        }
        out = pd.DataFrame(
            {
                "sample_id": split["test_ids"],
                "row_index": te,
                "repeat": ri,
                "fold": fi,
                "ACEGP_withoutBlock": result["QTL_informed"],
            }
        )
        conditions = ["Genome_only", "Genome_plus_QTL"] if cohort == "validation" else ["Genome_plus_QTL"]
        for condition in conditions:
            at = q[tr] if condition != "Genome_only" else np.empty((len(tr), 0))
            ax = q[te] if condition != "Genome_only" else np.empty((len(te), 0))
            cache = base.prepare_inner_fold_cache(condition, g[tr], at, folds, config)
            outer = prepare(condition, g[tr], g[te], at, ax, config)
            for number, model in enumerate(["GBLUP", "RKHS"]):
                _, hyper, score = inner_results(
                    model, condition, g[tr], at, y, folds, config, seed + 10000 + number, cache
                )
                label = model + ("_GW" if condition == "Genome_only" else "_GW_QTL")
                out[label] = base.fit_predict_prepared(
                    model, condition, outer, y, hyper, seed + 7000
                )
                audit["learner_controls"][label] = {"hyper": hyper, "training_inner_OOF_r": score}
    if cohort == "development":
        inherited_path = RESULT_ROOT / "model_comparisons" / cohort / "all_oof_predictions.csv"
        inherited = read_saved_fold("model_comparisons", cohort, ri, fi, split["test_ids"])
        assert inherited.sample_id.tolist() == out.sample_id.tolist()
        for label in ["GBLUP_GW", "RKHS_GW"]:
            out[label] = inherited[label].to_numpy()
            audit["learner_controls"][label] = {
                "reused_unchanged_GW_control": True,
                "source": str(inherited_path),
                "source_sha256": sha(inherited_path),
            }
    assert np.isfinite(out[MODELS].to_numpy()).all()
    out.to_csv(folder / "predictions.csv", index=False)
    atomic(folder / "choices.json", audit)
    atomic(
        folder / "done.json",
        {
            "state": "PASS_REGISTERED_CONTROL_FOLD",
            "cohort": cohort,
            "repeat": ri,
            "fold": fi,
            "elapsed_seconds": time.monotonic() - start,
            "predictions_sha256": sha(folder / "predictions.csv"),
            "choices_sha256": sha(folder / "choices.json"),
            "core_model_code_hash": CODE_HASH,
            "registration_sha256": sha(DATA_ROOT / "controls_registration.json"),
            "heldout_phenotype_passed_to_fitters": False,
            "GW_max_abs_error": err,
        },
    )
    cache = (folder / "cache").resolve()
    assert cache.is_relative_to((ROOT / "controls").resolve()) and not cache.is_symlink()
    if cache.exists():
        shutil.rmtree(cache)


def summarize(cohort):
    frames = [
        pd.read_csv(ROOT / "controls" / cohort / f"r{r:02d}_f{f}/predictions.csv")
        for r in range(1, 21)
        for f in range(1, 6)
    ]
    all_predictions = pd.concat(frames)
    all_predictions.to_csv(ROOT / "controls" / cohort / "all_oof_predictions.csv", index=False)
    atomic(
        ROOT / "controls" / cohort / "COMPLETE.json",
        {
            "state": "COMPLETE_20x5_REGISTERED_CONTROLS",
            "folds": 100,
            "models": MODELS,
            "predictions_sha256": sha(ROOT / "controls" / cohort / "all_oof_predictions.csv"),
        },
    )


def configure_output(path=None):
    """Set output paths for an explicit execution request."""
    global ROOT
    ROOT = output_root(path)
    ROOT.mkdir(parents=True, exist_ok=True)
