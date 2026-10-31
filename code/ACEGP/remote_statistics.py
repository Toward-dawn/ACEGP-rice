from pathlib import Path
import sys
import json
import hashlib
import time
import traceback
import fcntl
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "src"))
from public_paths import DATA_ROOT, RESULT_ROOT, output_root, read_saved_fold

ROOT = None
import statistical_reporting as st
import reporting_wrappers as wr

OUT = None
CHANNELS = ["Genome_wide", "QTL_informed"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(p, d):
    t = Path(str(p) + ".partial")
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


def analyze(cohort):
    directory = RESULT_ROOT / "repeated_cv" / cohort
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["state"] == "COMPLETE_20x5_FIXED_PANEL"
    samples = pd.read_csv(DATA_ROOT / "inputs" / f"{cohort}_samples.csv", dtype={"sample_id": str})
    y = samples.loss_SF.to_numpy(float)
    ids = samples.sample_id.to_numpy(str)
    splits = json.loads((DATA_ROOT / "inputs" / f"{cohort}_20x5.json").read_text())
    folds = []
    frames = []
    source = {}
    for ri in range(1, 21):
        for fi in range(1, 6):
            p = directory / "all_oof_predictions.csv"
            item = splits["repeats"][ri - 1]["folds"][fi - 1]
            te = np.array(item["test_rows"], int)
            frame = read_saved_fold("repeated_cv", cohort, ri, fi, item["test_ids"])
            assert (
                frame.row_index.tolist() == te.tolist()
                and frame.sample_id.tolist() == item["test_ids"]
            )
            metrics = {c: st.metrics(y[te], frame[c].to_numpy(float)) for c in CHANNELS}
            folds.append(
                {
                    "repeat": ri,
                    "fold": fi,
                    "n_train": len(item["train_rows"]),
                    "n_test": len(te),
                    "metrics": metrics,
                }
            )
            frames.append(frame)
            source[str(p.relative_to(RESULT_ROOT))] = sha(p)
    full = pd.concat(frames)
    records = []
    selection = []
    calibration = []
    for ri in range(1, 21):
        d = full[full["repeat"] == ri].sort_values("row_index")
        assert d.row_index.tolist() == list(range(len(y))) and d.sample_id.tolist() == ids.tolist()
        metrics = {c: st.metrics(y, d[c].to_numpy(float)) for c in CHANNELS}
        qr, gr = metrics["QTL_informed"]["pearson_r"], metrics["Genome_wide"]["pearson_r"]
        records.append(
            {
                "repeat": ri,
                "metrics": metrics,
                "delta_r": None if qr is None or gr is None else qr - gr,
            }
        )
        for c in CHANNELS:
            calibration.append(
                {"repeat": ri, "model": c, **st.calibration(y, d[c].to_numpy(float))}
            )
        for fraction in [0.10, 0.20]:
            selection.append(
                {
                    "repeat": ri,
                    **wr.selection_with_tie_audit(
                        y,
                        d.QTL_informed.to_numpy(float),
                        d.Genome_wide.to_numpy(float),
                        ids,
                        fraction,
                    ),
                }
            )
    tr = [r["n_train"] for r in folds]
    te = [r["n_test"] for r in folds]
    q = [r["metrics"]["QTL_informed"]["pearson_r"] for r in folds]
    g = [r["metrics"]["Genome_wide"]["pearson_r"] for r in folds]
    primary = wr.guarded_fisher_comparison(
        [np.nan if v is None else v for v in q], [np.nan if v is None else v for v in g], tr, te
    )
    primary.update(
        cohort=cohort,
        confidence_interval_units="Fisher_z_difference",
        historical_selection_bias_removed=False,
        fold_count=100,
        ordinary_t_on20_repeat_scores=False,
    )
    mae = [r["metrics"]["Genome_wide"]["mae"] - r["metrics"]["QTL_informed"]["mae"] for r in folds]
    secondary = st.corrected_paired_test(mae, tr, te)
    if secondary["status"] == "ZERO_VARIANCE":
        secondary.update(p_value=None, CI=None)
    secondary.update(estimand="paired_outer_fold_MAE_GW_minus_QTL", units="percentage_points")
    means = {}
    for c in CHANNELS:
        means[c] = {}
        for metric in ["pearson_r", "spearman_rho", "rmse", "mae"]:
            a = [r["metrics"][c][metric] for r in records]
            means[c][metric] = (
                {
                    "mean": float(np.mean(a)),
                    "SD": float(np.std(a, ddof=1)),
                    "min": float(np.min(a)),
                    "max": float(np.max(a)),
                }
                if all(v is not None for v in a)
                else {"state": "UNDEFINED_REGISTERED_REPEAT_NO_DROPPING"}
            )
    dr = [r["delta_r"] for r in records]
    assert all(v is not None for v in dr)
    observed = {
        "mean_delta_r": float(np.mean(dr)),
        "SD_delta_r": float(np.std(dr, ddof=1)),
        "positive_repeats": int(np.sum(np.array(dr) > 0)),
        "relative_gain_percent": (
            100 * float(np.mean(dr)) / means["Genome_wide"]["pearson_r"]["mean"]
            if means["Genome_wide"]["pearson_r"]["mean"] > 0
            else None
        ),
    }
    assert abs(observed["mean_delta_r"] - summary["mean_delta_r"]) < 1e-12
    report = {
        "state": "PASS_100_FOLD_PREDICTION_AUDIT_AND_REGISTERED_STATISTICS",
        "cohort": cohort,
        "samples": len(y),
        "repeats": 20,
        "folds_per_repeat": 5,
        "models": means,
        "improvement": observed,
        "primary": primary,
        "MAE": secondary,
        "fold_records": folds,
        "repeat_records": records,
        "calibration": calibration,
        "selection": selection,
        "bootstrap": "DEFERRED_UNTIL_MATERIAL_DEPENDENCE_UNITS_JUSTIFIED",
        "warnings": [
            "These20CVrepeats are not independent biological replicates.",
            "Fixed panel was historically selected; nominal tests do not remove selection bias.",
            "Shared-fit branch predictions are diagnostic, not independently refitted branch controls.",
        ],
        "source_files": source,
        "phenotypes_sha256": sha(DATA_ROOT / "inputs" / f"{cohort}_samples.csv"),
        "splits_sha256": sha(DATA_ROOT / "inputs" / f"{cohort}_20x5.json"),
    }
    atomic(OUT / (cohort + ".json"), report)
    pd.DataFrame(calibration).to_csv(OUT / (cohort + "_calibration.csv"), index=False)
    return report


def complete(test):
    return {
        "status": (
            "COMPLETE"
            if test["status"] == "OK" and test.get("p_value") is not None
            else test["status"]
        ),
        "p_value": test.get("p_value"),
    }


def configure_output(path=None):
    """Set output paths for an explicit execution request."""
    global ROOT, OUT
    ROOT = output_root(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT = ROOT / "statistics"
    OUT.mkdir(parents=True, exist_ok=True)
