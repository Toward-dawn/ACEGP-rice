from pathlib import Path
import json
import time
import sys
import hashlib
import fcntl
import traceback
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "src"))
from public_paths import DATA_ROOT, RESULT_ROOT, output_root, read_saved_fold

ROOT = None
import statistical_reporting as st
import reporting_wrappers as wr

OUT = None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(p, d):
    t = Path(str(p) + ".partial")
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False))
    t.replace(p)


def pair_test(cohort, first, second, frame, description):
    ids = pd.read_csv(
        DATA_ROOT / "inputs" / f"{cohort}_samples.csv", dtype={"sample_id": str}
    ).set_index("sample_id")
    schedule = json.loads((DATA_ROOT / "inputs" / f"{cohort}_20x5.json").read_text())
    q = []
    g = []
    tr = []
    te = []
    for ri in range(1, 21):
        for fi in range(1, 6):
            d = frame[(frame["repeat"] == ri) & (frame["fold"] == fi)]
            s = schedule["repeats"][ri - 1]["folds"][fi - 1]
            assert d.sample_id.tolist() == s["test_ids"]
            y = ids.loc[d.sample_id].loss_SF.to_numpy(float)
            q.append(st.metrics(y, d[first].to_numpy(float))["pearson_r"])
            g.append(st.metrics(y, d[second].to_numpy(float))["pearson_r"])
            tr.append(len(s["train_rows"]))
            te.append(len(s["test_rows"]))
    result = wr.guarded_fisher_comparison(
        [np.nan if v is None else v for v in q], [np.nan if v is None else v for v in g], tr, te
    )
    result.update(description=description, selected_fixed_panel_bias_removed=False)
    repeat = []
    for ri in range(1, 21):
        d = frame[frame["repeat"] == ri].sort_values("row_index")
        assert len(d) == len(ids) and d.sample_id.nunique() == len(ids)
        y = ids.loc[d.sample_id].loss_SF.to_numpy(float)
        repeat.append(
            {
                "repeat": ri,
                first: st.metrics(y, d[first].to_numpy(float)),
                second: st.metrics(y, d[second].to_numpy(float)),
            }
        )
    return {"test": result, "repeat_metrics": repeat}


def models(cohort):
    core_report = json.loads((OUT / (cohort + ".json")).read_text())
    assert core_report["state"] == "PASS_100_FOLD_PREDICTION_AUDIT_AND_REGISTERED_STATISTICS"
    source = RESULT_ROOT / "model_comparisons" / cohort / "all_oof_predictions.csv"
    accepted = json.loads((source.parent / "COMPLETE.json").read_text())
    assert sha(source) == accepted["predictions_sha256"]
    core = pd.read_csv(
        RESULT_ROOT / "repeated_cv" / cohort / "all_oof_predictions.csv", dtype={"sample_id": str}
    )
    extra = pd.read_csv(source, dtype={"sample_id": str})
    keys = ["sample_id", "row_index", "repeat", "fold"]
    frame = core.merge(extra, on=keys, validate="one_to_one", sort=False)
    assert len(frame) == 20 * {"development": 241, "validation": 151}[cohort]
    result = {}
    for c in ["GBLUP_GW", "RKHS_GW", "GBLUP_GW_QTL", "RKHS_GW_QTL", "ACEGP_withoutBlock"]:
        result[cohort + "_QTL_ACEGP_vs_" + c + "_Fisher_z"] = pair_test(
            cohort, "QTL_informed", c, frame, "FullACEGP QTL vs " + c
        )
    result[cohort + "_GBLUP_GW_QTL_vs_GBLUP_GW_Fisher_z"] = pair_test(
        cohort,
        "GBLUP_GW_QTL",
        "GBLUP_GW",
        frame,
        "Regionkernel incremental contribution under simple GBLUP",
    )
    result[cohort + "_QTL_vs_GW_MAE"] = {"test": core_report["MAE"]}
    atomic(
        OUT / (cohort + "_model_comparisons.json"),
        {"source_control_sha256": sha(source), "comparisons": result},
    )
    return result


def dense():
    c = "development"
    source = RESULT_ROOT / "dense_gblup/all_oof_predictions.csv"
    accepted = json.loads((source.parent / "COMPLETE.json").read_text())
    assert sha(source) == accepted["predictions_sha256"]
    core = pd.read_csv(
        RESULT_ROOT / "repeated_cv" / c / "all_oof_predictions.csv", dtype={"sample_id": str}
    )
    extra = pd.read_csv(source, dtype={"sample_id": str})
    frame = core.merge(
        extra, on=["sample_id", "row_index", "repeat", "fold"], validate="one_to_one", sort=False
    )
    result = pair_test(
        c,
        "QTL_informed",
        "dense_GBLUP",
        frame,
        "Sparse-background QTL ACEGP vs4817964-marker dense GBLUP",
    )
    atomic(OUT / "dense_model_comparison.json", result)
    return {"development_QTL_ACEGP_vs_dense_GBLUP_Fisher_z": result}


def configure_output(path=None):
    """Set output paths for an explicit execution request."""
    global ROOT, OUT
    ROOT = output_root(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT = ROOT / "statistics"
    OUT.mkdir(parents=True, exist_ok=True)
