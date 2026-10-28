"""Locate packaged inputs and the selected output directory."""

from pathlib import Path
import json
import os

PACKAGE = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE / "02_data/prediction_inputs"
RESULT_ROOT = PACKAGE / "02_data/prediction_results"
RANDOM_ROOT = PACKAGE / "02_data/random_controls"
CODE_ROOT = Path(__file__).resolve().parent


def output_root(path=None):
    value = path if path is not None else os.environ.get("ACEGP_RUN_ROOT")
    if not value:
        raise RuntimeError("Use run_analysis.py with an explicit output directory.")
    root = Path(value).resolve()
    if root == PACKAGE or root.is_relative_to(PACKAGE):
        raise RuntimeError("Outputs must be outside the archived public package.")
    return root


def read_saved_fold(section, cohort, repeat, fold, order=None):
    import pandas as pd

    path = RESULT_ROOT / section / cohort / "all_oof_predictions.csv"
    frame = pd.read_csv(path, dtype={"sample_id": str})
    frame = frame[(frame["repeat"] == repeat) & (frame["fold"] == fold)].copy()
    if order is not None:
        assert frame.sample_id.is_unique and set(frame.sample_id) == set(order)
        frame = frame.set_index("sample_id", drop=False).loc[order].reset_index(drop=True)
    return frame


def read_validation_real_fold(fold, order):
    import pandas as pd

    path = RANDOM_ROOT / "validation/results/real7/all_oof_predictions.csv"
    frame = pd.read_csv(path, dtype={"sample_id": str})
    frame = (
        frame[frame["fold"] == fold]
        .set_index("sample_id", drop=False)
        .loc[order]
        .reset_index(drop=True)
    )
    return frame
