"""Validate task outputs against the archived study and combine complete folds."""

import csv
import math
from pathlib import Path

from public_execution import read_json, task_folder, write_json

# Absolute tolerance in predicted fertility-loss percentage points, not in r.
PREDICTION_TOLERANCE_PP = 1e-8
MODEL_COLUMNS = {
    "prediction": ["Genome_wide", "QTL_informed"],
    "controls": ["ACEGP_withoutBlock", "GBLUP_GW_QTL", "RKHS_GW_QTL", "GBLUP_GW", "RKHS_GW"],
    "quality": ["Genome_wide", "QTL_informed"],
    "genetic-group": ["Genome_wide", "QTL_informed"],
    "ablation": ["Genome_wide", "QTL_without_region", "full_QTL"],
    "primary-random": ["Genome_wide", "random_QTL"],
    "validation-random": ["Genome_wide", "candidate_QTL"],
}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def archived_predictions(package, args, cohort, task, region=None):
    results = package / "02_data/prediction_results"
    if args.analysis == "prediction":
        folder = results / "repeated_cv" / cohort
    elif args.analysis == "controls":
        folder = results / "model_comparisons" / cohort
    elif args.analysis == "quality":
        folder = results / "genotype_quality_subsets" / f"subset{args.subset}"
    elif args.analysis == "genetic-group":
        return results / "genetic_group_cv" / cohort / "leave_group_oof_predictions.csv"
    elif args.analysis == "ablation":
        folder = results / "region_ablation" / cohort / region
    else:
        population = "primary" if args.analysis == "primary-random" else "validation"
        name = (
            f"panel_{task['panel']:03d}"
            if population == "primary"
            else ("real7" if task["panel"] == 0 else f"random_{task['panel']:03d}")
        )
        folder = package / "02_data/random_controls" / population / "results" / name
    return folder / "all_oof_predictions.csv"


def compare_predictions(actual, reference, task, models):
    """Check every held-out row and model; never substitute saved predictions."""
    keys = ["group_id"] if "group" in task else ["repeat", "fold"]
    selected = [
        row
        for row in reference
        if all(int(row[key]) == task["group" if key == "group_id" else key] for key in keys)
    ]
    if not selected or len(actual) != len(selected):
        raise ValueError(f"Incomplete or duplicated predictions: {task['key']}")
    row_column = "original_row" if "original_row" in selected[0] else "row_index"
    actual = sorted(actual, key=lambda row: int(row[row_column]))
    selected = sorted(selected, key=lambda row: int(row[row_column]))
    if len({row["sample_id"] for row in actual}) != len(actual):
        raise ValueError(f"Duplicate sample IDs: {task['key']}")
    maximum = 0.0
    for observed, expected in zip(actual, selected):
        identity = ["sample_id", row_column, *keys]
        identity += [key for key in ("panel", "excluded_region") if key in expected]
        if any(observed[key] != expected[key] for key in identity):
            raise ValueError(f"Sample, partition or region mismatch: {task['key']}")
        for model in models:
            value, previous = float(observed[model]), float(expected[model])
            if not (math.isfinite(value) and math.isfinite(previous)):
                raise ValueError(f"Non-finite prediction: {task['key']} / {model}")
            difference = abs(value - previous)
            maximum = max(maximum, difference)
            if difference > PREDICTION_TOLERANCE_PP:
                raise ValueError(
                    f"Prediction differs from archived study: {task['key']} / {model}; absolute difference {difference:.12g} pp"
                )
    return maximum


def validate_predictions(package, args, cohort, task, folder):
    if args.analysis == "statistics":
        return
    if args.analysis == "ablation":
        regions = [
            row["qid"]
            for row in read_rows(package / f"02_data/prediction_inputs/inputs/{cohort}_regions.csv")
        ]
    else:
        regions = [None]
    for region in regions:
        source = (folder / region if region else folder) / "predictions.csv"
        archive = archived_predictions(package, args, cohort, task, region)
        compare_predictions(
            read_rows(source), read_rows(archive), task, MODEL_COLUMNS[args.analysis]
        )


def summarize_predictions(package, output, args, cohort, tasks):
    """Compute metrics on complete OOF vectors, never means of fold correlations."""
    import numpy as np
    import pandas as pd
    from statistical_reporting import metrics

    ids = pd.read_csv(
        package / f"02_data/prediction_inputs/inputs/{cohort}_samples.csv", dtype={"sample_id": str}
    )
    truth = ids.set_index("sample_id").loss_SF
    panels = sorted({task.get("panel", -1) for task in tasks})
    panel_rows = []
    for panel in panels:
        selected = [task for task in tasks if task.get("panel", -1) == panel]
        frames = [
            pd.read_csv(
                task_folder(output, args.analysis, cohort, args.subset, task) / "predictions.csv",
                dtype={"sample_id": str},
            )
            for task in selected
        ]
        frame = pd.concat(frames, ignore_index=True).sort_values(["repeat", "row_index"])
        expected_repeats = range(1, 21) if args.analysis == "prediction" else [1]
        if sorted(frame["repeat"].unique()) != list(expected_repeats):
            raise ValueError("Summary requires every registered repeat.")
        records = []
        for repeat in expected_repeats:
            block = frame[frame["repeat"] == repeat]
            if (
                block.row_index.tolist() != list(range(len(ids)))
                or block.sample_id.tolist() != ids.sample_id.tolist()
            ):
                raise ValueError("OOF rows do not cover every material exactly once.")
            y = truth.loc[block.sample_id].to_numpy(float)
            measured = {
                model: metrics(y, block[model].to_numpy(float))
                for model in MODEL_COLUMNS[args.analysis]
            }
            genome, qtl = MODEL_COLUMNS[args.analysis]
            row = {"repeat": repeat}
            for model, values in measured.items():
                row.update({f"{model}_{key}": value for key, value in values.items()})
            a, b = measured[genome]["pearson_r"], measured[qtl]["pearson_r"]
            row["delta_r"] = None if a is None or b is None else b - a
            records.append(row)
        destination = task_folder(output, args.analysis, cohort, args.subset, selected[0]).parent
        frame.to_csv(destination / "all_oof_predictions.csv", index=False)
        pd.DataFrame(records).to_csv(destination / "paired_repeat_results.csv", index=False)
        if args.analysis == "prediction":
            summary = {"repeats": 20, "folds_per_repeat": 5, "models": {}}
            for model in MODEL_COLUMNS[args.analysis]:
                values = [row[f"{model}_pearson_r"] for row in records]
                summary["models"][model] = {
                    "mean_r": None if any(v is None for v in values) else float(np.mean(values)),
                    "sample_SD_r": (
                        None if any(v is None for v in values) else float(np.std(values, ddof=1))
                    ),
                }
            gains = [row["delta_r"] for row in records]
            summary["mean_delta_r"] = (
                None if any(v is None for v in gains) else float(np.mean(gains))
            )
            summary["positive_gain_repeats"] = (
                None if any(v is None for v in gains) else sum(v > 0 for v in gains)
            )
            write_json(destination / "SUMMARY.json", summary)
        else:
            panel_rows.append({"panel": panel, **records[0]})
    if panel_rows:
        base = task_folder(output, args.analysis, cohort, args.subset, tasks[0]).parent.parent
        # A per-panel summary must not overwrite a previous all-panel summary.
        name = "all_panel_metrics.csv" if args.all_panels else f"panel_{panels[0]:03d}_metrics.csv"
        pd.DataFrame(panel_rows).to_csv(base / name, index=False)
