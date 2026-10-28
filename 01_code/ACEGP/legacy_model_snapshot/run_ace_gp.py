"""Canonical runner for Adaptive Consensus Ensemble genomic prediction.

ACE-GP evaluates candidate predictors within each outer-training partition,
constructs adaptive and conservative screening branches, and averages their
outer-test predictions with fixed equal weights. Trait-informed marker panels
are external inputs and are not embedded in the model definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
METHOD = "ACE-GP"
VERSION = "1.0.0-incremental-preserve-old-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
        "spearman_rho": float(pd.Series(observed).corr(pd.Series(predicted), method="spearman")),
        "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "mae": float(np.mean(np.abs(observed - predicted))),
    }


def build_consensus(work: Path, label: str) -> None:
    source_adaptive = pd.read_csv(work / "adaptive_branch" / label / "selected_oof_predictions.csv")
    source_screened = pd.read_csv(work / "screened_branch" / label / "oof_predictions.csv")
    output = work / "final"
    output.mkdir(parents=True, exist_ok=True)
    keys = ["repeat", "fold", "sample_id"]
    prediction_frames = []
    metric_rows = []
    for condition in ("Genome_only", "Genome_plus_QTL"):
        adaptive = source_adaptive[
            (source_adaptive.condition == condition) & (source_adaptive.model == "AMSE10_RepeatedBlend")
        ]
        screened = source_screened[
            (source_screened.condition == condition)
            & (source_screened.model == "AMSE18_ConservativeAnchorBlend")
        ]
        merged = adaptive[keys + ["observed", "predicted"]].rename(
            columns={"predicted": "adaptive_prediction"}
        ).merge(
            screened[keys + ["observed", "predicted"]].rename(
                columns={"observed": "screened_observed", "predicted": "screened_prediction"}
            ), on=keys, validate="one_to_one",
        )
        if len(merged) != len(adaptive) or len(merged) != len(screened):
            raise RuntimeError(f"Incomplete branch merge for {condition}")
        if not np.allclose(merged.observed, merged.screened_observed, atol=1e-12, rtol=0):
            raise RuntimeError(f"Observed phenotypes disagree for {condition}")
        merged["predicted"] = 0.5 * merged.adaptive_prediction + 0.5 * merged.screened_prediction
        merged["condition"] = condition
        merged["model"] = METHOD
        prediction_frames.append(merged)
        for repeat, group in merged.groupby("repeat"):
            metric_rows.append({"condition": condition, "repeat": int(repeat),
                                **metrics(group.observed, group.predicted)})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    repeat_metrics = pd.DataFrame(metric_rows)
    summary = repeat_metrics.groupby("condition").agg(
        pearson_mean=("pearson_r", "mean"), pearson_sd=("pearson_r", "std"),
        spearman_mean=("spearman_rho", "mean"), rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
    ).reset_index()
    genome = repeat_metrics[repeat_metrics.condition == "Genome_only"].set_index("repeat")
    qtl = repeat_metrics[repeat_metrics.condition == "Genome_plus_QTL"].set_index("repeat")
    deltas = pd.DataFrame({
        "repeat": genome.index,
        "delta_pearson": qtl.pearson_r - genome.pearson_r,
        "delta_rmse": qtl.rmse - genome.rmse,
        "delta_mae": qtl.mae - genome.mae,
    })
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    repeat_metrics.to_csv(output / "repeat_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    deltas.to_csv(output / "repeat_deltas.csv", index=False)


def main(config: Path, output: Path, cohort: str, repeats: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    status.write_text(json.dumps({"state": "RUNNING", "method": METHOD, "version": VERSION}, indent=2),
                      encoding="utf-8")
    base_output = output / "base_predictions"
    screened_output = output / "screened_branch"
    adaptive_output = output / "adaptive_branch"
    base_command = [sys.executable, "run_amse8_qtl_expert.py", "--output", str(base_output),
                    "--repeats", str(repeats), "--cohort", cohort, "--config", str(config)]
    screened_command = [sys.executable, "run_amse18_conservative_anchor_blend.py",
                        "--output", str(screened_output), "--repeats", str(repeats),
                        "--preset", "current", "--config", str(config), "--label", cohort]
    try:
        with (output / "base_predictions.log").open("w", encoding="utf-8") as base_log, \
             (output / "screened_branch.log").open("w", encoding="utf-8") as screened_log:
            base_process = subprocess.Popen(base_command, cwd=HERE, stdout=base_log, stderr=subprocess.STDOUT)
            screened_process = subprocess.Popen(
                screened_command, cwd=HERE, stdout=screened_log, stderr=subprocess.STDOUT
            )
            base_return = base_process.wait()
            screened_return = screened_process.wait()
        if base_return or screened_return:
            raise RuntimeError(f"Branch failure: base={base_return}, screened={screened_return}")
        run([sys.executable, "run_amse10_repeated_blend.py", "--source", str(base_output),
             "--output", str(adaptive_output), "--cohort", cohort, "--config", str(config)],
            output / "adaptive_branch.log")
        build_consensus(output, cohort)
        dependencies = [
            Path(__file__), HERE / "run_amse8_qtl_expert.py", HERE / "run_amse10_repeated_blend.py",
            HERE / "run_amse18_conservative_anchor_blend.py", config,
        ]
        protocol = {
            "method": METHOD, "version": VERSION, "repeats": repeats, "outer_folds": 5,
            "inner_folds": 3, "final_branch_weights": {"adaptive": 0.5, "screened": 0.5},
            "external_prior_embedded_in_model": False,
            "external_prior_supplied_by_config": True,
            "outer_test_phenotypes_used_for_selection_or_weighting": False,
            "dependencies": {str(path): sha256(path) for path in dependencies},
        }
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        status.write_text(json.dumps({"state": "COMPLETE", "method": METHOD, "version": VERSION}, indent=2),
                          encoding="utf-8")
    except Exception as error:
        status.write_text(json.dumps({"state": "FAILED", "error": str(error)}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("pmb", "external"), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    main(args.config, args.output, args.cohort, args.repeats)
