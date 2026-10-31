"""Repeated-inner-CV selection of a smooth anchor/QTL-block blend."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
AMSE8_PATH = HERE / "run_amse8_qtl_expert.py"
SPEC = importlib.util.spec_from_file_location("amse8", AMSE8_PATH)
amse8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(amse8)
base, block = amse8.base, amse8.block

INNER_REPEATS = 3
STABILITY_PENALTY = 0.25
WEIGHTS = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 1.0)
NEAR_BEST_MARGIN = 0.002


def dump(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def stable(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values) - STABILITY_PENALTY * np.std(values, ddof=0))


def choose_weight(repeated_anchor, repeated_block, repeated_y):
    records = []
    for weight in WEIGHTS:
        scores, rmses = [], []
        for anchor, block_prediction, y in zip(repeated_anchor, repeated_block, repeated_y):
            prediction = (1.0 - weight) * anchor + weight * block_prediction
            scores.append(base.pearson(y, prediction))
            rmses.append(float(np.sqrt(np.mean((y - prediction) ** 2))))
        records.append({"weight": weight, "repeat_r": scores, "repeat_rmse": rmses,
                        "stable_r": stable(scores), "mean_r": float(np.mean(scores)),
                        "mean_rmse": float(np.mean(rmses))})
    best_score = max(row["stable_r"] for row in records)
    eligible = [row for row in records if row["stable_r"] >= best_score - NEAR_BEST_MARGIN]
    chosen = min(eligible, key=lambda row: (row["weight"], row["mean_rmse"]))
    return chosen, records


def evaluate_cohort(cohort, source, output, config_override=None):
    output.mkdir(parents=True, exist_ok=True)
    config_path = config_override or (
        HERE.parent / "amse_gp_multienvironment" / f"config_{cohort}_qtl_block_ard_5x5.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("lambda_grid", [0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    config.setdefault("functional_weight_grid", [0.25, 0.5, 0.75])
    config.setdefault("rkhs_bandwidth_scale_grid", [0.5, 1.0, 2.0])
    ids, genome, panels, panel_markers, y, _, audit = base.load_inputs(config)
    panel_name, qtl = next(iter(panels.items()))
    names, indices = block.qtl_groups(
        panel_markers[panel_name], config["qtl_loci_path"], config.get("qtl_marker_coordinate_path")
    )
    groups = list(zip(names, indices))
    source_predictions = pd.read_csv(source / "oof_predictions.csv")
    repeats = int(source_predictions.repeat.max())
    dump(output / "protocol.json", {
        "method": "AMSE10_RepeatedBlend", "cohort": cohort,
        "source_amse8": str(source), "source_oof_sha256": base.sha256(source / "oof_predictions.csv"),
        "input_audit": audit, "inner_repeats": INNER_REPEATS,
        "candidate_block_weights": list(WEIGHTS),
        "selection_score": "mean inner r - 0.25 SD",
        "near_best_rule": f"smallest weight within {NEAR_BEST_MARGIN} of best stable score",
        "outer_predictions_reused_without_refitting": True,
        "interpretation": "exploratory; previously examined cohorts",
    })
    selected_rows = []
    total = len(config.get("analysis_repeat_ids", range(1, repeats + 1))) * len(config.get("analysis_fold_ids") or range(1, int(config["outer_folds"]) + 1))
    completed, started = 0, time.time()
    for repeat in config.get("analysis_repeat_ids", range(1, repeats + 1)):
        outer_folds = base.folds_for(len(y), int(config["seed"]) + repeat, int(config["outer_folds"]))
        for fold, test in enumerate(outer_folds, 1):
            if config.get("analysis_fold_ids") and fold not in config["analysis_fold_ids"]:
                continue
            fold_started = time.time()
            train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
            y_train = y[train]
            repeated = []
            for inner_repeat in range(1, INNER_REPEATS + 1):
                inner_seed = int(config["seed"]) + repeat * 1000 + fold + inner_repeat * 100000
                inner = base.folds_for(len(train), inner_seed, int(config["inner_folds"]))
                model_seed = int(config["seed"]) + repeat * 10000 + fold * 100 + inner_repeat * 1000000
                condition_predictions = {}
                for condition, addon in (("Genome_only", np.zeros((len(y), 1))), ("Genome_plus_QTL", qtl)):
                    cache = base.prepare_inner_fold_cache(condition, genome[train], addon[train], inner, config)
                    for number, model in enumerate(("GBLUP", "RKHS")):
                        prediction, _, score = base.inner_oof(
                            model, condition, genome[train], addon[train], y_train,
                            inner, config, model_seed + number, fold_cache=cache,
                        )
                        condition_predictions[(condition, model)] = (prediction, score)
                block_prediction, _, _ = amse8.block_inner_oof(
                    y_train, genome[train], qtl[train], groups, inner, config, model_seed + 5000
                )
                repeated.append({"inner_repeat": inner_repeat, "folds": inner,
                    "predictions": condition_predictions, "block": block_prediction})

            genome_stable = {
                model: stable([item["predictions"][("Genome_only", model)][1] for item in repeated])
                for model in ("GBLUP", "RKHS")
            }
            anchor = max(("GBLUP", "RKHS"), key=lambda model: (genome_stable[model], model))
            repeated_anchor = [item["predictions"][("Genome_plus_QTL", anchor)][0] for item in repeated]
            repeated_block = [item["block"] for item in repeated]
            chosen, weight_records = choose_weight(
                repeated_anchor, repeated_block, [y_train] * INNER_REPEATS
            )
            weight = float(chosen["weight"])

            fold_source = source_predictions[
                (source_predictions.repeat == repeat) & (source_predictions.fold == fold)
            ]
            genome_source = fold_source[
                (fold_source.condition == "Genome_only") & (fold_source.model == anchor)
            ].copy()
            anchor_qtl = fold_source[
                (fold_source.condition == "Genome_plus_QTL") & (fold_source.model == anchor)
            ].sort_values("sample_id").reset_index(drop=True)
            block_qtl = fold_source[
                (fold_source.condition == "Genome_plus_QTL") & (fold_source.model == "QTL_Block_ARD")
            ].sort_values("sample_id").reset_index(drop=True)
            if len(genome_source) != len(test) or len(anchor_qtl) != len(test) or len(block_qtl) != len(test):
                raise RuntimeError("Missing AMSE8 outer predictions")
            if not anchor_qtl.sample_id.equals(block_qtl.sample_id):
                raise RuntimeError("Outer prediction rows are misaligned")
            qtl_selected = anchor_qtl.copy()
            qtl_selected["predicted"] = (
                (1.0 - weight) * anchor_qtl.predicted.to_numpy()
                + weight * block_qtl.predicted.to_numpy()
            )
            genome_source["model"] = "AMSE10_RepeatedBlend"
            genome_source["block_weight"] = 0.0
            genome_source["selected_anchor"] = anchor
            qtl_selected["model"] = "AMSE10_RepeatedBlend"
            qtl_selected["block_weight"] = weight
            qtl_selected["selected_anchor"] = anchor
            selected_rows.extend([genome_source, qtl_selected])
            dump(output / f"selection_r{repeat}_f{fold}.json", {
                "repeat": repeat, "fold": fold, "genome_stable_scores": genome_stable,
                "selected_anchor": anchor, "selected_block_weight": weight,
                "selected_weight_record": chosen, "all_weight_records": weight_records,
            })
            completed += 1
            elapsed = time.time() - started
            dump(output / "status.json", {"state": "RUNNING", "completed_folds": completed,
                "total_folds": total, "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": elapsed / completed * (total - completed)})
            print(f"{cohort} {completed}/{total} anchor={anchor} block_weight={weight:.2f} seconds={time.time()-fold_started:.1f}", flush=True)

    selected = pd.concat(selected_rows, ignore_index=True)
    comparison = pd.concat([
        source_predictions[source_predictions.model.isin({"GBLUP", "RKHS", "QTL_Block_ARD"})], selected
    ], ignore_index=True)
    metrics = []
    for (condition, model, repeat), group in comparison.groupby(["condition", "model", "repeat"]):
        if len(group) == expected_oof_count(config, base, len(ids), repeat):
            metrics.append({"condition": condition, "model": model, "repeat": repeat,
                            **base.metrics(group.observed.to_numpy(), group.predicted.to_numpy())})
    metric_frame = pd.DataFrame(metrics)
    selected.to_csv(output / "selected_oof_predictions.csv", index=False)
    metric_frame.to_csv(output / "repeat_metrics.csv", index=False)
    metric_frame.groupby(["condition", "model"]).agg(
        pearson_mean=("pearson_r", "mean"), pearson_sd=("pearson_r", "std"),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean")
    ).reset_index().to_csv(output / "summary.csv", index=False)
    dump(output / "status.json", {"state": "COMPLETE", "completed_folds": total,
                                   "elapsed_seconds": time.time() - started})


def self_test():
    y = np.arange(10, dtype=float)
    anchor = y + np.array([1, -1] * 5)
    block_prediction = y + np.array([0.1, -0.1] * 5)
    chosen, records = choose_weight([anchor] * 3, [block_prediction] * 3, [y] * 3)
    assert chosen["weight"] > 0 and len(records) == len(WEIGHTS)
    print(f"PASS selected_weight={chosen['weight']}", flush=True)


# BOUNDARY_RUNTIME_V1
from runtime_support import install, expected_oof_count
install(base, block)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--cohort", choices=("pmb", "external", "both"), default="both")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        dump(args.output / "job_status.json", {"state": "RUNNING", "pid": os.getpid(),
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        cohorts = ("external", "pmb") if args.cohort == "both" else (args.cohort,)
        try:
            with threadpool_limits(limits=2):
                for cohort in cohorts:
                    evaluate_cohort(cohort, args.source / cohort, args.output / cohort, args.config)
            dump(args.output / "job_status.json", {"state": "COMPLETE",
                                                   "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        except Exception:
            dump(args.output / "job_status.json", {"state": "FAILED", "traceback": traceback.format_exc()})
            raise
