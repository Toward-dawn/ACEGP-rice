"""Full-pool adaptive anchor selection followed by guarded QTL-expert blending.

AMSE11 retains the prespecified 16-learner pool from AMSE-GP. Within every
outer-training partition, all learners compete on the genome-wide baseline in
repeated inner cross-validation. The most stable learner becomes the anchor.
The same anchor is then evaluated with the QTL representation and blended with
QTL Block-ARD using a weight selected entirely within the outer-training data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "run_scheme2_tuned.py"
BLOCK_PATH = HERE / "run_qtl_block_ard.py"
if not BLOCK_PATH.exists():
    BLOCK_PATH = HERE.parent / "amse_gp_multienvironment" / "run_qtl_block_ard.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("amse11_base", BASE_PATH)
block = load_module("amse11_block", BLOCK_PATH)

CANDIDATE_MODELS = (
    "GBLUP", "RKHS", "Ridge", "ElasticNet", "LASSO", "BayesianRidge",
    "PLS", "SVR_RBF", "KNN", "ExtraTrees", "RandomForest",
    "GradientBoosting", "HistGradientBoosting", "AdaBoost", "MLP",
    "CatBoost_D4",
)
INNER_REPEATS = 3
STABILITY_PENALTY = 0.25
BLOCK_WEIGHTS = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 1.0)
NEAR_BEST_MARGIN = 0.002


def dump(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def stable(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values) - STABILITY_PENALTY * np.std(values, ddof=0))


def choose_anchor(score_by_model):
    records = []
    for order, model in enumerate(CANDIDATE_MODELS):
        scores = [float(value) for value in score_by_model[model]]
        records.append({
            "model": model,
            "repeat_r": scores,
            "mean_r": float(np.mean(scores)),
            "sd_r": float(np.std(scores, ddof=0)),
            "stable_r": stable(scores),
            "prespecified_order": order,
        })
    records.sort(key=lambda row: (-row["stable_r"], row["prespecified_order"]))
    return records[0]["model"], records


def choose_weight(repeated_anchor, repeated_block, repeated_y):
    records = []
    for weight in BLOCK_WEIGHTS:
        scores, rmses = [], []
        for anchor, block_prediction, y in zip(
            repeated_anchor, repeated_block, repeated_y
        ):
            prediction = (1.0 - weight) * anchor + weight * block_prediction
            scores.append(base.pearson(y, prediction))
            rmses.append(float(np.sqrt(np.mean((y - prediction) ** 2))))
        records.append({
            "weight": weight,
            "repeat_r": scores,
            "repeat_rmse": rmses,
            "mean_r": float(np.mean(scores)),
            "sd_r": float(np.std(scores, ddof=0)),
            "stable_r": stable(scores),
            "mean_rmse": float(np.mean(rmses)),
        })
    best_score = max(row["stable_r"] for row in records)
    eligible = [
        row for row in records
        if row["stable_r"] >= best_score - NEAR_BEST_MARGIN
    ]
    chosen = min(eligible, key=lambda row: (row["weight"], row["mean_rmse"]))
    return chosen, records


def modal_hyper(hypers, scores):
    """Choose the most frequent inner-selected setting, then its best mean score."""
    keys = [json.dumps(item, sort_keys=True) for item in hypers]
    counts = Counter(keys)
    top_count = max(counts.values())
    candidates = [key for key, count in counts.items() if count == top_count]
    mean_scores = {
        key: float(np.mean([score for k, score in zip(keys, scores) if k == key]))
        for key in candidates
    }
    selected = max(candidates, key=lambda key: (mean_scores[key], key))
    return json.loads(selected), {
        "selection": "modal inner-selected hyperparameter; ties by mean inner r",
        "frequency": int(counts[selected]),
        "mean_inner_r": mean_scores[selected],
        "all": [
            {"hyper": hyper, "inner_r": float(score)}
            for hyper, score in zip(hypers, scores)
        ],
    }


def apply_frozen_pmb_overrides(config):
    config.update({
        "seed": 20260805,
        "outer_folds": 5,
        "inner_folds": 3,
        "min_maf": 0.0,
        "max_marker_missing_rate": 1.0,
        "functional_min_maf": 0.0,
        "functional_max_marker_missing_rate": 1.0,
        "genome_pcs": 60,
        "lambda_grid": [1.0],
        "functional_weight_grid": [0.25, 0.5, 0.75],
        "rkhs_bandwidth_scale_grid": [1.0],
        "ridge_alpha_grid": [10.0],
        "elasticnet_alpha_grid": [0.1],
        "elasticnet_l1_ratio_grid": [0.5],
        "lasso_alpha_grid": [0.1],
        "pls_component_grid": [5],
        "svr_c_grid": [10.0],
        "svr_epsilon_grid": [1.0],
        "knn_grid": [10],
        "mlp_hidden_grid": [32],
    })
    return config


def block_inner_oof(y, genome, qtl, groups, folds, config, seed):
    grid = block.hyper_grid(config)
    predictions = [np.full(len(y), np.nan, dtype=float) for _ in grid]
    for fold_no, valid in enumerate(folds):
        fit = np.setdiff1d(np.arange(len(y)), valid, assume_unique=True)
        designs = block.design_matrices(
            genome[fit], genome[valid], qtl[fit], qtl[valid], groups,
            config, seed + fold_no * 97,
        )
        for candidate_no, hyper in enumerate(grid):
            predictions[candidate_no][valid] = block.fit_predict(
                y[fit], designs, hyper
            )
    scored = []
    for hyper, prediction in zip(grid, predictions):
        score = base.pearson(y, prediction)
        if np.isfinite(score):
            scored.append((float(score), hyper, prediction))
    if not scored:
        raise RuntimeError("QTL Block-ARD inner screening failed")
    scored.sort(key=lambda item: -item[0])
    return scored[0][2], scored[0][1], scored[0][0]


def load_checkpoint(output, ids, resume):
    checkpoint = output / "oof_predictions.checkpoint.csv"
    if not resume or not checkpoint.exists():
        return [], set(), 0.0
    frame = pd.read_csv(checkpoint)
    required = {
        ("Genome_only", "AMSE11_FullPoolBlend"),
        ("Genome_plus_QTL", "SelectedAnchor"),
        ("Genome_plus_QTL", "QTL_Block_ARD"),
        ("Genome_plus_QTL", "AMSE11_FullPoolBlend"),
    }
    completed = set()
    for (repeat, fold), group in frame.groupby(["repeat", "fold"]):
        present = set(zip(group["condition"], group["model"]))
        sample_counts = group.groupby(["condition", "model"])["sample_id"].nunique()
        if present != required or sample_counts.nunique() != 1:
            raise RuntimeError(f"Incomplete checkpoint for repeat={repeat}, fold={fold}")
        completed.add((int(repeat), int(fold)))
    if frame.duplicated(["repeat", "fold", "condition", "model", "sample_id"]).any():
        raise RuntimeError("Duplicate prediction rows in checkpoint")
    previous_elapsed = 0.0
    status_path = output / "status.json"
    if status_path.exists():
        previous_elapsed = float(json.loads(status_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    return frame.to_dict("records"), completed, previous_elapsed


def evaluate_cohort(cohort, output, repeats, preset, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    config_path = (
        HERE.parent / "amse_gp_multienvironment"
        / f"config_{cohort}_qtl_block_ard_5x5.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["outer_repeats"] = repeats
    if preset == "frozen-pmb":
        if cohort != "pmb":
            raise ValueError("The frozen-pmb preset can only be used for PMB2026")
        apply_frozen_pmb_overrides(config)
        config["outer_repeats"] = repeats
    else:
        config.setdefault("lambda_grid", [0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
        config.setdefault("functional_weight_grid", [0.25, 0.5, 0.75])
        config.setdefault("rkhs_bandwidth_scale_grid", [0.5, 1.0, 2.0])

    ids, genome, panels, panel_markers, y, _, audit = base.load_inputs(config)
    panel_name, qtl = next(iter(panels.items()))
    names, indices = block.qtl_groups(
        panel_markers[panel_name], config["qtl_loci_path"],
        config.get("qtl_marker_coordinate_path"),
    )
    groups = list(zip(names, indices))
    protocol = {
        "method": "AMSE11_FullPoolBlend",
        "cohort": cohort,
        "preset": preset,
        "config": config,
        "input_audit": audit,
        "base_sha256": base.sha256(BASE_PATH),
        "block_runner_sha256": base.sha256(BLOCK_PATH),
        "candidate_models": list(CANDIDATE_MODELS),
        "anchor_selection": "maximum mean inner r minus 0.25 SD",
        "inner_repeats": INNER_REPEATS,
        "candidate_block_weights": list(BLOCK_WEIGHTS),
        "weight_selection": (
            "smallest weight within 0.002 of the best repeated-inner stable score"
        ),
        "same_anchor_identity_used_for_both_representations": True,
        "outer_test_phenotype_used_for_selection": False,
        "interpretation": "exploratory; previously examined cohorts",
    }
    protocol_path = output / "protocol.json"
    if resume and protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("method", "cohort", "preset", "candidate_models"):
            if previous.get(key) != protocol.get(key):
                raise RuntimeError(f"Resume protocol mismatch for {key}")
    else:
        dump(protocol_path, protocol)

    rows, completed_keys, previous_elapsed = load_checkpoint(output, ids, resume)
    total = repeats * int(config["outer_folds"])
    completed, started = len(completed_keys), time.time()
    for repeat in range(1, repeats + 1):
        outer = base.folds_for(
            len(y), int(config["seed"]) + repeat, int(config["outer_folds"])
        )
        for fold, test in enumerate(outer, 1):
            if (repeat, fold) in completed_keys:
                print(f"{cohort} skip completed repeat={repeat} fold={fold}", flush=True)
                continue
            fold_started = time.time()
            train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
            y_train = y[train]
            repeated_genome = []
            score_by_model = {model: [] for model in CANDIDATE_MODELS}

            for inner_repeat in range(1, INNER_REPEATS + 1):
                inner_seed = (
                    int(config["seed"]) + repeat * 1000 + fold
                    + inner_repeat * 100000
                )
                inner = base.folds_for(
                    len(train), inner_seed, int(config["inner_folds"])
                )
                model_seed = (
                    int(config["seed"]) + repeat * 10000 + fold * 100
                    + inner_repeat * 1000000
                )
                empty = np.zeros((len(train), 1))
                cache = base.prepare_inner_fold_cache(
                    "Genome_only", genome[train], empty, inner, config
                )
                model_results = {}
                for number, model in enumerate(CANDIDATE_MODELS):
                    prediction, hyper, score = base.inner_oof(
                        model, "Genome_only", genome[train], empty, y_train,
                        inner, config, model_seed + number, fold_cache=cache,
                    )
                    model_results[model] = {
                        "prediction": prediction,
                        "hyper": hyper,
                        "score": score,
                    }
                    score_by_model[model].append(score)
                repeated_genome.append({
                    "inner_repeat": inner_repeat,
                    "inner": inner,
                    "model_seed": model_seed,
                    "models": model_results,
                })

            anchor, anchor_ranking = choose_anchor(score_by_model)
            genome_hyper, genome_hyper_audit = modal_hyper(
                [item["models"][anchor]["hyper"] for item in repeated_genome],
                [item["models"][anchor]["score"] for item in repeated_genome],
            )

            qtl_inner_predictions, qtl_hypers, qtl_scores = [], [], []
            block_predictions, block_hypers, block_scores = [], [], []
            for item in repeated_genome:
                cache = base.prepare_inner_fold_cache(
                    "Genome_plus_QTL", genome[train], qtl[train],
                    item["inner"], config,
                )
                prediction, hyper, score = base.inner_oof(
                    anchor, "Genome_plus_QTL", genome[train], qtl[train],
                    y_train, item["inner"], config,
                    item["model_seed"] + 2000, fold_cache=cache,
                )
                block_prediction, block_hyper, block_score = block_inner_oof(
                    y_train, genome[train], qtl[train], groups, item["inner"],
                    config, item["model_seed"] + 5000,
                )
                qtl_inner_predictions.append(prediction)
                qtl_hypers.append(hyper)
                qtl_scores.append(score)
                block_predictions.append(block_prediction)
                block_hypers.append(block_hyper)
                block_scores.append(block_score)

            qtl_hyper, qtl_hyper_audit = modal_hyper(qtl_hypers, qtl_scores)
            block_hyper, block_hyper_audit = modal_hyper(
                block_hypers, block_scores
            )
            chosen_weight, weight_records = choose_weight(
                qtl_inner_predictions, block_predictions,
                [y_train] * INNER_REPEATS,
            )
            weight = float(chosen_weight["weight"])

            empty_train = np.zeros((len(train), 1))
            empty_test = np.zeros((len(test), 1))
            genome_prediction = base.fit_predict(
                anchor, "Genome_only", genome[train], genome[test],
                empty_train, empty_test, y_train, genome_hyper,
                int(config["seed"]) + repeat * 10000 + fold * 100 + 7000,
                config,
            )
            qtl_anchor_prediction = base.fit_predict(
                anchor, "Genome_plus_QTL", genome[train], genome[test],
                qtl[train], qtl[test], y_train, qtl_hyper,
                int(config["seed"]) + repeat * 10000 + fold * 100 + 8000,
                config,
            )
            block_design = block.design_matrices(
                genome[train], genome[test], qtl[train], qtl[test], groups,
                config, int(config["seed"]) + repeat * 10000 + fold * 100 + 9000,
            )
            block_prediction = block.fit_predict(
                y_train, block_design, block_hyper
            )
            blended = (
                (1.0 - weight) * qtl_anchor_prediction
                + weight * block_prediction
            )

            outputs = {
                ("Genome_only", "AMSE11_FullPoolBlend"): genome_prediction,
                ("Genome_plus_QTL", "SelectedAnchor"): qtl_anchor_prediction,
                ("Genome_plus_QTL", "QTL_Block_ARD"): block_prediction,
                ("Genome_plus_QTL", "AMSE11_FullPoolBlend"): blended,
            }
            for (condition, model), prediction in outputs.items():
                for sample, observed, value in zip(ids[test], y[test], prediction):
                    rows.append({
                        "repeat": repeat,
                        "fold": fold,
                        "condition": condition,
                        "model": model,
                        "sample_id": sample,
                        "observed": float(observed),
                        "predicted": float(value),
                        "selected_anchor": anchor,
                        "block_weight": weight if condition != "Genome_only" else 0.0,
                    })

            dump(output / f"selection_r{repeat}_f{fold}.json", {
                "repeat": repeat,
                "fold": fold,
                "selected_anchor": anchor,
                "anchor_ranking": anchor_ranking,
                "genome_hyper": genome_hyper,
                "genome_hyper_audit": genome_hyper_audit,
                "qtl_hyper": qtl_hyper,
                "qtl_hyper_audit": qtl_hyper_audit,
                "block_hyper": block_hyper,
                "block_hyper_audit": block_hyper_audit,
                "selected_block_weight": weight,
                "selected_weight_record": chosen_weight,
                "all_weight_records": weight_records,
                "qtl_blocks_retained": block_design[-1],
            })
            completed += 1
            elapsed = previous_elapsed + time.time() - started
            pd.DataFrame(rows).to_csv(
                output / "oof_predictions.checkpoint.csv", index=False
            )
            dump(output / "status.json", {
                "state": "RUNNING",
                "completed_folds": completed,
                "total_folds": total,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": elapsed / completed * (total - completed),
                "last_anchor": anchor,
                "last_block_weight": weight,
            })
            print(
                f"{cohort} {completed}/{total} anchor={anchor} "
                f"block_weight={weight:.2f} seconds={time.time()-fold_started:.1f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    metric_rows = []
    for (condition, model, repeat), group in frame.groupby(
        ["condition", "model", "repeat"]
    ):
        if len(group) != len(ids):
            raise RuntimeError("Incomplete OOF prediction set")
        metric_rows.append({
            "condition": condition,
            "model": model,
            "repeat": repeat,
            **base.metrics(
                group.observed.to_numpy(), group.predicted.to_numpy()
            ),
        })
    metric_frame = pd.DataFrame(metric_rows)
    frame.to_csv(output / "oof_predictions.csv", index=False)
    metric_frame.to_csv(output / "repeat_metrics.csv", index=False)
    metric_frame.groupby(["condition", "model"]).agg(
        pearson_mean=("pearson_r", "mean"),
        pearson_sd=("pearson_r", "std"),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
    ).reset_index().to_csv(output / "summary.csv", index=False)
    dump(output / "status.json", {
        "state": "COMPLETE",
        "completed_folds": total,
        "elapsed_seconds": previous_elapsed + time.time() - started,
    })


def self_test():
    score_by_model = {
        model: [0.2, 0.2, 0.2] for model in CANDIDATE_MODELS
    }
    score_by_model["RKHS"] = [0.40, 0.42, 0.41]
    anchor, records = choose_anchor(score_by_model)
    assert anchor == "RKHS" and len(records) == 16
    y = np.arange(12, dtype=float)
    anchor_prediction = y + np.array([1.0, -1.0] * 6)
    block_prediction = y + np.array([0.2, -0.2] * 6)
    chosen, weights = choose_weight(
        [anchor_prediction] * 3,
        [block_prediction] * 3,
        [y] * 3,
    )
    assert chosen["weight"] > 0 and len(weights) == len(BLOCK_WEIGHTS)
    hyper, audit = modal_hyper(
        [{"alpha": 1}, {"alpha": 1}, {"alpha": 2}], [0.2, 0.3, 0.4]
    )
    assert hyper == {"alpha": 1} and audit["frequency"] == 2
    print("PASS full-pool anchor and QTL blend selection", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cohort", choices=("pmb", "external", "both"), default="pmb")
    parser.add_argument("--preset", choices=("current", "frozen-pmb"), default="current")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        dump(args.output / "job_status.json", {
            "state": "RUNNING",
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        cohorts = ("external", "pmb") if args.cohort == "both" else (args.cohort,)
        try:
            with threadpool_limits(limits=2):
                for cohort in cohorts:
                    evaluate_cohort(
                        cohort, args.output / cohort, args.repeats, args.preset, args.resume
                    )
            dump(args.output / "job_status.json", {
                "state": "COMPLETE",
                "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception:
            dump(args.output / "job_status.json", {
                "state": "FAILED",
                "traceback": traceback.format_exc(),
            })
            raise
