"""Conservatively screen a stable anchor, then apply the AMSE10 QTL blend."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
AMSE11_PATH = HERE / "run_amse11_full_pool_blend.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a11 = load_module("amse11_shared", AMSE11_PATH)
base, block = a11.base, a11.block

TOP_N = 6
MIN_TOP_COUNT = 2
SHORTLIST_MARGIN = 0.03
BASE_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
NEAR_BEST_MARGIN = 0.002
BALANCED_QTL_WEIGHT = 0.50
BRIDGE_WEIGHTS = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 1.0)
ANCHOR_INNER_REPEATS = 2
ANCHOR_INNER_FOLDS = 5


def dump(path, payload):
    a11.dump(path, payload)


def build_shortlist(repeated_genome):
    models = a11.CANDIDATE_MODELS
    ranks = {model: [] for model in models}
    scores = {model: [] for model in models}
    for item in repeated_genome:
        ordered = sorted(
            models,
            key=lambda model: (-item["models"][model]["score"], models.index(model)),
        )
        rank = {model: number + 1 for number, model in enumerate(ordered)}
        for model in models:
            ranks[model].append(rank[model])
            scores[model].append(float(item["models"][model]["score"]))
    records = []
    for order, model in enumerate(models):
        values = scores[model]
        records.append({
            "model": model,
            "repeat_r": values,
            "repeat_rank": ranks[model],
            "top6_count": int(sum(rank <= TOP_N for rank in ranks[model])),
            "mean_rank": float(np.mean(ranks[model])),
            "mean_r": float(np.mean(values)),
            "sd_r": float(np.std(values, ddof=0)),
            "stable_r": a11.stable(values),
            "prespecified_order": order,
        })
    records.sort(key=lambda row: (-row["stable_r"], row["prespecified_order"]))
    best = records[0]["stable_r"]
    eligible = [
        row["model"] for row in records
        if row["top6_count"] >= MIN_TOP_COUNT
        and row["stable_r"] >= best - SHORTLIST_MARGIN
    ]
    if len(eligible) < 2:
        eligible = [row["model"] for row in records[:2]]
    return eligible, records


def choose_base_predictor(repeated_genome, shortlist, repeated_y):
    candidates = []
    for left_no, left in enumerate(shortlist):
        for right in shortlist[left_no:]:
            weights = (0.0,) if left == right else BASE_WEIGHTS
            for weight in weights:
                repeat_r, repeat_rmse = [], []
                for item, y in zip(repeated_genome, repeated_y):
                    left_pred = item["models"][left]["prediction"]
                    right_pred = item["models"][right]["prediction"]
                    prediction = (1.0 - weight) * left_pred + weight * right_pred
                    repeat_r.append(base.pearson(y, prediction))
                    repeat_rmse.append(float(np.sqrt(np.mean((y - prediction) ** 2))))
                effective_models = 1 if left == right or weight in (0.0, 1.0) else 2
                candidates.append({
                    "left_model": left,
                    "right_model": right,
                    "right_weight": float(weight),
                    "effective_models": effective_models,
                    "repeat_r": repeat_r,
                    "repeat_rmse": repeat_rmse,
                    "mean_r": float(np.mean(repeat_r)),
                    "sd_r": float(np.std(repeat_r, ddof=0)),
                    "stable_r": a11.stable(repeat_r),
                    "mean_rmse": float(np.mean(repeat_rmse)),
                })
    best = max(row["stable_r"] for row in candidates)
    eligible = [row for row in candidates if row["stable_r"] >= best - NEAR_BEST_MARGIN]
    chosen = min(
        eligible,
        key=lambda row: (
            row["effective_models"], row["mean_rmse"],
            row["left_model"], row["right_model"], row["right_weight"],
        ),
    )
    return chosen, candidates


def combine(left, right, right_weight):
    return (1.0 - right_weight) * left + right_weight * right


def choose_bridge_weight(records):
    candidates = []
    for weight in BRIDGE_WEIGHTS:
        values = [
            base.pearson(
                item["observed"],
                combine(item["base_prediction"], item["block_prediction"], weight),
            )
            for item in records
        ]
        candidates.append({
            "weight": float(weight),
            "repeat_r": values,
            "mean_r": float(np.mean(values)),
            "sd_r": float(np.std(values, ddof=0)),
            "stable_r": a11.stable(values),
        })
    best = max(row["stable_r"] for row in candidates)
    eligible = [row for row in candidates if row["stable_r"] >= best - NEAR_BEST_MARGIN]
    chosen = min(eligible, key=lambda row: row["weight"])
    return chosen, candidates


def choose_anchor(records, shortlist, n_training):
    candidates = []
    for model in shortlist:
        values = [item["scores"][model] for item in records]
        candidates.append({
            "model": model,
            "repeat_r": values,
            "mean_r": float(np.mean(values)),
            "sd_r": float(np.std(values, ddof=0)),
            "stable_r": a11.stable(values),
            "prespecified_order": a11.CANDIDATE_MODELS.index(model),
        })
    candidates.sort(key=lambda row: (-row["stable_r"], row["prespecified_order"]))
    genetic = max(
        (row for row in candidates if row["model"] in ("GBLUP", "RKHS")),
        key=lambda row: (row["stable_r"], -row["prespecified_order"]),
    )
    fisher_se = 1.0 / math.sqrt(max(1, n_training - 3))
    promoted = []
    for row in candidates:
        if row["model"] in ("GBLUP", "RKHS"):
            continue
        repeat_gains = [
            challenger - baseline
            for challenger, baseline in zip(row["repeat_r"], genetic["repeat_r"])
        ]
        z_gain = math.atanh(np.clip(row["mean_r"], -0.999999, 0.999999)) - math.atanh(
            np.clip(genetic["mean_r"], -0.999999, 0.999999)
        )
        row["gain_over_genetic"] = float(row["mean_r"] - genetic["mean_r"])
        row["fisher_z_gain"] = float(z_gain)
        row["positive_in_all_anchor_repeats"] = bool(min(repeat_gains) > 0)
        row["passes_conservative_gate"] = bool(
            row["positive_in_all_anchor_repeats"] and z_gain > fisher_se
        )
        if row["passes_conservative_gate"]:
            promoted.append(row)
    chosen = promoted[0] if promoted else genetic
    decision = {
        "genetic_anchor": genetic["model"],
        "fisher_z_standard_error": float(fisher_se),
        "challenger_promoted": bool(promoted),
        "selected_model": chosen["model"],
    }
    return chosen, candidates, decision


def evaluate(output, repeats, preset, seed_override=None, config_override=None):
    output.mkdir(parents=True, exist_ok=True)
    config_path = config_override or (
        HERE.parent / "amse_gp_multienvironment" / "config_pmb_qtl_block_ard_5x5.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["outer_repeats"] = repeats
    if preset == "frozen-pmb":
        a11.apply_frozen_pmb_overrides(config)
        config["outer_repeats"] = repeats
    if seed_override is not None:
        config["seed"] = int(seed_override)
    config.setdefault("lambda_grid", [0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    config.setdefault("functional_weight_grid", [0.25, 0.5, 0.75])
    config.setdefault("rkhs_bandwidth_scale_grid", [0.5, 1.0, 2.0])
    config.setdefault("elasticnet_alpha_grid", [0.01, 0.1, 1.0])
    config.setdefault("elasticnet_l1_ratio_grid", [0.1, 0.5, 0.9])
    config.setdefault("lasso_alpha_grid", [0.01, 0.1, 1.0])
    config.setdefault("pls_component_grid", [2, 5, 10])
    config.setdefault("svr_c_grid", [1.0, 10.0, 100.0])
    config.setdefault("svr_epsilon_grid", [0.1, 1.0])
    config.setdefault("knn_grid", [5, 10, 20])
    config.setdefault("mlp_hidden_grid", [16, 32, 64])
    ids, genome, panels, panel_markers, y, _, audit = base.load_inputs(config)
    panel_name, qtl = next(iter(panels.items()))
    names, indices = block.qtl_groups(
        panel_markers[panel_name], config["qtl_loci_path"],
        config.get("qtl_marker_coordinate_path"),
    )
    groups = list(zip(names, indices))
    dump(output / "protocol.json", {
        "method": "AMSE18_ConservativeAnchorBlend",
        "preset": preset,
        "config": config,
        "input_audit": audit,
        "candidate_models": list(a11.CANDIDATE_MODELS),
        "inner_repeats": a11.INNER_REPEATS,
        "shortlist_rule": {
            "top_n": TOP_N,
            "minimum_top_n_occurrences": MIN_TOP_COUNT,
            "stable_score_margin": SHORTLIST_MARGIN,
        },
        "base_candidate_weights": list(BASE_WEIGHTS),
        "bridge_candidate_weights": list(BRIDGE_WEIGHTS),
        "anchor_inner_repeats": ANCHOR_INNER_REPEATS,
        "anchor_inner_folds": ANCHOR_INNER_FOLDS,
        "anchor_rule": "best GBLUP/RKHS anchor retained unless a shortlisted challenger wins in every repeated fivefold screen and exceeds one Fisher-z sampling SE",
        "fusion_rule": "AMSE10 repeated-inner-CV blend of the selected anchor and QTL Block-ARD expert",
        "all_selection_uses_outer_training_data_only": True,
        "outer_test_phenotype_used_for_selection": False,
        "interpretation": "exploratory; previously examined cohort",
    })

    rows, total = [], len(config.get("analysis_repeat_ids", range(1, repeats + 1))) * len(config.get("analysis_fold_ids") or range(1, int(config["outer_folds"]) + 1))
    started = time.time()
    for repeat in config.get("analysis_repeat_ids", range(1, repeats + 1)):
        outer = base.folds_for(len(y), int(config["seed"]) + repeat, int(config["outer_folds"]))
        for fold, test in enumerate(outer, 1):
            if config.get("analysis_fold_ids") and fold not in config["analysis_fold_ids"]:
                continue
            fold_started = time.time()
            train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
            y_train = y[train]
            repeated_genome = []
            for inner_repeat in range(1, a11.INNER_REPEATS + 1):
                inner_seed = int(config["seed"]) + repeat * 1000 + fold + inner_repeat * 100000
                inner = base.folds_for(len(train), inner_seed, int(config["inner_folds"]))
                model_seed = int(config["seed"]) + repeat * 10000 + fold * 100 + inner_repeat * 1000000
                empty = np.zeros((len(train), 1))
                cache = base.prepare_inner_fold_cache("Genome_only", genome[train], empty, inner, config)
                model_results = {}
                for number, model in enumerate(a11.CANDIDATE_MODELS):
                    prediction, hyper, score = base.inner_oof(
                        model, "Genome_only", genome[train], empty, y_train,
                        inner, config, model_seed + number, fold_cache=cache,
                    )
                    model_results[model] = {"prediction": prediction, "hyper": hyper, "score": score}
                repeated_genome.append({
                    "inner_repeat": inner_repeat, "inner": inner,
                    "model_seed": model_seed, "models": model_results,
                })

            shortlist, ranking = build_shortlist(repeated_genome)
            anchor_pool = list(dict.fromkeys(shortlist + ["GBLUP", "RKHS"]))
            anchor_records = []
            for anchor_repeat in range(1, ANCHOR_INNER_REPEATS + 1):
                anchor_seed = (
                    int(config["seed"]) + repeat * 100000 + fold * 1000
                    + anchor_repeat * 10000000
                )
                anchor_folds = base.folds_for(
                    len(train), anchor_seed, min(ANCHOR_INNER_FOLDS, len(train))
                )
                empty = np.zeros((len(train), 1))
                anchor_cache = base.prepare_inner_fold_cache(
                    "Genome_only", genome[train], empty, anchor_folds, config
                )
                anchor_scores = {}
                for number, model in enumerate(anchor_pool):
                    _, _, score = base.inner_oof(
                        model, "Genome_only", genome[train], empty, y_train,
                        anchor_folds, config, anchor_seed + number,
                        fold_cache=anchor_cache,
                    )
                    anchor_scores[model] = float(score)
                anchor_records.append({"scores": anchor_scores})
            anchor_choice, anchor_candidates, anchor_decision = choose_anchor(
                anchor_records, anchor_pool, len(train)
            )
            anchor = anchor_choice["model"]
            left = right = anchor
            base_weight = 0.0
            selected_models = [anchor]

            genome_hypers, qtl_hypers = {}, {}
            qtl_inner = {model: [] for model in selected_models}
            block_predictions, block_hypers, block_scores = [], [], []
            for model in selected_models:
                genome_hypers[model] = a11.modal_hyper(
                    [item["models"][model]["hyper"] for item in repeated_genome],
                    [item["models"][model]["score"] for item in repeated_genome],
                )[0]
            for item in repeated_genome:
                cache = base.prepare_inner_fold_cache(
                    "Genome_plus_QTL", genome[train], qtl[train], item["inner"], config
                )
                for number, model in enumerate(selected_models):
                    prediction, hyper, score = base.inner_oof(
                        model, "Genome_plus_QTL", genome[train], qtl[train], y_train,
                        item["inner"], config, item["model_seed"] + 2000 + number,
                        fold_cache=cache,
                    )
                    qtl_inner[model].append({"prediction": prediction, "hyper": hyper, "score": score})
                bpred, bhyper, bscore = a11.block_inner_oof(
                    y_train, genome[train], qtl[train], groups, item["inner"],
                    config, item["model_seed"] + 5000,
                )
                block_predictions.append(bpred)
                block_hypers.append(bhyper)
                block_scores.append(bscore)
            for model in selected_models:
                qtl_hypers[model] = a11.modal_hyper(
                    [item["hyper"] for item in qtl_inner[model]],
                    [item["score"] for item in qtl_inner[model]],
                )[0]

            block_hyper = a11.modal_hyper(block_hypers, block_scores)[0]

            bridge_records = [
                {
                    "observed": y_train,
                    "base_prediction": qtl_inner[anchor][i]["prediction"],
                    "block_prediction": block_predictions[i],
                }
                for i in range(a11.INNER_REPEATS)
            ]
            bridge_choice, bridge_candidates = choose_bridge_weight(bridge_records)
            qtl_weight = float(bridge_choice["weight"])

            empty_train, empty_test = np.zeros((len(train), 1)), np.zeros((len(test), 1))
            genome_outer, qtl_outer = {}, {}
            for number, model in enumerate(selected_models):
                genome_outer[model] = base.fit_predict(
                    model, "Genome_only", genome[train], genome[test], empty_train,
                    empty_test, y_train, genome_hypers[model],
                    int(config["seed"]) + repeat * 10000 + fold * 100 + 7000 + number,
                    config,
                )
                qtl_outer[model] = base.fit_predict(
                    model, "Genome_plus_QTL", genome[train], genome[test], qtl[train],
                    qtl[test], y_train, qtl_hypers[model],
                    int(config["seed"]) + repeat * 10000 + fold * 100 + 8000 + number,
                    config,
                )
            genome_base = combine(genome_outer[left], genome_outer[right], base_weight)
            qtl_base = combine(qtl_outer[left], qtl_outer[right], base_weight)
            block_design = block.design_matrices(
                genome[train], genome[test], qtl[train], qtl[test], groups, config,
                int(config["seed"]) + repeat * 10000 + fold * 100 + 9000,
            )
            block_outer = block.fit_predict(y_train, block_design, block_hyper)
            final = combine(qtl_base, block_outer, qtl_weight)
            outputs = {
                ("Genome_only", "AMSE18_ConservativeAnchorBlend"): genome_base,
                ("Genome_plus_QTL", "ShortlistedBase"): qtl_base,
                ("Genome_plus_QTL", "QTL_Block_ARD"): block_outer,
                ("Genome_plus_QTL", "AMSE18_ConservativeAnchorBlend"): final,
            }
            for (condition, model), prediction in outputs.items():
                for sample, observed, value in zip(ids[test], y[test], prediction):
                    rows.append({
                        "repeat": repeat, "fold": fold, "condition": condition,
                        "model": model, "sample_id": sample,
                        "observed": float(observed), "predicted": float(value),
                        "base_left": left, "base_right": right,
                        "base_right_weight": base_weight, "qtl_block_weight": qtl_weight,
                    })
            dump(output / f"selection_r{repeat}_f{fold}.json", {
                "repeat": repeat, "fold": fold, "shortlist": shortlist,
                "full_pool_ranking": ranking,
                "selected_anchor": anchor_choice,
                "all_anchor_candidates": anchor_candidates,
                "anchor_gate_decision": anchor_decision,
                "selected_qtl_weight": bridge_choice,
                "all_bridge_weight_candidates": bridge_candidates,
                "block_hyperparameter": block_hyper,
                "selected_qtl_anchor_hyper": qtl_hypers[anchor],
            })
            completed = list(config.get("analysis_repeat_ids", range(1, repeats + 1))).index(repeat) * len(config.get("analysis_fold_ids") or range(1, int(config["outer_folds"]) + 1)) + list(config.get("analysis_fold_ids") or range(1, int(config["outer_folds"]) + 1)).index(fold) + 1
            pd.DataFrame(rows).to_csv(output / "oof_predictions.checkpoint.csv", index=False)
            elapsed = time.time() - started
            dump(output / "status.json", {
                "state": "RUNNING", "completed_folds": completed, "total_folds": total,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": elapsed / completed * (total - completed),
                "last_shortlist": shortlist,
                "last_base": {"left": left, "right": right, "weight": base_weight},
                "last_qtl_block_weight": qtl_weight,
            })
            print(
                f"{output.name} {completed}/{total} shortlist={','.join(shortlist)} "
                f"base={left}+{base_weight:.2f}*{right} qtl={qtl_weight:.2f} "
                f"seconds={time.time()-fold_started:.1f}", flush=True,
            )

    frame = pd.DataFrame(rows)
    metrics = []
    for (condition, model, repeat), group in frame.groupby(["condition", "model", "repeat"]):
        if len(group) != expected_oof_count(config, base, len(ids), repeat):
            raise RuntimeError("Incomplete OOF prediction set")
        metrics.append({
            "condition": condition, "model": model, "repeat": repeat,
            **base.metrics(group.observed.to_numpy(), group.predicted.to_numpy()),
        })
    metric_frame = pd.DataFrame(metrics)
    frame.to_csv(output / "oof_predictions.csv", index=False)
    metric_frame.to_csv(output / "repeat_metrics.csv", index=False)
    metric_frame.groupby(["condition", "model"]).agg(
        pearson_mean=("pearson_r", "mean"), pearson_sd=("pearson_r", "std"),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
    ).reset_index().to_csv(output / "summary.csv", index=False)
    dump(output / "status.json", {
        "state": "COMPLETE", "completed_folds": total,
        "elapsed_seconds": time.time() - started,
    })


def self_test():
    rng = np.random.default_rng(12)
    y = np.arange(30, dtype=float)
    repeated = []
    for repeat in range(3):
        models = {}
        for number, model in enumerate(a11.CANDIDATE_MODELS):
            scale = 0.2 if model in ("GBLUP", "RKHS") else 2.0 + number / 10
            prediction = y + rng.normal(scale=scale, size=len(y))
            models[model] = {
                "prediction": prediction, "hyper": {},
                "score": base.pearson(y, prediction),
            }
        repeated.append({"models": models})
    shortlist, ranking = build_shortlist(repeated)
    chosen, candidates = choose_base_predictor(repeated, shortlist, [y] * 3)
    assert len(shortlist) >= 2 and len(ranking) == 16 and candidates
    assert chosen["left_model"] in shortlist and chosen["right_model"] in shortlist
    anchor_records = [
        {"scores": {"GBLUP": 0.40, "RKHS": 0.42, "ExtraTrees": 0.44}},
        {"scores": {"GBLUP": 0.41, "RKHS": 0.43, "ExtraTrees": 0.45}},
    ]
    anchor, _, decision = choose_anchor(
        anchor_records, ["GBLUP", "RKHS", "ExtraTrees"], 120
    )
    assert anchor["model"] == "RKHS" and not decision["challenger_promoted"]
    print(f"PASS shortlist={shortlist} selected={chosen['left_model']}/{chosen['right_model']}")


# BOUNDARY_RUNTIME_V1
from runtime_support import install, expected_oof_count
install(base, block)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--preset", choices=("current", "frozen-pmb"), default="frozen-pmb")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--label", default="pmb")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        dump(args.output / "job_status.json", {
            "state": "RUNNING", "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        try:
            with threadpool_limits(limits=2):
                evaluate(
                    args.output / args.label, args.repeats, args.preset,
                    args.seed, args.config,
                )
            dump(args.output / "job_status.json", {
                "state": "COMPLETE", "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception:
            dump(args.output / "job_status.json", {
                "state": "FAILED", "traceback": traceback.format_exc(),
            })
            raise
