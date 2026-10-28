"""AMSE-GP screen with a QTL-block ARD expert and guarded expert selection."""

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
BASE_PATH = HERE / "run_scheme2_tuned.py"
BLOCK_PATH = HERE / "run_qtl_block_ard.py"
if not BLOCK_PATH.exists():
    BLOCK_PATH = HERE.parent / "amse_gp_multienvironment" / "run_qtl_block_ard.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("amse_base", BASE_PATH)
block = load_module("qtl_block", BLOCK_PATH)
MIN_GAIN = 0.005
MIN_POSITIVE_FOLDS = 2


def dump(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


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
            predictions[candidate_no][valid] = block.fit_predict(y[fit], designs, hyper)
    scored = []
    for hyper, prediction in zip(grid, predictions):
        score = base.pearson(y, prediction)
        if np.isfinite(score):
            scored.append((score, hyper, prediction))
    if not scored:
        raise RuntimeError("QTL Block-ARD inner screening failed")
    scored.sort(key=lambda item: -item[0])
    score, hyper, prediction = scored[0]
    return prediction, hyper, float(score)


def choose_qtl_expert(y, folds, anchor_name, anchor_prediction, block_prediction):
    anchor_score = base.pearson(y, anchor_prediction)
    block_score = base.pearson(y, block_prediction)
    fold_gains = [
        base.pearson(y[idx], block_prediction[idx])
        - base.pearson(y[idx], anchor_prediction[idx])
        for idx in folds
    ]
    positive = int(sum(value > 0 for value in fold_gains))
    use_block = bool(block_score >= anchor_score + MIN_GAIN and positive >= MIN_POSITIVE_FOLDS)
    return ("QTL_Block_ARD" if use_block else anchor_name), {
        "anchor": anchor_name, "anchor_inner_r": anchor_score,
        "block_inner_r": block_score, "inner_delta": block_score - anchor_score,
        "fold_deltas": [float(value) for value in fold_gains],
        "positive_inner_folds": positive, "selected": "QTL_Block_ARD" if use_block else anchor_name,
    }


def evaluate_cohort(cohort, output, repeats, config_override=None):
    output.mkdir(parents=True, exist_ok=True)
    config_path = config_override or (
        HERE.parent / "amse_gp_multienvironment" / f"config_{cohort}_qtl_block_ard_5x5.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["outer_repeats"] = repeats
    config.setdefault("lambda_grid", [0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    config.setdefault("functional_weight_grid", [0.25, 0.5, 0.75])
    config.setdefault("rkhs_bandwidth_scale_grid", [0.5, 1.0, 2.0])
    ids, genome, panels, panel_markers, y, _, audit = base.load_inputs(config)
    panel_name, qtl = next(iter(panels.items()))
    names, indices = block.qtl_groups(
        panel_markers[panel_name], config["qtl_loci_path"], config.get("qtl_marker_coordinate_path")
    )
    groups = list(zip(names, indices))
    dump(output / "protocol.json", {
        "cohort": cohort, "method": "AMSE8_QTLExpert", "config": config,
        "input_audit": audit, "base_sha256": base.sha256(BASE_PATH),
        "block_runner_sha256": base.sha256(BLOCK_PATH),
        "genome_anchor_candidates": ["GBLUP", "RKHS"],
        "qtl_expert_candidates": ["selected genome anchor", "QTL_Block_ARD"],
        "minimum_block_gain": MIN_GAIN, "minimum_positive_inner_folds": MIN_POSITIVE_FOLDS,
        "all_selection_uses_outer_training_data_only": True,
        "interpretation": "exploratory; previously examined cohorts",
    })
    rows, selections = [], []
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
            inner = base.folds_for(len(train), int(config["seed"]) + repeat * 1000 + fold, int(config["inner_folds"]))
            seed = int(config["seed"]) + repeat * 10000 + fold * 100

            inner_predictions, hypers, scores = {}, {}, {}
            for condition, addon in (("Genome_only", np.zeros((len(y), 1))), ("Genome_plus_QTL", qtl)):
                cache = base.prepare_inner_fold_cache(
                    condition, genome[train], addon[train], inner, config
                )
                for number, model in enumerate(("GBLUP", "RKHS")):
                    prediction, hyper, score = base.inner_oof(
                        model, condition, genome[train], addon[train], y_train,
                        inner, config, seed + number, fold_cache=cache,
                    )
                    inner_predictions[(condition, model)] = prediction
                    hypers[(condition, model)] = hyper
                    scores[(condition, model)] = score

            genome_anchor = max(
                ("GBLUP", "RKHS"), key=lambda name: (scores[("Genome_only", name)], name)
            )
            qtl_anchor = genome_anchor
            block_oof, block_hyper, block_score = block_inner_oof(
                y_train, genome[train], qtl[train], groups, inner, config, seed + 5000
            )
            selected_qtl, gate = choose_qtl_expert(
                y_train, inner, qtl_anchor,
                inner_predictions[("Genome_plus_QTL", qtl_anchor)], block_oof,
            )

            outer_predictions = {}
            for condition, addon in (("Genome_only", np.zeros((len(y), 1))), ("Genome_plus_QTL", qtl)):
                prepared = base.prepare_fold_data(
                    condition, genome[train], genome[test], addon[train], addon[test], config
                )
                for number, model in enumerate(("GBLUP", "RKHS")):
                    outer_predictions[(condition, model)] = base.fit_predict_prepared(
                        model, condition, prepared, y_train, hypers[(condition, model)], seed + number
                    )
            block_design = block.design_matrices(
                genome[train], genome[test], qtl[train], qtl[test], groups, config, seed + 9000
            )
            outer_predictions[("Genome_plus_QTL", "QTL_Block_ARD")] = block.fit_predict(
                y_train, block_design, block_hyper
            )
            outputs = {
                ("Genome_only", "AMSE8_QTLExpert"): outer_predictions[("Genome_only", genome_anchor)],
                ("Genome_only", "GBLUP"): outer_predictions[("Genome_only", "GBLUP")],
                ("Genome_only", "RKHS"): outer_predictions[("Genome_only", "RKHS")],
                ("Genome_plus_QTL", "AMSE8_QTLExpert"): outer_predictions[("Genome_plus_QTL", selected_qtl)],
                ("Genome_plus_QTL", "GBLUP"): outer_predictions[("Genome_plus_QTL", "GBLUP")],
                ("Genome_plus_QTL", "RKHS"): outer_predictions[("Genome_plus_QTL", "RKHS")],
                ("Genome_plus_QTL", "QTL_Block_ARD"): outer_predictions[("Genome_plus_QTL", "QTL_Block_ARD")],
            }
            for (condition, model), prediction in outputs.items():
                for sample, observed, value in zip(ids[test], y[test], prediction):
                    rows.append({"repeat": repeat, "fold": fold, "condition": condition,
                                 "model": model, "sample_id": sample,
                                 "observed": float(observed), "predicted": float(value)})
            record = {
                "repeat": repeat, "fold": fold, "genome_anchor": genome_anchor,
                "genome_inner_scores": {name: scores[("Genome_only", name)] for name in ("GBLUP", "RKHS")},
                "qtl_anchor_inner_scores": {name: scores[("Genome_plus_QTL", name)] for name in ("GBLUP", "RKHS")},
                "qtl_anchor_hypers": {name: hypers[("Genome_plus_QTL", name)] for name in ("GBLUP", "RKHS")},
                "block_inner_r": block_score, "block_hyper": block_hyper,
                "qtl_gate": gate, "selected_qtl_expert": selected_qtl,
                "qtl_blocks_retained": block_design[-1],
            }
            selections.append(record)
            dump(output / f"selection_r{repeat}_f{fold}.json", record)
            completed += 1
            elapsed = time.time() - started
            dump(output / "status.json", {"state": "RUNNING", "completed_folds": completed,
                "total_folds": total, "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": elapsed / completed * (total - completed)})
            print(f"{cohort} {completed}/{total} genome={genome_anchor} qtl={selected_qtl} seconds={time.time()-fold_started:.1f}", flush=True)

    frame = pd.DataFrame(rows)
    metric_rows = []
    for (condition, model, repeat), group in frame.groupby(["condition", "model", "repeat"]):
        if len(group) != expected_oof_count(config, base, len(ids), repeat):
            raise RuntimeError("Incomplete OOF prediction set")
        metric_rows.append({"condition": condition, "model": model, "repeat": repeat,
                            **base.metrics(group.observed.to_numpy(), group.predicted.to_numpy())})
    metric_frame = pd.DataFrame(metric_rows)
    frame.to_csv(output / "oof_predictions.csv", index=False)
    metric_frame.to_csv(output / "repeat_metrics.csv", index=False)
    metric_frame.groupby(["condition", "model"]).agg(
        pearson_mean=("pearson_r", "mean"), pearson_sd=("pearson_r", "std"),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean")
    ).reset_index().to_csv(output / "summary.csv", index=False)
    dump(output / "status.json", {"state": "COMPLETE", "completed_folds": total,
                                   "elapsed_seconds": time.time() - started})


def self_test():
    rng = np.random.default_rng(23)
    y = rng.normal(size=90)
    folds = [np.arange(0, 30), np.arange(30, 60), np.arange(60, 90)]
    anchor = y + rng.normal(scale=0.9, size=len(y))
    better = y + rng.normal(scale=0.2, size=len(y))
    selected, audit = choose_qtl_expert(y, folds, "GBLUP", anchor, better)
    assert selected == "QTL_Block_ARD" and audit["positive_inner_folds"] == 3
    selected, _ = choose_qtl_expert(y, folds, "GBLUP", anchor, anchor.copy())
    assert selected == "GBLUP"
    print("PASS guarded expert selection", flush=True)


# BOUNDARY_RUNTIME_V1
from runtime_support import install, expected_oof_count
install(base, block)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
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
                    evaluate_cohort(cohort, args.output / cohort, args.repeats, args.config)
            dump(args.output / "job_status.json", {"state": "COMPLETE",
                                                   "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        except Exception:
            dump(args.output / "job_status.json", {"state": "FAILED", "traceback": traceback.format_exc()})
            raise
