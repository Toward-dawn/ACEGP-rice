"""Nested-CV screen for a QTL-block principal-component ARD model.

The model keeps the genome-wide background and the QTL representation separate:
genome-wide PCs enter a ridge model, while two fold-local PCs per frozen QTL
interval enter an ARD regression on the ridge residuals.  It is an exploratory
model screen, not a replacement for the frozen AMSE-GP analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import randomized_svd
from sklearn.linear_model import ARDRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_scheme2_tuned as base  # noqa: E402


EPS = 1e-12


def marker_coordinates(marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse the frozen BIM IDs or CHR:POS:REF:ALT external marker IDs."""
    chrom, pos = [], []
    for marker in marker_names:
        fields = str(marker).split(":")
        if len(fields) >= 2:
            chrom.append(int(fields[0].replace("chr", "")))
            pos.append(int(fields[1]))
        else:
            raise ValueError(f"Cannot parse chromosome and position from marker {marker!r}")
    return np.asarray(chrom), np.asarray(pos)


def qtl_groups(marker_names: list[str], loci_path: str, coordinate_path: str | None = None) -> tuple[list[str], list[np.ndarray]]:
    loci = pd.read_csv(loci_path)
    required = {"qid", "chr", "start", "end"}
    if not required.issubset(loci.columns):
        raise ValueError(f"QTL coordinate file lacks required columns: {required}")
    if coordinate_path:
        coordinates = pd.read_csv(coordinate_path)
        if len(coordinates) != len(marker_names) or not {"chr", "pos"}.issubset(coordinates.columns):
            raise ValueError("QTL marker coordinate file must have chr and pos columns in panel-column order")
        chrom = coordinates["chr"].to_numpy(int)
        pos = coordinates["pos"].to_numpy(int)
    else:
        chrom, pos = marker_coordinates(marker_names)
    names, groups = [], []
    for row in loci.itertuples(index=False):
        idx = np.flatnonzero(
            (chrom == int(row.chr)) & (pos >= int(row.start)) & (pos <= int(row.end))
        )
        if idx.size:
            names.append(str(row.qid))
            groups.append(idx)
    if len(groups) < 1:
        raise RuntimeError("No QTL blocks contain markers")
    return names, groups


def filtered_imputed(train: np.ndarray, test: np.ndarray, min_maf: float, max_missing: float):
    missing = 1.0 - np.isfinite(train).mean(axis=0)
    af = np.nanmean(train, axis=0) / 2.0
    maf = np.minimum(af, 1.0 - af)
    keep = (
        np.isfinite(maf) & (maf >= min_maf) & (missing <= max_missing)
        & (np.nanvar(train, axis=0) > EPS)
    )
    if not keep.any():
        raise RuntimeError("No markers passed fold-local QC")
    x_train, x_test = base.impute_from_train(train[:, keep], test[:, keep])
    return x_train, x_test, keep


def randomized_pcs(train: np.ndarray, test: np.ndarray, n_components: int, random_state: int):
    train, test = base.impute_from_train(train, test)
    center = train.mean(axis=0)
    train = train - center
    test = test - center
    maximum = min(int(n_components), train.shape[0] - 1, train.shape[1])
    if maximum < 1:
        return np.zeros((len(train), 0)), np.zeros((len(test), 0))
    u, singular, vt = randomized_svd(
        train, n_components=maximum, n_iter=4, random_state=random_state
    )
    return u * singular, test @ vt.T


def block_seed(seed: int, block_name: str, config: dict, ordinal: int) -> int:
    """Keep unaffected blocks' randomized PCA stable when panels lose rows.

    The opt-in policy is confined to this isolated exploratory runner.  The
    legacy ordinal policy is retained as a fallback for source comparison.
    """
    policy = config.get("qtl_block_seed_policy", "legacy_ordinal")
    if policy == "legacy_ordinal":
        return seed + ordinal
    if policy != "identity_v1":
        raise ValueError(f"Unknown QTL block seed policy: {policy}")
    digest = hashlib.sha256(block_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (int(seed) + offset) % (2**32 - 1)


def design_matrices(genome_train, genome_test, qtl_train, qtl_test, groups, config, seed):
    genome_train, genome_test, _ = filtered_imputed(
        genome_train, genome_test, float(config["min_maf"]), float(config["max_marker_missing_rate"])
    )
    # QTL QC is applied before block-specific PCs.  The retained-index mapping
    # keeps every QTL block aligned with the fold-local genotype matrix.
    qtl_train, qtl_test, qtl_keep = filtered_imputed(
        qtl_train, qtl_test,
        float(config["functional_min_maf"]), float(config["functional_max_marker_missing_rate"]),
    )
    original_to_filtered = np.full(len(qtl_keep), -1, dtype=int)
    original_to_filtered[qtl_keep] = np.arange(int(qtl_keep.sum()))

    genome_pc_train, genome_pc_test = randomized_pcs(
        genome_train, genome_test, int(config["genome_pcs"]), seed
    )
    qtl_train_parts, qtl_test_parts, kept_blocks = [], [], []
    for block_name, original_idx in groups:
        block_idx = original_to_filtered[original_idx]
        block_idx = block_idx[block_idx >= 0]
        if block_idx.size < 2:
            continue
        pc_train, pc_test = randomized_pcs(
            qtl_train[:, block_idx], qtl_test[:, block_idx],
            int(config["qtl_pcs_per_block"]),
            block_seed(seed, str(block_name), config, len(kept_blocks) + 1),
        )
        if pc_train.shape[1]:
            qtl_train_parts.append(pc_train)
            qtl_test_parts.append(pc_test)
            kept_blocks.append(block_name)
    if not qtl_train_parts:
        raise RuntimeError("No QTL blocks retained after fold-local QC")
    return genome_pc_train, genome_pc_test, np.hstack(qtl_train_parts), np.hstack(qtl_test_parts), kept_blocks


def fit_predict(y_train, designs, hyper):
    genome_train, genome_test, qtl_train, qtl_test, _ = designs
    genome_train, genome_test = base.standardize(genome_train, genome_test)
    qtl_train, qtl_test = base.standardize(qtl_train, qtl_test)
    baseline = Ridge(alpha=float(hyper["ridge_alpha"]))
    baseline.fit(genome_train, y_train)
    residual = y_train - baseline.predict(genome_train)
    ard = ARDRegression(
        max_iter=int(hyper["ard_max_iter"]),
        threshold_lambda=float(hyper["ard_threshold_lambda"]),
    )
    ard.fit(qtl_train, residual)
    return baseline.predict(genome_test) + float(hyper["qtl_scale"]) * ard.predict(qtl_test)


def genome_only_design(genome_train, genome_test, config, seed):
    genome_train, genome_test, _ = filtered_imputed(
        genome_train, genome_test, float(config["min_maf"]), float(config["max_marker_missing_rate"])
    )
    return randomized_pcs(genome_train, genome_test, int(config["genome_pcs"]), seed)


def baseline_predict(y_train, design, ridge_alpha):
    train, test = base.standardize(*design)
    model = Ridge(alpha=float(ridge_alpha)).fit(train, y_train)
    return model.predict(test)


def tune_baseline(y, genome, train_idx, config, seed):
    candidates = [float(value) for value in config["ridge_alpha_grid"]]
    predicted = [[] for _ in candidates]
    observed = []
    for split, valid_local in enumerate(base.folds_for(len(train_idx), seed, int(config["inner_folds"]))):
        valid_idx = train_idx[valid_local]
        inner_train = np.setdiff1d(train_idx, valid_idx, assume_unique=True)
        design = genome_only_design(genome[inner_train], genome[valid_idx], config, seed + split * 97)
        for index, alpha in enumerate(candidates):
            predicted[index].append(baseline_predict(y[inner_train], design, alpha))
        observed.append(y[valid_idx])
    observed = np.concatenate(observed)
    scores = [pearsonr(observed, np.concatenate(values)).statistic for values in predicted]
    best = int(np.nanargmax(scores))
    return candidates[best], float(scores[best])


def hyper_grid(config):
    return [
        {
            "ridge_alpha": float(a), "qtl_scale": float(b),
            "ard_threshold_lambda": float(c), "ard_max_iter": int(config["ard_max_iter"]),
        }
        for a, b, c in itertools.product(
            config["ridge_alpha_grid"], config["qtl_scale_grid"], config["ard_threshold_lambda_grid"]
        )
    ]


def tune(y, genome, qtl, groups, train_idx, config, seed):
    grid = hyper_grid(config)
    predicted = [[] for _ in grid]
    observed = []
    for split, valid_local in enumerate(base.folds_for(len(train_idx), seed, int(config["inner_folds"]))):
        valid_idx = train_idx[valid_local]
        inner_train = np.setdiff1d(train_idx, valid_idx, assume_unique=True)
        designs = design_matrices(
            genome[inner_train], genome[valid_idx], qtl[inner_train], qtl[valid_idx],
            groups, config, seed + split * 97,
        )
        for index, hyper in enumerate(grid):
            predicted[index].append(fit_predict(y[inner_train], designs, hyper))
        observed.append(y[valid_idx])
    observed = np.concatenate(observed)
    scores = [
        pearsonr(observed, np.concatenate(values)).statistic if np.std(np.concatenate(values)) > EPS else -np.inf
        for values in predicted
    ]
    best = int(np.nanargmax(scores))
    return grid[best], float(scores[best])


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    ids, genome, panels, panel_markers, y, _, audit = base.load_inputs(config)
    panel_name, qtl = next(iter(panels.items()))
    block_names, block_indices = qtl_groups(
        panel_markers[panel_name], config["qtl_loci_path"], config.get("qtl_marker_coordinate_path")
    )
    groups = list(zip(block_names, block_indices))
    records, settings = [], []
    for repeat in range(1, int(config["outer_repeats"]) + 1):
        for fold, test_idx in enumerate(base.folds_for(len(ids), int(config["seed"]) + repeat, int(config["outer_folds"])), 1):
            train_idx = np.setdiff1d(np.arange(len(ids)), test_idx, assume_unique=True)
            if bool(config.get("include_genome_pc_ridge_baseline", False)):
                alpha, baseline_score = tune_baseline(
                    y, genome, train_idx, config, int(config["seed"]) + repeat * 100 + fold
                )
                baseline_design = genome_only_design(
                    genome[train_idx], genome[test_idx], config, int(config["seed"]) + repeat * 1000 + fold
                )
                baseline_prediction = baseline_predict(y[train_idx], baseline_design, alpha)
                for sample_id, observed, predicted in zip(ids[test_idx], y[test_idx], baseline_prediction):
                    records.append({"repeat": repeat, "fold": fold, "model": "Genome_PC_Ridge", "sample_id": sample_id, "observed": observed, "predicted": predicted})
                settings.append({"repeat": repeat, "fold": fold, "model": "Genome_PC_Ridge", "inner_oof_r": baseline_score, "ridge_alpha": alpha})
            hyper, score = tune(y, genome, qtl, groups, train_idx, config, int(config["seed"]) + repeat * 100 + fold)
            designs = design_matrices(
                genome[train_idx], genome[test_idx], qtl[train_idx], qtl[test_idx],
                groups, config, int(config["seed"]) + repeat * 1000 + fold,
            )
            prediction = fit_predict(y[train_idx], designs, hyper)
            for sample_id, observed, predicted in zip(ids[test_idx], y[test_idx], prediction):
                records.append({"repeat": repeat, "fold": fold, "model": "QTL_Block_ARD", "sample_id": sample_id, "observed": observed, "predicted": predicted})
            settings.append({
                "repeat": repeat, "fold": fold, "model": "QTL_Block_ARD", "inner_oof_r": score, **hyper,
                "qtl_blocks_retained": ";".join(designs[-1]), "qtl_block_count": len(designs[-1]),
            })
            print(f"DONE repeat={repeat} fold={fold} inner_r={score:.4f}", flush=True)
    pred = pd.DataFrame(records)
    metrics = []
    for (repeat, model), frame in pred.groupby(["repeat", "model"]):
        metrics.append({
            "repeat": repeat, "model": model,
            "pearson_r": pearsonr(frame.observed, frame.predicted).statistic,
            "spearman_rho": spearmanr(frame.observed, frame.predicted).statistic,
            "rmse": mean_squared_error(frame.observed, frame.predicted) ** 0.5,
            "mae": mean_absolute_error(frame.observed, frame.predicted),
        })
    metric_df = pd.DataFrame(metrics)
    summary = metric_df.groupby("model", as_index=False).agg(
        pearson_mean=("pearson_r", "mean"), pearson_sd=("pearson_r", "std"),
        spearman_mean=("spearman_rho", "mean"), rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
    ).sort_values("pearson_mean", ascending=False)
    pred.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(settings).to_csv(output / "selected_hyperparameters.csv", index=False)
    metric_df.to_csv(output / "repeat_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv")
    (output / "manifest.json").write_text(json.dumps({"input_audit": audit, "config": config}, indent=2, default=str), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run(json.loads(Path(arguments.config).read_text(encoding="utf-8")), Path(arguments.output))
