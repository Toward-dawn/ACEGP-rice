"""Adaptive functional-prior genomic prediction.

The outer test fold is never used for model screening, hyperparameter tuning,
feature construction, or ensemble weighting. Candidate background learners
are screened from inner out-of-fold predictions, then the selected learners
are evaluated with genome-only, QTL-DEG, and matched-random representations.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


EPS = 1e-12
CORE_MODELS = (
    "GBLUP", "RKHS", "Ridge", "ElasticNet", "LASSO", "BayesianRidge",
    "PLS", "SVR_RBF", "KNN", "ExtraTrees", "RandomForest",
    "GradientBoosting", "HistGradientBoosting", "AdaBoost", "MLP",
    "CatBoost_D4",
)
OPTIONAL_MODELS = ("XGBoost", "LightGBM")
DEFAULT_MODELS = CORE_MODELS


def available_models() -> tuple[str, ...]:
    core = [
        model for model in CORE_MODELS
        if model != "CatBoost_D4" or CatBoostRegressor is not None
    ]
    optional = []
    if XGBRegressor is not None:
        optional.append("XGBoost")
    if LGBMRegressor is not None:
        optional.append("LightGBM")
    return tuple(core) + tuple(optional)


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def pearson(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) <= EPS or np.std(pred) <= EPS:
        return float("nan")
    return float(pearsonr(y, pred)[0])


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": pearson(y, pred),
        "spearman_rho": float(spearmanr(y, pred).statistic),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
    }


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def folds_for(n: int, seed: int, folds: int) -> list[np.ndarray]:
    if folds < 2 or folds > n:
        raise ValueError(f"Invalid folds={folds} for n={n}")
    order = np.random.default_rng(seed).permutation(n)
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(order, folds)]


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_wide_table(path: Path, options: dict[str, Any] | None = None) -> pd.DataFrame:
    options = options or {}
    if path.suffix.lower() == ".raw":
        return pd.read_csv(path, sep=r"\s+", na_values=["NA", "NaN", ""])
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", na_values=["NA", "NaN", ""])
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=options.get("sheet_name", 0), na_values=["NA", "NaN", ""])
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    if path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        key = options.get("hdf_key")
        if not key:
            raise ValueError("HDF5 input requires hdf_key in the configuration")
        return pd.read_hdf(path, key=key)
    return pd.read_csv(path, na_values=["NA", "NaN", ""])


def read_plink_bed(bed_path: Path, bim_path: Path, fam_path: Path):
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, dtype=str)
    bim = pd.read_csv(bim_path, sep=r"\s+", header=None, dtype=str)
    if fam.shape[1] < 2 or bim.shape[1] < 2:
        raise ValueError("Invalid PLINK FAM or BIM file")
    sample_ids = fam.iloc[:, 1].astype(str).to_numpy()
    marker_names = bim.iloc[:, 1].astype(str).tolist()
    n_samples, n_markers = len(sample_ids), len(marker_names)
    bytes_per_marker = (n_samples + 3) // 4
    with bed_path.open("rb") as handle:
        magic = handle.read(3)
    if magic != b"\x6c\x1b\x01":
        raise ValueError("BED must be standard PLINK 1 SNP-major format")
    expected_size = 3 + n_markers * bytes_per_marker
    if bed_path.stat().st_size != expected_size:
        raise ValueError(
            f"BED size mismatch: expected {expected_size}, observed {bed_path.stat().st_size}"
        )
    encoded = np.memmap(
        bed_path, mode="r", dtype=np.uint8, offset=3,
        shape=(n_markers, bytes_per_marker),
    )
    sample_index = np.arange(n_samples)
    codes = (
        np.asarray(encoded)[:, sample_index // 4]
        >> (2 * (sample_index % 4))
    ) & 3
    lookup = np.asarray([2.0, np.nan, 1.0, 0.0], dtype=np.float32)
    matrix = lookup[codes].T.copy()
    return sample_ids, matrix, marker_names


def read_vcf_dosage(path: Path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    sample_ids = None
    markers = []
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                sample_ids = np.asarray(fields[9:], dtype=str)
                continue
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if sample_ids is None or len(fields) < 10:
                raise ValueError("VCF header or genotype columns are missing")
            chrom, pos, variant_id, ref, alt = fields[:5]
            if "," in alt:
                continue
            marker = variant_id if variant_id not in {"", "."} else f"{chrom}:{pos}:{ref}:{alt}"
            format_fields = fields[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF variant {marker} has no GT field")
            gt_index = format_fields.index("GT")
            dosage = []
            for sample_field in fields[9:]:
                parts = sample_field.split(":")
                gt = parts[gt_index] if gt_index < len(parts) else "."
                alleles = gt.replace("|", "/").split("/")
                if len(alleles) != 2 or any(allele == "." for allele in alleles):
                    dosage.append(np.nan)
                elif all(allele in {"0", "1"} for allele in alleles):
                    dosage.append(float(sum(allele == "1" for allele in alleles)))
                else:
                    dosage.append(np.nan)
            markers.append(marker)
            rows.append(dosage)
    if sample_ids is None:
        raise ValueError("VCF #CHROM header was not found")
    if not rows:
        raise ValueError("VCF contains no usable biallelic variants")
    matrix = np.asarray(rows, dtype=np.float32).T
    return sample_ids, matrix, markers


def load_matrix(
    path: Path,
    id_column: str,
    marker_start_column: int,
    options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    options = options or {}
    input_format = str(options.get("format", "auto")).lower()
    if input_format == "plink" or path.suffix.lower() == ".bed":
        prefix = path.with_suffix("") if path.suffix.lower() == ".bed" else path
        return read_plink_bed(
            prefix.with_suffix(".bed"),
            Path(options.get("bim_path", prefix.with_suffix(".bim"))),
            Path(options.get("fam_path", prefix.with_suffix(".fam"))),
        )
    if input_format == "vcf" or path.name.lower().endswith((".vcf", ".vcf.gz")):
        return read_vcf_dosage(path)
    if input_format == "npy" or path.suffix.lower() == ".npy":
        matrix = np.asarray(np.load(path, mmap_mode="r"), dtype=float)
        order_path = Path(options["sample_order_path"])
        order_frame = read_wide_table(order_path, options)
        ids = order_frame[id_column].astype(str).to_numpy()
        marker_path = options.get("marker_names_path")
        if marker_path:
            marker_frame = read_wide_table(Path(marker_path), options)
            marker_column = options.get("marker_name_column", "marker_id")
            marker_columns = marker_frame[marker_column].astype(str).tolist()
        else:
            marker_columns = [f"feature_{index + 1}" for index in range(matrix.shape[1])]
        if matrix.shape[0] != len(ids):
            raise ValueError("NPY rows do not match sample_order_path")
        return ids, matrix, marker_columns
    frame = read_wide_table(path, options)
    if id_column not in frame.columns:
        raise ValueError(f"{path} does not contain sample ID column {id_column!r}")
    ids = frame[id_column].astype(str).to_numpy()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    marker_columns = [column for column in frame.columns[marker_start_column:] if column != id_column]
    if not marker_columns:
        raise ValueError(f"No marker columns found in {path}")
    numeric = frame[marker_columns].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(float)
    return ids, matrix, marker_columns


def load_functional_matrix(
    functional: dict[str, Any], target_ids: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    path = Path(functional["path"])
    if path.suffix.lower() == ".npy":
        matrix = np.asarray(np.load(path, mmap_mode="r"), dtype=float)
        order_path = Path(functional["sample_order_path"])
        order_frame = read_wide_table(order_path, functional)
        order_column = functional.get("sample_id_column", "sample_id")
        source_ids = order_frame[order_column].astype(str).to_numpy()
        marker_path = functional.get("marker_names_path")
        if marker_path:
            marker_frame = read_wide_table(Path(marker_path), functional)
            marker_column = functional.get("marker_name_column", "marker_id")
            marker_names = marker_frame[marker_column].astype(str).tolist()
        elif functional.get("allow_positional_marker_alignment", False):
            marker_names = [f"feature_{index + 1}" for index in range(matrix.shape[1])]
        else:
            raise ValueError(
                f"NPY functional panel {path} requires marker_names_path. "
                "Set allow_positional_marker_alignment=true only when training and prediction "
                "matrices are guaranteed to have exactly the same column order."
            )
    else:
        source_ids, matrix, marker_names = load_matrix(
            path,
            functional.get("sample_id_column", "sample_id"),
            int(functional.get("marker_start_column", 1)),
            functional,
        )
    if matrix.shape[0] != len(source_ids):
        raise ValueError("Functional matrix rows do not match its sample-order file")
    if matrix.shape[1] != len(marker_names):
        raise ValueError("Functional matrix columns do not match its marker-name file")
    if len(marker_names) != len(set(marker_names)):
        raise ValueError(f"Duplicate marker names in functional panel {path}")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"Duplicate sample IDs in functional panel {path}")
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in target_ids if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"Functional panel is missing {len(missing)} accessions: {missing[:5]}")
    return matrix[[lookup[sample_id] for sample_id in target_ids]], marker_names


def functional_panel_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    panels = config.get("functional_panels")
    if panels is None and config.get("functional_panel"):
        panels = [{"name": "QTLDEG", **config["functional_panel"]}]
    if panels is None:
        return []
    if not panels:
        return []
    names = [str(panel.get("name", "")).strip() for panel in panels]
    if any(not name for name in names):
        raise ValueError("Every functional panel requires a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate functional panel names: {names}")
    if any(name == "Genome_only" or name.startswith("RandomMatched_") for name in names):
        raise ValueError("Panel names cannot be Genome_only or start with RandomMatched_")
    return panels


def load_inputs(config: dict[str, Any]):
    genotype_cfg = config["genotype"]
    phenotype_cfg = config["phenotype"]
    genotype_path = Path(genotype_cfg["path"])
    phenotype_path = Path(phenotype_cfg["path"])
    genotype_ids, genotype, marker_names = load_matrix(
        genotype_path,
        genotype_cfg.get("sample_id_column", "IID"),
        int(genotype_cfg.get("marker_start_column", 6)),
        genotype_cfg,
    )
    if len(genotype_ids) != len(set(genotype_ids)):
        raise ValueError("Duplicate sample IDs in genotype input")
    if len(marker_names) != len(set(marker_names)):
        duplicates = pd.Series(marker_names)[pd.Series(marker_names).duplicated()].head(10).tolist()
        raise ValueError(f"Duplicate marker names in genotype input: {duplicates}")
    if genotype.ndim != 2 or genotype.shape[0] != len(genotype_ids) or genotype.shape[1] == 0:
        raise ValueError("Genotype input is empty or has inconsistent dimensions")
    finite_genotype = genotype[np.isfinite(genotype)]
    if finite_genotype.size == 0:
        raise ValueError("Genotype input contains no finite values")
    if not genotype_cfg.get("allow_outside_dosage", False):
        if float(finite_genotype.min()) < 0.0 or float(finite_genotype.max()) > 2.0:
            raise ValueError(
                "Genotype dosage must be within [0, 2]. Set allow_outside_dosage=true "
                "only when the genome input intentionally contains continuous features."
            )
    phenotype = read_wide_table(phenotype_path, phenotype_cfg)
    phenotype_id = phenotype_cfg.get("sample_id_column", "sample_id")
    if phenotype_id not in phenotype.columns:
        raise ValueError(f"Phenotype file does not contain {phenotype_id!r}")
    phenotype[phenotype_id] = phenotype[phenotype_id].astype(str)
    if phenotype[phenotype_id].duplicated().any():
        raise ValueError("Duplicate phenotype sample IDs")
    phenotype = phenotype.set_index(phenotype_id)
    shared_ids = np.asarray([sample_id for sample_id in genotype_ids if sample_id in phenotype.index], dtype=str)
    genotype_only_ids = int(sum(sample_id not in phenotype.index for sample_id in genotype_ids))
    phenotype_only_ids = int(sum(sample_id not in set(genotype_ids) for sample_id in phenotype.index))
    expected = config.get("expected_accessions")
    if expected is not None and len(shared_ids) != int(expected):
        raise RuntimeError(f"Expected {expected} matched accessions, found {len(shared_ids)}")
    if len(shared_ids) < 20:
        raise RuntimeError(f"Only {len(shared_ids)} matched accessions; genomic prediction is not reliable")
    genotype_lookup = {sample_id: index for index, sample_id in enumerate(genotype_ids)}
    genotype = genotype[[genotype_lookup[sample_id] for sample_id in shared_ids]]
    phenotype = phenotype.loc[shared_ids]
    target_name = phenotype_cfg.get("target", "loss_SF")
    if target_name in phenotype.columns:
        y = phenotype[target_name].to_numpy(float)
    elif target_name == "loss_SF":
        y = (
            phenotype[phenotype_cfg.get("control_column", "SF_CONT")]
            - phenotype[phenotype_cfg.get("heat_column", "SF_HT")]
        ).to_numpy(float)
    else:
        raise ValueError(f"Cannot construct target {target_name!r}")
    finite_target = np.isfinite(y)
    dropped_target = int((~finite_target).sum())
    if dropped_target:
        shared_ids = shared_ids[finite_target]
        genotype = genotype[finite_target]
        phenotype = phenotype.iloc[np.flatnonzero(finite_target)]
        y = y[finite_target]
    sample_missing = 1.0 - np.isfinite(genotype).mean(axis=1)
    observed = np.isfinite(genotype)
    sample_heterozygosity = ((genotype == 1) & observed).sum(axis=1) / np.maximum(observed.sum(axis=1), 1)
    max_sample_missing = float(config.get("max_sample_missing_rate", 1.0))
    max_sample_heterozygosity = float(config.get("max_sample_heterozygosity", 1.0))
    sample_pass = (
        (sample_missing <= max_sample_missing)
        & (sample_heterozygosity <= max_sample_heterozygosity)
    )
    dropped_sample_qc = int((~sample_pass).sum())
    if dropped_sample_qc:
        shared_ids = shared_ids[sample_pass]
        genotype = genotype[sample_pass]
        phenotype = phenotype.iloc[np.flatnonzero(sample_pass)]
        y = y[sample_pass]
    if len(y) < 20 or np.std(y) <= EPS:
        raise RuntimeError("Too few samples or no target variation after sample-level QC")
    if len(y) < 20 or np.std(y) <= EPS:
        raise RuntimeError("Target has too few finite observations or no usable variation")
    panel_cfgs = functional_panel_configs(config)
    loaded_panels = {
        str(panel["name"]): load_functional_matrix(panel, shared_ids)
        for panel in panel_cfgs
    }
    panels = {name: loaded[0] for name, loaded in loaded_panels.items()}
    panel_marker_names = {name: loaded[1] for name, loaded in loaded_panels.items()}
    for name, matrix in panels.items():
        if matrix.ndim != 2 or matrix.shape[0] != len(shared_ids) or matrix.shape[1] == 0:
            raise ValueError(f"Functional panel {name!r} is empty or dimensionally inconsistent")
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            raise ValueError(f"Functional panel {name!r} contains no finite values")
        if not config.get("allow_outside_dosage", False):
            if float(finite.min()) < 0.0 or float(finite.max()) > 2.0:
                raise ValueError(f"Functional panel {name!r} dosage must be within [0, 2]")
    audit = {
        "n_accessions": int(len(shared_ids)),
        "n_genome_markers": int(genotype.shape[1]),
        "genome_genotyping_rate": float(np.isfinite(genotype).mean()),
        "target": target_name,
        "dropped_missing_target": dropped_target,
        "dropped_sample_qc": dropped_sample_qc,
        "genotype_only_sample_ids": genotype_only_ids,
        "phenotype_only_sample_ids": phenotype_only_ids,
        "max_sample_missing_rate": max_sample_missing,
        "max_sample_heterozygosity": max_sample_heterozygosity,
        "sample_missing_rate_mean": float(sample_missing[sample_pass].mean()),
        "sample_missing_rate_max": float(sample_missing[sample_pass].max()),
        "sample_heterozygosity_mean": float(sample_heterozygosity[sample_pass].mean()),
        "sample_heterozygosity_max": float(sample_heterozygosity[sample_pass].max()),
        "target_mean": float(np.mean(y)),
        "target_sd": float(np.std(y, ddof=1)),
        "target_min": float(np.min(y)),
        "target_q1": float(np.quantile(y, 0.25)),
        "target_median": float(np.median(y)),
        "target_q3": float(np.quantile(y, 0.75)),
        "target_max": float(np.max(y)),
        "genotype_path": str(genotype_path),
        "phenotype_path": str(phenotype_path),
        "genotype_sha256": sha256(genotype_path),
        "phenotype_sha256": sha256(phenotype_path),
    }
    for panel in panel_cfgs:
        name = str(panel["name"])
        audit[f"{name}_markers"] = int(panels[name].shape[1])
        audit[f"{name}_genotyping_rate"] = float(np.isfinite(panels[name]).mean())
        audit[f"{name}_path"] = str(panel["path"])
    return shared_ids, genotype, panels, panel_marker_names, y, marker_names, audit


def impute_from_train(train: np.ndarray, test: np.ndarray):
    means = np.nanmean(train, axis=0)
    means[~np.isfinite(means)] = 0.0
    return np.where(np.isfinite(train), train, means), np.where(np.isfinite(test), test, means)


def filter_markers_from_training(
    train: np.ndarray,
    test: np.ndarray,
    min_maf: float,
    max_missing_rate: float,
    label: str,
):
    """Apply marker QC using training genotypes only and mirror it to test data."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError(f"{label} train/test matrices have incompatible dimensions")
    missing_rate = 1.0 - np.isfinite(train).mean(axis=0)
    allele_frequency = np.nanmean(train, axis=0) / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    variance = np.nanvar(train, axis=0)
    keep = (
        np.isfinite(maf)
        & (missing_rate <= float(max_missing_rate))
        & (maf >= float(min_maf))
        & (variance > EPS)
    )
    if not np.any(keep):
        raise RuntimeError(
            f"No {label} markers passed training-fold QC: "
            f"min_maf={min_maf}, max_missing_rate={max_missing_rate}"
        )
    audit = {
        "label": label,
        "raw_markers": int(train.shape[1]),
        "passed_markers": int(keep.sum()),
        "removed_markers": int((~keep).sum()),
        "removed_missing": int((missing_rate > float(max_missing_rate)).sum()),
        "removed_low_maf_or_monomorphic": int(
            ((maf < float(min_maf)) | (variance <= EPS) | ~np.isfinite(maf)).sum()
        ),
        "min_maf": float(min_maf),
        "max_missing_rate": float(max_missing_rate),
    }
    return train[:, keep], test[:, keep], audit


def pca_scores(train: np.ndarray, test: np.ndarray, components: int):
    train, test = impute_from_train(np.asarray(train, float), np.asarray(test, float))
    center = train.mean(axis=0)
    train = train - center
    test = test - center
    if train.shape[1] == 0:
        return np.zeros((len(train), 0)), np.zeros((len(test), 0))
    u, s, vt = np.linalg.svd(train, full_matrices=False)
    keep = min(int(components), len(s), max(1, len(train) - 1))
    return u[:, :keep] * s[:keep], test @ vt[:keep].T


def build_feature_sets(genome_train, genome_test, addon_train, addon_test, config):
    genome_pc_train, genome_pc_test = pca_scores(
        genome_train, genome_test, int(config.get("genome_pcs", 60))
    )
    observed_train = np.isfinite(genome_train)
    observed_test = np.isfinite(genome_test)
    extra_train = np.column_stack((
        1.0 - observed_train.mean(axis=1),
        ((genome_train == 1) & observed_train).sum(axis=1) / np.maximum(observed_train.sum(axis=1), 1),
    ))
    extra_test = np.column_stack((
        1.0 - observed_test.mean(axis=1),
        ((genome_test == 1) & observed_test).sum(axis=1) / np.maximum(observed_test.sum(axis=1), 1),
    ))
    base_train = np.column_stack((genome_pc_train, extra_train))
    base_test = np.column_stack((genome_pc_test, extra_test))
    addon_pc_train, addon_pc_test = pca_scores(
        addon_train, addon_test, int(config.get("functional_pcs", 5))
    )
    return base_train, base_test, np.column_stack((base_train, addon_pc_train)), np.column_stack((base_test, addon_pc_test))


def standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd <= EPS] = 1.0
    return (train - mean) / sd, (test - mean) / sd


def linear_kernel(train: np.ndarray, test: np.ndarray):
    train, test = impute_from_train(train, test)
    center = train.mean(axis=0)
    train = train - center
    test = test - center
    k_train = train @ train.T / max(1, train.shape[1])
    k_test = test @ train.T / max(1, train.shape[1])
    scale = max(float(np.mean(np.diag(k_train))), EPS)
    return k_train / scale, k_test / scale


def rbf_kernel(train: np.ndarray, test: np.ndarray, bandwidth_scale: float = 1.0):
    train, test = standardize(train, test)
    train_dot = train @ train.T / max(1, train.shape[1])
    test_dot = test @ train.T / max(1, train.shape[1])
    train_diag = np.diag(train_dot)
    test_diag = np.sum(test * test, axis=1) / max(1, train.shape[1])
    train_d2 = np.clip(train_diag[:, None] + train_diag[None, :] - 2 * train_dot, 0, None)
    test_d2 = np.clip(test_diag[:, None] + train_diag[None, :] - 2 * test_dot, 0, None)
    upper = train_d2[np.triu_indices_from(train_d2, k=1)]
    bandwidth = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    bandwidth *= float(bandwidth_scale)
    return np.exp(-train_d2 / bandwidth), np.exp(-test_d2 / bandwidth)


def kernel_predict(k_train, k_test, y_train, lam):
    center = float(np.mean(y_train))
    alpha = np.linalg.solve(k_train + float(lam) * np.eye(len(y_train)), y_train - center)
    return center + k_test @ alpha


def prepare_fold_data(condition, genome_train, genome_test, addon_train, addon_test, config):
    """Compute fold-local transformations once and reuse them across models."""
    genome_train, genome_test, genome_qc = filter_markers_from_training(
        genome_train, genome_test,
        float(config.get("min_maf", 0.0)),
        float(config.get("max_marker_missing_rate", 1.0)),
        "genome",
    )
    addon_qc = None
    if condition != "Genome_only":
        addon_train, addon_test, addon_qc = filter_markers_from_training(
            addon_train, addon_test,
            float(config.get("functional_min_maf", config.get("min_maf", 0.0))),
            float(config.get(
                "functional_max_marker_missing_rate",
                config.get("max_marker_missing_rate", 1.0),
            )),
            "functional_panel",
        )
    base_train, base_test, augmented_train, augmented_test = build_feature_sets(
        genome_train, genome_test, addon_train, addon_test, config
    )
    feature_train, feature_test = (
        (base_train, base_test)
        if condition == "Genome_only"
        else (augmented_train, augmented_test)
    )
    train_z, test_z = standardize(feature_train, feature_test)
    bg_k_train, bg_k_test = linear_kernel(genome_train, genome_test)
    addon_k_train = addon_k_test = None
    if condition != "Genome_only":
        addon_k_train, addon_k_test = linear_kernel(addon_train, addon_test)
    rbf_kernels = {
        float(scale): rbf_kernel(feature_train, feature_test, float(scale))
        for scale in config.get("rkhs_bandwidth_scale_grid", [1.0])
    }
    return {
        "train_z": train_z,
        "test_z": test_z,
        "bg_k_train": bg_k_train,
        "bg_k_test": bg_k_test,
        "addon_k_train": addon_k_train,
        "addon_k_test": addon_k_test,
        "rbf_kernels": rbf_kernels,
        "genome_marker_qc": genome_qc,
        "functional_marker_qc": addon_qc,
    }


def model_grid(model: str, condition: str, config: dict[str, Any]):
    lambda_grid = config.get("lambda_grid", [0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    if model == "GBLUP":
        weights = [0.0] if condition == "Genome_only" else config.get("functional_weight_grid", [0.25, 0.5, 0.75])
        return [{"lambda": float(lam), "weight": float(weight)} for lam, weight in itertools.product(lambda_grid, weights)]
    if model == "RKHS":
        return [
            {"lambda": float(lam), "bandwidth_scale": float(scale)}
            for lam, scale in itertools.product(
                lambda_grid, config.get("rkhs_bandwidth_scale_grid", [1.0])
            )
        ]
    if model == "Ridge":
        return [{"alpha": float(alpha)} for alpha in config.get("ridge_alpha_grid", [1.0, 10.0, 30.0])]
    if model == "ElasticNet":
        return [
            {"alpha": float(alpha), "l1_ratio": float(ratio)}
            for alpha, ratio in itertools.product(
                config.get("elasticnet_alpha_grid", [0.01, 0.1, 1.0]),
                config.get("elasticnet_l1_ratio_grid", [0.1, 0.5, 0.9]),
            )
        ]
    if model == "LASSO":
        return [{"alpha": float(alpha)} for alpha in config.get("lasso_alpha_grid", [0.01, 0.1, 1.0])]
    if model == "PLS":
        return [{"components": int(value)} for value in config.get("pls_component_grid", [2, 5, 10])]
    if model == "SVR_RBF":
        return [
            {"C": float(c), "epsilon": float(epsilon)}
            for c, epsilon in itertools.product(
                config.get("svr_c_grid", [1.0, 10.0, 100.0]),
                config.get("svr_epsilon_grid", [0.1, 1.0]),
            )
        ]
    if model == "KNN":
        return [{"neighbors": int(value)} for value in config.get("knn_grid", [5, 10, 20])]
    if model == "MLP":
        return [{"hidden": int(value)} for value in config.get("mlp_hidden_grid", [16, 32, 64])]
    return [{}]


def fit_predict_prepared(model, condition, prepared, y_train, hyper, seed):
    if model == "GBLUP":
        bg_train = prepared["bg_k_train"]
        bg_test = prepared["bg_k_test"]
        weight = float(hyper.get("weight", 0.0))
        if weight > 0:
            bg_train = (1.0 - weight) * bg_train + weight * prepared["addon_k_train"]
            bg_test = (1.0 - weight) * bg_test + weight * prepared["addon_k_test"]
        return kernel_predict(bg_train, bg_test, y_train, hyper["lambda"])
    if model == "RKHS":
        rbf_train, rbf_test = prepared["rbf_kernels"][float(hyper.get("bandwidth_scale", 1.0))]
        return kernel_predict(
            rbf_train, rbf_test, y_train, hyper["lambda"]
        )
    train_z, test_z = prepared["train_z"], prepared["test_z"]
    if model == "Ridge":
        estimator = Ridge(alpha=hyper["alpha"])
    elif model == "ElasticNet":
        estimator = ElasticNet(
            alpha=hyper["alpha"], l1_ratio=hyper["l1_ratio"], max_iter=10000, random_state=seed
        )
    elif model == "LASSO":
        estimator = Lasso(alpha=hyper["alpha"], max_iter=10000, random_state=seed)
    elif model == "BayesianRidge":
        estimator = BayesianRidge()
    elif model == "PLS":
        estimator = PLSRegression(n_components=min(hyper["components"], train_z.shape[1], len(train_z) - 1))
    elif model == "SVR_RBF":
        estimator = SVR(kernel="rbf", C=hyper["C"], epsilon=hyper["epsilon"], gamma="scale")
    elif model == "KNN":
        estimator = KNeighborsRegressor(n_neighbors=min(hyper["neighbors"], len(train_z) - 1), weights="distance")
    elif model == "ExtraTrees":
        estimator = ExtraTreesRegressor(
            n_estimators=300, max_features=0.5, min_samples_leaf=4,
            max_depth=8, random_state=seed, n_jobs=-1,
        )
    elif model == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=400, max_features=0.7, min_samples_leaf=5,
            max_depth=6, random_state=seed, n_jobs=-1,
        )
    elif model == "CatBoost_D4":
        estimator = CatBoostRegressor(
            iterations=250, depth=4, learning_rate=0.035, l2_leaf_reg=12.0,
            loss_function="RMSE", random_seed=seed, verbose=False,
            allow_writing_files=False, thread_count=-1,
        )
    elif model == "GradientBoosting":
        estimator = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=2,
            min_samples_leaf=5, loss="huber", random_state=seed,
        )
    elif model == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=5.0, min_samples_leaf=10, random_state=seed,
        )
    elif model == "AdaBoost":
        estimator = AdaBoostRegressor(
            n_estimators=300, learning_rate=0.025, loss="square", random_state=seed,
        )
    elif model == "MLP":
        estimator = MLPRegressor(
            hidden_layer_sizes=(hyper["hidden"],), activation="relu", solver="adam",
            alpha=1.0, learning_rate_init=0.003, max_iter=2000,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=50,
            random_state=seed,
        )
    elif model == "XGBoost":
        if XGBRegressor is None:
            raise RuntimeError("XGBoost requested but xgboost is not installed")
        estimator = XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.025, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=10.0, objective="reg:squarederror",
            random_state=seed, n_jobs=-1,
        )
    elif model == "LightGBM":
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM requested but lightgbm is not installed")
        estimator = LGBMRegressor(
            n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown model {model}")
    estimator.fit(train_z, y_train)
    return np.asarray(estimator.predict(test_z), dtype=float).reshape(-1)


def fit_predict(model, condition, genome_train, genome_test, addon_train, addon_test, y_train, hyper, seed, config):
    prepared = prepare_fold_data(
        condition, genome_train, genome_test, addon_train, addon_test, config
    )
    return fit_predict_prepared(model, condition, prepared, y_train, hyper, seed)


def prepare_inner_fold_cache(condition, genome, addon, folds, config):
    cache = []
    for valid in folds:
        fit = np.setdiff1d(np.arange(len(genome)), valid, assume_unique=True)
        prepared = prepare_fold_data(
            condition, genome[fit], genome[valid], addon[fit], addon[valid], config
        )
        cache.append((fit, valid, prepared))
    return cache


def inner_oof(model, condition, genome, addon, y, folds, config, seed, fold_cache=None):
    candidates = model_grid(model, condition, config)
    predictions = [np.full(len(y), np.nan) for _ in candidates]
    if fold_cache is None:
        fold_cache = prepare_inner_fold_cache(condition, genome, addon, folds, config)
    for fold_no, (fit, valid, prepared) in enumerate(fold_cache):
        for candidate_no, hyper in enumerate(candidates):
            predictions[candidate_no][valid] = fit_predict_prepared(
                model, condition, prepared, y[fit], hyper,
                seed + candidate_no * 100 + fold_no,
            )
    best_score = -np.inf
    best_hyper = None
    best_prediction = None
    for hyper, prediction in zip(candidates, predictions):
        score = pearson(y, prediction)
        if np.isfinite(score) and score > best_score:
            best_score, best_hyper, best_prediction = score, hyper, prediction
    if best_hyper is None or best_prediction is None:
        raise RuntimeError(f"Inner screening failed for {model}/{condition}")
    return best_prediction, best_hyper, float(best_score)


def save_prediction_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_prediction_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("bundle_format") != "functional_gp_prediction_bundle_v1":
        raise ValueError("Unsupported or invalid prediction bundle")
    return bundle


def align_markers(
    matrix: np.ndarray,
    observed_names: list[str],
    required_names: list[str],
    missing_policy: str = "error",
    max_missing_fraction: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    if len(observed_names) != len(set(observed_names)):
        raise ValueError("Duplicate marker names in prediction genotype")
    lookup = {name: index for index, name in enumerate(observed_names)}
    missing = [name for name in required_names if name not in lookup]
    missing_fraction = len(missing) / max(len(required_names), 1)
    if missing and missing_policy == "error":
        preview = ", ".join(missing[:10])
        raise ValueError(f"Prediction genotype is missing {len(missing)} required markers: {preview}")
    if missing and missing_policy != "mean_impute":
        raise ValueError(f"Unknown missing_marker_policy {missing_policy!r}")
    if missing_fraction > float(max_missing_fraction):
        raise ValueError(
            f"Prediction input is missing {len(missing)}/{len(required_names)} required markers "
            f"({missing_fraction:.2%}), exceeding max_missing_marker_fraction="
            f"{float(max_missing_fraction):.2%}"
        )
    aligned = np.full((matrix.shape[0], len(required_names)), np.nan, dtype=float)
    present_positions = [index for index, name in enumerate(required_names) if name in lookup]
    source_positions = [lookup[required_names[index]] for index in present_positions]
    if present_positions:
        aligned[:, present_positions] = matrix[:, source_positions]
    return aligned, {
        "required_markers": len(required_names),
        "available_markers": len(observed_names),
        "missing_required_markers": len(missing),
        "missing_required_marker_fraction": float(missing_fraction),
    }


def load_unphenotyped_inputs(
    config: dict[str, Any], bundle: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    genotype_cfg = config["genotype"]
    genotype_ids, genotype, marker_names = load_matrix(
        Path(genotype_cfg["path"]),
        genotype_cfg.get("sample_id_column", "IID"),
        int(genotype_cfg.get("marker_start_column", 6)),
        genotype_cfg,
    )
    if len(genotype_ids) != len(set(genotype_ids)):
        raise ValueError("Duplicate sample IDs in prediction genotype")
    missing_policy = str(config.get("missing_marker_policy", "error"))
    max_missing_fraction = float(config.get("max_missing_marker_fraction", 0.0))
    genotype, marker_audit = align_markers(
        genotype, marker_names, list(bundle["genome_marker_names"]),
        missing_policy, max_missing_fraction,
    )
    condition = str(bundle["condition"])
    if condition == "Genome_only":
        addon = np.zeros((len(genotype_ids), 1), dtype=float)
    else:
        panel_cfg = config.get("functional_panel")
        if panel_cfg is None:
            panels = config.get("functional_panels", [])
            expected_name = bundle.get("panel_name")
            panel_cfg = next((panel for panel in panels if panel.get("name") == expected_name), None)
        if panel_cfg is None:
            raise ValueError(
                f"Prediction config must provide functional panel {bundle.get('panel_name')!r}"
            )
        addon, addon_marker_names = load_functional_matrix(
            panel_cfg, np.asarray(genotype_ids, dtype=str)
        )
        addon, addon_audit = align_markers(
            addon, addon_marker_names, list(bundle["functional_marker_names"]),
            missing_policy, max_missing_fraction,
        )
    audit = {
        **marker_audit,
        "n_prediction_accessions": int(len(genotype_ids)),
        "prediction_genotyping_rate": float(np.isfinite(genotype).mean()),
    }
    if condition != "Genome_only":
        audit.update({f"functional_{key}": value for key, value in addon_audit.items()})
    return np.asarray(genotype_ids, dtype=str), genotype, addon, audit


def predict_from_bundle(
    bundle_path: Path, prediction_config: dict[str, Any], output: Path
) -> dict[str, Any]:
    bundle = load_prediction_bundle(bundle_path)
    ids, genome, addon, audit = load_unphenotyped_inputs(prediction_config, bundle)
    condition = str(bundle["condition"])
    prepared = prepare_fold_data(
        condition,
        bundle["reference_genome"], genome,
        bundle["reference_addon"], addon,
        bundle["model_config"],
    )
    columns = []
    for model_number, model in enumerate(bundle["models"]):
        columns.append(fit_predict_prepared(
            model, condition, prepared, bundle["reference_target"],
            bundle["hyperparameters"][model],
            int(bundle["seed"]) + model_number,
        ))
    matrix = np.column_stack(columns)
    if bundle["strategy"] == "adaptive":
        prediction = float(bundle["intercept"]) + matrix @ np.asarray(bundle["weights"], float)
    elif bundle["strategy"] == "geg2p":
        prediction = matrix @ np.asarray(bundle["weights"], float)
    else:
        raise ValueError(f"Unknown deployment strategy {bundle['strategy']!r}")
    reference_mean = float(bundle.get("reference_mean", np.mean(bundle["reference_target"])))
    reference_sd = float(bundle.get("reference_sd", np.std(bundle["reference_target"], ddof=1)))
    genomic_breeding_value = prediction - reference_mean
    standardized_gebv = genomic_breeding_value / max(reference_sd, EPS)
    cv_rmse = float(bundle.get("deployment_cv_rmse", np.nan))
    base_reliability = float(bundle.get("deployment_cv_reliability", np.nan))
    marker_coverage = 1.0 - float(audit.get("missing_required_marker_fraction", 0.0))
    functional_coverage = 1.0 - float(
        audit.get("functional_missing_required_marker_fraction", 0.0)
    )
    reliability = np.clip(base_reliability * min(marker_coverage, functional_coverage), 0.0, 1.0)
    model_disagreement = (
        np.std(matrix, axis=1, ddof=1) if matrix.shape[1] > 1 else np.zeros(len(ids))
    )
    selection_direction = str(bundle.get("selection_direction", "higher_is_better"))
    lower_is_better = selection_direction == "lower_is_better"
    prediction_rank = pd.Series(prediction).rank(
        method="min", ascending=lower_is_better
    ).astype(int)
    if np.isfinite(cv_rmse):
        interval_low = prediction - 1.96 * cv_rmse
        interval_high = prediction + 1.96 * cv_rmse
    else:
        interval_low = np.full(len(ids), np.nan)
        interval_high = np.full(len(ids), np.nan)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({
        "sample_id": ids,
        "predicted_phenotype": prediction,
        f"predicted_{bundle['target']}": prediction,
        "genomic_breeding_value": genomic_breeding_value,
        "standardized_GEBV": standardized_gebv,
        "reliability": np.full(len(ids), reliability),
        "prediction_interval_low": interval_low,
        "prediction_interval_high": interval_high,
        "base_model_disagreement": model_disagreement,
        "breeding_rank": prediction_rank,
        "selection_direction": selection_direction,
    })
    result.to_csv(output, index=False)
    audit_path = output.with_name(output.stem + "_audit.json")
    audit_path.write_text(json.dumps({
        **audit, "target": bundle["target"], "strategy": bundle["strategy"],
        "condition": condition, "models": bundle["models"],
        "reference_mean": reference_mean, "reference_sd": reference_sd,
        "deployment_cv_rmse": cv_rmse,
        "deployment_cv_reliability": base_reliability,
        "reported_reliability_after_marker_coverage": float(reliability),
        "selection_direction": selection_direction,
        "gebv_definition": (
            "predicted phenotype minus reference-population mean; this is a genomic "
            "prediction score and is a classical additive GEBV only when the selected "
            "learner is an additive genomic model"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "success", "predictions": str(output.resolve()),
            "audit": str(audit_path.resolve()), "n_accessions": len(ids)}


def select_background_models(scores: dict[str, float], config: dict[str, Any]):
    finite_scores = {name: score for name, score in scores.items() if np.isfinite(score)}
    if not finite_scores:
        raise RuntimeError("All candidate background models failed inner-CV screening")
    ordered = sorted(finite_scores, key=lambda name: finite_scores[name], reverse=True)
    mode = str(config.get("selection_mode", "top_n")).lower()
    if mode == "top_n":
        top_n = int(config.get("selection_top_n", config.get("selection_top_k", 6)))
        if top_n < 1:
            raise ValueError("selection_top_n must be at least 1")
        return ordered[: min(top_n, len(ordered))]
    if mode != "within_margin":
        raise ValueError("selection_mode must be 'top_n' or 'within_margin'")
    best_score = scores[ordered[0]]
    margin = float(config.get("selection_margin", 0.03))
    minimum = float(config.get("minimum_inner_r", 0.0))
    top_k = int(config.get("selection_top_k", 3))
    selected = [name for name in ordered if scores[name] >= best_score - margin and scores[name] >= minimum][:top_k]
    return selected or [ordered[0]]


def select_diverse_models(
    scores: dict[str, float],
    predictions: dict[str, np.ndarray],
    y: np.ndarray,
    top_n: int = 5,
    diversity_penalty: float = 0.05,
):
    """Select one performance anchor and complementary residual patterns."""
    finite = [name for name, score in scores.items() if np.isfinite(score)]
    if not finite:
        raise RuntimeError("All candidate models failed diversity-aware screening")
    ordered = sorted(finite, key=lambda name: scores[name], reverse=True)
    selected = [ordered[0]]
    residuals = {name: np.asarray(y, float) - np.asarray(predictions[name], float) for name in finite}
    while len(selected) < min(int(top_n), len(ordered)):
        best_name = None
        best_value = -np.inf
        for name in ordered:
            if name in selected:
                continue
            correlations = []
            for chosen in selected:
                corr = pearson(residuals[name], residuals[chosen])
                correlations.append(abs(corr) if np.isfinite(corr) else 1.0)
            redundancy = max(correlations) if correlations else 0.0
            value = float(scores[name]) - float(diversity_penalty) * redundancy
            if value > best_value:
                best_name, best_value = name, value
        selected.append(best_name)
    return selected


def simplex_ensemble_fit_predict(
    inner_matrix: np.ndarray,
    y_train: np.ndarray,
    test_matrix: np.ndarray,
    ridge: float = 0.1,
    anchor_weight: float = 0.5,
):
    """Fit stable non-negative weights constrained to sum to one."""
    n_models = inner_matrix.shape[1]
    if n_models == 1:
        return test_matrix[:, 0], np.asarray([1.0]), 0.0
    prior = np.full(n_models, (1.0 - anchor_weight) / (n_models - 1), dtype=float)
    prior[0] = float(anchor_weight)
    scale = max(float(np.var(y_train)), EPS)

    def objective(weights):
        residual = y_train - inner_matrix @ weights
        return float(np.mean(residual ** 2) / scale + float(ridge) * np.sum((weights - prior) ** 2))

    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = np.asarray(result.x if result.success else prior, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights /= max(float(weights.sum()), EPS)
    intercept = float(np.mean(y_train - inner_matrix @ weights))
    return intercept + test_matrix @ weights, weights, intercept


def ensemble_fit_predict(inner_matrix, y_train, test_matrix):
    if inner_matrix.shape[1] == 1:
        return test_matrix[:, 0], np.asarray([1.0]), 0.0
    estimator = LinearRegression(positive=True, fit_intercept=True)
    estimator.fit(inner_matrix, y_train)
    coefficients = np.asarray(estimator.coef_, dtype=float)
    if coefficients.sum() <= EPS:
        coefficients = np.full(inner_matrix.shape[1], 1.0 / inner_matrix.shape[1])
        intercept = float(np.mean(y_train) - np.mean(inner_matrix @ coefficients))
    else:
        intercept = float(estimator.intercept_)
    return intercept + test_matrix @ coefficients, coefficients, intercept


def cross_fitted_anchor_gate(
    inner_matrix: np.ndarray,
    y_train: np.ndarray,
    inner_folds: list[np.ndarray],
    ridge: float = 0.1,
    anchor_weight: float = 0.7,
    margin: float = 0.0,
):
    """Compare a simplex ensemble with its anchor using a second CV layer."""
    n = len(y_train)
    if inner_matrix.shape[0] != n or inner_matrix.shape[1] < 1:
        raise ValueError("Invalid inner prediction matrix")
    meta_prediction = np.full(n, np.nan, dtype=float)
    all_indices = np.arange(n)
    for validation_idx in inner_folds:
        validation_idx = np.asarray(validation_idx, dtype=int)
        fit_idx = np.setdiff1d(all_indices, validation_idx, assume_unique=True)
        if len(fit_idx) < 2 or len(validation_idx) == 0:
            raise ValueError("Each gate fold needs training and validation samples")
        fold_prediction, _, _ = simplex_ensemble_fit_predict(
            inner_matrix[fit_idx], y_train[fit_idx], inner_matrix[validation_idx],
            ridge=ridge, anchor_weight=anchor_weight,
        )
        meta_prediction[validation_idx] = fold_prediction
    if not np.isfinite(meta_prediction).all():
        raise RuntimeError("Anchor gate did not produce complete cross-fitted predictions")
    anchor_score = pearson(y_train, inner_matrix[:, 0])
    ensemble_score = pearson(y_train, meta_prediction)
    use_ensemble = bool(
        np.isfinite(ensemble_score)
        and np.isfinite(anchor_score)
        and ensemble_score > anchor_score + float(margin)
    )
    return use_ensemble, anchor_score, ensemble_score


def anchor_residual_fit_predict(
    inner_matrix: np.ndarray,
    y_train: np.ndarray,
    test_matrix: np.ndarray,
    alpha: float,
):
    """Correct an anchor prediction with regularized cross-model deviations."""
    p = np.asarray(inner_matrix, dtype=float)
    y = np.asarray(y_train, dtype=float)
    test = np.asarray(test_matrix, dtype=float)
    if p.ndim != 2 or test.ndim != 2 or p.shape[1] != test.shape[1]:
        raise ValueError("Inner and test prediction matrices must have the same number of columns")
    if p.shape[0] != len(y) or p.shape[1] < 1:
        raise ValueError("Invalid anchor-residual prediction matrix")
    if not np.isfinite(p).all() or not np.isfinite(test).all() or not np.isfinite(y).all():
        raise ValueError("Anchor-residual inputs must be finite")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if p.shape[1] == 1:
        return test[:, 0], np.empty(0, dtype=float), 0.0

    anchor = p[:, 0]
    deviations = p[:, 1:] - anchor[:, None]
    test_deviations = test[:, 1:] - test[:, [0]]
    center = deviations.mean(axis=0)
    scale = deviations.std(axis=0, ddof=0)
    scale = np.where(scale > EPS, scale, 1.0)
    deviations_z = (deviations - center) / scale
    test_deviations_z = (test_deviations - center) / scale
    residual = y - anchor
    estimator = Ridge(alpha=float(alpha), fit_intercept=True)
    estimator.fit(deviations_z, residual)
    correction = estimator.predict(test_deviations_z)
    return anchor_test_plus_correction(test[:, 0], correction), estimator.coef_, estimator.intercept_


def anchor_test_plus_correction(anchor_test: np.ndarray, correction: np.ndarray):
    return np.asarray(anchor_test, dtype=float) + np.asarray(correction, dtype=float)


def cross_fitted_anchor_residual(
    inner_matrix: np.ndarray,
    y_train: np.ndarray,
    inner_folds: list[np.ndarray],
    alpha_grid: list[float],
    margin: float = 0.0,
):
    """Choose residual-correction strength using only cross-fitted inner predictions."""
    p = np.asarray(inner_matrix, dtype=float)
    y = np.asarray(y_train, dtype=float)
    anchor_score = pearson(y, p[:, 0])
    all_indices = np.arange(len(y))
    candidates = []
    for alpha in alpha_grid:
        meta_prediction = np.full(len(y), np.nan, dtype=float)
        for validation_idx in inner_folds:
            validation_idx = np.asarray(validation_idx, dtype=int)
            fit_idx = np.setdiff1d(all_indices, validation_idx, assume_unique=True)
            if len(fit_idx) < 2 or len(validation_idx) == 0:
                raise ValueError("Each residual-gate fold needs training and validation samples")
            prediction, _, _ = anchor_residual_fit_predict(
                p[fit_idx], y[fit_idx], p[validation_idx], alpha=float(alpha)
            )
            meta_prediction[validation_idx] = prediction
        if not np.isfinite(meta_prediction).all():
            raise RuntimeError("Residual gate did not produce complete cross-fitted predictions")
        score = pearson(y, meta_prediction)
        candidates.append((float(score), float(alpha), meta_prediction))
    candidates.sort(key=lambda item: (-item[0] if np.isfinite(item[0]) else np.inf, item[1]))
    best_score, best_alpha, best_prediction = candidates[0]
    use_correction = bool(
        np.isfinite(best_score)
        and np.isfinite(anchor_score)
        and best_score > anchor_score + float(margin)
    )
    return use_correction, anchor_score, best_score, best_alpha, best_prediction


def train_final_adaptive(config: dict[str, Any], bundle_path: Path) -> dict[str, Any]:
    """Lock a final adaptive model on all phenotyped reference accessions."""
    ids, genome, panels, panel_marker_names, y, marker_names, audit = load_inputs(config)
    seed = int(config.get("seed", 20260804))
    inner_folds_count = int(config.get("deployment_inner_folds", config.get("inner_folds", 5)))
    folds = folds_for(len(ids), seed + 700001, inner_folds_count)
    models = tuple(config.get("candidate_models", DEFAULT_MODELS))
    unknown = sorted(set(models) - set(available_models()))
    if unknown:
        raise ValueError(f"Unknown or unavailable candidate models: {unknown}")
    first_panel_name = next(iter(panels), None)
    requested_panel = config.get("deployment_panel", first_panel_name)
    if requested_panel in {None, "Genome_only"}:
        panel_name, condition, addon = None, "Genome_only", np.zeros((len(ids), 1))
    else:
        if requested_panel not in panels:
            raise ValueError(f"deployment_panel {requested_panel!r} is not in functional_panels")
        panel_name = str(requested_panel)
        condition = f"Genome_plus_{panel_name}"
        addon = panels[panel_name]

    background_addon = np.zeros((len(ids), 1))
    background_cache = prepare_inner_fold_cache(
        "Genome_only", genome, background_addon, folds, config
    )
    background_scores = {}
    for model_number, model in enumerate(models):
        _, _, score = inner_oof(
            model, "Genome_only", genome, background_addon, y, folds, config,
            seed + model_number, fold_cache=background_cache,
        )
        background_scores[model] = score
    selected = select_background_models(background_scores, config)

    condition_cache = prepare_inner_fold_cache(condition, genome, addon, folds, config)
    inner_columns, hyperparameters = [], {}
    for model_number, model in enumerate(selected):
        prediction, hyper, _ = inner_oof(
            model, condition, genome, addon, y, folds, config,
            seed + 1000 + model_number, fold_cache=condition_cache,
        )
        inner_columns.append(prediction)
        hyperparameters[model] = hyper
    inner_matrix = np.column_stack(inner_columns)
    deployment_oof, weights, intercept = ensemble_fit_predict(inner_matrix, y, inner_matrix)
    deployment_rmse = float(np.sqrt(np.mean((y - deployment_oof) ** 2)))
    phenotype_variance = float(np.var(y, ddof=1))
    cv_reliability = float(np.clip(
        1.0 - (deployment_rmse ** 2 / max(phenotype_variance, EPS)), 0.0, 1.0
    ))
    selection_direction = str(config.get("selection_direction", "higher_is_better"))
    if selection_direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("selection_direction must be higher_is_better or lower_is_better")
    bundle = {
        "bundle_format": "functional_gp_prediction_bundle_v1",
        "strategy": "adaptive", "target": audit["target"], "condition": condition,
        "panel_name": panel_name, "models": list(selected),
        "hyperparameters": hyperparameters, "weights": weights,
        "intercept": float(intercept), "seed": seed,
        "reference_ids": ids, "reference_genome": genome,
        "reference_addon": addon, "reference_target": y,
        "genome_marker_names": marker_names,
        "functional_marker_names": (
            panel_marker_names[panel_name] if panel_name is not None else []
        ),
        "model_config": config,
        "training_audit": audit, "background_inner_scores": background_scores,
        "reference_mean": float(np.mean(y)),
        "reference_sd": float(np.std(y, ddof=1)),
        "deployment_cv_rmse": deployment_rmse,
        "deployment_cv_reliability": cv_reliability,
        "selection_direction": selection_direction,
    }
    save_prediction_bundle(bundle, bundle_path)
    metadata_path = bundle_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({
        "strategy": "adaptive", "target": audit["target"], "condition": condition,
        "panel_name": panel_name, "n_reference_accessions": len(ids),
        "n_genome_markers": genome.shape[1], "n_panel_markers": addon.shape[1],
        "selected_models": selected, "weights": weights.tolist(),
        "intercept": float(intercept), "hyperparameters": hyperparameters,
        "deployment_cv_rmse": deployment_rmse,
        "deployment_cv_reliability": cv_reliability,
        "selection_direction": selection_direction,
        "test_phenotype_used_for_training_selection": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "success", "bundle": str(bundle_path.resolve()),
            "metadata": str(metadata_path.resolve()), "models": selected}


def preprocess_dataset(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Audit and standardize a reference dataset without fitting prediction models."""
    output.mkdir(parents=True, exist_ok=True)
    ids, genome, panels, panel_marker_names, y, marker_names, audit = load_inputs(config)
    missing_rate = 1.0 - np.isfinite(genome).mean(axis=0)
    allele_frequency = np.nanmean(genome, axis=0) / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    variance = np.nanvar(genome, axis=0)
    min_maf = float(config.get("min_maf", 0.05))
    max_missing = float(config.get("max_marker_missing_rate", 0.20))
    marker_pass = (
        np.isfinite(maf) & (maf >= min_maf)
        & (missing_rate <= max_missing) & (variance > EPS)
    )
    marker_qc = pd.DataFrame({
        "marker_id": marker_names,
        "missing_rate": missing_rate,
        "allele_frequency": allele_frequency,
        "maf": maf,
        "variance": variance,
        "pass_qc": marker_pass,
    })
    marker_qc_mode = str(config.get("marker_qc_output", "failed_only"))
    marker_qc_path = None
    if marker_qc_mode not in {"none", "failed_only", "all"}:
        raise ValueError("marker_qc_output must be none, failed_only or all")
    if marker_qc_mode != "none":
        marker_qc_path = output / "marker_qc.csv.gz"
        frame = marker_qc if marker_qc_mode == "all" else marker_qc.loc[~marker_pass]
        frame.to_csv(marker_qc_path, index=False, compression="gzip")

    sample_missing = 1.0 - np.isfinite(genome).mean(axis=1)
    observed = np.isfinite(genome)
    heterozygosity = ((genome == 1) & observed).sum(axis=1) / np.maximum(observed.sum(axis=1), 1)
    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))
    robust_z = np.zeros(len(y)) if mad <= EPS else 0.67448975 * (y - median) / mad
    phenotype_outlier_threshold = float(config.get("phenotype_outlier_robust_z", 3.5))
    samples = pd.DataFrame({
        "sample_id": ids,
        "target": y,
        "genotype_missing_rate": sample_missing,
        "heterozygosity_rate": heterozygosity,
        "phenotype_robust_z": robust_z,
        "phenotype_outlier_flag": np.abs(robust_z) > phenotype_outlier_threshold,
    })
    samples.to_csv(output / "sample_qc_and_phenotype.csv", index=False)

    passed_markers = np.asarray(marker_names, dtype=object)[marker_pass]
    pd.DataFrame({"marker_id": passed_markers}).to_csv(
        output / "passed_marker_names.csv", index=False
    )
    pd.DataFrame({"sample_id": ids}).to_csv(output / "sample_order.csv", index=False)
    if config.get("export_clean_matrices", False):
        np.save(output / "genotype_qc.npy", np.asarray(genome[:, marker_pass], dtype=np.float32))
        for panel_name, panel in panels.items():
            np.save(output / f"functional_panel_{panel_name}.npy", np.asarray(panel, dtype=np.float32))
            pd.DataFrame({"marker_id": panel_marker_names[panel_name]}).to_csv(
                output / f"functional_panel_{panel_name}_markers.csv", index=False
            )

    preprocessing_summary = {
        **audit,
        "raw_markers": int(genome.shape[1]),
        "passed_markers": int(marker_pass.sum()),
        "removed_markers": int((~marker_pass).sum()),
        "phenotype_outliers_flagged": int((np.abs(robust_z) > phenotype_outlier_threshold).sum()),
        "phenotype_outliers_removed": 0,
        "note": "Phenotype outliers are flagged, not automatically removed.",
        "marker_qc_output": marker_qc_mode,
        "clean_matrices_exported": bool(config.get("export_clean_matrices", False)),
    }
    (output / "preprocessing_summary.json").write_text(
        json.dumps(preprocessing_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "status": "success",
        "output_dir": str(output.resolve()),
        "n_accessions": int(len(ids)),
        "passed_markers": int(marker_pass.sum()),
        "summary": str((output / "preprocessing_summary.json").resolve()),
        "sample_qc": str((output / "sample_qc_and_phenotype.csv").resolve()),
        "marker_qc": str(marker_qc_path.resolve()) if marker_qc_path else None,
    }


def run(config: dict[str, Any], output: Path, progress_callback=None):
    output.mkdir(parents=True, exist_ok=True)
    ids, genome, panels, panel_marker_names, y, marker_names, audit = load_inputs(config)
    seed = int(config.get("seed", 20260804))
    repeats = int(config.get("outer_repeats", 10))
    outer_folds = int(config.get("outer_folds", 5))
    inner_folds = int(config.get("inner_folds", 3))
    models = tuple(config.get("candidate_models", DEFAULT_MODELS))
    unknown = sorted(set(models) - set(CORE_MODELS + OPTIONAL_MODELS))
    if unknown:
        raise ValueError(f"Unknown candidate models: {unknown}")
    unavailable = [model for model in models if model not in available_models()]
    if unavailable:
        raise RuntimeError(
            f"Requested optional models are not installed: {unavailable}. "
            f"Currently available models: {list(available_models())}"
        )
    background_addon = np.zeros((len(ids), 1), dtype=float)
    condition_addons: dict[str, np.ndarray] = {"Genome_only": background_addon}
    random_panel_markers: dict[str, int] = {}
    include_random_controls = bool(config.get("include_random_matched_controls", True))
    for panel_no, (panel_name, panel_matrix) in enumerate(panels.items()):
        condition_addons[f"Genome_plus_{panel_name}"] = panel_matrix
        if include_random_controls:
            random_columns = np.random.default_rng(seed + 99 + panel_no).choice(
                genome.shape[1], size=min(panel_matrix.shape[1], genome.shape[1]), replace=False
            )
            random_name = f"Genome_plus_RandomMatched_{panel_name}"
            condition_addons[random_name] = genome[:, random_columns]
            random_panel_markers[panel_name] = int(len(random_columns))
    conditions = tuple(condition_addons)
    prediction_rows = []
    screening_rows = []
    selection_rows = []
    ranking_rows = []
    weight_rows = []
    gate_rows = []
    residual_gate_rows = []
    marker_qc_rows = []
    for repeat in range(1, repeats + 1):
        for fold, test_idx in enumerate(folds_for(len(ids), seed + repeat, outer_folds), start=1):
            train_idx = np.setdiff1d(np.arange(len(ids)), test_idx, assume_unique=True)
            inner_folds_local = folds_for(len(train_idx), seed + repeat * 1000 + fold, inner_folds)
            inner_predictions: dict[tuple[str, str], np.ndarray] = {}
            best_hypers: dict[tuple[str, str], dict[str, float]] = {}
            inner_scores: dict[tuple[str, str], float] = {}
            addon = condition_addons["Genome_only"][train_idx]
            background_cache = prepare_inner_fold_cache(
                "Genome_only", genome[train_idx], addon, inner_folds_local, config
            )
            for model_no, model in enumerate(models):
                inner_pred, hyper, score = inner_oof(
                    model, "Genome_only", genome[train_idx], addon, y[train_idx], inner_folds_local,
                    config, seed + repeat * 10000 + fold * 100 + model_no,
                    fold_cache=background_cache,
                )
                inner_predictions[("Genome_only", model)] = inner_pred
                best_hypers[("Genome_only", model)] = hyper
                inner_scores[("Genome_only", model)] = score
                screening_rows.append({
                    "repeat": repeat, "fold": fold, "condition": "Genome_only",
                    "model": model, "inner_pearson_r": score,
                    "selected_hyperparameters": json.dumps(hyper, sort_keys=True),
                })
            background_scores = {model: inner_scores[("Genome_only", model)] for model in models}
            selected = select_background_models(background_scores, config)
            diverse_selected = select_diverse_models(
                background_scores,
                {model: inner_predictions[("Genome_only", model)] for model in models},
                y[train_idx],
                top_n=int(config.get("diverse_selection_top_n", 5)),
                diversity_penalty=float(config.get("diversity_penalty", 0.05)),
            )
            background_anchor_model = max(
                (model for model in ("GBLUP", "RKHS") if model in background_scores),
                key=lambda model: background_scores[model],
            )
            ordered_background = sorted(
                background_scores,
                key=lambda name: (
                    not np.isfinite(background_scores[name]),
                    -background_scores[name] if np.isfinite(background_scores[name]) else np.inf,
                ),
            )
            for rank, model in enumerate(ordered_background, start=1):
                ranking_rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "rank": rank,
                    "model": model,
                    "inner_pearson_r": background_scores[model],
                    "selected_for_next_stage": model in selected,
                })
            panel_models = (
                models
                if config.get("evaluate_all_models_with_panels", False)
                else tuple(dict.fromkeys(
                    list(selected)
                    + list(diverse_selected)
                    + [model for model in ("GBLUP", "RKHS") if model in models]
                ))
            )
            for condition in conditions[1:]:
                addon = condition_addons[condition][train_idx]
                panel_cache = prepare_inner_fold_cache(
                    condition, genome[train_idx], addon, inner_folds_local, config
                )
                for model_no, model in enumerate(panel_models):
                    inner_pred, hyper, score = inner_oof(
                        model, condition, genome[train_idx], addon, y[train_idx], inner_folds_local,
                        config, seed + repeat * 10000 + fold * 100 + model_no,
                        fold_cache=panel_cache,
                    )
                    inner_predictions[(condition, model)] = inner_pred
                    best_hypers[(condition, model)] = hyper
                    inner_scores[(condition, model)] = score
                    screening_rows.append({
                        "repeat": repeat, "fold": fold, "condition": condition,
                        "model": model, "inner_pearson_r": score,
                        "selected_hyperparameters": json.dumps(hyper, sort_keys=True),
                    })
            for rank, model in enumerate(sorted(selected, key=lambda name: background_scores[name], reverse=True), start=1):
                selection_rows.append({
                    "repeat": repeat, "fold": fold, "model": model, "rank": rank,
                    "background_inner_r": background_scores[model], "selection_strategy": "performance_top6",
                })
            for rank, model in enumerate(diverse_selected, start=1):
                selection_rows.append({
                    "repeat": repeat, "fold": fold, "model": model, "rank": rank,
                    "background_inner_r": background_scores[model], "selection_strategy": "anchor_diverse_top5",
                })

            outer_predictions: dict[tuple[str, str], np.ndarray] = {}
            for condition in conditions:
                addon_all = condition_addons[condition]
                condition_models = models if condition == "Genome_only" else panel_models
                outer_prepared = prepare_fold_data(
                    condition,
                    genome[train_idx], genome[test_idx],
                    addon_all[train_idx], addon_all[test_idx], config,
                )
                for qc_name in ("genome_marker_qc", "functional_marker_qc"):
                    qc = outer_prepared.get(qc_name)
                    if qc is not None:
                        marker_qc_rows.append({
                            "repeat": repeat, "fold": fold, "condition": condition, **qc
                        })
                for model_no, model in enumerate(condition_models):
                    prediction = fit_predict_prepared(
                        model, condition, outer_prepared, y[train_idx],
                        best_hypers[(condition, model)],
                        seed + repeat * 10000 + fold * 100 + model_no,
                    )
                    outer_predictions[(condition, model)] = prediction
                    for sample, observed, predicted in zip(ids[test_idx], y[test_idx], prediction):
                        prediction_rows.append({
                            "repeat": repeat, "fold": fold, "condition": condition,
                            "model": model, "sample_id": sample,
                            "observed": float(observed), "predicted": float(predicted),
                        })

            for condition in conditions:
                inner_matrix = np.column_stack([inner_predictions[(condition, model)] for model in selected])
                test_matrix = np.column_stack([outer_predictions[(condition, model)] for model in selected])
                ensemble_prediction, coefficients, intercept = ensemble_fit_predict(inner_matrix, y[train_idx], test_matrix)
                ensemble_name = f"Adaptive_Selected_{condition}"
                for model, coefficient in zip(selected, coefficients):
                    weight_rows.append({
                        "repeat": repeat, "fold": fold, "condition": condition,
                        "ensemble": ensemble_name, "component": model,
                        "coefficient": float(coefficient), "intercept": intercept,
                    })
                for sample, observed, predicted in zip(ids[test_idx], y[test_idx], ensemble_prediction):
                    prediction_rows.append({
                        "repeat": repeat, "fold": fold, "condition": condition,
                        "model": "Adaptive_Selected", "sample_id": sample,
                        "observed": float(observed), "predicted": float(predicted),
                    })

                diverse_inner = np.column_stack([
                    inner_predictions[(condition, model)] for model in diverse_selected
                ])
                diverse_test = np.column_stack([
                    outer_predictions[(condition, model)] for model in diverse_selected
                ])
                for ensemble_name, ridge, anchor_weight in (
                    ("AMSE2_Diverse5_Simplex", 0.0, 0.5),
                    ("AMSE2_Diverse5_AnchorShrink", 0.1, 0.5),
                    ("AMSE2_Diverse5_StrongAnchor", 0.1, 0.7),
                ):
                    prediction, coefficients, intercept = simplex_ensemble_fit_predict(
                        diverse_inner, y[train_idx], diverse_test,
                        ridge=ridge, anchor_weight=anchor_weight,
                    )
                    for model, coefficient in zip(diverse_selected, coefficients):
                        weight_rows.append({
                            "repeat": repeat, "fold": fold, "condition": condition,
                            "ensemble": ensemble_name, "component": model,
                            "coefficient": float(coefficient), "intercept": intercept,
                        })
                    for sample, observed, predicted in zip(ids[test_idx], y[test_idx], prediction):
                        prediction_rows.append({
                            "repeat": repeat, "fold": fold, "condition": condition,
                            "model": ensemble_name, "sample_id": sample,
                            "observed": float(observed), "predicted": float(predicted),
                        })

                anchor_candidates = [
                    model for model in ("GBLUP", "RKHS")
                    if (condition, model) in inner_predictions
                    and (condition, model) in outer_predictions
                ]
                if not anchor_candidates:
                    raise RuntimeError("No GBLUP or RKHS anchor is available")
                anchor_model = max(
                    anchor_candidates,
                    key=lambda model: pearson(
                        y[train_idx], inner_predictions[(condition, model)]
                    ),
                )
                gate_models = [anchor_model] + [
                    model for model in diverse_selected if model != anchor_model
                ]
                gate_inner = np.column_stack([
                    inner_predictions[(condition, model)] for model in gate_models
                ])
                gate_test = np.column_stack([
                    outer_predictions[(condition, model)] for model in gate_models
                ])
                gate_margin = float(config.get("anchor_gate_margin", 0.0))
                use_ensemble, anchor_inner_r, ensemble_inner_r = cross_fitted_anchor_gate(
                    gate_inner,
                    y[train_idx],
                    inner_folds_local,
                    ridge=float(config.get("anchor_gate_ridge", 0.1)),
                    anchor_weight=float(config.get("anchor_gate_weight", 0.7)),
                    margin=gate_margin,
                )
                if use_ensemble:
                    gate_prediction, gate_weights, gate_intercept = simplex_ensemble_fit_predict(
                        gate_inner,
                        y[train_idx],
                        gate_test,
                        ridge=float(config.get("anchor_gate_ridge", 0.1)),
                        anchor_weight=float(config.get("anchor_gate_weight", 0.7)),
                    )
                    deployed_method = "ensemble"
                    for model, coefficient in zip(gate_models, gate_weights):
                        weight_rows.append({
                            "repeat": repeat, "fold": fold, "condition": condition,
                            "ensemble": "AMSE3_AnchorGate", "component": model,
                            "coefficient": float(coefficient),
                            "intercept": gate_intercept,
                        })
                else:
                    gate_prediction = outer_predictions[(condition, anchor_model)]
                    gate_weights = np.asarray([1.0] + [0.0] * (len(gate_models) - 1))
                    gate_intercept = 0.0
                    deployed_method = "anchor"
                gate_rows.append({
                    "repeat": repeat, "fold": fold, "condition": condition,
                    "anchor_model": anchor_model,
                    "anchor_inner_r": float(anchor_inner_r),
                    "ensemble_inner_cross_fitted_r": float(ensemble_inner_r),
                    "inner_delta": float(ensemble_inner_r - anchor_inner_r),
                    "gate_margin": gate_margin,
                    "gate_selected_ensemble": use_ensemble,
                    "deployed_method": deployed_method,
                    "gate_models": ",".join(gate_models),
                })
                for sample, observed, predicted in zip(ids[test_idx], y[test_idx], gate_prediction):
                    prediction_rows.append({
                        "repeat": repeat, "fold": fold, "condition": condition,
                        "model": "AMSE3_AnchorGate", "sample_id": sample,
                        "observed": float(observed), "predicted": float(predicted),
                    })

                residual_models = [background_anchor_model] + [
                    model for model in diverse_selected if model != background_anchor_model
                ]
                residual_inner = np.column_stack([
                    inner_predictions[(condition, model)] for model in residual_models
                ])
                residual_test = np.column_stack([
                    outer_predictions[(condition, model)] for model in residual_models
                ])
                alpha_grid = [
                    float(alpha) for alpha in config.get(
                        "anchor_residual_alpha_grid", [0.1, 1.0, 10.0, 100.0]
                    )
                ]
                residual_margin = float(config.get("anchor_residual_margin", 0.0))
                (
                    use_correction,
                    anchor_inner_r,
                    correction_inner_r,
                    residual_alpha,
                    _,
                ) = cross_fitted_anchor_residual(
                    residual_inner,
                    y[train_idx],
                    inner_folds_local,
                    alpha_grid=alpha_grid,
                    margin=residual_margin,
                )
                if use_correction:
                    residual_prediction, residual_coefficients, residual_intercept = anchor_residual_fit_predict(
                        residual_inner,
                        y[train_idx],
                        residual_test,
                        alpha=residual_alpha,
                    )
                    residual_deployed_method = "anchor_residual_correction"
                    weight_rows.append({
                        "repeat": repeat, "fold": fold, "condition": condition,
                        "ensemble": "AMSE4_AnchorResidualGate", "component": background_anchor_model,
                        "coefficient": 1.0, "intercept": float(residual_intercept),
                    })
                    for model, coefficient in zip(residual_models[1:], residual_coefficients):
                        weight_rows.append({
                            "repeat": repeat, "fold": fold, "condition": condition,
                            "ensemble": "AMSE4_AnchorResidualGate", "component": model,
                            "coefficient": float(coefficient), "intercept": float(residual_intercept),
                        })
                else:
                    residual_prediction = outer_predictions[(condition, background_anchor_model)]
                    residual_deployed_method = "anchor"
                residual_gate_rows.append({
                    "repeat": repeat, "fold": fold, "condition": condition,
                    "anchor_model": background_anchor_model,
                    "anchor_inner_r": float(anchor_inner_r),
                    "correction_inner_cross_fitted_r": float(correction_inner_r),
                    "inner_delta": float(correction_inner_r - anchor_inner_r),
                    "selected_alpha": float(residual_alpha),
                    "alpha_grid": ",".join(str(alpha) for alpha in alpha_grid),
                    "gate_margin": residual_margin,
                    "gate_selected_correction": use_correction,
                    "deployed_method": residual_deployed_method,
                    "residual_models": ",".join(residual_models),
                })
                for sample, observed, predicted in zip(ids[test_idx], y[test_idx], residual_prediction):
                    prediction_rows.append({
                        "repeat": repeat, "fold": fold, "condition": condition,
                        "model": "AMSE4_AnchorResidualGate", "sample_id": sample,
                        "observed": float(observed), "predicted": float(predicted),
                    })
            print(
                f"DONE repeat={repeat} fold={fold} top6={','.join(selected)} "
                f"diverse5={','.join(diverse_selected)}",
                flush=True,
            )
            if progress_callback is not None:
                progress_callback({
                    "repeat": repeat,
                    "fold": fold,
                    "total_repeats": repeats,
                    "total_folds": outer_folds,
                    "fraction_complete": ((repeat - 1) * outer_folds + fold) / (repeats * outer_folds),
                    "selected_models": list(selected),
                })

    predictions = pd.DataFrame(prediction_rows)
    repeat_rows = []
    for (condition, model, repeat), frame in predictions.groupby(["condition", "model", "repeat"]):
        n_oof = int(frame["sample_id"].nunique())
        repeat_rows.append({
            "condition": condition, "model": model, "repeat": int(repeat),
            "n_oof_samples": n_oof,
            "oof_coverage": float(n_oof / len(ids)),
            "complete_oof": bool(n_oof == len(ids)),
            **metrics(frame.observed.to_numpy(float), frame.predicted.to_numpy(float)),
        })
    repeat_metrics = pd.DataFrame(repeat_rows)
    complete_metrics = repeat_metrics.query("complete_oof == True").copy()
    summary = complete_metrics.groupby(["condition", "model"], as_index=False).agg(
        pearson_mean=("pearson_r", "mean"), pearson_sd=("pearson_r", "std"),
        spearman_mean=("spearman_rho", "mean"), rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
    )
    comparisons = []
    for model in list(models) + ["Adaptive_Selected", "AMSE3_AnchorGate", "AMSE4_AnchorResidualGate"]:
        base = complete_metrics.query("condition == 'Genome_only' and model == @model").set_index("repeat")["pearson_r"]
        for panel_name in panels:
            panel_condition = f"Genome_plus_{panel_name}"
            random_condition = f"Genome_plus_RandomMatched_{panel_name}"
            panel_result = complete_metrics.query("condition == @panel_condition and model == @model").set_index("repeat")["pearson_r"]
            random_result = complete_metrics.query("condition == @random_condition and model == @model").set_index("repeat")["pearson_r"]
            if panel_result.empty or random_result.empty or base.empty:
                continue
            for label, current, reference in [
                ("Panel_vs_Genome", panel_result, base),
                ("Panel_vs_RandomMatched", panel_result, random_result),
            ]:
                common = current.index.intersection(reference.index)
                difference = current.loc[common] - reference.loc[common]
                paired_t_p = float("nan")
                wilcoxon_p = float("nan")
                if len(common) >= 2:
                    paired_t_p = float(ttest_rel(current.loc[common], reference.loc[common]).pvalue)
                    wilcoxon_p = float(wilcoxon(difference).pvalue) if np.any(difference != 0) else 1.0
                comparisons.append({
                    "panel": panel_name, "model": model, "comparison": label,
                    "delta_mean": float(difference.mean()), "delta_sd": float(difference.std()),
                    "positive_repeats": int((difference > 0).sum()), "n_repeats": int(len(difference)),
                    "paired_t_p": paired_t_p, "wilcoxon_p": wilcoxon_p,
                })
    comparison_frame = pd.DataFrame(comparisons)
    if not comparison_frame.empty:
        comparison_frame["paired_t_bh"] = bh_adjust(comparison_frame["paired_t_p"])
        comparison_frame["wilcoxon_bh"] = bh_adjust(comparison_frame["wilcoxon_p"])
    selections = pd.DataFrame(selection_rows)
    selection_frequency = selections.groupby(["selection_strategy", "model"], as_index=False).agg(
        selected_folds=("model", "size"), mean_rank=("rank", "mean"),
        mean_background_inner_r=("background_inner_r", "mean"),
    ).sort_values(["selected_folds", "mean_rank"], ascending=[False, True])
    predictions.to_csv(output / "adaptive_oof_predictions.csv", index=False)
    repeat_metrics.to_csv(output / "adaptive_repeat_metrics.csv", index=False)
    summary.to_csv(output / "adaptive_summary.csv", index=False)
    pd.DataFrame(screening_rows).to_csv(output / "inner_model_screening.csv", index=False)
    selections.to_csv(output / "selected_models_by_fold.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(output / "background_model_ranking_by_fold.csv", index=False)
    selection_frequency.to_csv(output / "model_selection_frequency.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "adaptive_ensemble_weights.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output / "anchor_gate_decisions.csv", index=False)
    pd.DataFrame(residual_gate_rows).to_csv(output / "anchor_residual_gate_decisions.csv", index=False)
    comparison_frame.to_csv(output / "functional_prior_comparisons.csv", index=False)
    pd.DataFrame([audit]).to_csv(output / "data_audit.csv", index=False)
    pd.DataFrame(marker_qc_rows).to_csv(output / "fold_marker_qc.csv", index=False)
    manifest = {
        **audit,
        "candidate_models": list(models),
        "outer_cv": f"{repeats} repeats x {outer_folds} folds",
        "inner_cv": f"{inner_folds} folds for model screening and tuning",
        "selection_rule": {
            "mode": str(config.get("selection_mode", "top_n")),
            "top_n": int(config.get("selection_top_n", config.get("selection_top_k", 6))),
            "top_k": int(config.get("selection_top_k", 3)),
            "within_best_margin": float(config.get("selection_margin", 0.03)),
            "minimum_inner_r": float(config.get("minimum_inner_r", 0.0)),
            "diverse_top_n": int(config.get("diverse_selection_top_n", 5)),
            "diversity_penalty": float(config.get("diversity_penalty", 0.05)),
        },
        "test_phenotype_used_for_selection": False,
        "functional_panels": list(panels),
        "random_panel_markers": random_panel_markers,
        "random_control": "one fixed genotype-only random panel matched to each functional panel's SNP count",
        "anchor_gate": {
            "method": "second-layer cross-fitted comparison of simplex ensemble with GBLUP/RKHS anchor",
            "margin": float(config.get("anchor_gate_margin", 0.0)),
            "ridge": float(config.get("anchor_gate_ridge", 0.1)),
            "anchor_weight": float(config.get("anchor_gate_weight", 0.7)),
            "outer_test_phenotypes_used": False,
        },
        "anchor_residual_gate": {
            "method": "fixed genome-only GBLUP/RKHS anchor with regularized cross-model deviation correction",
            "alpha_grid": [
                float(alpha) for alpha in config.get(
                    "anchor_residual_alpha_grid", [0.1, 1.0, 10.0, 100.0]
                )
            ],
            "margin": float(config.get("anchor_residual_margin", 0.0)),
            "anchor_identity_source": "Genome-only inner OOF ranking",
            "outer_test_phenotypes_used": False,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# 自适应功能先验GP结果",
        "",
        "候选背景模型只在每个外层训练折的内层OOF预测中筛选。外层测试表型不参与模型选择、调参、PCA或集成权重估计。",
        "",
        "## 模型选择频率",
        "",
        "```text",
        selection_frequency.to_string(index=False),
        "```",
        "",
        "## 外层无偏结果",
        "",
        "```text",
        summary.sort_values("pearson_mean", ascending=False).to_string(index=False),
        "```",
        "",
        "## 功能先验增量",
        "",
        "```text",
        comparison_frame.to_string(index=False),
        "```",
    ]
    (output / "自适应功能先验GP报告.md").write_text("\n".join(report), encoding="utf-8")
    # Rewrite the human-readable report with verified UTF-8 Chinese text.
    report = [
        "# 自适应功能先验GP结果",
        "",
        "候选背景模型只在每个外层训练折的内层OOF预测中筛选。外层测试表型不参与模型选择、调参、PCA或集成权重估计。",
        "",
        "## 模型选择频率",
        "",
        "```text",
        selection_frequency.to_string(index=False),
        "```",
        "",
        "## 外层无偏结果",
        "",
        "```text",
        summary.sort_values("pearson_mean", ascending=False).to_string(index=False),
        "```",
        "",
        "## 功能先验增量",
        "",
        "```text",
        comparison_frame.to_string(index=False),
        "```",
    ]
    (output / "自适应功能先验GP报告.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    report = [
        "# \u81ea\u9002\u5e94\u529f\u80fd\u5148\u9a8cGP\u7ed3\u679c",
        "",
        "\u5019\u9009\u80cc\u666f\u6a21\u578b\u53ea\u5728\u6bcf\u4e2a\u5916\u5c42\u8bad\u7ec3\u6298\u7684\u5185\u5c42OOF\u9884\u6d4b\u4e2d\u7b5b\u9009\u3002\u5916\u5c42\u6d4b\u8bd5\u8868\u578b\u4e0d\u53c2\u4e0e\u6a21\u578b\u9009\u62e9\u3001\u8c03\u53c2\u3001PCA\u6216\u96c6\u6210\u6743\u91cd\u4f30\u8ba1\u3002",
        "",
        "## \u6a21\u578b\u9009\u62e9\u9891\u7387",
        "",
        "```text",
        selection_frequency.to_string(index=False),
        "```",
        "",
        "## \u5916\u5c42\u65e0\u504f\u7ed3\u679c",
        "",
        "```text",
        summary.sort_values("pearson_mean", ascending=False).to_string(index=False),
        "```",
        "",
        "## \u529f\u80fd\u5148\u9a8c\u589e\u91cf",
        "",
        "```text",
        comparison_frame.to_string(index=False),
        "```",
    ]
    report_name = "\u81ea\u9002\u5e94\u529f\u80fd\u5148\u9a8cGP\u62a5\u544a.md"
    (output / report_name).write_text("\n".join(report), encoding="utf-8")
    print("\nSELECTION\n" + selection_frequency.to_string(index=False))
    print("\nSUMMARY\n" + summary.sort_values("pearson_mean", ascending=False).to_string(index=False))
    ranked_summary = summary.sort_values("pearson_mean", ascending=False)
    return {
        "status": "success",
        "output_dir": str(output.resolve()),
        "n_accessions": int(len(ids)),
        "target": audit["target"],
        "candidate_models": list(models),
        "functional_panels": list(panels),
        "best_result": ranked_summary.iloc[0].to_dict() if not ranked_summary.empty else None,
        "summary": ranked_summary.to_dict(orient="records"),
        "model_selection_frequency": selection_frequency.to_dict(orient="records"),
        "functional_prior_comparisons": comparison_frame.to_dict(orient="records"),
        "files": {
            "oof_predictions": str((output / "adaptive_oof_predictions.csv").resolve()),
            "repeat_metrics": str((output / "adaptive_repeat_metrics.csv").resolve()),
            "summary": str((output / "adaptive_summary.csv").resolve()),
            "comparisons": str((output / "functional_prior_comparisons.csv").resolve()),
            "manifest": str((output / "manifest.json").resolve()),
            "report_utf8": str((output / report_name).resolve()),
            "report": str((output / "自适应功能先验GP报告.md").resolve()),
            "background_ranking": str((output / "background_model_ranking_by_fold.csv").resolve()),
            "fold_marker_qc": str((output / "fold_marker_qc.csv").resolve()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive functional-prior genomic prediction")
    parser.add_argument(
        "--mode", choices=["preprocess", "evaluate", "train", "predict"],
        default="evaluate",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Prediction bundle for predict mode")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.mode == "preprocess":
        print(json.dumps(preprocess_dataset(config, args.output), ensure_ascii=False, indent=2))
    elif args.mode == "evaluate":
        run(config, args.output)
    elif args.mode == "train":
        print(json.dumps(train_final_adaptive(config, args.output), ensure_ascii=False, indent=2))
    else:
        if args.model is None:
            parser.error("--model is required in predict mode")
        print(json.dumps(predict_from_bundle(args.model, config, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
