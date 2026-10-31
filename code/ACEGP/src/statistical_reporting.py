"""Registered descriptive endpoints and dependence-corrected paired tests."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import pearsonr, spearmanr, t as student_t

def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if y.ndim != 1 or len(y) == 0 or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Complete paired prediction and phenotype arrays required")
    finite_correlation = len(y) >= 3 and np.std(y) > 0 and np.std(pred) > 0
    return {"n": len(y), "pearson_r": float(pearsonr(y, pred).statistic) if finite_correlation else None, "spearman_rho": float(spearmanr(y, pred).statistic) if finite_correlation else None, "rmse": float(np.sqrt(np.mean((y-pred)**2))), "mae": float(np.mean(np.abs(y-pred)))}

def corrected_paired_test(differences, train_sizes, test_sizes, alpha=.05):
    d = np.asarray(differences, float)
    training, test = np.asarray(train_sizes, float), np.asarray(test_sizes, float)
    if d.ndim != 1 or training.ndim != 1 or test.ndim != 1 or len(d) < 2 or len(training) != len(d) or len(test) != len(d):
        raise ValueError("One paired difference and train/test size per outer fold required")
    if not np.isfinite(d).all() or not np.isfinite(training).all() or not np.isfinite(test).all() or not (training > 0).all() or not (test > 0).all():
        return {"status": "UNDEFINED_INPUT", "folds_expected": len(d), "invalid_differences": int((~np.isfinite(d)).sum()), "p_value": None}
    mean, variance = float(d.mean()), float(d.var(ddof=1))
    q = float(np.mean(test / training))
    se = float(math.sqrt((1/len(d)+q)*variance))
    if se == 0:
        return {"status": "ZERO_VARIANCE", "mean_difference": mean, "df": len(d)-1, "p_value": 1.0 if mean == 0 else None, "CI": [0., 0.] if mean == 0 else None, "approximate_inference": True}
    statistic = mean/se
    critical = float(student_t.ppf(1-alpha/2, len(d)-1))
    return {"status": "OK", "mean_difference": mean, "sample_variance": variance, "mean_test_train_ratio": q, "standard_error": se, "t_statistic": statistic, "df": len(d)-1, "p_value": float(2*student_t.sf(abs(statistic), len(d)-1)), "CI": [mean-critical*se, mean+critical*se], "approximate_inference": True, "CV_folds_are_independent_biological_replicates": False}

def fisher_fold_comparison(qtl_r, genome_r, train_sizes, test_sizes):
    q, g = np.asarray(qtl_r, float), np.asarray(genome_r, float)
    if q.ndim != 1 or g.ndim != 1 or len(q) != len(g): raise ValueError("Paired fold correlations must align")
    if np.any(np.abs(q[np.isfinite(q)]) > 1) or np.any(np.abs(g[np.isfinite(g)]) > 1):
        raise ValueError("Correlation outside [-1,1]")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        # An undefined registered fold is reported, never clipped into a finite
        # perfect correlation or removed to create a more favorable family.
        valid = np.isfinite(q) & np.isfinite(g)
        differences = np.full(len(q), np.nan)
        differences[valid] = np.arctanh(np.clip(q[valid], -1+1e-12, 1-1e-12))-np.arctanh(np.clip(g[valid], -1+1e-12, 1-1e-12))
        result = corrected_paired_test(differences, train_sizes, test_sizes)
        return {**result, "estimand": "paired_outer_fold_Fisher_z_difference", "CI_is_raw_Pearson_r_difference": False}
    differences = np.arctanh(np.clip(q, -1+1e-12, 1-1e-12)) - np.arctanh(np.clip(g, -1+1e-12, 1-1e-12))
    result = corrected_paired_test(differences, train_sizes, test_sizes)
    return {**result, "estimand": "paired_outer_fold_Fisher_z_difference", "CI_is_raw_Pearson_r_difference": False, "perfect_correlation_clip_epsilon": 1e-12}

def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    if not np.isfinite(pvalues).all() or np.any(pvalues < 0) or np.any(pvalues > 1):
        raise ValueError("Register and report incomplete hypothesis families explicitly")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1))
    result = np.empty(len(order))
    result[order] = np.minimum(adjusted, 1)
    return result

def calibration(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if y.ndim != 1 or len(y) == 0 or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all(): raise ValueError("Complete aligned calibration records required")
    if len(y) < 3 or np.std(pred) == 0:
        return {"intercept": None, "slope": None, "mean_bias": float(np.mean(pred-y)), "status": "UNDEFINED_SLOPE", "predictions_recalibrated": False}
    intercept, slope = np.linalg.lstsq(np.column_stack((np.ones(len(y)), pred)), y, rcond=None)[0]
    return {"intercept": float(intercept), "slope": float(slope), "mean_bias": float(np.mean(pred-y)), "status": "OK", "predictions_recalibrated": False, "OLS_independence_P_values_reported": False}

def selection_metrics(y, qtl_prediction, genome_prediction, material_ids, fraction):
    y, q, g = np.asarray(y, float), np.asarray(qtl_prediction, float), np.asarray(genome_prediction, float)
    ids = np.asarray(material_ids, str)
    if any(v.ndim != 1 for v in (y, q, g, ids)) or not (len(y) == len(q) == len(g) == len(ids)) or len(y) == 0: raise ValueError("Sample order mismatch")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Material selection fraction must be in (0,1]")
    if not (np.isfinite(y).all() and np.isfinite(q).all() and np.isfinite(g).all()): raise ValueError("Incomplete prediction bank")
    k = int(math.ceil(float(fraction)*len(y)))
    slots = np.arange(len(y))
    # Slots resolve clone ties during bootstrap without removing copies.
    truth = np.lexsort((slots, ids, y))[:k]
    selected_q = np.lexsort((slots, ids, q))[:k]
    selected_g = np.lexsort((slots, ids, g))[:k]
    hit_q = len(np.intersect1d(truth, selected_q))/k
    hit_g = len(np.intersect1d(truth, selected_g))/k
    lq, lg = float(y[selected_q].mean()), float(y[selected_g].mean())
    return {"n": len(y), "k": k, "fraction": fraction, "QTL_selected_actual_mean_loss": lq, "GW_selected_actual_mean_loss": lg, "GW_minus_QTL_selection_loss": lg-lq, "QTL_hit_rate": hit_q, "GW_hit_rate": hit_g, "QTL_minus_GW_hit_rate": hit_q-hit_g, "QTL_gain_relative_to_population_mean": float(y.mean()-lq), "GW_gain_relative_to_population_mean": float(y.mean()-lg), "next_generation_genetic_gain_measured": False}

def empirical_random_tail(observed, random_values, expected_random_count=100):
    values = np.asarray(random_values, float)
    if expected_random_count < 1 or values.ndim != 1 or len(values) != expected_random_count or not np.isfinite(values).all() or not np.isfinite(observed):
        raise ValueError("All registered random pools must finish; no failed-pool replacement by score")
    exceedances = int((values >= observed).sum())
    return {"P_emp": (1+exceedances)/(expected_random_count+1), "random_values_at_least_observed": exceedances, "random_count": expected_random_count, "comparison_is_first_fixed_partition_only": True}

def prediction_bank_summary(y, qtl_predictions, genome_predictions, material_ids, fractions=(.10, .20)):
    """Recalculate each repeat first, then average its metric across repeats."""
    y = np.asarray(y, float)
    q, g = np.asarray(qtl_predictions, float), np.asarray(genome_predictions, float)
    ids = np.asarray(material_ids, str)
    if q.ndim == 1: q = q[:, None]
    if g.ndim == 1: g = g[:, None]
    if y.ndim != 1 or ids.ndim != 1 or q.ndim != 2 or q.shape != g.shape or q.shape[0] != len(y) or len(ids) != len(y) or q.shape[1] == 0:
        raise ValueError("Rows must be aligned materials and columns complete CV repeats")
    if not np.isfinite(y).all() or not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Complete prediction banks required; missing repeats cannot be dropped")
    values = {}
    def append(key, value):
        values.setdefault(key, []).append(float(value) if value is not None else np.nan)
    for repeat in range(q.shape[1]):
        mq, mg = metrics(y, q[:, repeat]), metrics(y, g[:, repeat])
        for key in ("pearson_r", "spearman_rho", "rmse", "mae"):
            append("QTL_"+key, mq[key]); append("GW_"+key, mg[key])
        for key in ("pearson_r", "spearman_rho"):
            append("QTL_minus_GW_"+key, None if mq[key] is None or mg[key] is None else mq[key]-mg[key])
        for key in ("rmse", "mae"):
            append("GW_minus_QTL_"+key, mg[key]-mq[key])
        for label, prediction in (("QTL", q[:, repeat]), ("GW", g[:, repeat])):
            c = calibration(y, prediction)
            for key in ("intercept", "slope", "mean_bias"):
                append(label+"_calibration_"+key, c[key])
        for fraction in fractions:
            s = selection_metrics(y, q[:, repeat], g[:, repeat], ids, fraction)
            for key in ("QTL_selected_actual_mean_loss", "GW_selected_actual_mean_loss", "GW_minus_QTL_selection_loss", "QTL_hit_rate", "GW_hit_rate", "QTL_minus_GW_hit_rate"):
                append(f"selection_{fraction:g}_"+key, s[key])
    return {key: float(np.mean(v)) if np.isfinite(v).all() else None for key, v in values.items()}

def conditional_material_bootstrap(y, qtl_predictions, genome_predictions, material_ids,
                                   *, seed, material_resampling_justified,
                                   resamples=2000, minimum_valid_fraction=.95,
                                   fractions=(.10, .20), verified_material_clusters=None):
    """Paired fixed-prediction sensitivity intervals, conditional on fitted models.

    The caller must establish that row or verified-cluster resampling is a
    defensible unit. Genetic-PC groups alone do not satisfy that requirement.
    This function never retrains, reselects regions, or treats CV repeats as rows.
    """
    if not material_resampling_justified:
        raise ValueError("Material dependence/resampling unit must be justified before reporting intervals")
    if int(resamples) != resamples or resamples < 2 or not 0 < minimum_valid_fraction <= 1:
        raise ValueError("Invalid registered bootstrap count or validity threshold")
    y = np.asarray(y, float)
    q, g = np.asarray(qtl_predictions, float), np.asarray(genome_predictions, float)
    if q.ndim == 1: q = q[:, None]
    if g.ndim == 1: g = g[:, None]
    ids = np.asarray(material_ids, str)
    observed = prediction_bank_summary(y, q, g, ids, fractions)
    generator = np.random.default_rng(seed)
    cluster_members = None
    if verified_material_clusters is not None:
        clusters = np.asarray(verified_material_clusters, str)
        if clusters.ndim != 1 or len(clusters) != len(y) or len(np.unique(clusters)) < 2:
            raise ValueError("Verified material clusters must align and include at least two units")
        cluster_members = [np.flatnonzero(clusters == label) for label in np.unique(clusters)]
    distributions = {key: np.full(int(resamples), np.nan) for key in observed}
    for draw in range(int(resamples)):
        if cluster_members is None:
            selected = generator.integers(0, len(y), size=len(y))
        else:
            selected = np.concatenate([cluster_members[i] for i in generator.integers(0, len(cluster_members), size=len(cluster_members))])
        summary = prediction_bank_summary(y[selected], q[selected], g[selected], ids[selected], fractions)
        for key, value in summary.items():
            if value is not None: distributions[key][draw] = value
    endpoints = {}
    for key, distribution in distributions.items():
        valid = np.isfinite(distribution)
        fraction_valid = float(valid.mean())
        pass_validity = observed[key] is not None and fraction_valid >= minimum_valid_fraction
        endpoints[key] = {"observed": observed[key], "valid_resamples": int(valid.sum()), "valid_fraction": fraction_valid,
                          "status": "OK_CONDITIONAL_SENSITIVITY" if pass_validity else "INSUFFICIENT_VALID_RESAMPLES",
                          "percentile_95_interval": np.quantile(distribution[valid], [.025, .975]).tolist() if pass_validity else None}
    return {"resamples": int(resamples), "seed": int(seed), "materials": len(y), "prediction_repeats": q.shape[1],
            "resampling_unit": "verified_material_clusters" if cluster_members is not None else "material_rows",
            "shared_indices_for_models_and_all_repeats": True, "bootstrap_clones_retained": True,
            "metric_averaged_after_per_repeat_recalculation": True, "full_selection_training_uncertainty_included": False,
            "conditional_intervals_replace_primary_corrected_tests": False, "endpoints": endpoints,
            "replicate_values": {key: [float(v) if np.isfinite(v) else None for v in value] for key, value in distributions.items()}}
