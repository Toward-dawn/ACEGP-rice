"""Train-only ACEGP interface for a single holdout partition.

The adaptive branch retains AMSE8 outer hyperparameter fits and AMSE10
repeated-inner blend selection. The screening branch retains AMSE18's full
16-learner shortlist and conservative anchor gate. No holdout phenotype is
accepted by this API. Historical incremental feature layers are never loaded.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import ARDRegression, Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "legacy_model_snapshot"
VERSION = "uniform_train_local_acegp_v1"

def stable_seed(*parts):
    literal = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(literal).digest()[:8], "big") % (2**32)

def load_base():
    spec = importlib.util.spec_from_file_location("uniform_acegp_base", SNAPSHOT / "run_scheme2_tuned.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = load_base()
MODELS = tuple(base.CORE_MODELS)

def load_helpers(filename, function_names, globals_dict):
    parsed = ast.parse((SNAPSHOT / filename).read_text(encoding="utf-8-sig"))
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in function_names]
    assert {n.name for n in nodes} == set(function_names)
    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SNAPSHOT / filename), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in function_names})

def stable(values):
    values = np.asarray(values, float)
    return float(values.mean() - 0.25 * values.std(ddof=0))

a11 = load_helpers("run_amse11_full_pool_blend.py", ["modal_hyper"], {"np": np, "json": json, "Counter": __import__("collections").Counter})
a11.stable, a11.CANDIDATE_MODELS = stable, MODELS
a10 = load_helpers("run_amse10_repeated_blend.py", ["choose_weight"], {"np": np, "base": base, "stable": stable, "WEIGHTS": (0, .10, .25, .40, .55, .70, 1), "NEAR_BEST_MARGIN": .002})
a18 = load_helpers("run_amse18_conservative_anchor_blend.py", ["combine", "build_shortlist", "choose_anchor", "choose_bridge_weight"], {"np": np, "base": base, "a11": a11, "math": math, "TOP_N": 6, "MIN_TOP_COUNT": 2, "SHORTLIST_MARGIN": .03, "NEAR_BEST_MARGIN": .002, "BRIDGE_WEIGHTS": (0, .10, .25, .40, .55, .70, 1)})

def array_hash(value):
    a = np.asarray(value)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    view = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    h.update(view)
    return h.hexdigest()

CODE_HASH = hashlib.sha256(Path(__file__).read_bytes() + b"".join(p.read_bytes() for p in sorted(SNAPSHOT.glob("*.py")))).hexdigest()

def cache_key(kind, args):
    h = hashlib.sha256((VERSION + CODE_HASH + kind).encode())
    def add(x):
        if isinstance(x, np.ndarray):
            h.update(array_hash(x).encode())
        elif isinstance(x, dict):
            for k, v in sorted(x.items()):
                h.update(str(k).encode()); add(v)
        elif isinstance(x, (list, tuple)):
            for item in x: add(item)
        else:
            h.update(json.dumps(x, sort_keys=True, default=str).encode())
    add(args)
    return h.hexdigest()

def enforce_cache_budget(config, root, protected=None):
    """Evict only reproducible pickle caches under this registered run directory."""
    root = Path(root).resolve()
    minimum = float(config.get("minimum_free_disk_GiB", 10)) * (1 << 30)
    maximum = float(config.get("cache_max_GiB", 20)) * (1 << 30)
    files = [p for p in root.rglob("*.pkl") if p.is_file() and not p.is_symlink()]
    total = sum(p.stat().st_size for p in files)
    for p in sorted(files, key=lambda x: x.stat().st_mtime):
        if total <= maximum and shutil.disk_usage(root).free >= minimum:
            break
        if protected is not None and p == protected:
            continue
        if not p.resolve().is_relative_to(root):
            raise RuntimeError("Cache path escapes registered directory")
        size = p.stat().st_size
        p.unlink()
        total -= size
    if shutil.disk_usage(root).free < minimum:
        raise RuntimeError("Insufficient free disk after releasing reproducible caches")

def cached(config, kind, args, compute):
    root = config.get("new_cache_directory")
    if not root:
        return compute()
    directory = Path(root) / kind
    directory.mkdir(parents=True, exist_ok=True)
    enforce_cache_budget(config, root)
    target = directory / (cache_key(kind, args) + ".pkl")
    if target.is_file():
        with target.open("rb") as f:
            return pickle.load(f)
    result = compute()
    temp = target.with_name(target.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(target)
    enforce_cache_budget(config, root, target)
    return result

def qc(train, test, minimum, maximum):
    train, test = np.asarray(train, float), np.asarray(test, float)
    if train.shape[1] == 0:
        return train, test, np.zeros(0, dtype=bool)
    observed = np.isfinite(train)
    counts = observed.sum(0)
    means = np.divide(np.nansum(train, axis=0), counts, out=np.zeros(train.shape[1]), where=counts > 0)
    maf = np.minimum(means / 2, 1 - means / 2)
    variance = np.divide(np.where(observed, (train - means)**2, 0).sum(0), counts, out=np.zeros(train.shape[1]), where=counts > 0)
    keep = (counts > 0) & (maf >= minimum) & ((1 - counts / len(train)) <= maximum) & (variance > 1e-12)
    return train[:, keep], test[:, keep], keep

def pcs(train, test, count):
    """Exact PCs by the small training Gram matrix; no fitted test statistics."""
    if train.shape[1] == 0 or len(train) < 2:
        return np.zeros((len(train), 0)), np.zeros((len(test), 0))
    train, test = base.impute_from_train(train, test)
    mean = train.mean(0)
    centered = train - mean
    values, vectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    threshold = max(float(values[0]), 1) * 1e-12
    k = min(int(count), int((values > threshold).sum()), len(train)-1, train.shape[1])
    if k == 0:
        return np.zeros((len(train), 0)), np.zeros((len(test), 0))
    vectors = vectors[:, :k]
    singular = np.sqrt(np.maximum(values[:k], 0))
    # Resolve each PC sign from its largest-magnitude training loading.
    loadings = (vectors.T @ centered) / singular[:, None]
    signs = np.sign(loadings[np.arange(k), np.argmax(np.abs(loadings), axis=1)])
    signs[signs == 0] = 1
    loadings *= signs[:, None]
    return (vectors * singular) * signs, (test - mean) @ loadings.T

def auxiliaries(genotypes):
    observed = np.isfinite(genotypes)
    return np.column_stack((1 - observed.mean(1), ((genotypes == 1) & observed).sum(1) / np.maximum(observed.sum(1), 1)))

def prepare(condition, gt, gx, at, ax, config):
    gt, gx, gkeep = qc(gt, gx, float(config["min_maf"]), float(config["max_marker_missing_rate"]))
    if not gkeep.any(): raise RuntimeError("No background SNPs passed training-local QC")
    bgpc = cached(config, "background_pc", (gt, gx, config["genome_pcs"]), lambda: pcs(gt, gx, config["genome_pcs"]))
    feature_t = np.column_stack((bgpc[0], auxiliaries(gt)))
    feature_x = np.column_stack((bgpc[1], auxiliaries(gx)))
    bgkt, bgkx = cached(config, "background_kernel", (gt, gx), lambda: base.linear_kernel(gt, gx))
    aqc = None
    akt, akx = bgkt, bgkx
    if condition != "Genome_only":
        at, ax, akeep = qc(at, ax, float(config["functional_min_maf"]), float(config["functional_max_marker_missing_rate"]))
        apc = cached(config, "union_pc", (at, ax, int(config.get("qtl_union_pcs", 5))), lambda: pcs(at, ax, int(config.get("qtl_union_pcs", 5))))
        feature_t, feature_x = np.column_stack((feature_t, apc[0])), np.column_stack((feature_x, apc[1]))
        if akeep.any(): akt, akx = base.linear_kernel(at, ax)
        aqc = {"raw_markers": int(len(akeep)), "passed_markers": int(akeep.sum()), "empty_union_is_background_only": bool(not akeep.any())}
    tz, xz = base.standardize(feature_t, feature_x)
    return {"train_z": tz, "test_z": xz, "bg_k_train": bgkt, "bg_k_test": bgkx, "addon_k_train": akt, "addon_k_test": akx, "rbf_kernels": {float(s): base.rbf_kernel(feature_t, feature_x, float(s)) for s in config["rkhs_bandwidth_scale_grid"]}, "genome_marker_qc": {"raw_markers": int(len(gkeep)), "passed_markers": int(gkeep.sum())}, "functional_marker_qc": aqc}

base.prepare_fold_data = prepare
original_fit_prepared = base.fit_predict_prepared
def seeded_fit(model, condition, prepared, y, hyper, seed):
    return original_fit_prepared(model, condition, prepared, y, hyper, int(seed) % (2**32))
base.fit_predict_prepared = seeded_fit

def block_design(gt, gx, at, ax, groups, config, seed):
    gt, gx, _ = qc(gt, gx, float(config["min_maf"]), float(config["max_marker_missing_rate"]))
    at, ax, keep = qc(at, ax, float(config["functional_min_maf"]), float(config["functional_max_marker_missing_rate"]))
    gpcs = cached(config, "background_pc", (gt, gx, config["genome_pcs"]), lambda: pcs(gt, gx, config["genome_pcs"]))
    mapping = np.full(len(keep), -1, dtype=int)
    mapping[keep] = np.arange(keep.sum())
    tparts, xparts, names = [], [], []
    for signal_id, original in groups:
        selected = mapping[np.asarray(original, dtype=int)]
        selected = selected[selected >= 0]
        if len(selected) == 0: continue
        # PC computation is exact; the identity seed remains registered.
        identity_seed = stable_seed(config["seed"], seed, signal_id, "block_pc")
        part = cached(config, "block_pc", (signal_id, at[:, selected], ax[:, selected], config["qtl_pcs_per_block"], identity_seed), lambda: pcs(at[:, selected], ax[:, selected], config["qtl_pcs_per_block"]))
        if part[0].shape[1]:
            tparts.append(part[0]); xparts.append(part[1]); names.append(signal_id)
    return (*gpcs, np.column_stack(tparts) if tparts else np.zeros((len(gt), 0)), np.column_stack(xparts) if xparts else np.zeros((len(gx), 0)), names)

def block_hyper_grid(config):
    import itertools
    return [{"ridge_alpha": float(a), "qtl_scale": float(s), "ard_threshold_lambda": float(t), "ard_max_iter": int(config["ard_max_iter"])} for a, s, t in itertools.product(config["ridge_alpha_grid"], config["qtl_scale_grid"], config["ard_threshold_lambda_grid"])]

def block_predict(y, design, hyper):
    gt, gx, at, ax, names = design
    gt, gx = base.standardize(gt, gx)
    baseline = Ridge(alpha=hyper["ridge_alpha"]).fit(gt, y)
    prediction = baseline.predict(gx)
    if at.shape[1] == 0: return prediction
    at, ax = base.standardize(at, ax)
    residual = y - baseline.predict(gt)
    estimator = ARDRegression(max_iter=hyper["ard_max_iter"], threshold_lambda=hyper["ard_threshold_lambda"])
    estimator.fit(at, residual)
    return prediction + hyper["qtl_scale"] * estimator.predict(ax)

def block_oof(y, genome, qtl, groups, folds, config, seed):
    hypers = block_hyper_grid(config)
    predictions = [np.full(len(y), np.nan) for h in hypers]
    for number, valid in enumerate(folds):
        train = np.setdiff1d(np.arange(len(y)), valid, assume_unique=True)
        design = block_design(genome[train], genome[valid], qtl[train], qtl[valid], groups, config, seed + number * 97)
        for j, hyper in enumerate(hypers):
            predictions[j][valid] = block_predict(y[train], design, hyper)
    scores = [base.pearson(y, p) for p in predictions]
    if not np.isfinite(scores).any(): raise RuntimeError("All block parameter candidates failed")
    chosen = int(np.nanargmax(scores))
    return predictions[chosen], hypers[chosen], float(scores[chosen])

def inner_results(model, condition, genome, addon, y, folds, config, seed, prepared=None):
    relevant = {k: config[k] for k in config if k.endswith("_grid") or k in ["seed", "genome_pcs", "qtl_union_pcs", "min_maf", "max_marker_missing_rate", "functional_min_maf", "functional_max_marker_missing_rate"]}
    actual_addon = np.zeros((len(genome), 0)) if condition == "Genome_only" else addon
    return cached(config, "inner_oof", (model, condition, genome, actual_addon, y, folds, relevant, seed), lambda: base.inner_oof(model, condition, genome, actual_addon, y, folds, config, seed, fold_cache=prepared))

def fit_holdout(genome_train, genome_test, y_train, qtl_train, qtl_test, groups, config, context_seed):
    """Returns predictions and training-only choices. Does not accept y_test."""
    assert len(genome_train) == len(y_train) == len(qtl_train)
    assert genome_train.shape[1] == genome_test.shape[1]
    assert qtl_train.shape[1] == qtl_test.shape[1]
    if tuple(config["learner_pool"]) != MODELS: raise RuntimeError("Learner pool changed")
    if len(base.available_models()) < 16 or base.CatBoostRegressor is None: raise RuntimeError("All 16 registered learners must be available")
    with threadpool_limits(limits=int(config.get("blas_threads", 2))):
        return _fit_holdout(genome_train, genome_test, np.asarray(y_train, float), qtl_train, qtl_test, groups, config, int(context_seed))

def _fit_holdout(g, gx, y, q, qx, groups, config, seed):
    if not np.isfinite(y).all(): raise RuntimeError("Nonfinite training target")
    if not np.std(y) > 1e-12: raise RuntimeError("Constant training phenotype")
    empty, empty_x = np.zeros((len(y), 0)), np.zeros((len(gx), 0))
    _, _, qkeep = qc(q, qx, config["functional_min_maf"], config["functional_max_marker_missing_rate"])
    active_qtl = bool(qkeep.any())
    first_inner = base.folds_for(len(y), seed + 1001, int(config["inner_folds"]))
    first_hypers = {}
    prep_first = base.prepare_inner_fold_cache("Genome_only", g, empty, first_inner, config)
    for number, model in enumerate(("GBLUP", "RKHS")):
        _, first_hypers[model], _ = inner_results(model, "Genome_only", g, empty, y, first_inner, config, seed + 10000 + number, prep_first)
    repeated = []
    for repeat in range(1, 4):
        folds = base.folds_for(len(y), seed + repeat * 100000, int(config["inner_folds"]))
        model_seed = seed + repeat * 1000000
        prep_bg = base.prepare_inner_fold_cache("Genome_only", g, empty, folds, config)
        models = {}
        for number, model in enumerate(MODELS):
            p, h, score = inner_results(model, "Genome_only", g, empty, y, folds, config, model_seed + number, prep_bg)
            models[model] = {"prediction": p, "hyper": h, "score": score}
        repeated.append({"inner": folds, "model_seed": model_seed, "models": models})
    adaptive_anchor = max(("GBLUP", "RKHS"), key=lambda name: (stable([r["models"][name]["score"] for r in repeated]), name))
    shortlist, ranking = a18.build_shortlist(repeated)
    anchor_pool = list(dict.fromkeys(shortlist + ["GBLUP", "RKHS"]))
    anchor_records = []
    # Preserve AMSE18's additional 2 x 5 conservative anchor screen.
    for repeat in range(1, 3):
        as_seed = seed + repeat * 10000000
        folds = base.folds_for(len(y), as_seed, min(5, len(y)))
        prep = base.prepare_inner_fold_cache("Genome_only", g, empty, folds, config)
        scores = {model: inner_results(model, "Genome_only", g, empty, y, folds, config, as_seed + MODELS.index(model), prep)[2] for model in anchor_pool}
        anchor_records.append({"scores": scores})
    anchor_choice, anchor_candidates, anchor_decision = a18.choose_anchor(anchor_records, anchor_pool, len(y))
    screened_anchor = anchor_choice["model"]
    bg_prepared = prepare("Genome_only", g, gx, empty, empty_x, config)
    adaptive_gw = base.fit_predict_prepared(adaptive_anchor, "Genome_only", bg_prepared, y, first_hypers[adaptive_anchor], seed + 7000)
    screened_hyper = a11.modal_hyper([r["models"][screened_anchor]["hyper"] for r in repeated], [r["models"][screened_anchor]["score"] for r in repeated])[0]
    screened_gw = base.fit_predict_prepared(screened_anchor, "Genome_only", bg_prepared, y, screened_hyper, seed + 17000)
    gw = .5 * adaptive_gw + .5 * screened_gw
    audit = {"active_QTL": active_qtl, "adaptive_anchor": adaptive_anchor, "screened_anchor": screened_anchor, "shortlist": shortlist, "screened_gate": anchor_decision, "screened_candidates": anchor_candidates, "full_pool_ranking": ranking, "feature_policy": VERSION, "internal_repeated_screen": "3 x 3", "additional_conservative_anchor_screen": "2 x 5", "y_holdout_accepted_by_API": False}
    if not active_qtl:
        return {"Genome_wide": gw, "QTL_informed": gw.copy(), "adaptive_GW": adaptive_gw, "screened_GW": screened_gw, "adaptive_QTL": adaptive_gw.copy(), "screened_QTL": screened_gw.copy(), "audit": {**audit, "empty_QTL_recovery": True, "adaptive_block_weight": 0.0, "screened_block_weight": 0.0}}
    prep_first_q = base.prepare_inner_fold_cache("Genome_plus_QTL", g, q, first_inner, config)
    aq_hyper = inner_results(adaptive_anchor, "Genome_plus_QTL", g, q, y, first_inner, config, seed + 10000 + (0 if adaptive_anchor == "GBLUP" else 1), prep_first_q)[1]
    _, first_block_hyper, _ = block_oof(y, g, q, groups, first_inner, config, seed + 15000)
    needed = list(dict.fromkeys([adaptive_anchor, screened_anchor]))
    qrepeated = {m: [] for m in needed}
    bpreds, bhypers, bscores = [], [], []
    for r in repeated:
        prep = base.prepare_inner_fold_cache("Genome_plus_QTL", g, q, r["inner"], config)
        for m in needed:
            p, h, score = inner_results(m, "Genome_plus_QTL", g, q, y, r["inner"], config, r["model_seed"] + 2000 + MODELS.index(m), prep)
            qrepeated[m].append({"prediction": p, "hyper": h, "score": score})
        bp, bh, bs = block_oof(y, g, q, groups, r["inner"], config, r["model_seed"] + 5000)
        bpreds.append(bp); bhypers.append(bh); bscores.append(bs)
    adaptive_choice, aw_records = a10.choose_weight([r["prediction"] for r in qrepeated[adaptive_anchor]], bpreds, [y]*3)
    screened_choice, sw_records = a18.choose_bridge_weight([{"observed": y, "base_prediction": r["prediction"], "block_prediction": b} for r, b in zip(qrepeated[screened_anchor], bpreds)])
    qprepared = prepare("Genome_plus_QTL", g, gx, q, qx, config)
    adaptive_outer = base.fit_predict_prepared(adaptive_anchor, "Genome_plus_QTL", qprepared, y, aq_hyper, seed + 7000)
    screened_q_hyper = a11.modal_hyper([r["hyper"] for r in qrepeated[screened_anchor]], [r["score"] for r in qrepeated[screened_anchor]])[0]
    screened_outer = base.fit_predict_prepared(screened_anchor, "Genome_plus_QTL", qprepared, y, screened_q_hyper, seed + 18000)
    ad = block_design(g, gx, q, qx, groups, config, seed + 9000)
    sd = block_design(g, gx, q, qx, groups, config, seed + 19000)
    adaptive_block = block_predict(y, ad, first_block_hyper)
    screened_bhyper = a11.modal_hyper(bhypers, bscores)[0]
    screened_block = block_predict(y, sd, screened_bhyper)
    aw, sw = adaptive_choice["weight"], screened_choice["weight"]
    adaptive_q = (1-aw)*adaptive_outer + aw*adaptive_block
    screened_q = (1-sw)*screened_outer + sw*screened_block
    return {"Genome_wide": gw, "QTL_informed": .5*adaptive_q + .5*screened_q, "adaptive_GW": adaptive_gw, "screened_GW": screened_gw, "adaptive_QTL": adaptive_q, "screened_QTL": screened_q, "audit": {**audit, "adaptive_block_weight": aw, "screened_block_weight": sw, "adaptive_weight_grid": aw_records, "screened_weight_grid": sw_records, "blocks_retained": sd[-1], "adaptive_hyper": aq_hyper, "screened_hyper": screened_q_hyper, "adaptive_block_hyper": first_block_hyper, "screened_block_hyper": screened_bhyper}}
