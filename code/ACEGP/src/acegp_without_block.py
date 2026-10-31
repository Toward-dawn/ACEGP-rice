"""Separate no-Block ablation; retain complete16 screening, kernels and registered grids."""
import acegp_uniform as original
globals().update({k:v for k,v in vars(original).items() if not k.startswith("__")})
def _fit_without_block(g, gx, y, q, qx, groups, config, seed):
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
    needed = list(dict.fromkeys([adaptive_anchor, screened_anchor]))
    qrepeated = {m: [] for m in needed}
    for r in repeated:
        prep = base.prepare_inner_fold_cache("Genome_plus_QTL", g, q, r["inner"], config)
        for m in needed:
            p, h, score = inner_results(m, "Genome_plus_QTL", g, q, y, r["inner"], config, r["model_seed"] + 2000 + MODELS.index(m), prep)
            qrepeated[m].append({"prediction": p, "hyper": h, "score": score})
    qprepared = prepare("Genome_plus_QTL", g, gx, q, qx, config)
    adaptive_outer = base.fit_predict_prepared(adaptive_anchor, "Genome_plus_QTL", qprepared, y, aq_hyper, seed + 7000)
    screened_q_hyper = a11.modal_hyper([r["hyper"] for r in qrepeated[screened_anchor]], [r["score"] for r in qrepeated[screened_anchor]])[0]
    screened_outer = base.fit_predict_prepared(screened_anchor, "Genome_plus_QTL", qprepared, y, screened_q_hyper, seed + 18000)
    return {"Genome_wide": gw, "QTL_informed": .5*adaptive_outer+.5*screened_outer, "adaptive_QTL": adaptive_outer, "screened_QTL": screened_outer, "audit": {**audit, "without_Block_ARD": True, "adaptive_hyper": aq_hyper, "screened_hyper": screened_q_hyper, "adaptive_block_weight": 0.0, "screened_block_weight": 0.0}}

def fit_without_block(g,gx,y,q,qx,groups,config,seed):
    assert tuple(config['learner_pool'])==MODELS and len(base.available_models())>=16 and base.CatBoostRegressor is not None
    with threadpool_limits(limits=int(config.get('blas_threads',2))):
        return _fit_without_block(g,gx,np.asarray(y,float),q,qx,groups,config,int(seed))
