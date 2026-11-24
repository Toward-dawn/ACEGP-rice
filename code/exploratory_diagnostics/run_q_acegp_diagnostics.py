"""Reproduce Table S3 Section C exploratory diagnostic models (train-only, no holdout phenotype).

Models evaluated directly on QTL-informed inputs across the frozen 20 x 5 outer splits
and 3 x 3 inner cross-validation splits (strictly confined to each outer training fold):
  - Q-ACEGP-3          : simplex-weighted consensus of regional-kernel GBLUP, RKHS, and QTL Block-ARD
  - Q-ACEGP-2          : simplex-weighted two-kernel consensus of regional-kernel GBLUP and RKHS (no Block-ARD)
  - GBLUP-QTL (3x3 CV) : standalone regional-kernel GBLUP component tuned via 3 x 3 modal inner CV

This script imports `code/ACEGP/src/acegp_uniform.py` read-only and does not modify
any frozen primary ACEGP code or configuration.

Example usage (when placed under `<package_root>/code/exploratory_diagnostics/`):
    python run_q_acegp_diagnostics.py --output ./exploratory_outputs --population all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

STEP = 0.1
TOLERANCE = 0.002
SIMPLEX_3 = [
    (round(g, 1), round(k, 1), round(1.0 - g - k, 1))
    for g in np.arange(0.0, 1.0001, STEP)
    for k in np.arange(0.0, 1.0001 - g, STEP)
]
WEIGHTS_2 = [round(k, 1) for k in np.arange(0.0, 1.0001, STEP)]


def parse_range(spec: str, lo: int, hi: int) -> list[int]:
    if "-" in spec:
        a, b = map(int, spec.split("-", 1))
        vals = list(range(a, b + 1))
    else:
        vals = [int(x) for x in spec.split(",")]
    if not vals or min(vals) < lo or max(vals) > hi:
        raise ValueError(f"Range {spec!r} outside [{lo}, {hi}]")
    return vals


def choose_simplex_3(oofs, ys, base, stable):
    recs = []
    for w in SIMPLEX_3:
        wv = np.array(w, dtype=float)
        scores = [base.pearson(y, P @ wv) for P, y in zip(oofs, ys)]
        recs.append({"w": w, "score": stable(scores)})
    best = max(r["score"] for r in recs)
    near = [r for r in recs if r["score"] >= best - TOLERANCE]
    chosen = max(near, key=lambda r: (r["w"][0], -r["w"][2]))
    return tuple(float(x) for x in chosen["w"])


def choose_simplex_2(oofs, ys, base, stable):
    recs = []
    for wk in WEIGHTS_2:
        wg = round(1.0 - wk, 1)
        scores = [base.pearson(y, wg * P[:, 0] + wk * P[:, 1]) for P, y in zip(oofs, ys)]
        recs.append({"w": (wg, wk), "score": stable(scores)})
    best = max(r["score"] for r in recs)
    near = [r for r in recs if r["score"] >= best - TOLERANCE]
    chosen = max(near, key=lambda r: r["w"][0])
    return tuple(float(x) for x in chosen["w"])


def fit_fold_diagnostics(U, g, gx, y, q, qx, groups, config, seed):
    base, a11, stable = U.base, U.a11, U.stable
    y = np.asarray(y, dtype=float)
    with U.threadpool_limits(limits=int(config.get("blas_threads", 2))):
        oofs_3 = []
        hy = {"GBLUP": [], "RKHS": []}
        sc = {"GBLUP": [], "RKHS": []}
        bh, bs = [], []
        for r_in in range(1, 4):
            folds = base.folds_for(len(y), seed + r_in * 100000, int(config["inner_folds"]))
            ms = seed + r_in * 1000000
            prep = base.prepare_inner_fold_cache("Genome_plus_QTL", g, q, folds, config)
            cols = []
            for m in ("GBLUP", "RKHS"):
                p, h, s = U.inner_results(
                    m, "Genome_plus_QTL", g, q, y, folds, config,
                    ms + 2000 + U.MODELS.index(m), prep,
                )
                cols.append(p)
                hy[m].append(h)
                sc[m].append(s)
            bp, bhyp, bsc = U.block_oof(y, g, q, groups, folds, config, ms + 5000)
            cols.append(bp)
            bh.append(bhyp)
            bs.append(bsc)
            oofs_3.append(np.column_stack(cols))

        w3 = choose_simplex_3(oofs_3, [y] * 3, base, stable)
        w2 = choose_simplex_2([P[:, :2] for P in oofs_3], [y] * 3, base, stable)

        qp = U.prepare("Genome_plus_QTL", g, gx, q, qx, config)
        h_g = a11.modal_hyper(hy["GBLUP"], sc["GBLUP"])[0]
        h_k = a11.modal_hyper(hy["RKHS"], sc["RKHS"])[0]
        pred_g = base.fit_predict_prepared("GBLUP", "Genome_plus_QTL", qp, y, h_g, seed + 7000)
        pred_k = base.fit_predict_prepared("RKHS", "Genome_plus_QTL", qp, y, h_k, seed + 7001)
        if w3[2] > 0:
            design = U.block_design(g, gx, q, qx, groups, config, seed + 9000)
            pred_b = U.block_predict(y, design, a11.modal_hyper(bh, bs)[0])
        else:
            pred_b = np.zeros(len(gx), dtype=float)

        pred_q3 = w3[0] * pred_g + w3[1] * pred_k + w3[2] * pred_b
        pred_q2 = w2[0] * pred_g + w2[1] * pred_k
        return {
            "Q_ACEGP_3": pred_q3,
            "Q_ACEGP_2": pred_q2,
            "GBLUP_QTL_3x3": pred_g,
            "RKHS_QTL_3x3": pred_k,
            "audit": {
                "weights_Q_ACEGP_3_G_K_B": list(w3),
                "weights_Q_ACEGP_2_G_K": list(w2),
                "modal_hyper_GBLUP": h_g,
                "modal_hyper_RKHS": h_k,
            },
        }


def summarize_population(pop: str, all_df: pd.DataFrame, samples_df: pd.DataFrame) -> list[dict]:
    d = all_df.merge(samples_df[["sample_id", "loss_SF"]], on="sample_id")
    specs = [
        ("Q_ACEGP_3", "Q-ACEGP-3", "3x3 modal + inner-OOF simplex"),
        ("GBLUP_QTL_3x3", "GBLUP-QTL (3x3 CV)", "3x3 modal inner CV"),
        ("Q_ACEGP_2", "Q-ACEGP-2", "3x3 modal + inner-OOF simplex"),
    ]
    rows = []
    for col, label, tuning in specs:
        rep_metrics = []
        for rep, g in d.groupby("repeat"):
            y, p = g["loss_SF"].to_numpy(float), g[col].to_numpy(float)
            rep_metrics.append({
                "r": float(np.corrcoef(y, p)[0, 1]),
                "rho": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "mae": float(np.mean(np.abs(y - p))),
            })
        rm = pd.DataFrame(rep_metrics)
        rows.append({
            "Population": pop.capitalize(),
            "Model": label,
            "Inner_CV_Tuning": tuning,
            "Complete_Repeats": len(rm),
            "Pearson_r_mean": rm["r"].mean(),
            "Pearson_r_SD": rm["r"].std(ddof=1) if len(rm) > 1 else np.nan,
            "Spearman_rho_mean": rm["rho"].mean(),
            "Spearman_rho_SD": rm["rho"].std(ddof=1) if len(rm) > 1 else np.nan,
            "RMSE_mean": rm["rmse"].mean(),
            "RMSE_SD": rm["rmse"].std(ddof=1) if len(rm) > 1 else np.nan,
            "MAE_mean": rm["mae"].mean(),
            "MAE_SD": rm["mae"].std(ddof=1) if len(rm) > 1 else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root directory of the ACEGP release package (contains code/ and 02_data/).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory outside this package and archived results.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow reuse of an existing output directory containing predictions.csv after independent provenance checks.",
    )
    parser.add_argument("--population", choices=["primary", "validation", "all"], default="all")
    parser.add_argument("--repeats", default="1-20", help="Repeat range, e.g. '1-20' or '1'.")
    parser.add_argument("--folds", default="1-5", help="Fold range, e.g. '1-5' or '1'.")
    parser.add_argument("--blas-threads", type=int, default=2)
    args = parser.parse_args()

    package = args.package_root.resolve()
    out_root = args.output.resolve()
    try:
        out_root.relative_to(package)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Refusing output inside package root {package}; choose an external output directory."
        )
    existing_predictions = list(out_root.rglob("predictions.csv")) if out_root.is_dir() else []
    if existing_predictions and not args.resume:
        sample = ", ".join(str(p) for p in existing_predictions[:3])
        more = " ..." if len(existing_predictions) > 3 else ""
        raise ValueError(
            "Output directory already contains predictions.csv; use a fresh directory "
            f"or pass --resume after independent provenance checks ({sample}{more})."
        )

    code_dir = package / "code/ACEGP"
    sys.path[:0] = [str(code_dir), str(code_dir / "src")]
    import acegp_uniform as U
    from revised_inputs import load_arrays

    data_root = package / "02_data/prediction_inputs"
    config_base = json.loads((data_root / "inputs/model_config.json").read_text(encoding="utf-8"))
    repeats = parse_range(args.repeats, 1, 20)
    folds = parse_range(args.folds, 1, 5)
    pops = ["primary", "validation"] if args.population == "all" else [args.population]

    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for pop in pops:
        cohort = "development" if pop == "primary" else "validation"
        ids, genome, qtl, groups = load_arrays(data_root, cohort)
        design = json.loads((data_root / f"inputs/{cohort}_20x5.json").read_text(encoding="utf-8"))
        pop_out = out_root / pop
        pop_out.mkdir(parents=True, exist_ok=True)
        frames = []

        for rep in repeats:
            for fold in folds:
                fold_dir = pop_out / f"r{rep:02d}_f{fold}"
                pred_csv = fold_dir / "predictions.csv"
                if pred_csv.is_file() and args.resume:
                    frames.append(pd.read_csv(pred_csv))
                    continue
                if pred_csv.is_file():
                    raise ValueError(
                        f"Found existing prediction file {pred_csv}; use --resume only after verifying it."
                    )
                fold_dir.mkdir(parents=True, exist_ok=True)
                split = design["repeats"][rep - 1]["folds"][fold - 1]
                tr = np.asarray(split["train_rows"], dtype=int)
                te = np.asarray(split["test_rows"], dtype=int)
                assert ids.sample_id.iloc[te].tolist() == split["test_ids"]
                seed = U.stable_seed(20261013, cohort, rep, fold, "outer_refit")
                cfg = dict(config_base, blas_threads=args.blas_threads, new_cache_directory=None)

                res = fit_fold_diagnostics(
                    U, genome[tr], genome[te], ids.loss_SF.to_numpy(float)[tr],
                    qtl[tr], qtl[te], groups, cfg, seed,
                )
                df = pd.DataFrame({
                    "sample_id": ids.sample_id.iloc[te].to_numpy(),
                    "row_index": te,
                    "repeat": rep,
                    "fold": fold,
                    "Q_ACEGP_3": res["Q_ACEGP_3"],
                    "Q_ACEGP_2": res["Q_ACEGP_2"],
                    "GBLUP_QTL_3x3": res["GBLUP_QTL_3x3"],
                    "RKHS_QTL_3x3": res["RKHS_QTL_3x3"],
                    "wG_Q2": res["audit"]["weights_Q_ACEGP_2_G_K"][0],
                    "wK_Q2": res["audit"]["weights_Q_ACEGP_2_G_K"][1],
                })
                df.to_csv(pred_csv, index=False)
                (fold_dir / "audit.json").write_text(
                    json.dumps(res["audit"], indent=2, default=str), encoding="utf-8"
                )
                frames.append(df)
                print(
                    f"{pop} r{rep:02d} f{fold} | w3={res['audit']['weights_Q_ACEGP_3_G_K_B']} "
                    f"| w2={res['audit']['weights_Q_ACEGP_2_G_K']}",
                    flush=True,
                )

        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(pop_out / "all_oof_predictions.csv", index=False)
        if len(folds) == 5:
            summary_rows.extend(summarize_population(pop, all_df, ids))

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        sum_df.to_csv(out_root / "table_s3_section_c_summary.csv", index=False)
        print("\n=== Table S3 Section C Summary ===")
        print(sum_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
