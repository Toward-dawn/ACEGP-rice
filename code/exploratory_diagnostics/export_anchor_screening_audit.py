"""Export the 200-fold anchor selection and 16-learner screening gate audit tables.

Reads strictly from the frozen primary `02_data/prediction_results/repeated_cv/`
training_choices.json records (never from legacy shared_fit_diagnostics files).

Example usage (when placed under `<package_root>/code/exploratory_diagnostics/`):
    python export_anchor_screening_audit.py --output ./audit_outputs
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

REPRESENTATIVE_CASES = [
    ("Primary", 13, 3),
    ("Primary", 20, 3),
    ("Validation", 11, 4),
    ("Validation", 2, 4),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root directory of the ACEGP release package (contains 02_data/).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = args.package_root.resolve()
    rep_root = package / "02_data/prediction_results/repeated_cv"
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {}

    for pop, coh in [("Primary", "development"), ("Validation", "validation")]:
        files = sorted((rep_root / coh).glob("r*_f*/training_choices.json"))
        if len(files) != 100:
            raise RuntimeError(f"Expected 100 training_choices.json files for {coh}, found {len(files)}")
        anchor_pairs = Counter()
        both_zero_block = 0
        beat_mean_folds = 0
        beat_both_folds = 0
        promoted_folds = 0

        for path in files:
            tag = path.parent.name
            rep, fold = int(tag[1:3]), int(tag[5:])
            j = json.loads(path.read_text(encoding="utf-8"))

            a_anc, s_anc = j["adaptive_anchor"], j["screened_anchor"]
            a_bw, s_bw = float(j["adaptive_block_weight"]), float(j["screened_block_weight"])
            anchor_pairs[f"{a_anc}/{s_anc}"] += 1
            if a_bw == 0.0 and s_bw == 0.0:
                both_zero_block += 1

            cands = j["screened_candidates"]
            g_name = j["screened_gate"]["genetic_anchor"]
            se = float(j["screened_gate"]["fisher_z_standard_error"])
            promoted = bool(j["screened_gate"]["challenger_promoted"])
            if promoted:
                promoted_folds += 1

            g_row = next(c for c in cands if c["model"] == g_name)
            g_r1, g_r2 = float(g_row["repeat_r"][0]), float(g_row["repeat_r"][1])
            g_mean = float(g_row["mean_r"])

            nk = [c for c in cands if c["model"] not in ("GBLUP", "RKHS")]
            if any(float(c["mean_r"]) > g_mean for c in nk):
                beat_mean_folds += 1
            if any(float(c["repeat_r"][0]) > g_r1 and float(c["repeat_r"][1]) > g_r2 for c in nk):
                beat_both_folds += 1

            if nk:
                best = max(nk, key=lambda c: float(c["mean_r"]))
                nk_m = best["model"]
                nk_r1, nk_r2 = float(best["repeat_r"][0]), float(best["repeat_r"][1])
                nk_mean = float(best["mean_r"])
                dm = nk_mean - g_mean
                zg = math.atanh(np.clip(nk_mean, -1 + 1e-12, 1 - 1e-12)) - math.atanh(
                    np.clip(g_mean, -1 + 1e-12, 1 - 1e-12)
                )
                won_both = bool(nk_r1 > g_r1 and nk_r2 > g_r2)
            else:
                nk_m, nk_r1, nk_r2, nk_mean, dm, zg, won_both = (
                    "None", np.nan, np.nan, np.nan, np.nan, np.nan, False
                )

            rows.append({
                "Population": pop,
                "Repeat": rep,
                "Fold": fold,
                "Adaptive_anchor": a_anc,
                "Screened_anchor": s_anc,
                "Adaptive_block_weight": a_bw,
                "Screened_block_weight": s_bw,
                "Default_genetic_anchor_2x5": g_name,
                "Genetic_r1": g_r1,
                "Genetic_r2": g_r2,
                "Genetic_mean_r": g_mean,
                "Top_nonkernel_2x5": nk_m,
                "Nonkernel_r1": nk_r1,
                "Nonkernel_r2": nk_r2,
                "Nonkernel_mean_r": nk_mean,
                "Mean_r_gain": dm,
                "Fisher_z_gain": zg,
                "Fisher_z_SE_threshold": se,
                "Won_both_inner_repeats": won_both,
                "Challenger_promoted": promoted,
            })

        summary[pop] = {
            "total_folds": len(files),
            "anchor_pairs_adaptive_screened": dict(anchor_pairs),
            "both_block_weights_zero_folds": both_zero_block,
            "nonkernel_higher_mean_2x5_folds": beat_mean_folds,
            "nonkernel_won_both_repeats_2x5_folds": beat_both_folds,
            "nonkernel_promoted_folds": promoted_folds,
        }

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(out_dir / "Supplementary_Data_200fold_anchor_screening_audit.csv", index=False)

    rep_df = pd.concat(
        [
            audit_df[(audit_df.Population == pop) & (audit_df.Repeat == r) & (audit_df.Fold == f)]
            for pop, r, f in REPRESENTATIVE_CASES
        ],
        ignore_index=True,
    )
    rep_df.to_csv(out_dir / "Supplementary_Data_4_representative_screening_cases.csv", index=False)
    (out_dir / "anchor_screening_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
