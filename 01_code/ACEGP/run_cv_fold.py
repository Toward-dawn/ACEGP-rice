"""Fit ACEGP on one saved outer fold and write its held-out predictions."""

from pathlib import Path
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True, choices=["primary", "validation"])
    parser.add_argument("--repeat", type=int, required=True, choices=range(1, 21))
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output == package or output.is_relative_to(package):
        parser.error("Choose a new output location outside the archived package.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be new or empty.")
    sys.path[:0] = [
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve().parent / "src"),
    ]
    import numpy as np
    import pandas as pd
    from revised_inputs import load_arrays
    from acegp_uniform import fit_holdout, stable_seed

    cohort = "development" if args.population == "primary" else "validation"
    data = package / "02_data/prediction_inputs"
    ids, genome, qtl, groups = load_arrays(data, cohort)
    design = json.loads((data / f"inputs/{cohort}_20x5.json").read_text(encoding="utf-8"))
    split = design["repeats"][args.repeat - 1]["folds"][args.fold - 1]
    train = np.array(split["train_rows"], dtype=int)
    test = np.array(split["test_rows"], dtype=int)
    assert ids.sample_id.iloc[test].tolist() == split["test_ids"]
    config = json.loads((data / "inputs/model_config.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    config.update(blas_threads=2, new_cache_directory=str(output / "cache"))
    context = stable_seed(20261013, cohort, args.repeat, args.fold, "outer_refit")
    result = fit_holdout(
        genome[train],
        genome[test],
        ids.loss_SF.to_numpy(float)[train],
        qtl[train],
        qtl[test],
        groups,
        config,
        context,
    )
    pd.DataFrame(
        {
            "sample_id": ids.sample_id.iloc[test].to_numpy(),
            "row_index": test,
            "repeat": args.repeat,
            "fold": args.fold,
            "Genome_wide": result["Genome_wide"],
            "QTL_informed": result["QTL_informed"],
        }
    ).to_csv(output / "predictions.csv", index=False)
    (output / "training_choices.json").write_text(
        json.dumps(result["audit"], indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
