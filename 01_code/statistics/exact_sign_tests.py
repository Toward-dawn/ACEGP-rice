"""Reproduce the two reported post hoc sign tests from saved repeat results."""

from pathlib import Path
from collections import defaultdict
from fractions import Fraction
from math import comb
import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(package):
        parser.error("Output must be outside the archive.")
    if args.output.exists():
        parser.error("Preserve existing outputs; choose a new file.")
    g = defaultdict(dict)
    for population, cohort in [("Primary", "development"), ("Validation", "validation")]:
        source = (
            package
            / "02_data/prediction_results/repeated_cv"
            / cohort
            / "paired_repeat_results.csv"
        )
        with source.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (population, int(row["repeat"]))
                g[key]["ACEGP QTL-informed"] = float(row["QTL_informed_r"])
                g[key]["ACEGP genome-wide"] = float(row["Genome_wide_r"])
    signs = []
    for population in ["Primary", "Validation"]:
        pairs = [
            (repeat, values) for (pop, repeat), values in sorted(g.items()) if pop == population
        ]
        assert [r for r, v in pairs] == list(range(1, 21))
        delta = [v["ACEGP QTL-informed"] - v["ACEGP genome-wide"] for r, v in pairs]
        n = len(delta)
        positive = sum(x > 0 for x in delta)
        negative = sum(x < 0 for x in delta)
        ties = sum(x == 0 for x in delta)
        assert n == 20 and ties == 0
        t = min(positive, negative)
        raw = Fraction(2 * sum(comb(n, j) for j in range(t + 1)), 2**n)
        extreme = sum(mask.bit_count() <= t or mask.bit_count() >= n - t for mask in range(2**n))
        assert raw == Fraction(extreme, 2**n)
        signs.append(
            {
                "population": population,
                "n": n,
                "positive": positive,
                "negative": negative,
                "ties": ties,
                "effect": positive / n,
                "raw_fraction": str(raw),
                "raw_p": float(raw),
                "extreme_sign_patterns": extreme,
                "repeat_differences": delta,
            }
        )
    assert [r["positive"] for r in signs] == [20, 19]
    previous = Fraction(0)
    for rank, index in enumerate(sorted(range(2), key=lambda i: signs[i]["raw_p"])):
        adjusted = max(
            previous, min(Fraction(1), Fraction(signs[index]["raw_fraction"]) * (2 - rank))
        )
        signs[index]["holm_p"] = float(adjusted)
        signs[index]["holm_fraction"] = str(adjusted)
        previous = adjusted
    assert signs[0]["raw_p"] == 1.9073486328125e-6 and signs[0]["holm_p"] == 3.814697265625e-6
    assert signs[1]["raw_p"] == signs[1]["holm_p"] == 0.0000400543212890625
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(signs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
