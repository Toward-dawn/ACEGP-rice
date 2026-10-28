"""Load the fixed genotype matrices, sample order and region columns."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHAPES = {
    "development": (241, 32670, 67985, 8),
    "validation": (151, 32666, 67104, 7),
}


def load_primary_matrices(root, sample_ids):
    raw = pd.read_csv(root / "inputs/development_background.raw", sep=r"\s+", dtype={"IID": str})
    assert raw.IID.tolist() == sample_ids
    genome = raw.iloc[:, 6:].to_numpy(float)
    genome[genome < 0] = np.nan
    order = (
        pd.read_csv(root / "inputs/development_qtl_sample_order.csv", dtype=str).iloc[:, 0].tolist()
    )
    qtl = np.asarray(np.load(root / "inputs/development_qtl.npy", mmap_mode="r"), float)
    markers = pd.read_csv(root / "inputs/development_qtl_markers.csv")
    return genome, qtl, markers, order


def load_validation_matrices(root):
    receipt = json.loads((root / "validation_inputs/INPUT_ACCEPTANCE.json").read_text())
    assert receipt["state"] == "PASS_UNIFORM_GT_INPUTS_AND_MASKED_QUALITY"
    order = pd.read_csv(root / "validation_inputs/sample_order.csv", dtype=str).sample_id.tolist()
    genome = np.asarray(np.load(root / "validation_inputs/background.npy", mmap_mode="r"), float)
    qtl = np.asarray(np.load(root / "validation_inputs/qtl.npy", mmap_mode="r"), float)
    markers = pd.read_csv(root / "validation_inputs/qtl_markers.csv")
    return genome, qtl, markers, order


def assign_region_columns(markers, regions, marker_count, expected_regions):
    """Return columns for each fixed region, with exactly one region per SNP."""
    groups = []
    coverage = np.zeros(marker_count, int)
    for region in regions.itertuples():
        columns = np.flatnonzero(
            (markers.chr.to_numpy() == region.chr)
            & (markers.pos.to_numpy() >= region.start_bp)
            & (markers.pos.to_numpy() <= region.end_bp)
        )
        assert len(columns) > 0
        coverage[columns] += 1
        groups.append((region.qid, columns))
    assert len(groups) == expected_regions
    assert np.all(coverage == 1)
    assert "zhu17_01" not in [region_id for region_id, _ in groups]
    return groups


def load_arrays(root, cohort):
    root = Path(root)
    assert cohort in EXPECTED_SHAPES
    assert (root / "INPUT_AND_REUSE_ACCEPTANCE.json").exists()
    ids = pd.read_csv(root / f"inputs/{cohort}_samples.csv", dtype={"sample_id": str})
    if cohort == "development":
        genome, qtl, markers, order = load_primary_matrices(root, ids.sample_id.tolist())
    else:
        genome, qtl, markers, order = load_validation_matrices(root)

    sample_count, genome_count, qtl_count, region_count = EXPECTED_SHAPES[cohort]
    assert order == ids.sample_id.tolist()
    assert genome.shape == (sample_count, genome_count)
    assert qtl.shape == (sample_count, qtl_count)
    regions = pd.read_csv(root / f"inputs/{cohort}_regions.csv")
    groups = assign_region_columns(markers, regions, qtl.shape[1], region_count)
    return ids, genome, qtl, groups
