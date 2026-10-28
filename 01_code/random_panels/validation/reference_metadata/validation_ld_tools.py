import numpy as np
import hashlib
from pathlib import Path

BINS = [
    (0, 10000),
    (10000, 25000),
    (25000, 50000),
    (50000, 100000),
    (100000, 250000),
    (250000, 500000),
]
MAX_PAIRS = 500
BASE_SEED = 20261013


def retained_segments(start, end, holes=()):
    assert not holes
    return [(start, end)]


class PositionIndex:
    """The same sorted original BIM rows as MarkerIndex; no histogram caches."""

    def __init__(self, chrom, position):
        self.by_chr = {}
        for c in np.unique(chrom):
            rows = np.flatnonzero(chrom == c)
            rows = rows[np.argsort(position[rows], kind="stable")]
            self.by_chr[int(c)] = {"rows": rows, "position": position[rows].astype(np.int64)}

    def bounds(self, chrom, start, end):
        p = self.by_chr[int(chrom)]["position"]
        return int(np.searchsorted(p, start, side="left")), int(
            np.searchsorted(p, end, side="right")
        )

    def region(self, chrom, start, end, holes=()):
        d = self.by_chr[int(chrom)]
        pieces = [
            np.arange(*self.bounds(chrom, l, r)) for l, r in retained_segments(start, end, holes)
        ]
        ix = np.concatenate(pieces) if pieces else np.empty(0, int)
        return d["position"][ix], d["rows"][ix]


class BedReader:
    def __init__(self, path, n, m):
        self.n, self.m, self.stride = n, m, (n + 3) // 4
        with path.open("rb") as f:
            assert f.read(3) == b"\x6c\x1b\x01", "Not a SNP-major PLINK1 BED"
        assert path.stat().st_size == 3 + self.stride * m
        self.raw = np.memmap(path, dtype=np.uint8, mode="r", offset=3, shape=(m, self.stride))
        self.lookup = np.array([2, -1, 1, 0], dtype=np.int8)
        self.sample_byte = np.arange(n) // 4
        self.sample_shift = (np.arange(n) % 4) * 2

    def __call__(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        assert rows.ndim == 1 and (not len(rows) or (rows.min() >= 0 and rows.max() < self.m))
        code = (self.raw[rows][:, self.sample_byte] >> self.sample_shift) & 3
        return self.lookup[code].T


def pair_values(g, a, b):
    """Pairwise-complete dosage correlation squared, >=3 nonconstant pairs.

    Float64 integer sufficient statistics are exact here (241 diploid calls),
    so this vectorization retains the original helper's missing-data rule.
    """
    x, y = g[:, a].T.astype(np.float64), g[:, b].T.astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    n = valid.sum(axis=1)
    x = np.where(valid, x, 0.0)
    y = np.where(valid, y, 0.0)
    sx, sy = x.sum(axis=1), y.sum(axis=1)
    xx = n * (x * x).sum(axis=1) - sx * sx
    yy = n * (y * y).sum(axis=1) - sy * sy
    xy = n * (x * y).sum(axis=1) - sx * sy
    ok = (n >= 3) & (xx > 0) & (yy > 0)
    r2 = np.full(len(n), np.nan)
    r2[ok] = np.clip(xy[ok] ** 2 / (xx[ok] * yy[ok]), 0, 1)
    return r2, n, ok


def quantiles(v, q=(0.1, 0.5, 0.9)):
    return np.quantile(v, q).tolist() if len(v) else None


def stable_seed(pool_id, variant_id):
    key = f"{BASE_SEED}|validation151|{pool_id}|{variant_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def region_diagnostics(index, reader, signal, variant, pool_id):
    c, start, end = int(signal["chr"]), int(variant["start_bp"]), int(variant["end_bp"])
    holes = variant.get("excluded_subranges_bp", [])
    pos, rows = index.region(c, start, end, holes)
    assert len(rows) > 0
    rng = np.random.default_rng(stable_seed(pool_id, variant["variant_id"]))
    collected, traces, base = [], [], np.arange(len(pos)) + 1
    for l, r in BINS:
        lo = np.maximum(base, np.searchsorted(pos, pos + l, side="left"))
        hi = np.searchsorted(pos, pos + r, side="left")
        count = np.maximum(hi - lo, 0).astype(np.int64)
        cum = np.cumsum(count)
        possible = int(cum[-1]) if len(cum) else 0
        flat = (
            rng.choice(possible, size=min(MAX_PAIRS, possible), replace=False)
            if possible
            else np.empty(0, int)
        )
        a = np.searchsorted(cum, flat, side="right")
        before = np.where(a > 0, cum[np.maximum(a - 1, 0)], 0)
        b = lo[a] + flat - before
        assert len(np.unique(flat)) == len(flat)
        assert np.all(a < b) and np.all(pos[b] - pos[a] >= l) and np.all(pos[b] - pos[a] < r)
        traces.append(np.column_stack((rows[a], rows[b])).astype("<u4").tobytes())
        collected.append((possible, a, b))
    needed = np.unique(np.concatenate([np.concatenate((a, b)) for _, a, b in collected]))
    g = reader(rows[needed])
    diagnostics = []
    for (l, r), (possible, a, b) in zip(BINS, collected):
        r2, n, ok = pair_values(g, np.searchsorted(needed, a), np.searchsorted(needed, b))
        values = r2[ok]
        diagnostics.append(
            {
                "distance_bp": [l, r],
                "upper_edge_exclusive": True,
                "possible_pairs": possible,
                "sampled_pairs": len(a),
                "computable_pairs": int(ok.sum()),
                "undefined_nonconstant_or_insufficient_pairs": int((~ok).sum()),
                "mean_r_squared": float(values.mean()) if len(values) else None,
                "median_r_squared": float(np.median(values)) if len(values) else None,
                "r_squared_quantiles_10_50_90": quantiles(values),
                "pairwise_observed_samples_min": int(n.min()) if len(n) else None,
                "pairwise_observed_samples_median": float(np.median(n)) if len(n) else None,
            }
        )
    d = np.diff(pos)
    return {
        "pool_index": pool_id,
        "signal_id": signal["signal_id"],
        "canonical_name": signal["name"],
        "variant_id": variant["variant_id"],
        "chr": c,
        "start_bp": start,
        "end_bp": end,
        "excluded_subranges_bp": holes,
        "raw_SNPs": len(rows),
        "window_length_bp": end - start + 1,
        "seed": stable_seed(pool_id, variant["variant_id"]),
        "sampled_original_BIM_row_pairs_SHA256": hashlib.sha256(b"".join(traces)).hexdigest(),
        "marker_spacing_bp_quantiles_10_50_90": quantiles(d),
        "marker_spacing_mean_bp": float(d.mean()) if len(d) else None,
        "marker_spacing_max_bp": int(d.max()) if len(d) else None,
        "marker_spacing_zero_gaps": int((d == 0).sum()),
        "LD_by_distance": diagnostics,
        "LD_matching_assumed": False,
    }


def union_spacing(signals, index):
    """Count each selected original marker once; never take interchromosome gaps."""
    parts = {}
    for s in signals:
        for v in s["variants"]:
            _, rows = index.region(
                s["chr"], v["start_bp"], v["end_bp"], v.get("excluded_subranges_bp", [])
            )
            parts.setdefault(int(s["chr"]), []).append(rows)
    result, diffs, union_rows = [], [], []
    for chrom, lists in sorted(parts.items()):
        rows = np.unique(np.concatenate(lists))
        data = index.by_chr[chrom]
        p = data["position"][np.searchsorted(data["rows"], rows)]
        # Original BIM is position-sorted, verified by the raw-metadata manifest.
        assert np.all(data["rows"][np.searchsorted(data["rows"], rows)] == rows)
        p = np.sort(p)
        ds = np.diff(p)
        result.append(
            {
                "chr": chrom,
                "unique_SNPs": len(rows),
                "adjacent_pairs": len(ds),
                "spacing_quantiles_10_50_90_bp": quantiles(ds),
            }
        )
        diffs.append(ds)
        union_rows.append(rows)
    all_d = np.concatenate(diffs)
    rows = np.sort(np.concatenate(union_rows))
    return {
        "raw_unique_SNPs": len(rows),
        "adjacent_pairs_excluding_interchromosome_pairs": len(all_d),
        "spacing_quantiles_10_50_90_bp": quantiles(all_d),
        "spacing_mean_bp_includes_between_region_gaps": float(all_d.mean()),
        "spacing_max_bp_includes_between_region_gaps": int(all_d.max()),
        "within_chromosome_distribution_includes_region_and_source_hole_gaps": True,
        "chromosome_diagnostics": result,
    }, rows
