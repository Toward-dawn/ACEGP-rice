"""Disk-backed equivalent of the frozen chromosome/GT-stratified blind masking."""

from pathlib import Path
import gzip
import json
import hashlib
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gt_class(gt):
    x = gt.replace("|", "/").split("/")
    return int(x[0]) + int(x[1]) if len(x) == 2 and all(z in ["0", "1"] for z in x) else -1


def eligible_class(fmt, cell):
    d = dict(zip(fmt, cell.split(":")))
    klass = gt_class(d.get("GT", "."))
    if klass < 0:
        return -1
    try:
        dp = int(d.get("DP", "."))
        ad = list(map(int, d.get("AD", ".").split(",")))
        gq = int(d.get("GQ", ".")) if d.get("GQ", ".") != "." else -1
        support = (
            min(ad[:2])
            if klass == 1 and len(ad) >= 2
            else (ad[1] if klass == 2 and len(ad) >= 2 else (ad[0] if klass == 0 else 0))
        )
    except (ValueError, IndexError):
        return -1
    return klass if dp >= 10 and support >= 3 and (klass == 0 or gq >= 30) else -1


def build_cache(source, folder, expected_sites, expected_samples):
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    accepted = folder / "CACHE_ACCEPTANCE.json"
    digest = sha(source)
    ns = len(expected_samples)
    if accepted.exists():
        d = json.loads(accepted.read_text())
        assert (
            d["source_sha256"] == digest
            and d["samples"] == expected_samples
            and d["sites"] == expected_sites
        )
        for name, h in d["files"].items():
            assert sha(folder / name) == h
        return d
    code = np.lib.format.open_memmap(
        folder / "eligible_GT_code.npy", mode="w+", dtype=np.int8, shape=(expected_sites, ns)
    )
    code[:] = -1
    chrom = np.lib.format.open_memmap(
        folder / "chromosome.npy", mode="w+", dtype=np.int16, shape=(expected_sites,)
    )
    row = 0
    header_seen = False
    with gzip.open(source, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                assert line.rstrip("\n").split("\t")[9:] == expected_samples
                header_seen = True
            if line.startswith("#"):
                continue
            v = line.rstrip("\n").split("\t")
            assert header_seen and len(v) == ns + 9 and row < expected_sites
            chrom[row] = int(v[0][3:9]) - 29255
            fmt = v[8].split(":")
            code[row] = [eligible_class(fmt, cell) for cell in v[9:]]
            row += 1
    assert (
        row == expected_sites
        and np.all(chrom[1:] >= chrom[:-1])
        and 1 <= chrom.min()
        and chrom.max() <= 12
    )
    code.flush()
    chrom.flush()
    strata = []
    files = {}
    for c in range(1, 13):
        start = int(np.searchsorted(chrom, c, side="left"))
        end = int(np.searchsorted(chrom, c, side="right"))
        for klass in range(3):
            cells = (
                np.flatnonzero((code[start:end] == klass).reshape(-1)).astype(np.uint64)
                + start * ns
            )
            assert not len(cells) or cells[-1] < 2**32
            file = folder / f"chr{c}_class{klass}.npy"
            np.save(file, cells.astype(np.uint32), allow_pickle=False)
            strata.append({"chr": c, "class": klass, "eligible": len(cells), "file": file.name})
            files[file.name] = sha(file)
    for name in ["eligible_GT_code.npy", "chromosome.npy"]:
        files[name] = sha(folder / name)
    d = {
        "state": "PASS_DISK_BACKED_FROZEN_ELIGIBILITY_CACHE",
        "sites": expected_sites,
        "samples": expected_samples,
        "source_sha256": digest,
        "strata": strata,
        "files": files,
        "truth_rule": {"DP_min": 10, "support_min": 3, "nonreference_GQ_min": 30},
        "no_missing_GT_filled": True,
        "phenotypes_accessed": False,
    }
    accepted.write_text(json.dumps(d, indent=2))
    return d


def select_truth(folder, seed):
    folder = Path(folder)
    cache = json.loads((folder / "CACHE_ACCEPTANCE.json").read_text())
    ns = len(cache["samples"])
    rng = np.random.default_rng(seed)
    parts = []
    counts = {}
    for r in cache["strata"]:
        n = r["eligible"]
        if not n:
            continue
        number = max(1, int(n * 0.10))
        pick = rng.choice(n, size=number, replace=False)
        eligible = np.load(folder / r["file"], mmap_mode="r")
        cells = eligible[pick].astype(np.int64)
        parts.append(
            np.column_stack(
                (cells // ns, cells % ns, np.full(number, r["class"], np.int64))
            ).astype(np.int32)
        )
        counts[str((r["chr"], r["class"]))] = {"eligible": n, "masked": number}
    truth = np.concatenate(parts)
    assert truth.shape[1] == 3
    return truth, counts


def masked_vcf_stream(source, truth, samples, output=None):
    """Hash and optionally emit exactly the original masked VCF in bounded memory."""
    n = len(samples)
    cells = np.sort(truth[:, 0].astype(np.int64) * n + truth[:, 1])
    assert len(np.unique(cells)) == len(cells)
    digest = hashlib.sha256()
    at = 0
    row = 0
    cleared = 0
    with gzip.open(source, "rt") as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("#CHROM"):
                    assert line.rstrip("\n").split("\t")[9:] == samples
                data = line.encode()
            else:
                fields = line.rstrip("\n").split("\t")
                assert len(fields) == n + 9
                fields[5] = "."
                fields[6] = "."
                fields[7] = "."
                end = int(np.searchsorted(cells, (row + 1) * n, side="left"))
                for value in cells[at:end]:
                    si = int(value - row * n)
                    assert 0 <= si < n
                    fields[si + 9] = ":".join(
                        ["./." if name == "GT" else "." for name in fields[8].split(":")]
                    )
                    cleared += 1
                at = end
                data = ("\t".join(fields) + "\n").encode()
                row += 1
            digest.update(data)
            if output is not None:
                output.write(data)
    assert at == len(cells) and cleared == len(truth)
    return {
        "uncompressed_masked_VCF_SHA256": digest.hexdigest(),
        "masked_cells": cleared,
        "sites": row,
        "all_masked_FORMAT_fields_cleared": True,
        "site_QUAL_FILTER_INFO_cleared": True,
    }
