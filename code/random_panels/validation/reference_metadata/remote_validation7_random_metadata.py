"""Validation seven-region metadata matching; genotype-only, no model imports."""

from pathlib import Path
import os
import sys
import csv
import json
import hashlib
import fcntl
import time
import subprocess
import argparse
import traceback
import collections

for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
import numpy as np

PACKAGE = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "documentation/PACKAGE_STATUS.json").is_file()
)
CODE = PACKAGE / "code/ACEGP"
sys.path[:0] = [str(CODE), str(CODE / "src")]
from public_paths import RANDOM_ROOT, DATA_ROOT, output_root

ROOT = Path(__file__).resolve().parent
VMETA = RANDOM_ROOT / "validation/metadata"
SOURCE = RANDOM_ROOT / "validation/PENDING_common_genotypes/observed_catalog"
import validation_ld_tools as ld

OUT = None
REG = json.loads((VMETA / "original_matching_registration.json").read_text())
EDGE = np.array([np.inf if x == "infinity" else x for x in REG["spacing_edges_bp"]], float)


class MatrixReader:
    def __init__(self, path, n, m):
        self.n, self.m = n, m
        self.raw = np.load(path, mmap_mode="r", allow_pickle=False)
        assert self.raw.shape == (m, n) and self.raw.dtype == np.int8

    def __call__(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        assert rows.ndim == 1 and (not len(rows) or (rows.min() >= 0 and rows.max() < self.m))
        return np.asarray(self.raw[rows, :].T, dtype=np.int8)


def stop_requested():
    p = ROOT.parent / "STOP_REQUESTED"
    return p.exists() and p.read_text() != "SCHEDULER_HANDOFF_AFTER_ACCEPTED_MASKED_CHR1_V1\n"


def resources_ok():
    try:
        lines = Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
        anon = int(dict(line.split() for line in lines)["anon"]) / (1024**3)
    except Exception:
        anon = float("inf")
    import shutil

    free = shutil.disk_usage(ROOT).free / (1024**3)
    return (
        anon < REG["resource_gate"]["anonymous_memory_start_below_GiB"]
        and free > REG["resource_gate"]["free_disk_above_GiB"]
    )


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for z in iter(lambda: f.read(1 << 20), b""):
            h.update(z)
    return h.hexdigest()


def atomic(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = Path(str(p) + ".partial." + str(os.getpid()))
    q.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False))
    q.replace(p)


def regions():
    with (DATA_ROOT / "inputs/validation_regions.csv").open() as f:
        return [
            {**r, **{k: int(r[k]) for k in ["chr", "start_bp", "end_bp"]}}
            for r in csv.DictReader(f)
        ]


def rng(label):
    return np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(f"{REG['seed']}|{REG['domain_namespace']}|{label}".encode()).digest()[
                :8
            ],
            "little",
        )
    )


def data():
    source = SOURCE / "raw_metadata_validation151.npz"
    with np.load(source, allow_pickle=False) as f:
        raw = {k: f[k] for k in ["chromosome", "position_bp", "raw_MAF", "raw_missing_rate"]}
    assert len(raw["position_bp"]) == REG["source_sites"]
    return (
        raw,
        ld.PositionIndex(raw["chromosome"], raw["position_bp"]),
        MatrixReader(SOURCE / "observed_A1_151.npy", 151, len(raw["position_bp"])),
    )


def profile(index, reader, r, start):
    v = {
        "variant_id": r["qid"],
        "start_bp": int(start),
        "end_bp": int(start + r["end_bp"] - r["start_bp"]),
    }
    s = {"chr": r["chr"], "signal_id": r["qid"], "name": r["name"]}
    records = [
        ld.region_diagnostics(index, reader, s, v, int(ns + start)) for ns in REG["LD_namespaces"]
    ]
    return {
        "LD": [[b["mean_r_squared"] for b in d["LD_by_distance"]] for d in records],
        "pair_hashes": [d["sampled_original_BIM_row_pairs_SHA256"] for d in records],
        "spacing_q": records[0]["marker_spacing_bp_quantiles_10_50_90"],
        "n": records[0]["raw_SNPs"],
    }


def record(index, reader, raw, r, start):
    end = start + r["end_bp"] - r["start_bp"]
    pos, rows = index.region(r["chr"], start, end)
    d = np.diff(pos)
    if len(rows) < 2:
        return None
    p = profile(index, reader, r, start)
    maf = raw["raw_MAF"][rows]
    maf_hist = np.r_[np.histogram(maf[np.isfinite(maf)], REG["MAF_bins"])[0], np.isnan(maf).sum()]
    assert int(maf_hist.sum()) == len(rows), "Undefined MAF must be an explicit category"
    return {
        "qid": r["qid"],
        "chr": r["chr"],
        "start_bp": int(start),
        "end_bp": int(end),
        **p,
        "first": int(pos[0]),
        "last": int(pos[-1]),
        "MAF_hist": maf_hist.tolist(),
        "missing_hist": np.histogram(raw["raw_missing_rate"][rows], REG["missing_bins"])[
            0
        ].tolist(),
        "spacing_hist": np.histogram(d, EDGE)[0].tolist(),
    }


def local_gate(a, t):
    if a is None:
        return False
    ratio = np.array(a["spacing_q"][1:]) / np.array(t["spacing_q"][1:])
    if np.any(ratio < 0.5) or np.any(ratio > 2):
        return False
    for p, q in zip(a["LD"], t["LD"]):
        if [x is None for x in p] != [x is None for x in q]:
            return False
        diff = [abs(x - y) for x, y in zip(p, q) if x is not None]
        if not diff or max(diff) > 0.1 or np.mean(diff) > 0.075:
            return False
    return True


def mean_ld(records):
    return np.array(
        [
            [
                np.mean([r["LD"][s][b] for r in records if r["LD"][s][b] is not None])
                for b in range(6)
            ]
            for s in range(2)
        ]
    )


def panel(index, raw, records, exact=True):
    by = {}
    for r in records:
        by.setdefault(r["chr"], []).append(r)
    cross = []
    for c, a in by.items():
        a.sort(key=lambda r: r["start_bp"])
        if any(x["end_bp"] >= y["start_bp"] for x, y in zip(a, a[1:])):
            return None, None
        cross.extend(y["first"] - x["last"] for x, y in zip(a, a[1:]))
    n = sum(r["n"] for r in records)
    h = sum(np.array(r["spacing_hist"]) for r in records) + np.histogram(cross, EDGE)[0]
    result = {
        "n": n,
        "MAF": (sum(np.array(r["MAF_hist"]) for r in records) / n).tolist(),
        "missing": (sum(np.array(r["missing_hist"]) for r in records) / n).tolist(),
        "spacing_hist": (h / h.sum()).tolist(),
        "LD": mean_ld(records).tolist(),
    }
    rows = None
    if exact:
        diffs = []
        rr = []
        for c, a in sorted(by.items()):
            lists = [index.region(c, r["start_bp"], r["end_bp"]) for r in a]
            pos = np.concatenate([p for p, _ in lists])
            ids = np.concatenate([i for _, i in lists])
            assert len(ids) == len(np.unique(ids))
            diffs.append(np.diff(pos))
            rr.append(ids)
        rows = np.sort(np.concatenate(rr))
        assert len(rows) == n
        result["spacing_q"] = np.quantile(np.concatenate(diffs), [0.1, 0.5, 0.9]).tolist()
    return result, rows


def domains(r, allregions):
    width = r["end_bp"] - r["start_bp"] + 1
    maximum = REG["reference_lengths"][str(r["chr"])] - width + 1
    pieces = [(1, maximum)]
    for t in allregions:
        if t["chr"] != r["chr"]:
            continue
        left = max(1, t["start_bp"] - width + 1)
        right = min(maximum, t["end_bp"])
        new = []
        for a, b in pieces:
            if b < left or a > right:
                new.append((a, b))
            else:
                if a < left:
                    new.append((a, left - 1))
                if b > right:
                    new.append((right + 1, b))
        pieces = new
    lengths = np.array([b - a + 1 for a, b in pieces], np.int64)
    cum = lengths.cumsum()
    flat = rng("domain_" + r["qid"]).choice(
        int(cum[-1]),
        min(int(cum[-1]), REG["maximum_uniform_distinct_integer_start_proposals_per_region"]),
        replace=False,
    )
    which = np.searchsorted(cum, flat, side="right")
    before = np.where(which > 0, cum[np.maximum(0, which - 1)], 0)
    return np.array(
        [pieces[k][0] + int(x - pre) for k, x, pre in zip(which, flat, before)], np.int64
    )


def bank(k):
    r = regions()[k]
    folder = OUT / "banks" / r["qid"]
    folder.mkdir(parents=True, exist_ok=True)
    held = (folder / "run.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw, index, reader = data()
    starts = domains(r, regions())
    truth = json.loads((OUT / "template.json").read_text())["regions"][k]
    checkpoint = folder / "checkpoint.json"
    old = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {"next_position": 0, "accepted": []}
    )
    accepted = old["accepted"]
    fail = collections.Counter(old.get("rejections", {}))
    start = int(old["next_position"])
    for i in range(start, len(starts)):
        if stop_requested():
            return
        while not resources_ok():
            if stop_requested():
                return
            time.sleep(20)
        a = record(index, reader, raw, r, int(starts[i]))
        if local_gate(a, truth):
            accepted.append(a)
        else:
            fail["local_LD_or_spacing"] += 1
        if i % 16 == 0 or len(accepted) >= REG["qualified_bank_target"] or i == len(starts) - 1:
            atomic(
                checkpoint,
                {
                    "next_position": i + 1,
                    "accepted": accepted,
                    "rejections": dict(fail),
                    "protocol_sha256": REG["protocol_sha256"],
                    "starts_sha256": hashlib.sha256(starts.astype("<i8").tobytes()).hexdigest(),
                },
            )
        if len(accepted) >= REG["qualified_bank_target"]:
            break
    assert len(accepted) > 0, "No locally qualified positions under frozen calipers"
    atomic(
        folder / "COMPLETE.json",
        {
            "positions_evaluated": i + 1,
            "qualified_positions": len(accepted),
            "checkpoint_sha256": sha(checkpoint),
            "sampling_order_frozen": True,
        },
    )


def joint(index, raw):
    banks = [
        json.loads((OUT / "banks" / r["qid"] / "checkpoint.json").read_text())["accepted"]
        for r in regions()
    ]
    truth = json.loads((OUT / "template.json").read_text())["panel"]
    p = OUT / "joint_checkpoint.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    generator = rng("whole_panel_joint")
    generator.bit_generator.state = old.get("rng_state", generator.bit_generator.state)
    accepted = old.get("accepted", [])
    reject = collections.Counter(old.get("rejections", {}))
    seen_g = {x["geometry_sha256"] for x in accepted}
    seen_s = {x["SNP_rows_sha256"] for x in accepted}
    start = int(old.get("proposals", 0))
    for i in range(start, REG["maximum_whole_panel_proposals"]):
        if stop_requested():
            return
        select = [int(generator.integers(len(b))) for b in banks]
        a = [b[j] for b, j in zip(banks, select)]
        v, _ = panel(index, raw, a, False)
        why = None
        if v is None:
            why = "geometry"
        elif abs(v["n"] / truth["n"] - 1) > 0.1:
            why = "SNP_count"
        elif np.abs(np.array(v["MAF"]) - truth["MAF"]).sum() > 0.1:
            why = "MAF"
        elif np.abs(np.array(v["missing"]) - truth["missing"]).sum() > 0.1:
            why = "missingness"
        elif np.max(np.abs(np.array(v["LD"]) - truth["LD"])) > 0.01:
            why = "LD"
        elif np.abs(np.array(v["spacing_hist"]) - truth["spacing_hist"]).sum() > 0.1:
            why = "spacing_histogram"
        else:
            v, rows = panel(index, raw, a, True)
            if np.any(np.abs(np.array(v["spacing_q"]) / truth["spacing_q"] - 1) > 0.1):
                why = "spacing_quantiles"
            else:
                gh = hashlib.sha256(
                    json.dumps([[r["qid"], r["start_bp"], r["end_bp"]] for r in a]).encode()
                ).hexdigest()
                rh = hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
                if gh in seen_g or rh in seen_s:
                    why = "duplicate"
                else:
                    number = len(accepted) + 1
                    file = OUT / "panels" / f"panel_{number:03d}.json"
                    atomic(
                        file,
                        {
                            "protocol_sha256": REG["protocol_sha256"],
                            "proposal_index": i,
                            "bank_indices": select,
                            "regions": a,
                            "panel": v,
                        },
                    )
                    np.save(
                        OUT / "panels" / f"panel_{number:03d}_SNP_rows.npy",
                        rows,
                        allow_pickle=False,
                    )
                    accepted.append(
                        {
                            "index": number,
                            "proposal_index": i,
                            "geometry_sha256": gh,
                            "SNP_rows_sha256": rh,
                            "panel_json_sha256": sha(file),
                        }
                    )
                    seen_g.add(gh)
                    seen_s.add(rh)
        if why:
            reject[why] += 1
        if i % 1000 == 0 or why is None or i == REG["maximum_whole_panel_proposals"] - 1:
            atomic(
                p,
                {
                    "proposals": i + 1,
                    "rng_state": generator.bit_generator.state,
                    "accepted": accepted,
                    "rejections": dict(reject),
                    "protocol_sha256": REG["protocol_sha256"],
                },
            )
            atomic(
                OUT / "status.json",
                {
                    "state": "JOINT_METADATA_REJECTION_SAMPLING",
                    "accepted": len(accepted),
                    "proposals": i + 1,
                    "epoch": time.time(),
                },
            )
        if len(accepted) == 100:
            break
    atomic(
        OUT / "generation_report.json",
        {
            "state": (
                "GENERATED100_PENDING_INDEPENDENT_ACCEPTANCE"
                if len(accepted) == 100
                else "BUDGET_EXHAUSTED_NO_THRESHOLDS_RELAXED"
            ),
            "panels": len(accepted),
            "proposals": i + 1,
            "rejections": dict(reject),
            "phenotypes_read": False,
            "models_trained": False,
        },
    )


def configure_output(path=None):
    global OUT
    OUT = output_root(path) / "validation_matching_v1"
    OUT.mkdir(parents=True, exist_ok=True)
