"""Independent raw-marker/centered-correlation validation of eight-region panels."""

from pathlib import Path
import os
import sys
import json
import time
import fcntl
import traceback
import hashlib
import collections
from concurrent.futures import ProcessPoolExecutor

for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
import numpy as np
import remote_fixed8_random_metadata as m

ROOT = None
OUT = None


def actual_record(raw, index, reader, r):
    pos, rows = index.region(r["chr"], r["start_bp"], r["end_bp"])
    assert len(rows) > 1
    means = []
    hashes = []
    for ns in m.REG["LD_namespaces"]:
        generator = np.random.default_rng(m.ld.stable_seed(ns + r["start_bp"], r["qid"]))
        pairs = []
        traces = []
        needed = []
        for low, high in m.ld.BINS:
            lo = np.maximum(np.arange(len(pos)) + 1, np.searchsorted(pos, pos + low, side="left"))
            hi = np.searchsorted(pos, pos + high, side="left")
            counts = np.maximum(hi - lo, 0)
            cumulative = counts.cumsum()
            number = int(cumulative[-1])
            flat = (
                generator.choice(number, size=min(500, number), replace=False)
                if number
                else np.empty(0, int)
            )
            a = np.searchsorted(cumulative, flat, side="right")
            before = np.where(a > 0, cumulative[np.maximum(a - 1, 0)], 0)
            b = lo[a] + flat - before
            pairs.append((a, b))
            needed.extend([a, b])
            traces.append(np.column_stack((rows[a], rows[b])).astype("<u4").tobytes())
        ix = np.unique(np.concatenate(needed))
        g = reader(rows[ix]).astype(float)
        values = []
        for a, b in pairs:
            x = g[:, np.searchsorted(ix, a)].T
            y = g[:, np.searchsorted(ix, b)].T
            valid = (x >= 0) & (y >= 0) & np.isfinite(x) & np.isfinite(y)
            n = valid.sum(axis=1)
            x = np.where(valid, x, 0)
            y = np.where(valid, y, 0)
            mx = np.divide(x.sum(axis=1), n, out=np.zeros(len(n)), where=n > 0)
            my = np.divide(y.sum(axis=1), n, out=np.zeros(len(n)), where=n > 0)
            dx = np.where(valid, x - mx[:, None], 0)
            dy = np.where(valid, y - my[:, None], 0)
            vx = (dx * dx).sum(axis=1)
            vy = (dy * dy).sum(axis=1)
            ok = (n >= 3) & (vx > 0) & (vy > 0)
            r2 = np.clip((dx[ok] * dy[ok]).sum(axis=1) ** 2 / (vx[ok] * vy[ok]), 0, 1)
            values.append(float(r2.mean()) if len(r2) else None)
        means.append(values)
        hashes.append(hashlib.sha256(b"".join(traces)).hexdigest())
    d = np.diff(pos)
    return {
        "qid": r["qid"],
        "chr": r["chr"],
        "start_bp": r["start_bp"],
        "end_bp": r["end_bp"],
        "n": len(rows),
        "LD": means,
        "pair_hashes": hashes,
        "spacing_q": np.quantile(d, [0.1, 0.5, 0.9]).tolist(),
        "first": int(pos[0]),
        "last": int(pos[-1]),
        "MAF_hist": np.histogram(raw["raw_MAF"][rows], m.REG["MAF_bins"])[0].tolist(),
        "missing_hist": np.histogram(raw["raw_missing_rate"][rows], m.REG["missing_bins"])[
            0
        ].tolist(),
        "spacing_hist": np.histogram(d, m.EDGE)[0].tolist(),
    }, rows


def profile_parity(a, b):
    for key in ["n", "pair_hashes", "first", "last", "MAF_hist", "missing_hist", "spacing_hist"]:
        assert a[key] == b[key], key
    np.testing.assert_allclose(a["spacing_q"], b["spacing_q"], rtol=0, atol=0)
    for x, y in zip(a["LD"], b["LD"]):
        assert [v is None for v in x] == [v is None for v in y]
        np.testing.assert_allclose(
            [z for z in x if z is not None], [z for z in y if z is not None], rtol=1e-12, atol=1e-14
        )


def union(raw, rows, records):
    ix = np.unique(np.concatenate(rows))
    differences = []
    for c in np.unique(raw["chromosome"][ix]):
        p = np.sort(raw["position_bp"][ix[raw["chromosome"][ix] == c]])
        differences.append(np.diff(p))
    d = np.concatenate(differences)
    h = np.histogram(d, m.EDGE)[0]
    n = len(ix)
    return {
        "n": n,
        "MAF": (np.histogram(raw["raw_MAF"][ix], m.REG["MAF_bins"])[0] / n).tolist(),
        "missing": (
            np.histogram(raw["raw_missing_rate"][ix], m.REG["missing_bins"])[0] / n
        ).tolist(),
        "spacing_hist": (h / h.sum()).tolist(),
        "spacing_q": np.quantile(d, [0.1, 0.5, 0.9]).tolist(),
        "LD": m.mean_ld(records).tolist(),
    }, ix


def gates(records, value, truth):
    for a, t in zip(records, truth["regions"]):
        assert (
            a["qid"] == t["qid"]
            and a["chr"] == t["chr"]
            and a["end_bp"] - a["start_bp"] == t["end_bp"] - t["start_bp"]
        )
        assert m.local_gate(a, t)
        assert 1 <= a["start_bp"] <= a["end_bp"] <= m.REG["reference_lengths"][str(a["chr"])]
        for real in m.regions():
            if a["chr"] == real["chr"]:
                assert a["end_bp"] < real["start_bp"] or a["start_bp"] > real["end_bp"]
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            if a["chr"] == b["chr"]:
                assert a["end_bp"] < b["start_bp"] or a["start_bp"] > b["end_bp"]
    t = truth["panel"]
    assert abs(value["n"] / t["n"] - 1) <= 0.1
    for key in ["MAF", "missing", "spacing_hist"]:
        assert np.abs(np.array(value[key]) - t[key]).sum() <= 0.1
    assert np.max(np.abs(np.array(value["LD"]) - t["LD"])) <= 0.01
    assert np.all(np.abs(np.array(value["spacing_q"]) / t["spacing_q"] - 1) <= 0.1)


def verify_one(number):
    dest = OUT / "verification" / f"panel_{number:03d}.json"
    source = OUT / "panels" / f"panel_{number:03d}.json"
    if dest.exists():
        old = json.loads(dest.read_text())
        assert old["source_sha256"] == m.sha(source)
        return old
    raw, index, reader = m.data()
    a = json.loads(source.read_text())
    assert a["protocol_sha256"] == m.REG["protocol_sha256"]
    truth = json.loads((OUT / "verification/template.json").read_text())
    records = []
    rows = []
    for r in a["regions"]:
        current, ix = actual_record(raw, index, reader, r)
        profile_parity(current, r)
        records.append(current)
        rows.append(ix)
    value, ix = union(raw, rows, records)
    gates(records, value, truth)
    assert len(ix) == sum(len(x) for x in rows)
    for key in ["n", "MAF", "missing", "spacing_hist", "spacing_q", "LD"]:
        np.testing.assert_allclose(value[key], a["panel"][key], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(
        ix, np.load(OUT / "panels" / f"panel_{number:03d}_SNP_rows.npy", allow_pickle=False)
    )
    result = {
        "state": "PASS_INDEPENDENT_RAW_MARKER_AND_CENTERED_LD_GATES",
        "panel": number,
        "source_sha256": m.sha(source),
        "SNP_rows_file_sha256": m.sha(OUT / "panels" / f"panel_{number:03d}_SNP_rows.npy"),
        "SNP_rows_sha256": hashlib.sha256(ix.astype("<i8").tobytes()).hexdigest(),
        "n": len(ix),
        "max_LD_difference": float(np.max(np.abs(np.array(value["LD"]) - truth["panel"]["LD"]))),
    }
    m.atomic(dest, result)
    return result


def main():
    folder = OUT / "verification"
    folder.mkdir(exist_ok=True)
    held = (folder / "controller.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (OUT / "generation_report.json").exists():
        if (ROOT / "STOP_REQUESTED").exists():
            return
        if (OUT / "failure.json").exists():
            m.atomic(folder / "status.json", {"state": "PARKED_GENERATOR_FAILURE"})
            return
        m.atomic(
            folder / "status.json", {"state": "WAITING_FOR_GENERATED100", "epoch": time.time()}
        )
        time.sleep(30)
    report = json.loads((OUT / "generation_report.json").read_text())
    if report["panels"] != 100:
        m.atomic(
            folder / "status.json",
            {"state": "PARKED_FROZEN_GENERATION_BUDGET_EXHAUSTED", "panels": report["panels"]},
        )
        return
    if report["proposals"] > m.REG["maximum_whole_panel_proposals"]:
        extension = json.loads(
            (m.RANDOM_ROOT / "primary/metadata/fixed8_metadata_compute_extension.json").read_text()
        )
        assert (
            m.sha(m.RANDOM_ROOT / "primary/metadata/fixed8_random_metadata_registration.json")
            == extension["scientific_registration_file_sha256"]
        )
        assert report["proposals"] <= extension["effective_whole_panel_proposals"]
        assert (
            json.loads((OUT / "execution_extension_final.json").read_text())["state"]
            == "PASS_INITIAL_PANELS_UNCHANGED_AFTER_RANDOM_STREAM_CONTINUATION"
        )
    raw, index, reader = m.data()
    assert (
        m.sha(m.RANDOM_ROOT / "primary/genotype_source/PMB2026_full_241.bed")
        == m.REG["source_BED_sha256"]
    )
    old = json.loads((OUT / "template.json").read_text())
    records = []
    rows = []
    for r in old["regions"]:
        v, ix = actual_record(raw, index, reader, r)
        profile_parity(v, r)
        records.append(v)
        rows.append(ix)
    value, ix = union(raw, rows, records)
    assert value["n"] == 67985
    m.atomic(folder / "template.json", {"regions": records, "panel": value})
    with ProcessPoolExecutor(
        max_workers=4, initializer=configure_output, initargs=(OUT.parent,)
    ) as ex:
        receipts = []
        for record in ex.map(verify_one, range(1, 101)):
            receipts.append(record)
            m.atomic(
                folder / "status.json",
                {
                    "state": "INDEPENDENT_VERIFYING_100",
                    "complete": len(receipts),
                    "epoch": time.time(),
                },
            )
    assert len({r["SNP_rows_sha256"] for r in receipts}) == 100
    checkpoint = json.loads((OUT / "joint_checkpoint.json").read_text())
    assert len(checkpoint["accepted"]) == 100
    banks = [
        json.loads((OUT / "banks" / r["qid"] / "checkpoint.json").read_text())["accepted"]
        for r in m.regions()
    ]
    generator = m.rng("whole_panel_joint")
    byproposal = {x["proposal_index"]: x for x in checkpoint["accepted"]}
    seen = set()
    for i in range(checkpoint["proposals"]):
        select = [int(generator.integers(len(b))) for b in banks]
        if i in byproposal:
            hit = byproposal[i]
            saved = json.loads((OUT / "panels" / f"panel_{hit['index']:03d}.json").read_text())
            assert select == saved["bank_indices"]
            chosen = [b[j] for b, j in zip(banks, select)]
            assert chosen == saved["regions"]
            gh = hashlib.sha256(
                json.dumps([[r["qid"], r["start_bp"], r["end_bp"]] for r in chosen]).encode()
            ).hexdigest()
            assert gh == hit["geometry_sha256"] and gh not in seen
            seen.add(gh)
    assert generator.bit_generator.state == checkpoint["rng_state"]
    files = list((OUT / "panels").glob("*")) + list(folder.glob("panel_*.json"))
    manifest = {
        str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": m.sha(p)}
        for p in files
        if p.is_file()
    }
    m.atomic(
        OUT / "acceptance.json",
        {
            "state": "PASS_100_FIXED8_PANELS_ALL_FROZEN_METADATA_GATES",
            "protocol_sha256": m.REG["protocol_sha256"],
            "computational_extension_registration_sha256": (
                m.sha(m.RANDOM_ROOT / "primary/metadata/fixed8_metadata_compute_extension.json")
                if (
                    m.RANDOM_ROOT / "primary/metadata/fixed8_metadata_compute_extension.json"
                ).exists()
                else None
            ),
            "cohort": "development241",
            "regions_per_panel": 8,
            "panels": 100,
            "distinct_geometries": 100,
            "distinct_SNP_unions": 100,
            "independent_centered_LD_and_raw_union_recalculation": True,
            "accepted_coordinate_rng_trace_and_final_state_replayed": True,
            "exact_exchangeability_or_selection_adjusted_P_proved": False,
            "validation_panels_accepted": False,
            "predictive_models_trained": False,
            "files": manifest,
            "epoch": time.time(),
        },
    )
    m.atomic(
        folder / "status.json", {"state": "COMPLETE_INDEPENDENT_ACCEPTANCE", "epoch": time.time()}
    )


def configure_output(path=None):
    global ROOT, OUT
    m.configure_output(path)
    ROOT, OUT = m.ROOT, m.OUT


if __name__ == "__main__":
    configure_output()
    try:
        main()
    except Exception as e:
        m.atomic(
            OUT / "verification/failure.json",
            {"error": str(e), "traceback": traceback.format_exc()},
        )
        raise
