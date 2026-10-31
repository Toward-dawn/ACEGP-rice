"""Batch only the original early rejection gates; retain frozen final calculations."""

from pathlib import Path
import collections
import hashlib
import json
import time
import copy
import numpy as np
import remote_fixed8_random_metadata as m

REASONS = {1: "geometry", 2: "SNP_count", 3: "MAF", 4: "missingness"}


def tables(banks):
    assert len(banks) == 8 and all(len(b) == 128 for b in banks)
    return {
        "n": [np.array([r["n"] for r in b], np.int64) for b in banks],
        "MAF": [np.array([r["MAF_hist"] for r in b], np.int64) for b in banks],
        "missing": [np.array([r["missing_hist"] for r in b], np.int64) for b in banks],
        "start": [np.array([r["start_bp"] for r in b], np.int64) for b in banks],
        "end": [np.array([r["end_bp"] for r in b], np.int64) for b in banks],
        "chrom": [b[0]["chr"] for b in banks],
    }


def early(v, t, truth):
    n = sum(t["n"][k][v[:, k]] for k in range(8))
    why = np.zeros(len(v), np.int8)
    overlap = np.zeros(len(v), bool)
    for i in range(8):
        for j in range(i + 1, 8):
            if t["chrom"][i] == t["chrom"][j]:
                overlap |= (t["start"][i][v[:, i]] <= t["end"][j][v[:, j]]) & (
                    t["start"][j][v[:, j]] <= t["end"][i][v[:, i]]
                )
    why[overlap] = 1
    why[(why == 0) & (np.abs(n / truth["n"] - 1) > 0.1)] = 2
    maf = sum(t["MAF"][k][v[:, k]] for k in range(8)) / n[:, None]
    why[(why == 0) & (np.abs(maf - np.array(truth["MAF"])).sum(axis=1) > 0.1)] = 3
    miss = sum(t["missing"][k][v[:, k]] for k in range(8)) / n[:, None]
    why[(why == 0) & (np.abs(miss - np.array(truth["missing"])).sum(axis=1) > 0.1)] = 4
    return why


def scalar_early(a, index, raw, truth):
    v, _ = m.panel(index, raw, a, False)
    if v is None:
        return 1
    if abs(v["n"] / truth["n"] - 1) > 0.1:
        return 2
    if np.abs(np.array(v["MAF"]) - truth["MAF"]).sum() > 0.1:
        return 3
    if np.abs(np.array(v["missing"]) - truth["missing"]).sum() > 0.1:
        return 4
    return 0


def final(a, index, raw, truth):
    v, _ = m.panel(index, raw, a, False)
    assert v is not None and abs(v["n"] / truth["n"] - 1) <= 0.1
    assert np.abs(np.array(v["MAF"]) - truth["MAF"]).sum() <= 0.1
    assert np.abs(np.array(v["missing"]) - truth["missing"]).sum() <= 0.1
    if np.max(np.abs(np.array(v["LD"]) - truth["LD"])) > 0.01:
        return "LD", None, None
    if np.abs(np.array(v["spacing_hist"]) - truth["spacing_hist"]).sum() > 0.1:
        return "spacing_histogram", None, None
    v, rows = m.panel(index, raw, a, True)
    if np.any(np.abs(np.array(v["spacing_q"]) / truth["spacing_q"] - 1) > 0.1):
        return "spacing_quantiles", None, None
    return None, v, rows


def joint(index, raw):
    OUT = m.OUT
    REG = m.REG
    accept = m.RANDOM_ROOT / "primary/metadata/FAST_PREFILTER_ACCEPTANCE.json"
    assert (
        accept.exists()
        and json.loads(accept.read_text())["state"] == "PASS_EXACT_RNG_AND_SCALAR_EARLY_GATE_PARITY"
    )
    banks = [
        json.loads((OUT / "banks" / r["qid"] / "checkpoint.json").read_text())["accepted"]
        for r in m.regions()
    ]
    truth = json.loads((OUT / "template.json").read_text())["panel"]
    t = tables(banks)
    p = OUT / "joint_checkpoint.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    g = m.rng("whole_panel_joint")
    g.bit_generator.state = old.get("rng_state", g.bit_generator.state)
    accepted = old.get("accepted", [])
    reject = collections.Counter(old.get("rejections", {}))
    seen_g = {x["geometry_sha256"] for x in accepted}
    seen_s = {x["SNP_rows_sha256"] for x in accepted}
    start = int(old.get("proposals", 0))
    limit = REG["maximum_whole_panel_proposals"]
    ext = m.RANDOM_ROOT / "primary/metadata/fixed8_metadata_compute_extension.json"
    if ext.exists():
        extension = json.loads(ext.read_text())
        assert (
            m.sha(m.RANDOM_ROOT / "primary/metadata/fixed8_random_metadata_registration.json")
            == extension["scientific_registration_file_sha256"]
        )
        limit = extension["effective_whole_panel_proposals"]
    if len(accepted) == 100:
        return
    initial_accepted = copy.deepcopy(accepted)
    initial_prefix_hash = hashlib.sha256(
        json.dumps(initial_accepted, sort_keys=True).encode()
    ).hexdigest()
    while start < limit and len(accepted) < 100:
        if (m.ROOT / "STOP_REQUESTED").exists():
            return
        before = copy.deepcopy(g.bit_generator.state)
        count = min(8192, limit - start)
        vectors = g.integers(128, size=(count, 8))
        why = early(vectors, t, truth)
        used = count
        for offset, code in enumerate(why):
            i = start + offset
            if code:
                reject[REASONS[int(code)]] += 1
                continue
            chosen = [b[int(k)] for b, k in zip(banks, vectors[offset])]
            reason, v, rows = final(chosen, index, raw, truth)
            if reason:
                reject[reason] += 1
                continue
            gh = hashlib.sha256(
                json.dumps([[r["qid"], r["start_bp"], r["end_bp"]] for r in chosen]).encode()
            ).hexdigest()
            rh = hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
            if gh in seen_g or rh in seen_s:
                reject["duplicate"] += 1
                continue
            number = len(accepted) + 1
            file = OUT / "panels" / f"panel_{number:03d}.json"
            m.atomic(
                file,
                {
                    "protocol_sha256": REG["protocol_sha256"],
                    "proposal_index": i,
                    "bank_indices": vectors[offset].tolist(),
                    "regions": chosen,
                    "panel": v,
                },
            )
            np.save(OUT / "panels" / f"panel_{number:03d}_SNP_rows.npy", rows, allow_pickle=False)
            accepted.append(
                {
                    "index": number,
                    "proposal_index": i,
                    "geometry_sha256": gh,
                    "SNP_rows_sha256": rh,
                    "panel_json_sha256": m.sha(file),
                }
            )
            seen_g.add(gh)
            seen_s.add(rh)
            if len(accepted) == 100:
                used = offset + 1
                g.bit_generator.state = before
                g.integers(128, size=(used, 8))
                break
        start += used
        m.atomic(
            p,
            {
                "proposals": start,
                "rng_state": g.bit_generator.state,
                "accepted": accepted,
                "rejections": dict(reject),
                "protocol_sha256": REG["protocol_sha256"],
            },
        )
        m.atomic(
            OUT / "status.json",
            {
                "state": "JOINT_METADATA_REJECTION_SAMPLING_BATCHED_IDENTICAL_EARLY_GATES",
                "accepted": len(accepted),
                "proposals": start,
                "epoch": time.time(),
            },
        )
    assert accepted[: len(initial_accepted)] == initial_accepted
    if ext.exists():
        m.atomic(
            OUT / "execution_extension_final.json",
            {
                "state": "PASS_INITIAL_PANELS_UNCHANGED_AFTER_RANDOM_STREAM_CONTINUATION",
                "initial_accepted_count": len(initial_accepted),
                "initial_prefix_sha256": initial_prefix_hash,
                "final_proposals": start,
                "scientific_gates_or_rng_changed": False,
            },
        )
    m.atomic(
        OUT / "generation_report.json",
        {
            "state": (
                "GENERATED100_PENDING_INDEPENDENT_ACCEPTANCE"
                if len(accepted) == 100
                else "BUDGET_EXHAUSTED_NO_THRESHOLDS_RELAXED"
            ),
            "panels": len(accepted),
            "proposals": start,
            "rejections": dict(reject),
            "phenotypes_read": False,
            "models_trained": False,
            "speed_implementation_only": True,
        },
    )
