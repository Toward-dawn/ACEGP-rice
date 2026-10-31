"""Independent raw-GT, SNP-union and centered-LD acceptance of v2 controls."""

import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
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
SOURCE = ROOT / "reference_metadata"
SOURCE_OUT = None
OUT = None
sys.path.insert(0, str(SOURCE))
import remote_validation7_random_verifier as independent
import remote_validation7_random_metadata as original


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def verify_one(number):
    registration = json.loads((VMETA / "constrained_generation_registration.json").read_text())
    source = OUT / "panels" / f"panel_{number:03d}.json"
    rows_file = OUT / "panels" / f"panel_{number:03d}_SNP_rows.npy"
    receipt = OUT / "verification" / f"panel_{number:03d}.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        assert old["panel_json_sha256"] == sha(source)
        assert old["SNP_rows_file_sha256"] == sha(rows_file)
        return old
    saved = json.loads(source.read_text())
    assert saved["registration_sha256"] == sha(VMETA / "constrained_generation_registration.json")
    assert saved["source_v1_protocol_sha256"] == original.REG["protocol_sha256"]
    assert len(saved["regions"]) == len(saved["bank_indices"]) == 7
    template = json.loads((SOURCE_OUT / "template.json").read_text())
    banks = [
        json.loads((SOURCE_OUT / "banks" / r["qid"] / "checkpoint.json").read_text())["accepted"]
        for r in template["regions"]
    ]
    assert saved["regions"] == [bank[i] for bank, i in zip(banks, saved["bank_indices"])]
    raw, index, reader = original.data()
    records, rows = [], []
    for region in saved["regions"]:
        recalculated, ix = independent.actual_record(raw, index, reader, region)
        independent.profile_parity(recalculated, region)
        records.append(recalculated)
        rows.append(ix)
    value, union = independent.union(raw, rows, records)
    independent.gates(records, value, template)
    assert len(union) == sum(len(item) for item in rows)
    for key in ("n", "MAF", "missing", "spacing_hist", "spacing_q", "LD"):
        np.testing.assert_allclose(value[key], saved["panel"][key], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(union, np.load(rows_file, allow_pickle=False))
    result = {
        "state": "PASS_INDEPENDENT_VALIDATION151_RAW_GT_UNION_CENTERED_LD_GATES",
        "panel": number,
        "panel_json_sha256": sha(source),
        "SNP_rows_file_sha256": sha(rows_file),
        "SNP_rows_sha256": hashlib.sha256(union.astype("<i8").tobytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(
            json.dumps([[r["qid"], r["start_bp"], r["end_bp"]] for r in saved["regions"]]).encode()
        ).hexdigest(),
        "SNP_count": len(union),
        "max_LD_difference": float(np.max(np.abs(np.array(value["LD"]) - template["panel"]["LD"]))),
    }
    atomic_json(receipt, result)
    return result


def main():
    import fcntl

    folder = OUT / "verification"
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = json.loads((OUT / "generation_report.json").read_text())
    assert report["state"] == "GENERATED100_PENDING_INDEPENDENT_ACCEPTANCE"
    assert report["panels"] == 100
    registration = json.loads((VMETA / "constrained_generation_registration.json").read_text())
    assert report["registration_sha256"] == sha(VMETA / "constrained_generation_registration.json")
    assert report["generator_sha256"] == sha(ROOT / "generate_panels.py")
    for field, path in [
        ("v1_registration_sha256", VMETA / "original_matching_registration.json"),
        ("template_sha256", SOURCE_OUT / "template.json"),
        (
            "source_matrix_sha256",
            RANDOM_ROOT
            / "validation/PENDING_common_genotypes/observed_catalog/observed_A1_151.npy",
        ),
        (
            "raw_metadata_sha256",
            RANDOM_ROOT
            / "validation/PENDING_common_genotypes/observed_catalog/raw_metadata_validation151.npz",
        ),
    ]:
        assert sha(path) == registration[field], field
    for qid, expected in registration["bank_checkpoint_sha256"].items():
        assert sha(SOURCE_OUT / "banks" / qid / "checkpoint.json") == expected, qid
    checkpoint = json.loads((OUT / "checkpoint.json").read_text())
    assert len(checkpoint["accepted"]) == 100
    assert checkpoint["attempts"] == report["attempts"] <= registration["maximum_solve_attempts"]
    assert [item["index"] for item in checkpoint["accepted"]] == list(range(1, 101))
    attempts = [item["attempt_index"] for item in checkpoint["accepted"]]
    assert attempts == sorted(set(attempts)) and max(attempts) < checkpoint["attempts"]
    raw, index, reader = original.data()
    template = json.loads((SOURCE_OUT / "template.json").read_text())
    real_records, real_rows = [], []
    for region in template["regions"]:
        recalculated, ix = independent.actual_record(raw, index, reader, region)
        independent.profile_parity(recalculated, region)
        real_records.append(recalculated)
        real_rows.append(ix)
    real_panel, real_union = independent.union(raw, real_rows, real_records)
    for key in ("n", "MAF", "missing", "spacing_hist", "spacing_q", "LD"):
        np.testing.assert_allclose(real_panel[key], template["panel"][key], rtol=1e-12, atol=1e-14)
    assert len(real_union) == 67104
    receipts = []
    with ProcessPoolExecutor(
        max_workers=4, initializer=configure_output, initargs=(OUT.parent,)
    ) as executor:
        for result in executor.map(verify_one, range(1, 101)):
            receipts.append(result)
            atomic_json(
                folder / "status.json",
                {
                    "state": "INDEPENDENT_RAW_GT_AND_LD_VERIFICATION",
                    "complete": len(receipts),
                    "epoch": time.time(),
                },
            )
    assert len({r["geometry_sha256"] for r in receipts}) == 100
    assert len({r["SNP_rows_sha256"] for r in receipts}) == 100
    for saved, verified in zip(checkpoint["accepted"], receipts):
        assert saved["panel_json_sha256"] == verified["panel_json_sha256"]
        assert saved["SNP_rows_file_sha256"] == verified["SNP_rows_file_sha256"]
        assert saved["SNP_rows_sha256"] == verified["SNP_rows_sha256"]
        assert saved["geometry_sha256"] == verified["geometry_sha256"]
    files = list((OUT / "panels").glob("*")) + list(folder.glob("panel_*.json"))
    manifest = {
        "output/"
        + path.relative_to(OUT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in files
        if path.is_file()
    }
    acceptance = {
        "state": "PASS_100_VALIDATION7_CONSTRAINED_PANELS_ALL_FROZEN_METADATA_GATES",
        "panels": 100,
        "regions_per_panel": 7,
        "distinct_geometries": 100,
        "distinct_raw_SNP_unions": 100,
        "observed_GT_raw_SNP_and_centered_LD_independently_recomputed": True,
        "all_v1_matching_thresholds_unchanged": True,
        "source_input_and_panel_SHA256_verified": True,
        "uniform_genomic_sampling_or_exact_permutation_calibration_proved": False,
        "common_imputation_quality_accepted_by_this_step": False,
        "prediction_models_trained_by_this_step": False,
        "registration_sha256": sha(VMETA / "constrained_generation_registration.json"),
        "generator_sha256": sha(ROOT / "generate_panels.py"),
        "files": manifest,
        "epoch": time.time(),
    }
    atomic_json(OUT / "acceptance.json", acceptance)
    atomic_json(
        folder / "status.json",
        {"state": "COMPLETE_INDEPENDENT_ACCEPTANCE", "complete": 100, "epoch": time.time()},
    )


def configure_output(path=None):
    global SOURCE_OUT, OUT
    run_root = output_root(path)
    SOURCE_OUT = run_root / "validation_matching_v1"
    OUT = run_root / "validation_constrained_generation"
    OUT.mkdir(parents=True, exist_ok=True)
    independent.configure_output(run_root)
    original.configure_output(run_root)


if __name__ == "__main__":
    configure_output()
    try:
        main()
    except Exception:
        (OUT / "verification").mkdir(parents=True, exist_ok=True)
        atomic_json(
            OUT / "verification/failure.json",
            {
                "state": "FAILED_NO_THRESHOLDS_RELAXED",
                "traceback": traceback.format_exc(),
                "epoch": time.time(),
            },
        )
        raise
