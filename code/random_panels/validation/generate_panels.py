"""Generate constrained seven-region panels from observed validation genotypes."""

import hashlib
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

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
REG = json.loads((VMETA / "constrained_generation_registration.json").read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def source_check():
    paths = {
        "v1_registration_sha256": VMETA / "original_matching_registration.json",
        "template_sha256": SOURCE_OUT / "template.json",
        "source_matrix_sha256": RANDOM_ROOT
        / "validation/PENDING_common_genotypes/observed_catalog/observed_A1_151.npy",
        "raw_metadata_sha256": RANDOM_ROOT
        / "validation/PENDING_common_genotypes/observed_catalog/raw_metadata_validation151.npz",
    }
    for field, path in paths.items():
        if sha(path) != REG[field]:
            raise RuntimeError(f"Frozen input differs: {field}")
    for qid, expected in REG["bank_checkpoint_sha256"].items():
        if sha(SOURCE_OUT / "banks" / qid / "checkpoint.json") != expected:
            raise RuntimeError(f"Frozen bank differs: {qid}")
    if not REG["no_matching_thresholds_relaxed"] or not REG["no_phenotypes"]:
        raise RuntimeError("Registration scope mismatch")


def source_module():
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location(
        "frozen_validation7_matching", SOURCE / "remote_validation7_random_metadata.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_output(SOURCE_OUT.parent)
    return module


def make_problem(banks, truth, v1, accepted):
    sizes = [len(bank) for bank in banks]
    offsets = np.cumsum([0] + sizes)
    binary = int(offsets[-1])
    count = np.array([entry["n"] for bank in banks for entry in bank], float)
    maf = np.array([entry["MAF_hist"] for bank in banks for entry in bank], float).T
    missing = np.array([entry["missing_hist"] for bank in banks for entry in bank], float).T
    maf_bins = maf.shape[0]
    missing_bins = missing.shape[0]
    variables = binary + maf_bins + missing_bins
    rows, lows, highs = [], [], []

    def add(coefficients, low=-np.inf, high=np.inf):
        row = np.zeros(variables)
        row[: len(coefficients)] = coefficients
        rows.append(row)
        lows.append(low)
        highs.append(high)

    for i in range(len(banks)):
        row = np.zeros(variables)
        row[offsets[i] : offsets[i + 1]] = 1
        rows.append(row)
        lows.append(1)
        highs.append(1)
    add(count, 0.9 * truth["n"], 1.1 * truth["n"])
    for sample in range(2):
        for distance_bin in range(6):
            defined = sum(bank[0]["LD"][sample][distance_bin] is not None for bank in banks)
            if not defined:
                raise RuntimeError("Undefined whole-panel LD bin")
            coefficients = np.array(
                [
                    (entry["LD"][sample][distance_bin] or 0) / defined
                    for bank in banks
                    for entry in bank
                ],
                float,
            )
            reference = truth["LD"][sample][distance_bin]
            add(coefficients, reference - 0.01, reference + 0.01)

    def add_l1(histograms, reference, start_slack, limit):
        for bin_index in range(histograms.shape[0]):
            delta = histograms[bin_index] - reference[bin_index] * count
            for direction in (1, -1):
                row = np.zeros(variables)
                row[:binary] = direction * delta
                row[start_slack + bin_index] = -1
                rows.append(row)
                lows.append(-np.inf)
                highs.append(0)
        row = np.zeros(variables)
        row[:binary] = -limit * count
        row[start_slack : start_slack + histograms.shape[0]] = 1
        rows.append(row)
        lows.append(-np.inf)
        highs.append(0)

    add_l1(maf, truth["MAF"], binary, 0.1)
    add_l1(missing, truth["missing"], binary + maf_bins, 0.1)
    for item in accepted:
        row = np.zeros(variables)
        for i, index in enumerate(item["bank_indices"]):
            row[offsets[i] + index] = 1
        rows.append(row)
        lows.append(-np.inf)
        highs.append(len(banks) - 1)
    lower = np.zeros(variables)
    upper = np.r_[np.ones(binary), np.full(maf_bins + missing_bins, np.inf)]
    integrality = np.r_[np.ones(binary), np.zeros(maf_bins + missing_bins)]
    return offsets, integrality, Bounds(lower, upper), LinearConstraint(np.stack(rows), lows, highs)


def full_check(v1, index, raw, banks, selected, truth):
    chosen = [bank[j] for bank, j in zip(banks, selected)]
    approximate, _ = v1.panel(index, raw, chosen, False)
    if approximate is None:
        return "geometry", None, None
    if abs(approximate["n"] / truth["n"] - 1) > 0.1:
        return "SNP_count", None, None
    if np.abs(np.array(approximate["MAF"]) - truth["MAF"]).sum() > 0.1:
        return "MAF", None, None
    if np.abs(np.array(approximate["missing"]) - truth["missing"]).sum() > 0.1:
        return "missingness", None, None
    if np.max(np.abs(np.array(approximate["LD"]) - truth["LD"])) > 0.01:
        return "LD", None, None
    if np.abs(np.array(approximate["spacing_hist"]) - truth["spacing_hist"]).sum() > 0.1:
        return "spacing_histogram", None, None
    exact, rows = v1.panel(index, raw, chosen, True)
    if np.any(np.abs(np.array(exact["spacing_q"]) / truth["spacing_q"] - 1) > 0.1):
        return "spacing_quantiles", None, None
    return None, exact, rows


def resources_ok():
    memory = dict(
        line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
    )
    anon = int(memory["anon"]) / 2**30
    free = shutil.disk_usage(ROOT).free / 2**30
    return (
        anon < REG["resource_gate"]["new_start_anonymous_memory_below_GiB"]
        and free > REG["resource_gate"]["minimum_free_disk_GiB"]
    )


def main(max_new_attempts=None):
    import fcntl

    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_check()
    v1 = source_module()
    raw, index, _ = v1.data()
    template = json.loads((SOURCE_OUT / "template.json").read_text())
    banks = [
        json.loads((SOURCE_OUT / "banks" / region["qid"] / "checkpoint.json").read_text())[
            "accepted"
        ]
        for region in template["regions"]
    ]
    truth = template["panel"]
    checkpoint = OUT / "checkpoint.json"
    old = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    accepted = old.get("accepted", [])
    rejects = Counter(old.get("rejections", {}))
    attempts = int(old.get("attempts", 0))
    for item in accepted:
        folder = OUT / "panels"
        if (
            sha(folder / f"panel_{item['index']:03d}.json") != item["panel_json_sha256"]
            or sha(folder / f"panel_{item['index']:03d}_SNP_rows.npy")
            != item["SNP_rows_file_sha256"]
        ):
            raise RuntimeError("Accepted checkpoint panel SHA mismatch")
    if old and old.get("registration_sha256") != sha(
        VMETA / "constrained_generation_registration.json"
    ):
        raise RuntimeError("Resume registration SHA mismatch")
    if len(accepted) >= REG["target_distinct_panels"]:
        atomic_json(
            OUT / "status.json",
            {
                "state": "ALREADY_GENERATED_PENDING_INDEPENDENT_ACCEPTANCE",
                "attempts": attempts,
                "accepted": len(accepted),
                "epoch": time.time(),
            },
        )
        return
    if attempts >= REG["maximum_solve_attempts"]:
        atomic_json(
            OUT / "status.json",
            {
                "state": "REGISTERED_ATTEMPT_BUDGET_ALREADY_EXHAUSTED",
                "attempts": attempts,
                "accepted": len(accepted),
                "epoch": time.time(),
            },
        )
        return
    started = time.time()
    limit = (
        REG["maximum_solve_attempts"]
        if max_new_attempts is None
        else min(REG["maximum_solve_attempts"], attempts + max_new_attempts)
    )
    for attempt in range(attempts, limit):
        if v1.stop_requested():
            atomic_json(
                OUT / "status.json",
                {
                    "state": "STOP_REQUESTED",
                    "attempts": attempt,
                    "accepted": len(accepted),
                    "epoch": time.time(),
                },
            )
            return
        while not resources_ok():
            if v1.stop_requested():
                return
            atomic_json(
                OUT / "status.json",
                {
                    "state": "WAITING_RESOURCE_GATE",
                    "attempts": attempt,
                    "accepted": len(accepted),
                    "epoch": time.time(),
                },
            )
            time.sleep(15)
        offsets, integrality, bounds, constraints = make_problem(banks, truth, v1, accepted)
        generator = np.random.default_rng(np.random.SeedSequence([REG["objective_seed"], attempt]))
        objective = np.zeros(len(integrality))
        objective[: offsets[-1]] = generator.standard_normal(offsets[-1])
        solution = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": REG["time_limit_seconds_per_solve"], "presolve": True},
        )
        reason = None
        if solution.x is None:
            reason = "solver_no_incumbent_" + str(solution.status)
        else:
            selected = [
                int(np.argmax(solution.x[offsets[i] : offsets[i + 1]])) for i in range(len(banks))
            ]
            if any(abs(solution.x[offsets[i] + selected[i]] - 1) > 1e-5 for i in range(len(banks))):
                reason = "solver_fractional"
            else:
                reason, exact, rows = full_check(v1, index, raw, banks, selected, truth)
                if reason is None:
                    chosen = [bank[j] for bank, j in zip(banks, selected)]
                    geometry = hashlib.sha256(
                        json.dumps(
                            [[entry["qid"], entry["start_bp"], entry["end_bp"]] for entry in chosen]
                        ).encode()
                    ).hexdigest()
                    row_sha = hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
                    if any(
                        item["geometry_sha256"] == geometry or item["SNP_rows_sha256"] == row_sha
                        for item in accepted
                    ):
                        reason = "duplicate"
                    else:
                        number = len(accepted) + 1
                        folder = OUT / "panels"
                        folder.mkdir(exist_ok=True)
                        panel_path = folder / f"panel_{number:03d}.json"
                        rows_path = folder / f"panel_{number:03d}_SNP_rows.npy"
                        atomic_json(
                            panel_path,
                            {
                                "registration_sha256": sha(
                                    VMETA / "constrained_generation_registration.json"
                                ),
                                "source_v1_protocol_sha256": v1.REG["protocol_sha256"],
                                "attempt_index": attempt,
                                "solver_status": int(solution.status),
                                "bank_indices": selected,
                                "regions": chosen,
                                "panel": exact,
                            },
                        )
                        np.save(rows_path, rows, allow_pickle=False)
                        accepted.append(
                            {
                                "index": number,
                                "attempt_index": attempt,
                                "bank_indices": selected,
                                "geometry_sha256": geometry,
                                "SNP_rows_sha256": row_sha,
                                "panel_json_sha256": sha(panel_path),
                                "SNP_rows_file_sha256": sha(rows_path),
                            }
                        )
        if reason:
            rejects[reason] += 1
        atomic_json(
            checkpoint,
            {
                "registration_sha256": sha(VMETA / "constrained_generation_registration.json"),
                "generator_sha256": sha(ROOT / "generate_panels.py"),
                "attempts": attempt + 1,
                "accepted": accepted,
                "rejections": dict(rejects),
                "epoch": time.time(),
            },
        )
        atomic_json(
            OUT / "status.json",
            {
                "state": "SEARCHING_FROZEN_METADATA_FULL_GATES",
                "attempts": attempt + 1,
                "accepted": len(accepted),
                "rejections": dict(rejects),
                "epoch": time.time(),
            },
        )
        if len(accepted) >= REG["target_distinct_panels"]:
            break
    if len(accepted) < REG["target_distinct_panels"] and limit < REG["maximum_solve_attempts"]:
        atomic_json(
            OUT / "status.json",
            {
                "state": "PAUSED_AFTER_PILOT",
                "attempts": attempt + 1,
                "accepted": len(accepted),
                "rejections": dict(rejects),
                "epoch": time.time(),
            },
        )
        return
    atomic_json(
        OUT / "generation_report.json",
        {
            "state": (
                "GENERATED100_PENDING_INDEPENDENT_ACCEPTANCE"
                if len(accepted) == 100
                else "BUDGET_EXHAUSTED_NO_THRESHOLDS_RELAXED"
            ),
            "attempts": attempt + 1,
            "panels": len(accepted),
            "rejections": dict(rejects),
            "elapsed_seconds_this_run": time.time() - started,
            "phenotypes_read": False,
            "models_trained": False,
            "registration_sha256": sha(VMETA / "constrained_generation_registration.json"),
            "generator_sha256": sha(ROOT / "generate_panels.py"),
        },
    )


def configure_output(path=None):
    global SOURCE_OUT, OUT
    run_root = output_root(path)
    SOURCE_OUT = run_root / "validation_matching_v1"
    OUT = run_root / "validation_constrained_generation"
    OUT.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    configure_output()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--max-new-attempts", type=int)
        args = parser.parse_args()
        if args.max_new_attempts is not None and args.max_new_attempts <= 0:
            raise ValueError("Pilot attempt count must be positive")
        main(args.max_new_attempts)
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(
            OUT / "failure.json",
            {
                "state": "FAILED_NO_THRESHOLDS_RELAXED",
                "traceback": traceback.format_exc(),
                "epoch": time.time(),
            },
        )
        raise
