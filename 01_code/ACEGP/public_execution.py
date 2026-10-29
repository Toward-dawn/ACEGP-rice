"""Input checks, run records and task bookkeeping for the public interface."""

import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def check_inputs(package, analysis):
    """Check every packaged runtime file against the release manifest."""
    dependencies = read_json(package / "04_documentation/DEPENDENCIES.json")
    required = dependencies["required_existing_files"]
    with (package / "FILE_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["Path"]: row for row in csv.DictReader(stream)}
    missing, changed = [], []
    for relative in required:
        path = package / relative
        if not path.is_file():
            missing.append(relative)
        elif relative not in manifest or sha256(path) != manifest[relative]["SHA256"]:
            changed.append(relative)
    if not (missing or changed):
        from validation_release import verify_validation_release

        try:
            verify_validation_release(package, include_raw=analysis == "validation-random")
        except ValueError as error:
            changed.append(str(error))
    versions = {}
    errors = []
    requirements = package / "01_code/environment/prediction_requirements.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        versions[name] = {"required": expected, "installed": installed}
        if installed != expected:
            errors.append(f"{name}: requires {expected}, found {installed or 'not installed'}")
    if os.name != "posix":
        errors.append("The public task runner requires Linux/POSIX file locking (fcntl).")
    return {
        "required_files_checked": len(required),
        "missing_inputs": missing,
        "changed_inputs": changed,
        "environment_errors": errors,
        "package_versions": versions,
        "python_version": sys.version.split()[0],
        "ready": not (missing or changed or errors),
        "training_started": False,
    }


def run_identity(package, args, population):
    return {
        "schema": 1,
        "analysis": args.analysis,
        "population": population,
        "quality_subset": args.subset if args.analysis == "quality" else None,
        "release_manifest_sha256": sha256(package / "FILE_MANIFEST.csv"),
    }


def prepare_run(path, package, identity, resume=False, summarize=False):
    if path is None:
        raise ValueError("--output is required.")
    root = path.resolve()
    if root == package or root.is_relative_to(package):
        raise ValueError("Output must be outside the public package.")
    if root.exists() and not root.is_dir():
        raise ValueError("Output must be a directory.")
    record = root / "RUN_MANIFEST.json"
    if resume or summarize:
        if not record.is_file() or read_json(record) != identity:
            raise ValueError("The run record does not match this analysis, population and release.")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Use a new directory, or --resume for a matching recorded run.")
        root.mkdir(parents=True, exist_ok=True)
        write_json(record, identity)
    return root


@contextlib.contextmanager
def run_lock(root):
    import fcntl

    with (root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def tasks_for(args):
    all_folds = args.all_folds or args.summarize
    if args.analysis == "statistics":
        return [{"key": "statistics", "folder": "statistics"}]
    if args.analysis == "genetic-group":
        groups = range(1, 6) if all_folds else [args.group]
        return [{"key": f"group{k}", "group": k} for k in groups]
    if args.analysis.endswith("-random"):
        panels = (
            range(0 if args.analysis == "validation-random" else 1, 101)
            if args.all_panels
            else [args.panel]
        )
        folds = range(1, 6) if all_folds else [args.fold]
        return [
            {"key": f"panel{panel:03d}_r01_f{fold}", "panel": panel, "repeat": 1, "fold": fold}
            for panel in panels
            for fold in folds
        ]
    repeats = range(1, 21) if all_folds else [args.repeat]
    folds = range(1, 6) if all_folds else [args.fold]
    return [
        {"key": f"r{repeat:02d}_f{fold}", "repeat": repeat, "fold": fold}
        for repeat in repeats
        for fold in folds
    ]


def task_folder(root, analysis, cohort, subset, task):
    if analysis == "statistics":
        return root / "statistics"
    if analysis == "genetic-group":
        return root / "auxiliary" / cohort / f"group{task['group']}"
    name = f"r{task['repeat']:02d}_f{task['fold']}"
    if analysis == "prediction":
        return root / "repeated_cv" / cohort / name
    if analysis == "controls":
        return root / "controls" / cohort / name
    if analysis == "ablation":
        return root / "region_ablation" / cohort / name
    if analysis == "quality":
        return root / "quality_sensitivity" / f"subset{subset}" / name
    if analysis == "primary-random":
        return root / "random_fixed8_models" / f"panel_{task['panel']:03d}" / name
    panel = "real7" if task["panel"] == 0 else f"random_{task['panel']:03d}"
    return root / "common_context_models" / panel / name


def receipt_path(root, task):
    return root / ".task_receipts" / (task["key"] + ".json")


def verify_task(root, folder, task):
    receipt = receipt_path(root, task)
    if not receipt.is_file():
        return False
    saved = read_json(receipt)
    if saved["task"] != task or not saved["files"]:
        raise ValueError(f"Invalid task receipt: {task['key']}")
    for relative, expected in saved["files"].items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(folder.resolve())
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError(f"Changed or missing completed output: {relative}")
    return True


def execute_tasks(root, args, cohort, tasks, execute, validate):
    """Run pending tasks; retain and verify completed task outputs."""
    completed = skipped = 0
    for task in tasks:
        folder = task_folder(root, args.analysis, cohort, args.subset, task)
        if verify_task(root, folder, task):
            skipped += 1
            continue
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Unrecorded partial task at {folder}; use a new run directory.")
        execute(task, folder)
        validate(task, folder)
        record_task(root, folder, task)
        completed += 1
        print(json.dumps({"task": task["key"], "status": "complete"}), flush=True)
    return {"completed": completed, "verified_and_skipped": skipped}


def record_task(root, folder, task):
    files = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.suffix in (".csv", ".json")
        and "cache" not in path.relative_to(folder).parts
    }
    if not files:
        raise ValueError(f"Task produced no recorded outputs: {task['key']}")
    write_json(receipt_path(root, task), {"task": task, "files": files})


def require_complete(root, args, cohort, tasks):
    missing = [
        task["key"]
        for task in tasks
        if not verify_task(root, task_folder(root, args.analysis, cohort, args.subset, task), task)
    ]
    if missing:
        raise ValueError(
            f"Summary requires {len(tasks)} completed tasks; missing: " + ", ".join(missing)
        )
