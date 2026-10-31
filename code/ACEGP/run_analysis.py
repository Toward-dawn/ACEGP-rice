"""Execute saved folds and summarize complete, verified runs."""

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import public_execution as execution

ANALYSES = (
    "prediction",
    "controls",
    "ablation",
    "quality",
    "genetic-group",
    "primary-random",
    "validation-random",
    "statistics",
)
FIXED_POPULATIONS = {
    "quality": "validation",
    "primary-random": "primary",
    "validation-random": "validation",
}
FOLD_WORKERS = {
    "controls": "remote_controls",
    "ablation": "remote_region_ablation",
    "quality": "remote_quality_sensitivity",
    "genetic-group": "remote_auxiliary",
}
RANDOM_WORKERS = {
    "primary-random": "primary/remote_fixed8_random_models.py",
    "validation-random": "validation/remote_common_context_models.py",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, choices=ANALYSES)
    parser.add_argument("--population", choices=["primary", "validation"])
    parser.add_argument("--repeat", type=int, choices=range(1, 21), default=1)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--subset", type=int, choices=[146, 142], default=146)
    parser.add_argument("--group", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--panel", type=int, choices=range(0, 101), default=0)
    parser.add_argument(
        "--all-folds", action="store_true", help="Run all registered folds or genetic groups."
    )
    parser.add_argument(
        "--all-panels",
        action="store_true",
        help="Use all 100 random panels; validation also includes real panel 0.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a matching recorded run; verify completed tasks before skipping.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize complete outputs without fitting models.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check packaged files, checksums and environment without creating output.",
    )
    return parser


def resolve_population(analysis, requested):
    """Resolve the population before selecting a worker."""
    if analysis == "statistics":
        if requested is not None:
            raise ValueError("Statistics cover both populations; omit --population.")
        return "both"
    if analysis in FIXED_POPULATIONS:
        population = FIXED_POPULATIONS[analysis]
        if requested is not None and requested != population:
            raise ValueError(
                f"{analysis} requires --population {population}; "
                "omit the option to use this population."
            )
        return population
    return requested or "primary"


def validate_task(args):
    population = resolve_population(args.analysis, args.population)
    if (
        args.analysis == "primary-random"
        and args.panel == 0
        and not (args.all_panels or args.check)
    ):
        raise ValueError("Primary random controls require panel 1-100 or --all-panels.")
    if args.analysis in RANDOM_WORKERS and args.repeat != 1:
        raise ValueError("Random controls use only prespecified repeat 1.")
    if args.all_panels and args.analysis not in RANDOM_WORKERS:
        raise ValueError("--all-panels applies only to random controls.")
    if args.analysis == "statistics" and (args.summarize or args.all_folds):
        raise ValueError(
            "Statistics already summarize the archived predictions; omit --summarize and --all-folds."
        )
    return population


def load_worker(analysis, package, output):
    if analysis in FOLD_WORKERS:
        worker = importlib.import_module(FOLD_WORKERS[analysis])
    elif analysis in RANDOM_WORKERS:
        path = package / "code/random_panels" / RANDOM_WORKERS[analysis]
        spec = importlib.util.spec_from_file_location("public_random_worker", path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
    else:
        return None
    worker.configure_output(output)
    return worker


def dispatch_task(args, population, package, output, worker, task, folder):
    cohort = "development" if population == "primary" else "validation"
    if args.analysis == "prediction":
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(package / "code/ACEGP/run_cv_fold.py"),
                "--population",
                population,
                "--repeat",
                str(task["repeat"]),
                "--fold",
                str(task["fold"]),
                "--output",
                str(folder),
            ],
            check=True,
        )
    elif args.analysis == "controls":
        worker.task(cohort, task["repeat"], task["fold"])
    elif args.analysis == "ablation":
        worker.work(cohort, task["repeat"], task["fold"])
    elif args.analysis == "quality":
        worker.fold(args.subset, task["repeat"], task["fold"])
    elif args.analysis == "genetic-group":
        worker.group_task(cohort, task["group"])
    elif args.analysis == "primary-random":
        worker.fold(task["panel"], task["fold"])
    elif args.analysis == "validation-random":
        worker.fold("real" if task["panel"] == 0 else "random", task["panel"], task["fold"])
    else:
        run_statistical_comparisons(package, output)


def summarize(args, population, package, output, worker, tasks):
    from public_results import validate_predictions, summarize_predictions

    cohort = "development" if population == "primary" else "validation"
    execution.require_complete(output, args, cohort, tasks)
    for task in tasks:
        folder = execution.task_folder(output, args.analysis, cohort, args.subset, task)
        validate_predictions(package, args, cohort, task, folder)
    if args.analysis in ("controls", "ablation"):
        worker.summarize(cohort)
    elif args.analysis == "quality":
        worker.summarize(args.subset)
    elif args.analysis == "genetic-group":
        worker.group_summary(cohort)
    else:
        summarize_predictions(package, output, args, cohort, tasks)


def run_statistical_comparisons(package, output):
    """Report the two t-test families and the separate two-population sign tests."""
    import remote_statistics as primary_statistics
    import remote_secondary_statistics as secondary_statistics
    import reporting_wrappers

    primary_statistics.configure_output(output)
    secondary_statistics.configure_output(output)

    cohorts = ("development", "validation")
    reports = {cohort: primary_statistics.analyze(cohort) for cohort in cohorts}
    completed = {}
    for cohort in cohorts:
        completed.update(secondary_statistics.models(cohort))
    completed.update(secondary_statistics.dense())

    registration_file = package / "02_data/prediction_inputs/statistical_families.json"
    registration = json.loads(registration_file.read_text(encoding="utf-8"))
    primary = {
        cohort + "_QTL_ACEGP_vs_GW_ACEGP_Fisher_z": primary_statistics.complete(report["primary"])
        for cohort, report in reports.items()
    }
    secondary = {}
    for comparison, record in completed.items():
        test = record["test"]
        valid = test["status"] == "OK" and test.get("p_value") is not None
        secondary[comparison] = {
            "status": "COMPLETE" if valid else test["status"],
            "p_value": test.get("p_value"),
        }

    primary_family = reporting_wrappers.registered_holm_family(
        primary, registration["primary_family"], family_id="two_population_primary"
    )
    primary_statistics.atomic(primary_statistics.OUT / "primary_Holm_family.json", primary_family)
    secondary_family = reporting_wrappers.registered_holm_family(
        secondary,
        registration["secondary_family"],
        family_id="registered15_model_component_MAE_comparisons",
    )
    primary_statistics.atomic(
        primary_statistics.OUT / "secondary_models_family.json", secondary_family
    )
    subprocess.run(
        [sys.executable, "-B", str(package / "code/statistics/exact_sign_tests.py"),
         "--output", str(primary_statistics.OUT / "sign_tests.json")],
        check=True,
    )


def main():
    sys.dont_write_bytecode = True
    parser = build_parser()
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    package = code.parents[1]
    try:
        population = validate_task(args)
        checks = execution.check_inputs(package, args.analysis)
        if args.check:
            print(
                json.dumps(
                    {"analysis": args.analysis, "population": population, **checks}, indent=2
                )
            )
            return 0 if checks["ready"] else 1
        if not checks["ready"]:
            parser.error(
                "Preflight failed; run --check for missing files, changed hashes or environment differences."
            )
        output = execution.prepare_run(
            args.output,
            package,
            execution.run_identity(package, args, population),
            args.resume,
            args.summarize,
        )
        sys.path[:0] = [str(code), str(code / "src")]
        from public_results import validate_predictions

        cohort = "development" if population == "primary" else "validation"
        tasks = execution.tasks_for(args)
        with execution.run_lock(output):
            worker = load_worker(args.analysis, package, output)
            if args.summarize:
                summarize(args, population, package, output, worker, tasks)
                print(json.dumps({"status": "summary_complete", "tasks": len(tasks)}))
            else:
                if args.analysis == "validation-random":
                    worker.verified_sources()
                result = execution.execute_tasks(
                    output,
                    args,
                    cohort,
                    tasks,
                    lambda task, folder: dispatch_task(
                        args, population, package, output, worker, task, folder
                    ),
                    lambda task, folder: validate_predictions(package, args, cohort, task, folder),
                )
                print(json.dumps(result))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
