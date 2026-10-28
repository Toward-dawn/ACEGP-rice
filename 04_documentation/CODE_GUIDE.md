# Code guide

Extract the companion data archive into the code root before using these commands. The code root must contain both `01_code` and `02_data`. This split delivery does not change the existing source files or execution procedures.

## Source organization

| File | Responsibility |
| --- | --- |
| `ACEGP/run_analysis.py` | Validate options, dispatch registered tasks and select the summary procedure. |
| `ACEGP/public_execution.py` | Check dependencies, record run identity and verify completed tasks before reuse. |
| `ACEGP/public_results.py` | Compare new predictions with the archived study and summarize complete OOF vectors. |
| `ACEGP/run_cv_fold.py` | Fit one saved outer fold using the frozen model implementation. |
| `ACEGP/revised_inputs.py` | Read matrices, check sample order and dimensions, and assign columns to fixed regions. |
| `ACEGP/public_paths.py` | Resolve packaged inputs and explicit external outputs. |
| `ACEGP/remote_*.py` | Apply the registered analysis procedures. Call `configure_output(path)` before execution. |
| `ACEGP/src` and `ACEGP/legacy_model_snapshot` | Provide the 16 byte-preserved scientific source files. |
| `anther/prepare_workspace.py` | Copy prepared anther inputs and results to an external workspace. |

Historical filenames and `development` labels are retained where they form part of the frozen implementation. `development` means the primary population. The fitter CODE_HASH includes all ten snapshot Python files. Do not rename or edit those files.

## Preflight

Run from the package root. The task runner requires Linux/POSIX (`fcntl`) and the versions in `01_code/environment/prediction_requirements.txt`.

```
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population primary --check
python -B 01_code/ACEGP/run_analysis.py --analysis validation-random --check
```

`--check` verifies every packaged code and data file in the dependency inventory against FILE_MANIFEST.csv, checks pinned Python packages, and reports the platform. It does not import fitters or create an output directory. Exit status is 0 only when all checks pass. Validation random controls additionally require the 16 pending files to match their historical SHA256 values. Optional panel-generation inputs are listed separately in GENERATION_REFERENCE.md.

## Repeated CV: execute, continue, summarize

The first two commands below fit models. The third reads completed outputs only.

```
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population primary --all-folds --output /path/to/primary-run
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population primary --all-folds --resume --output /path/to/primary-run
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population primary --summarize --output /path/to/primary-run
```

`--all-folds` uses the 20 registered repeats and five folds per repeat. Without it, `--repeat` and `--fold` select one task. Use `--resume` to append further tasks to the same run. A run records its analysis, population, quality-subset choice and release-manifest hash. Mismatched runs cannot be combined. Completed tasks are skipped only after their output hashes pass verification. An incomplete task without a valid receipt is retained and rejected; use a new run directory rather than overwriting it. One runner may write to a run directory at a time.

Each new fold is checked against the archived predictions by sample ID, partition, region or panel, and model column. Absolute differences above 1e-8 percentage points stop the run. The check never replaces computed predictions with historical values. This tolerance follows the existing workers' prediction checks; it is not a significance threshold or an expected CV effect. Summary requires all registered tasks and rechecks their outputs. Correlations are calculated from complete OOF vectors within each repeat; SD is the sample SD across 20 repeats.

`run_cv_fold.py` remains available as the original low-level, one-fold fitter. It requires a new or empty output directory and does not create run receipts. Use `run_analysis.py` for the checked execution and summary workflow.

## Other analyses

`controls` and `ablation` use the same `--all-folds`, `--resume` and `--summarize` options. They default to primary; use `--population validation` for validation. `quality` uses validation only and requires a separate run directory for `--subset 146` and `--subset 142`. It retains the registered IDs and restricts the original 151-sample folds to those rows. `genetic-group --all-folds` executes the five saved genetic groups; `--group` selects one group. Existing scientific workers provide the controls, ablation, quality and genetic-group summaries.

Random panels use prespecified repeat 1 only:

```
python -B 01_code/ACEGP/run_analysis.py --analysis primary-random --all-panels --all-folds --output /path/to/primary-random-run
python -B 01_code/ACEGP/run_analysis.py --analysis primary-random --all-panels --summarize --output /path/to/primary-random-run
```

`primary-random` uses panels 1–100. `validation-random` uses panel 0 for the real seven-region panel and panels 1–100 for random panels; `--all-panels` includes all 101. `--panel N --all-folds` executes the five folds for one panel, and `--panel N --summarize` summarizes that panel. These analyses use the fixed supplied panels; no panels are regenerated or selected. Validation random controls remain blocked until their historical inputs are supplied and verified.

`--analysis statistics --output /path/to/statistics-run` recalculates the original 17 comparisons from the packaged historical predictions, not from a new run directory. Omit `--population`. The two primary corrected comparisons and 15 secondary comparisons retain separate Holm families. `01_code/statistics/exact_sign_tests.py` implements the two post hoc sign tests as a separate family. Final saved files are primary_Holm_family.json, secondary_models_family.json and sign_tests_as_reported.json. secondary_Holm_family.json is an earlier incomplete record.

## Anther workspace

```
python -B 01_code/anther/prepare_workspace.py --check
python -B 01_code/anther/prepare_workspace.py --output /path/to/new-anther-work
```

Preparing the workspace copies files only. `review_anther.R` performs expression and functional analyses, `map_gene_snps.py` maps genes to existing genotype inputs, and `plot_review.R` reads saved gene and enrichment tables. Mapping runs only through its main entry point; importing the module does not create directories or start an analysis.

## Reproduction status

This release was checked by source comparisons, input hashes, archived-output validation and synthetic execution tests. Models and research analyses were not rerun during packaging. Numerical reproduction in a supported execution environment remains to be demonstrated; the validation random-control inputs are still pending. The result checks above reject divergent new outputs rather than certifying them as reproduced.
