# Execution guide

## Input and output rules

Use the companion data matching code version `v1.0.0`. Extract `02_data/` at the repository root. Internal `development` labels refer to the primary population. All outputs must be outside the repository. Prediction fitting requires Linux/POSIX, the pinned Python dependencies and sufficient memory and disk for the frozen cache settings.

```bash
python -B 01_code/verify_release.py --code-only
python -B 01_code/verify_release.py
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population validation --check
python -B 01_code/ACEGP/run_analysis.py --analysis validation-random --check
```

The hash verifier imports no model code. `--check` also checks the environment and registered validation input version without fitting models or creating analysis output. Optional historical panel-generation inputs are not required for the fixed-panel analyses.

## Repeated cross-validation

```bash
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population validation --all-folds --output /path/to/validation-cv
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population validation --summarize --output /path/to/validation-cv
```

Use `--population primary` for the primary population. `--all-folds` runs the 20 × 5 registered tasks. `--repeat R --fold F` selects one task. `--resume` continues a matching run and verifies completed task receipts before skipping them. Incomplete tasks without valid receipts are retained and rejected; use a new output directory.

The run identity records the analysis, population, quality subset and release-manifest hash. Predictions are checked by sample ID, partition, model and region or panel. Absolute differences above 1e-8 percentage points stop the run; the checker never replaces newly fitted values with saved predictions. This is a numerical comparison tolerance, not a statistical threshold. Summaries require complete OOF vectors for every repeat and report sample SD across repeats.

`run_cv_fold.py` is the one-fold fitting entry. Prefer `run_analysis.py` for task receipts, continuation, reference checks and summaries.

## Model comparisons and sensitivity analyses

Use a separate output directory for every analysis and subset:

```bash
python -B 01_code/ACEGP/run_analysis.py --analysis controls --population validation --all-folds --output /path/to/validation-controls
python -B 01_code/ACEGP/run_analysis.py --analysis controls --population validation --summarize --output /path/to/validation-controls
python -B 01_code/ACEGP/run_analysis.py --analysis ablation --population validation --all-folds --output /path/to/validation-ablation
python -B 01_code/ACEGP/run_analysis.py --analysis ablation --population validation --summarize --output /path/to/validation-ablation
python -B 01_code/ACEGP/run_analysis.py --analysis quality --subset 146 --all-folds --output /path/to/quality146
python -B 01_code/ACEGP/run_analysis.py --analysis quality --subset 146 --summarize --output /path/to/quality146
python -B 01_code/ACEGP/run_analysis.py --analysis genetic-group --population validation --all-folds --output /path/to/validation-groups
python -B 01_code/ACEGP/run_analysis.py --analysis genetic-group --population validation --summarize --output /path/to/validation-groups
```

Repeat quality analysis with `--subset 142` and a different output directory. Both subsets retain their original independently selected members and restrict the original 151-accession folds to those members. Genetic-group CV uses the original five groups; it does not recluster the final imputed genotypes.

Validation controls refit GBLUP and RKHS for both genome-only and genome-plus-QTL representations, together with ACEGP without block features. Primary controls retain the existing workflow and reuse their unchanged archived genome-only control predictions. The frozen fitting algorithms, grids and seeds are unchanged.

## Fixed random panels

```bash
python -B 01_code/ACEGP/run_analysis.py --analysis validation-random --all-panels --all-folds --output /path/to/validation-random
python -B 01_code/ACEGP/run_analysis.py --analysis validation-random --all-panels --summarize --output /path/to/validation-random
```

`validation-random` includes real panel 0 and random panels 1–100. Use `--analysis primary-random` for the 100 primary panels. `--panel N --all-folds` selects one panel. All random-panel comparisons use prespecified repeat 1, with five folds per panel. They do not perform 20 repeats per panel or generate new panels.

## Statistics

```bash
python -B 01_code/ACEGP/run_analysis.py --analysis statistics --output /path/to/statistics
```

This command reads the matching archived predictions, not a separate new training-run directory. It reports 17 variance-corrected comparisons and two post hoc exact sign tests. Holm correction remains separate for the two primary comparisons, 15 secondary comparisons and two sign tests. Outputs include `statistics/primary_Holm_family.json`, `statistics/secondary_models_family.json` and `statistics/sign_tests.json`. The complete accepted 19-record table is in the companion data at `02_data/prediction_results/statistics/statistical_comparisons_19.csv`. CSV row order is aligned to the registered sample IDs before fold-level tests. CV repeats and folds are partitions, not biological replicates.

## Anther analyses

```bash
python -B 01_code/anther/prepare_workspace.py --check
python -B 01_code/anther/prepare_workspace.py --output /path/to/anther-work
```

Workspace preparation copies existing inputs. `review_anther.R` performs expression and functional analyses, `map_gene_snps.py` maps genes to markers. 

## Verification scope

The study workflows were rerun against their registered inputs and compared with archived outputs. Two-population repeated CV, fixed random panels, model comparisons, regional ablation, quality subsets, genetic-group CV and statistics were checked. The anther R workflow was also rerun; CAMERA correlation and P values showed cross-platform numerical sensitivity, while the reported FDR significance classification remained unchanged. These results do not guarantee bitwise identity across environments. The preflight checks above do not fit models.
