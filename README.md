# ACEGP: QTL-informed genomic prediction in rice

ACEGP combines 16 learners to compare genome-wide and QTL-informed genomic prediction of heat-induced spikelet fertility loss in rice. This repository accompanies *Curated heat-tolerance QTL priors improve genomic prediction of heat-induced spikelet fertility loss in rice*.

The primary population (241 accessions; eight regions) and validation population (151 accessions; seven regions) are trained and evaluated separately. Anther expression analyses provide biological context for the QTL regions.

## Contents

```text
code/ACEGP/           Prediction models and analysis runner
code/random_panels/   Original fixed-split random-panel workflows
code/high_density/    Common high-density background checks and bootstrap
code/anther/          Anther expression and gene–SNP analyses
code/statistics/       Statistical analysis
code/作图代码/         Figure 3–5 rendering scripts
code/environment/     Python and R dependency records
documentation/        Detailed guides, data definitions and provenance
documentation/FILE_MANIFEST.csv        Code and companion-data file checksums
documentation/CODE_FILE_MANIFEST.csv   Code-package file checksums
LICENSE                  Software licence
```

This repository provides the registered workflows and input conventions for the reported analyses; it is not a general-purpose interface for arbitrary datasets and does not by itself rerun every saved analysis. The current public entry points directly reproduce the main ACEGP workflows and the original fixed-split random-panel comparison. The additional sparse-background random-region comparison (100 panels per population × 20 repeated fivefold CV) is supplied as registered panel definitions and saved results in the matching Zenodo data package, but is not directly rerunnable from the current code entry points. See [Execution guide](documentation/CODE_GUIDE.md) for the supported commands and [Data definitions](documentation/DATA_DICTIONARY.md) for file formats, units and identifiers. Figure 3–5 rendering scripts and instructions are in [code/作图代码](code/作图代码/README.md); their inputs are supplied in the companion data package.

## Step 1: Set up the environment

The prediction runner requires Linux/POSIX, such as Ubuntu or Windows Subsystem for Linux (WSL), and the specified Python dependencies. GPU hardware is not required. Use a dedicated Python environment:

```bash
python -m pip install -r code/environment/prediction_requirements.txt
```

The validation runs used Python 3.12.3. Anther R package versions are recorded in `code/environment/anther/`. Optional panel-generation inputs are described in [Input versions](documentation/PENDING_INPUTS.md).

## Step 2: Prepare and check the companion data

The matching companion dataset has not yet been deposited. Place its `02_data/` directory beside `code/` in the repository root once it is available. The validation input version is `validation-imputed-v1`. Do not combine this code with the old `20261028_v2_split2` data archive.

```text
ACEGP-rice/
  code/
  02_data/          # matching companion data
  documentation/
  documentation/FILE_MANIFEST.csv
```

Check the code package, then the combined files and runtime requirements:

```bash
python -B code/verify_release.py --code-only
python -B code/verify_release.py
python -B code/ACEGP/run_analysis.py --analysis prediction --population validation --check
```

These checks do not train models. The hash verifier also runs on Windows; model execution requires Linux/POSIX. The data source and access details will be added when the dataset is deposited.

## Step 3: Run prediction

Choose a new output directory outside the repository. The following command fits one registered validation fold:

```bash
python -B code/ACEGP/run_analysis.py --analysis prediction --population validation --repeat 1 --fold 1 --output /path/to/validation-fold
```

To run and summarize all 20 repeats of fivefold cross-validation:

```bash
python -B code/ACEGP/run_analysis.py --analysis prediction --population validation --all-folds --output /path/to/validation-cv
python -B code/ACEGP/run_analysis.py --analysis prediction --population validation --summarize --output /path/to/validation-cv
```

Use `--population primary` for the primary population. Each population is evaluated within itself. The runner uses the registered partitions, region IDs and seeds, and compares new predictions with the matching archived results. A matching interrupted run can use `--resume`; see the [Execution guide](documentation/CODE_GUIDE.md) for incomplete-fold handling and commands for the other analyses.

## Output files

For a prediction run, results are written beneath the selected output directory:

```text
repeated_cv/<population>/r01_f1/predictions.csv   Held-out predictions for one fold
repeated_cv/<population>/r01_f1/training_choices.json   Model selection record
repeated_cv/<population>/all_oof_predictions.csv  Combined out-of-fold predictions
repeated_cv/<population>/paired_repeat_results.csv Metrics for each repeat
repeated_cv/<population>/SUMMARY.json             Summary across repeats
```

`<population>` is `development` for the primary population or `validation` for the validation population. The combined files are generated by `--summarize` after all registered folds finish. CV repeats are data partitions, not independent biological replicates.

The original fixed-split random-panel comparison uses the real panel and 100 random panels on the prespecified first fivefold split (Figure 5a,b). The additional **sparse-background random-region comparison** uses 100 random panels per population evaluated over 20 repeated fivefold splits (Figure 5c,d); its saved panel definitions and results are supplied in the matching Zenodo data package. The directory and result aliases `low_density` and `low_density_20x5` refer to this sparse-background comparison. The current code entry point implements the fixed-split comparison only and must not be described as a runner for the 100-panel × 20-repeat experiment. The **common high-density background comparison** (the `highdensity`/`high_density_common_background` result namespace) compares genome-wide and QTL-informed representations under a shared high-density background (Figure 5e,f); its saved results are auditable, but the public entry points do not refit that comparison.

## Computing resources

For reference, the other prediction and statistics workflows were verified on Ubuntu 22.04.1 with 32 logical CPU cores and a 64 GiB container memory limit. Ten analysis processes ran concurrently; the reported combined peak memory was approximately 14 GB. Runtime depends on the analysis and hardware. Validation random-panel runs require at least 25 GiB of free space under the configured cache policy. These figures describe that validation run and are not minimum requirements or a guarantee of runtime.

## Data and citation

The companion dataset is being prepared for deposit; the local Zenodo record is a private draft and is not yet a public data DOI. Please cite the accompanying paper and dataset when their final citations are available. Original software uses the MIT terms in [LICENSE](LICENSE). Original data and results use CC BY 4.0, while third-party materials retain their source terms. See [third-party notices](documentation/THIRD_PARTY_NOTICES.md).

Software version: `v1.0.0`. The software creator is `Anguo Gui`; repository citation metadata should be finalized together with the article citation.
