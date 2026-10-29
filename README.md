# ACEGP: QTL-informed genomic prediction in rice

Code accompanying *Curated heat-tolerance QTL priors improve genomic prediction of heat-induced spikelet fertility loss in rice*.

ACEGP combines 16 learners to compare genome-wide and QTL-informed marker representations. Models are fitted and cross-validated separately in the primary population (241 accessions, eight fixed regions) and validation population (151 accessions, seven regions). Anther expression analyses provide biological context for the QTL regions.

Repository: https://github.com/Toward-dawn/ACEGP-rice

## Code and companion data

This code version is `20261029_v3`. It uses the registered validation input version `validation_reconstructed_20261029` for repeated CV, fixed random panels, model comparisons and sensitivity analyses. Primary inputs and results are unchanged. The validation matrices are a protocol reconstruction, not a byte-identical recovery of the historical imputation output.

The companion data must be placed at `02_data/` beside `01_code/`. Data files are excluded from Git tracking. Do not combine this code with the old `20261028_v2_split2` data archive. The required files and hashes are recorded in `FILE_MANIFEST.csv` and `04_documentation/VALIDATION_DATA_VERSION.json`. See [input versions](04_documentation/PENDING_INPUTS.md) for the dataset layout and optional panel-generation inputs, and [data definitions](04_documentation/DATA_DICTIONARY.md) for units and identifiers.

```text
ACEGP-rice/
  01_code/
  02_data/                 # companion data
  04_documentation/
  FILE_MANIFEST.csv
  CODE_FILE_MANIFEST.csv
  SHA256SUMS_CODE.txt
```

## Installation and checks

The prediction runner requires Linux/POSIX and the package versions in `01_code/environment/prediction_requirements.txt`. Install them in a dedicated Python environment. Anther R dependencies are recorded separately in `01_code/environment/anther/`.

```bash
python -m pip install -r 01_code/environment/prediction_requirements.txt
python -B 01_code/verify_release.py --code-only
# After adding the matching companion data:
python -B 01_code/verify_release.py
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population validation --check
python -B 01_code/ACEGP/run_analysis.py --analysis validation-random --check
```

These checks do not train models. The hash-only verifier also works on Windows; execution requires Linux/POSIX. `--check` reports dependency or platform differences separately from file-integrity failures.

## Run an analysis

The following commands train models. Use a new output directory outside the repository.

```bash
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population validation --all-folds --output /path/to/validation-cv
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population validation --summarize --output /path/to/validation-cv
```

The runner uses the fixed partitions, region IDs and seeds. It checks newly computed predictions against the matching archived outputs. Complete commands for model comparisons, region ablation, genetic-group CV, quality subsets, random panels, statistics and anther analyses are in [CODE_GUIDE.md](04_documentation/CODE_GUIDE.md).

## Version and attribution

The frozen model libraries and configuration are unchanged. This update connects the public entry points to the accepted reconstructed inputs and results; it does not reselect regions or tune model parameters. The local integration checks are described in [CURATION_NOTES.md](04_documentation/CURATION_NOTES.md). A complete training rerun of this packaged interface has not been performed during packaging.

Original software uses the MIT terms in [LICENSE](LICENSE). The copyright-holder field still requires confirmation; see [LICENSE_STATUS.md](LICENSE_STATUS.md). Original data and results use CC BY 4.0; third-party materials retain their source terms. Citation fields and release identifiers are tracked in [CITATION_STATUS.md](CITATION_STATUS.md). Journal tables and Supplementary Data S1 are separate submission files.
