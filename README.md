# ACEGP: QTL-informed prediction of heat-induced fertility loss in rice

This source package accompanies *Curated heat-tolerance QTL priors improve genomic prediction of heat-induced spikelet fertility loss in rice*. It contains the ACEGP implementation, registered evaluation workflows and environment records. The primary population has 241 accessions and eight fixed regions; the validation population has 151 accessions and seven regions. Models were fitted and cross-validated separately within each population.

Local preparation version: `20261028_v2_split2`. Original software uses the MIT licence in `LICENSE`; third-party terms are retained. The copyright-holder name, confirmed creators and permanent identifiers remain to be completed; see `LICENSE_STATUS.md`, `THIRD_PARTY_NOTICES.md` and `CITATION_STATUS.md`.

## Contents

- `01_code/ACEGP`: model implementation, input loading, repeated CV, model comparisons, region ablation, quality subsets, genetic-group CV and result summaries.
- `01_code/random_panels`: fixed-panel model workflows and optional panel-generation methods.
- `01_code/anther`: expression and functional analysis, gene-to-SNP mapping and workspace preparation.
- `01_code/statistics`: the post hoc exact sign tests.
- `01_code/environment`: prediction dependency versions and anther environment records.
- `04_documentation`: execution guide, data definitions, dependency inventory and source-name mappings.

All 50 original code and environment files are unchanged from the combined v2 package. Frozen scientific sources, configuration values, region IDs and seed namespaces retain their original bytes. Internal `development` labels mean the primary population.

## Add the companion data

The local companion archive is `ACEGP_data_20261028_v2_split2.zip`. Its public DOI has not been assigned. Extract its contents directly into this repository root, without adding an extra enclosing folder. The resulting layout must be:

```text
repository-root/
  01_code/
  02_data/
  04_documentation/
  DATA_DOCUMENTATION/
  FILE_MANIFEST.csv
  CODE_FILE_MANIFEST.csv
  DATA_FILE_MANIFEST.csv
  SHA256SUMS_CODE.txt
  SHA256SUMS_DATA.txt
```

The archives have no conflicting file paths. `02_data` is intentionally excluded from Git tracking. Moving the data to an arbitrary external directory is not supported by the frozen path layout.

## Environment and checks

Use Linux/POSIX for the prediction runner; it uses `fcntl`. The historical prediction package versions are pinned in `01_code/environment/prediction_requirements.txt`. This preparation did not install or test a new complete environment. The anther R environment is recorded separately under `01_code/environment/anther`.

After adding the data, these commands only check files and the environment:

```bash
python -B 01_code/ACEGP/run_analysis.py --analysis prediction --population primary --check
python -B 01_code/anther/prepare_workspace.py --check
```

The prediction check examines the complete dependency inventory, including saved comparison results. It requires the full current companion data package. A nonzero exit code can report missing Python packages or an unsupported operating system even when all input hashes match. Consult `04_documentation/CODE_GUIDE.md` for analysis commands; those commands can train models or recompute results and were not run during packaging. Outputs must be written outside this repository.

## Coverage and pending inputs

Included data cover the two populations' repeated-CV inputs and saved results, secondary comparisons, primary random controls and anther analyses. Both populations' 100 fixed random panels and saved results are included. Validation random-control execution additionally needs 16 missing files: 12 common-imputation chromosome matrices, one site catalog and three allele-orientation arrays. The archived repeated-CV matrices cannot replace them. See `04_documentation/PENDING_INPUTS.md`.

Optional regeneration of validation random panels needs additional historical observed-genotype and candidate-bank inputs. Reusing the supplied fixed panels does not require their regeneration.

## Verification and citation

`CODE_FILE_MANIFEST.csv` inventories files in this code package. `SHA256SUMS_CODE.txt` also covers that manifest. `FILE_MANIFEST.csv` is the combined code-and-data payload inventory used by the existing runner; entries under `02_data` become present after the companion archive is extracted. The combined inventory excludes its own file and the code package's inventory/checksum files to avoid circular checksums.

This preparation checked byte identity, archive integrity and path assembly. It did not retrain models, recalculate scientific results or demonstrate complete numerical reproduction. No trained weights or runtime dependencies were generated during this step. Citation metadata will be completed when creators and repository identifiers are confirmed; MIT is already selected for original software.

Journal tables and Supplementary Data S1 remain separate submission files. They are not duplicated here.
