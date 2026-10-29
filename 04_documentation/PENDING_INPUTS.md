# Input versions and optional inputs

## Validation prediction inputs

Version `validation_reconstructed_20261029` replaces the historical validation prediction inputs. Its 4,817,083 sites, 151-accession order, counted A1 alleles and frozen panel definitions are retained. The 12 chromosome matrices differ from the historical imputation hashes and are explicitly registered as reconstructed inputs. Their new hashes are not substitutions for the historical recovery receipts.

Required companion-data paths:

- `02_data/random_controls/validation/genotypes/chr1/A1_GT.npy` through `chr12/A1_GT.npy`: sites × 151 accessions, uint8 A1 counts.
- `02_data/random_controls/validation/genotypes/raw_metadata_validation151.npz`: the site catalog.
- `02_data/random_controls/validation/genotypes/a1_is_ref.npy`, `reference_byte.npy`, `alternate_byte.npy`: allele orientation.
- `02_data/prediction_inputs/validation_inputs/background.npy` and `qtl.npy`: float64 arrays, 151 × 32,666 and 151 × 67,104, extracted in the frozen marker order. Their SHA256 values match the arrays used in the accepted supporting rerun.

All validation prediction branches use this version. The supplied 100 random panels are retained; prediction does not require panel regeneration. The complete local reconstructed source exists. Distribution of the companion dataset is held for author and rights metadata; no public dataset DOI is asserted here.

The original five-repeat blind-masking assessment, matching-QC records, genetic-group labels and quality-subset membership are retained as historical design evidence. They were not recomputed on the reconstructed matrices. Historical receipts are stored separately under `02_data/random_controls/validation/metadata/historical_input_records/`.

## Optional panel generation

Regenerating validation panels additionally requires historical observed genotypes, the original site metadata, candidate banks and their template. These are separate from the prediction inputs and remain outside the current package. See [GENERATION_REFERENCE.md](GENERATION_REFERENCE.md). Their old `PENDING_common_genotypes/observed_catalog/` paths are retained only by the optional historical generator/verifier code; the prediction runner reads `genotypes/`.
