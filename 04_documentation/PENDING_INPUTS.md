# Pending validation inputs

The 16 files required to rerun the **validation fixed-split random-control comparison** are listed, with historical SHA256 values, in:

`02_data/random_controls/validation/PENDING_common_genotypes/expected_files.csv`

- `common_final_matrix/chr1/A1_GT.npy` through `chr12/A1_GT.npy`: 12 chromosome matrices, with sites as rows and 151 accessions as columns.
- `observed_catalog/raw_metadata_validation151.npz`: the 4,817,083-site catalog.
- `common_final_matrix/a1_is_ref.npy`, `reference_byte.npy` and `alternate_byte.npy`: three allele-orientation arrays.

These relative names are rooted inside `PENDING_common_genotypes`, as expected by the current runtime. The expected-file list and historical acceptance records are receipts; they contain no genotype matrix. No placeholder arrays have been created.

Rebuilt files must be compared with the historical hashes and sample/site/allele order. A rebuilt file whose hash differs must not be silently labelled the original input or accepted by replacing the expected hash. Investigate the discrepancy separately. This preparation does not rebuild or analyse them.

Both populations' repeated-CV matrices and both sets of 100 fixed random panels and saved random-control results are included. Those results remain inspectable; the validation random-control comparison is not yet fully rerunnable from this deposit.

Optional **regeneration** of validation random panels additionally uses the historical pre-imputation `observed_A1_151.npy`, seven frozen 128-position candidate banks and their template. These are separate from the 16 prediction inputs and are not included. Using the supplied fixed panels does not require regenerating the panels. See `GENERATION_REFERENCE.md` in the same documentation directory.
