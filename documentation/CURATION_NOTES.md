# Release changes

## v1.0.0

- Register the reconstructed validation genotype version and its matching predictions. The derived background and QTL arrays match the accepted rerun's SHA256 values.
- Verify new validation hashes explicitly, while preserving historical receipts as provenance.
- Refit both validation kernel-control representations instead of inheriting old genome-only predictions.
- Align paired fold statistics by registered sample IDs; retain the existing test formulas and Holm families.
- Update the sign-test expected outcomes to 20/20 primary and 19/20 validation, and include sign tests in the statistics entry point.
- Add a model-free file verifier and refresh paths, dependencies and manifests.

The 16 files under `ACEGP/src` and `ACEGP/legacy_model_snapshot`, the model configuration, region IDs, sample order, folds and seeds retain their source bytes. The model library was not rewritten. Main primary-population results, anther data and optional panel-generation algorithms are unchanged.

The completed validation reruns preceded this packaging step. Integration validation does not include new model training. This public release pairs software `v1.0.0` (MIT License, Copyright (c) 2026 Anguo Gui) with the companion Zenodo dataset (`https://doi.org/10.5281/zenodo.23900534`).

## Earlier preparation

The 20261028 package separated code from data, normalized public file paths and execution entry points, and selected MIT for original code and CC BY 4.0 for original data/results. The complete preceding package and repository snapshot are retained in the local preparation backup, outside the upload contents.
