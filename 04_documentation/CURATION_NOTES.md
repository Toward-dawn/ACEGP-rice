# Revision 2 notes

The package retains frozen model sources, configuration, numerical data, region IDs, partitions and seeds. Input paths are relative to the package, and analysis outputs use a separate directory. Duplicate views, direct cross-population execution and server orchestration are excluded.

The code review corrected population handling in the public analysis entry point and output-directory checks in three entry points. Primary random-input checks no longer require a panel number. Reader-facing comments and instructions were shortened; the 16 frozen scientific source files were not edited.

Review covered training-fold preprocessing, inner model selection, seed propagation, statistical families and anther contrasts. Checks used synthetic fixtures, mocked worker calls, syntax parsing and file hashes. Research analyses were not rerun. Validation random-control inputs remain pending. The complete previous package and the detailed review record are stored outside this package.

The public Python layer uses separate imports and expanded statements. The analysis entry point, input loader and anther workspace preparation separate validation, reading and execution. The other 23 Python files received layout changes only; their executable syntax is equivalent after normalizing split import statements. The 16 frozen source files retain their original bytes.

The execution interface now supports registered multi-fold runs, explicit continuation and completeness-checked summaries. Workers initialize output paths only on request. The validation common-context worker uses distinct names for input, metadata, saved-result and output roots. New predictions must match archived held-out outputs within 1e-8 percentage points before task completion is recorded. No numerical model functions, configuration values, fixed partitions, seeds or data were changed. Earlier format-only counts describe the preceding review, not this execution-interface revision.


## Split distribution preparation

The combined v2 package was backed up before making separate code and data copies. All 50 existing code/environment files and all 1,084 existing data/result files remain byte-identical. Changes in this step are limited to release documentation, metadata drafts and inventories. The data archive restores `02_data` at the repository root, so no source-code path changes were needed. The archives do not include the original manuscript, journal tables or local backup. Authors and public record identifiers remain unresolved. Licence choices were subsequently confirmed as MIT for original software and CC BY 4.0 for original data/results, with third-party terms retained.


## Licence revision

Preparation `20261028_v2_split2` records the author's licence choices, adds the MIT text and CC BY 4.0 scope/legal text, and retains all third-party terms. The copyright-holder field is explicitly pending. README files, metadata drafts, manifests and archive hashes were updated; no file under `01_code` or `02_data` was changed. The preceding split1 delivery was backed up separately.
