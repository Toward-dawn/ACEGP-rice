# Optional random-panel generation

The fixed-split comparisons use the supplied 100 panels per population and their saved SNP row arrays. These panels can be used without regenerating candidate banks or selecting new panels.

Primary regeneration uses the included 241-sample BED/BIM/FAM and raw marker metadata. Validation regeneration additionally requires the historical pre-imputation `observed_A1_151.npy`, seven frozen 128-position candidate banks and their template. These generation inputs are not included. Their historical hashes are recorded in the registrations; final imputed genotypes are not substitutes.

Method source modules require an external working directory. Call `module.configure_output(path)` before calling their execution functions. Standalone generator and verifier scripts call this initializer from their main entry point and read `ACEGP_RUN_ROOT`; importing a module does not create its output directories. Validation v1 outputs use `validation_matching_v1`; constrained-generation outputs use `validation_constrained_generation` under that root. The original Linux resource checks remain in place.

Primary verifiers read source files from the packaged data directory. Validation manifest keys retain the historical `output/` prefix. The constrained generator and verifier use the same observed-input directory. These optional generation inputs are separate from the 16 pending prediction inputs listed in `PENDING_common_genotypes/expected_files.csv`.
