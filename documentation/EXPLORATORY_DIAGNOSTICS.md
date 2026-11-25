# Exploratory diagnostics

The main ACEGP code is unchanged. The companion dataset contains exploratory Table S3 results and anchor-screening audit tables under `02_data/prediction_results/exploratory_diagnostics/`. These are descriptive post hoc analyses and are not new significance tests.

The audit export reads the registered `training_choices.json` for the 200 outer folds, without fitting models:

```bash
python -B code/exploratory_diagnostics/export_anchor_screening_audit.py --output /path/to/new-audit
```

Each fold has a full initial ranking of 16 learners. `screened_candidates` is the subsequent shortlist plus anchors, not all 16 learners; its size varies across folds. Counts comparing non-kernel challengers to the genetic anchor refer only to these gate candidates. The 200-fold and four-case tables were checked against the supplied outputs.

The separate Q-ACEGP diagnostic runner fits exploratory models on the registered splits and can be computationally expensive:

```bash
python -B code/exploratory_diagnostics/run_q_acegp_diagnostics.py --output /path/to/new-diagnostic-run --population all
```

Choose a new output directory outside the repository and archived data. The runner was smoke-tested on registered fold primary r01_f1 against the supplied Table S3 Section C predictions (maximum absolute difference <= 1.4e-14); full 200-fold refitting was not rerun during packaging and should be executed into a fresh output directory. By default the runner refuses to use an output directory inside the package or an existing directory containing predictions.csv. Use --resume only after independently confirming the output provenance and completeness.
