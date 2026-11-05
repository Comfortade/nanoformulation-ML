# Bond-aware graph sensitivity experiment

Purpose: investigate sensitivity to reviewed chemical representations, not force graph superiority. This supplements the completed revised analysis. It does not replace it and does not reproduce the original manuscript's Korsmeyer–Peppas model.

## Run from the repository root in Git Bash
Activate the existing project environment, then run:

python scripts/graph_sensitivity/run_graph_sensitivity.py

Default: 500 epochs, seeds 11/42/97, CPU, both targets, all 19 splits. It may take longer than the previous notebook. No GPU is required. The existing notebook dependencies are reused. No PyTorch Geometric is required.

The workbook defaults to data/SMART NANO DATA OLA 2.xlsx, and chemical inputs to data/chemical_audit/graph_input_variants.json. Neither file is modified. Old notebooks and exports are retained. Each run creates a unique exports/graph_sensitivity/run_... folder and ZIP.

## Methods and interpretation
- Strictly the same grouped-condition/held-out-profile/forward-time splits as the prior revised implementation; no hyperparameter tuning.
- Training-only feature scaling, target scaling and empirical kinetic prior. Queries contain covariates only.
- Two-layer bond-aware message passing, hidden width 8, mean atom pooling then normalized selected-peak-area weighted compound pooling. Bond order and available E/Z information are actually consumed.
- Atom hybridization/CIP and bond properties use categorical one-hot encoding, including an unknown category. Unspecified stereochemistry is retained as unspecified.
- Same fixed extract context is appended to all observations. This is a dataset-level context experiment, not an assertion that the free-citral control contains extract or that all 13 compounds remain on the finished nanoparticles. The manuscript separately reports extract and nanoparticle profiles; this experiment cannot determine the actual retained mixture.
- Baseline uses zero context with the same prediction-head dimensions and seed initialization. Additional graph parameters still differ; comparisons do not isolate chemical information perfectly. A constant graph vector cannot establish chemical generalization or causal compound effects.
- Unconstrained models for both targets; cumulative-release target additionally uses the previous saturating-exponential constrained output and training-only empirical prior (weight 0.1). This is not a Korsmeyer–Peppas implementation or proof of a diffusion mechanism.
- Linear and random-forest benchmarks on the same rows; cumulative target additionally Higuchi and saturating-exponential benchmarks using documented prior-revision training-only profile fallback logic in revision_core.py.
- Six chemical variants: all reviewed entries; connectivity only; omit all three qualified entries; omit each individually. Excluded-entry pooling weights are renormalized computationally. These are not experimentally altered formulations.
- This is exploratory development after viewing previous results. It is not a fresh independent validation. Seed variability is not an experimental confidence interval. Do not choose compound identity based on the best prediction score.
- No new 120 h or pH 6.3 forecast is made here. Existing forecasts remain unvalidated by new laboratory experiments.

## Outputs
held_out_predictions.csv; comparison.csv; metrics_by_seed.csv; metrics_by_profile.csv; paired_prediction_changes.csv; split_assignments.csv; split_leakage_checks.csv; run_config.json; environment.json; completion.json. Predictions are saved after each completed fold; a run is complete only when completion.json exists and reports completed=true. Per-profile errors should be inspected alongside pooled R². NaN seed SD for deterministic benchmarks is expected.

## Verification performed before delivery
- All 19 folds and all model/variant paths executed using 2 epochs and one seed. Those scores are software-test outputs, not scientific results.
- Two repeated representative 500-epoch constrained graph fits checked for finite, exactly reproducible predictions on this CPU runtime.
- Focused tests: atom relabeling invariance; response to changed E/Z bond attributes; graph-layer gradients; finite outputs for all variants; zero initial constrained release, monotonicity and capacity bound at a fixed pH/formulation.
- The full default 500-epoch, three-seed comparison has not been executed here. Run it on the project computer and inspect the outputs before using results in the manuscript.

## Required inputs and data access

Experimental data are not distributed in this repository. Request the
experimental workbook and accompanying reviewed chemical-input files
from the corresponding author.

Place the supplied files at the "Data" local paths

The script checks that the mapping CSV matches the SHA-256 recorded in
the graph-input JSON. Use the matching files together.

The data and exports folders are excluded from Git tracking.
Reproducing the reported numerical results requires the original inputs
and the corresponding software environment.

To rerun focused checks:
python scripts/graph_sensitivity/check_graph_model.py
