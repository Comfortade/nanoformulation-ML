# Citral release modelling

Code for the revised analysis of citral release from a citral-loaded Ag-modified TiO2 nanocarrier and a free-citral control. The analysis evaluates concentration and cumulative release as separate outcomes using experimental values supplied in an Excel workbook.

Experimental data are not distributed with this version of the repository. Obtain the required inputs from the corresponding author upon reasonable request, as described in the manuscript's data-availability statement. Public code permits inspection of the method; reproducing numerical results also requires the matching inputs and software environment.

## Files

| File | Purpose |
| --- | --- |
| `notebooks/04_revised_analysis.ipynb` | Core model comparison, held-out evaluation and diagnostics |
| `scripts/compare_release_equations.py` | Standalone empirical power-law and saturating-exponential comparison |
| `scripts/check_release_equations.py` | Equation-fitting checks using known test curves; does not produce manuscript results |
| `scripts/graph_sensitivity/revision_core.py` | Shared workbook reader, core models and evaluation functions |
| `scripts/graph_sensitivity/run_graph_sensitivity.py` | Bond-aware extract-graph sensitivity analysis and zero-context baseline |
| `scripts/graph_sensitivity/check_graph_model.py` | Graph implementation checks |
| `scripts/graph_sensitivity/README.md` | Graph experiment instructions and interpretation |
| `scripts/plot_manuscript_results.py` | Manuscript figures from saved held-out predictions; no retraining |

## Private inputs

Create a local `data` folder and place the author-supplied files at:

- `data/SMART NANO DATA OLA 2.xlsx`
- `data/chemical_audit/graph_input_variants.json` (graph sensitivity and graph checks)
- `data/chemical_audit/reviewed_structure_mapping.csv` (graph sensitivity)

The graph runner verifies the mapping CSV against the SHA-256 recorded in the JSON; use matching files. The workbook readers expect the original sheet/cell layout and cached Excel formula results. The workbook is read without being modified. A differently structured spreadsheet cannot be substituted without adapting the reader.

The `data` and `exports` folders are ignored by Git. Outputs include observed values and should remain local unless their release is authorized. Clear notebook outputs before committing. These exclusions apply to the current checkout; removing previously tracked files does not erase them from Git history.

## Environment

The analysis was run in a Python 3.11.9 environment. Use a dedicated environment and select its kernel in VS Code. The code requires NumPy, pandas, SciPy, scikit-learn, Matplotlib, openpyxl, PyTorch and RDKit. PyTorch Geometric is not required. Models run on CPU.

For a new environment, install the dependencies:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib openpyxl rdkit
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

These commands install available versions, not a pinned reproduction environment. Use the package versions recorded with the original analysis when reproducing its reported numbers. The core notebook records actual package versions in each completed run. Exact equality across software versions and platforms is not guaranteed.

## Run the revised analyses

Run commands from the repository root with the project environment active.

### Core comparison

Open `notebooks/04_revised_analysis.ipynb`, select the project kernel and run all cells. The notebook locates the workbook in `data`. If multiple different matching workbooks exist, set `EXCEL_FILENAME` explicitly in its configuration cell.

Outputs are saved to a new `exports/revised_analysis/run_...` directory. They include held-out predictions, R2/RMSE/MAE summaries, split assignments and checks, source-cell/formula checks, configuration and diagnostic plots. A run finishes by writing `completion.json` with `status` set to `completed`.

Default neural settings are 500 epochs, learning rate 0.003, weight decay 1e-5, kinetic penalty weight 0.1 and seeds 11, 42 and 97. The random forest uses 200 trees and a minimum leaf size of two. These are fixed exploratory configurations, not a nested hyperparameter search.

### Standalone equation comparison

```bash
python scripts/compare_release_equations.py
```

Outputs are saved under `exports/powerlaw_benchmark/run_...`. This fits the empirical power law and saturating exponential on training observations only. It does not retrain the neural models. The power law is an empirical benchmark; its fitted exponent is not treated as proof of a release mechanism.

### Extract-graph sensitivity

```bash
python scripts/graph_sensitivity/run_graph_sensitivity.py
```

Supply the matching chemical-input files first. Outputs are saved under `exports/graph_sensitivity/run_...`. See the experiment's README for the chemical variants, settings and limitations.

### Manuscript figures

Replace the example directory names below with the completed local run directories you intend to plot:

```bash
python scripts/plot_manuscript_results.py --analysis exports/revised_analysis/YOUR_ANALYSIS_RUN --powerlaw exports/powerlaw_benchmark/YOUR_POWERLAW_RUN
```

The script checks agreement of evaluation conditions, observed values and shared exponential predictions between runs before plotting. PNG/PDF figures and plotted-value CSVs are saved under `exports/manuscript_figures`. Figure points average available seeds; analysis metrics are computed separately per seed.

## Evaluation and interpretation

- The workbook contains 108 concentration readings across 36 time/pH/formulation combinations, with three technical readings per combination, and 36 cumulative-release summaries. These are separate targets, not additional independent experiments.
- Concentration uses five-fold grouped evaluation, keeping technical readings from a condition together. Cumulative release uses six-fold complete-profile withholding. Each target also uses four forward-time evaluations at 12, 24, 48 and 96 h, trained on earlier observations only.
- Compared models share partitions within each target/protocol. Fitted scaling, response normalization and empirical reference curves use training observations only. Outcomes are not passed as query predictors.
- Concentration grouping does not separate entire profiles: different times from the same profile can appear in training and evaluation. Complete-profile withholding answers a different question. Neither protocol establishes independent-batch validation.
- Core graph results use one fixed citral-connectivity graph. Extract-graph sensitivity is a separate analysis using a fixed chemical context across observations. It cannot identify independent compound effects or establish generalization to new compositions.
- Release-constrained neural variants use a saturating-exponential output and a training-fitted reference penalty. They impose an empirical curve shape, not a verified diffusion PDE or the original Korsmeyer-Peppas architecture. Constraints are applied to cumulative release, not sample concentration.
- Read performance by target and protocol. Inspect RMSE, MAE and individual-profile errors alongside pooled R2. Seed variability describes model-run variation, not uncertainty across independent laboratory experiments. The comparisons do not establish consistent graph-model superiority.
- Supplied release values are retained without an extra multiplication by 100. Numerical errors follow that analysis scale; interpreting it as absolute percentage points requires confirmation of the experimental percentage convention.

## Optional exploratory predictions

The existing notebook and shared core retain a full-data neural fit generating pH 6.3 curves and predictions through 120 h. These are labelled unvalidated interpolation/extrapolation and are separate from the held-out evaluations reported in the revised ML results. Their presence does not establish a digital twin. They do not reproduce the manuscript's separate 120 h saturating-exponential bootstrap analysis, which is outside this code's scope.

## Implementation checks

```bash
python scripts/check_release_equations.py
python scripts/graph_sensitivity/check_graph_model.py
```

The equation checks use known generated curves to test parameter recovery and fallback behaviour. These values are not experimental data and do not generate the paper's results. The graph checks require the chemical-input JSON and examine permutation invariance, stereochemical sensitivity, gradients and constrained output shape. Passing these checks establishes specific software behaviour, not scientific validity or independent experimental confirmation.

## Earlier implementation

Superseded notebooks, figures, saved model weights and tracked inputs are removed from the revised repository version to avoid confusing them with the current analyses. Existing local copies can be retained. The revised work is exploratory development following earlier analyses and should be interpreted accordingly.
