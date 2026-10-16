# Code and data for low-GWP gas-insulated current transformers

This project studies how electrode geometry, gas composition and filling pressure affect candidate insulation designs for a 500 kV current transformer. It supplies the numerical inputs, implemented equations and saved experiments needed to recalculate the study or explore new screening cases.

The code and data accompany **Electric Field Uniformity Optimization and Gas Mixture Candidate Selection for 500 kV Current Transformers Insulated with Low GWP Gas**.

## What the project does

The workflow starts with eleven Morris designs for seven geometric inputs. A four-input Gaussian-process surrogate is validated against independent COMSOL cases and used for Sobol analysis and a five-objective NSGA-II search. AHP–entropy weights rank the candidates; Monte Carlo sampling and mesh comparisons examine their conditional margins and numerical limitations.

The independent surrogate validation gives R² approximately 0.996. The highest-ranked candidate in the 67-point reference subset contains 64.6% N2, 20.7% O2 and 14.7% He at approximately 0.79 MPa absolute pressure. Independent searches identify a similar high-pressure, N2-dominant region while selecting different helium fractions. These numerical screening results identify a region for subsequent physical testing.

## Quick start

Use Python 3.12. Open a terminal in the directory containing `main.py` and create an environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies and check the files:

```bash
python -m pip install -r requirements.txt
python main.py verify
```

Start by reproducing the independent surrogate validation:

```bash
python main.py analyze --task surrogate
```

Windows users can also run the environment's interpreter directly, without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py verify
.\.venv\Scripts\python.exe main.py analyze --task surrogate
.\.venv\Scripts\python.exe main.py analyze
```

On Linux/macOS the equivalent interpreter is `.venv/bin/python`. The remaining examples use `python` to mean this environment's interpreter.

Read `results/surrogate/validation_metrics.json` for R² and errors, and `results/surrogate/validation_predictions.csv` for all 30 independent comparisons. COMSOL is unnecessary for these commands. `verify` uses only the Python standard library and can also run before installation.

To recalculate all saved-data analyses:

```bash
python main.py analyze
```

This creates `results/summary.json`, task-specific CSV/JSON results and plots. It reads the supplied inputs without launching COMSOL or a new optimization. Running `python main.py` performs the same complete analysis. Generated results are not required inputs.

## Project structure

```text
main.py                 Command-line entry point
src/                    Analysis, scoring, sampling and simulation routines
src/comsol_java/        Java templates for new finite-element calculations
config/                 Physical inputs, scoring rules and experiment settings
data/morris/            Eleven 80-point designs and finite-element responses
data/surrogate/         100 training and 30 independent validation cases
data/reference/         Reference candidates, fixed weights and gas-property sources
data/optimization/      Final fronts, search histories and evaluated-objective traces
data/uncertainty/       67 candidates × 2,000 conditional Monte Carlo draws
data/mesh/              Mesh statistics, peak coordinates and field profiles
models/surrogate/       Fitted surrogate and model description
models/comsol/          Eleven solved COMSOL files
docs/data_dictionary.md Column definitions, units and output interpretation
CHECKSUMS.csv           SHA-256 checksums of the supplied project files
.gitattributes          Preserve file bytes across Git checkouts
results/                Created by analysis; not a required input
```

Identical numerical records are stored once. Explicit scenario mappings identify equivalent constraints, and longer search traces include their corresponding 200-generation prefixes. The repeated Morris reference design uses the same stored inputs and responses.

## Reproduce a particular result

Run `python main.py analyze --task TASK` with one of the following task names:

| Task | Calculation | Main result to inspect |
|---|---|---|
| `surrogate` | Validation and training leave-one-out estimates | `results/surrogate/validation_metrics.json` |
| `morris` | Elementary effects, ranking stability and trajectory resampling | `results/morris/summary.json` |
| `sobol` | First-order and total-effect sensitivity indices | `results/sensitivity/sobol_indices.csv` |
| `ranking` | AHP–entropy weights, candidate rankings and preference/input scenarios | `results/ranking/ranking_summary.json` |
| `scan` | Helium scan at fixed P01 geometry, pressure and N2:O2 ratio | `results/scan/controlled_scan_summary.json` |
| `optimization` | Hypervolumes, leaders and threshold comparisons | `results/optimization/optimization_summary.json` |
| `uncertainty` | Conditional margins and GP-scale comparison | `results/uncertainty/uncertainty_summary.json` |
| `mesh` | Mesh refinement, peak locations and profile differences | `results/mesh/local_mesh_peak_summary.csv` |

The [data and result guide](docs/data_dictionary.md) defines the keys and units and maps these outputs to the paper's tables and figures. Each task recalculates its derived outputs, so additional copies of generated tables and plots are unnecessary.

Choose a separate output directory with `--out`, for example:

```bash
python main.py analyze --task mesh --out my_results
```

## Run new calculations

```bash
python main.py fit --out generated
python main.py optimize --seed 20260816 --generations 200 --out generated
python main.py sample --seed 20260815 --samples 2000 --out generated
```

`fit` refits the surrogate from the training CSV. `optimize` runs a new NSGA-II search using the supplied fitted model; `--scenario` selects a threshold setting from `config/optimization.json`. `sample` generates a new Monte Carlo draw table using `config/uncertainty.json`.

Use the supplied fitted model for direct comparison with saved experiments; refitting can introduce small numerical differences. Configuration files specify the physical/scoring inputs and experiment settings. Numerical seeds identify designs and searches. New outputs remain separate from the supplied files.

## Generate finite-element data with COMSOL

New solutions require a licensed COMSOL Multiphysics 6.2 installation with the AC/DC Module. Prepare Java jobs first:

```bash
python main.py comsol --study local-study --out generated
python main.py comsol --study morris --seed 20260814 --out generated
python main.py comsol --study training --out generated
python main.py comsol --study export-fields --case rounded_R2 --level L03 --out generated
```

Add `--execute --comsol-root PATH_TO_COMSOL` to compile and run, replacing `PATH_TO_COMSOL` with your installation directory. `local-study` solves the two reference end shapes, `morris` and `training` create sampling calculations, and `export-fields` reads a solved model to export its full field without a new solve.

Executed jobs use a separate COMSOL preferences directory under the chosen output directory. This profile permits the model reads and data writes required by the Java jobs and disables external network/process capabilities. Completion is checked against the saved model and expected data files; a process return code alone is insufficient.

The eleven supplied MPH files retain their solved fields. Sampling models retain the last solved field and their complete scalar records; they do not store a full field for every sample. The supplied CSV files retain the mesh and profile measurements used in the analyses; full-domain field-point CSVs can be exported from the models when needed.

## Interpretation

The seven-input Morris geometry and four-input surrogate geometry are distinct equivalent axisymmetric models. The surrogate predicts the recorded rectangular-end, medium-mesh response. The rounded-end comparison is a separate reference study and does not establish the mesh error over the entire design space.

The mixture-breakdown rule, PD/impulse margins, thermal/detection indicators, costs and uncertainty distributions are declared screening assumptions. The calculations do not qualify a complete 500 kV device or determine a unique helium fraction. Device design requires corresponding breakdown data, resolved geometry and prototype tests.

## Citation

Use the associated paper's final publication citation when available. Authors: Shengbao Zhang, Hong Shen, Shixiang Wang, Xing Wang, Xiao Bin, Yang Zhao, Shuo Tong, Ke He and Jun Jia. Corresponding author: jiajun@tsinghua-eiri.org.

Python dependencies and COMSOL are distributed separately under their respective terms.
