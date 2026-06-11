# Financial AI Robustness Evaluation

This repository is organised around `Updated_Financial_AI_Robustness_Evaluation.ipynb` and one consolidated supervised drawdown-risk prediction pipeline.

## Structure

```text
.
|-- Updated_Financial_AI_Robustness_Evaluation.ipynb
|-- run_all.py
|-- scripts/
|   |-- README.md
|   `-- supervised_pipeline.py
|-- dissertation_outputs/
|   |-- data/
|   |-- results/
|   |-- figures/
|   |-- models/
|   `-- state/
|-- docs/
`-- reports/
```

## Run

```powershell
.\.venv\Scripts\python.exe run_all.py
```

or directly:

```powershell
.\.venv\Scripts\python.exe scripts\supervised_pipeline.py
```

The supervised target is `Target = 1` when the next 5-day return is less than or equal to -3%.

The consolidated pipeline also exports advanced validation artifacts including
threshold optimisation, PR/ROC/calibration curves, TimeSeriesSplit validation,
rolling backtests, baseline benchmarks, imbalance handling, hyperparameter
tuning, leakage checks, feature selection, permutation importance, SHAP model
comparison, SHAP drift, McNemar tests, cost-sensitive metrics, economic
backtests, robustness stress tests, ensembles, runtime benchmarks, and an
automated final summary dashboard.
