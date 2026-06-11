# Scripts

The active project pipeline is now merged into one file:

```text
scripts/supervised_pipeline.py
```

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\supervised_pipeline.py
```

or:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

## Pipeline Stages

1. Environment setup
2. Data import from the saved supervised dataset, or Yahoo Finance if needed
3. Feature engineering
4. Supervised target creation
5. Chronological train/validation/test split
6. Logistic Regression, Random Forest, and XGBoost/fallback training
7. Accuracy, Precision, Recall, F1, and ROC-AUC evaluation
8. Robustness testing under Gaussian noise
9. Financial stress scenario testing
10. SHAP explainability and SHAP stability
11. Threshold optimisation on validation data
12. Precision-recall, ROC, calibration, Brier, and PR-AUC diagnostics
13. Naive, historical volatility, and moving-average baselines
14. TimeSeriesSplit CV, rolling backtest, SMOTE/fallback imbalance handling, and hyperparameter tuning
15. Feature leakage checks, feature selection, permutation importance, SHAP comparison, and SHAP drift
16. McNemar/statistical testing, cost-sensitive evaluation, economic utility, and risk backtests
17. Missing-data, feature-corruption, regime-shift, correlation-breakdown, and flash-crash stress tests
18. Voting/stacking ensembles, runtime, latency, memory, final dashboard, and Chapter 4 visual manifest

## Target

```python
Target = 1 when the next 5-day return is <= -3%, else 0
```

Generated data, result tables, figures, models, and state files are saved under
`dissertation_outputs/`.
