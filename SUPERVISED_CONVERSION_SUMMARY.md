# Supervised Conversion Summary

The project has been converted to a supervised financial drawdown-risk
prediction pipeline.

## What Changed

The original active modelling route predicted anomalies without labels. The
new active route uses:

```python
X = feature_cols
y = Target
```

The target is:

```python
df["Future_Return_5D"] = df.groupby("Ticker")["Close"].shift(-5) / df["Close"] - 1
df["Target"] = (df["Future_Return_5D"] <= -0.03).astype(int)
```

`Target = 1` means the asset enters a 5-day drawdown-risk event.

## What Stayed

The pipeline still keeps:

- Setup and imports.
- Yahoo Finance data collection.
- EDA.
- Feature engineering.
- `feature_cols`.
- Chronological train/validation/test splitting.
- Robustness testing.
- Stress scenario testing.
- SHAP explainability.
- Export paths under `dissertation_outputs/`.
- Chapter 4 visual outputs.

## Active Supervised Scripts

`run_all.py` now runs the original Steps 1-6, then:

| Step | Script |
|---|---|
| 7 | `scripts/supervised_step_07_target_split.py` |
| 8 | `scripts/supervised_step_08_logistic.py` |
| 9 | `scripts/supervised_step_09_random_forest.py` |
| 10 | `scripts/supervised_step_10_xgboost.py` |
| 11 | `scripts/supervised_step_11_evaluation.py` |
| 12 | `scripts/supervised_step_12_robustness.py` |
| 13 | `scripts/supervised_step_13_stress.py` |
| 14 | `scripts/supervised_step_14_shap.py` |
| 15 | `scripts/supervised_step_15_shap_stability.py` |
| 16 | `scripts/supervised_step_16_summary.py` |
| 17 | `scripts/supervised_step_17_narrative.py` |
| 18 | `scripts/supervised_step_18_validation.py` |
| 19 | `scripts/supervised_step_19_visuals.py` |
| 20 | `scripts/supervised_step_20_chapter4_story.py` |

## Models

The active supervised models are:

- Logistic Regression with balanced class weights.
- Random Forest with balanced class weights.
- XGBoost when installed.
- Gradient Boosting fallback when `xgboost` is not installed.

In this environment, `xgboost` is not installed, so the verified run used the
Gradient Boosting fallback.

## Verified Run

The supervised Step 7-20 pipeline ran successfully with:

```powershell
.\.venv\Scripts\python.exe
```

Current test-set headline metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.478 | 0.235 | 0.745 | 0.358 | 0.612 |
| Random Forest | 0.805 | 0.600 | 0.006 | 0.012 | 0.549 |
| Gradient Boosting Fallback | 0.777 | 0.231 | 0.061 | 0.097 | 0.544 |

Main generated files:

- `dissertation_outputs/data/supervised_drawdown_dataset.csv`
- `dissertation_outputs/results/supervised_model_metrics.csv`
- `dissertation_outputs/results/supervised_robustness_results.csv`
- `dissertation_outputs/results/supervised_stress_scenario_results.csv`
- `dissertation_outputs/results/shap_feature_importance_test_sample.csv`
- `dissertation_outputs/results/final_model_summary_standardized.csv`
- `dissertation_outputs/figures/supervised_model_test_metrics.png`
- `dissertation_outputs/figures/supervised_model_performance_heatmap.png`
- `dissertation_outputs/figures/chapter4_visuals/01_supervised_framework_pipeline_diagram.png`

## How To Run

Use the virtual environment:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

Running with the system Python is not enough in this workspace because it is
missing project dependencies such as `joblib`.
