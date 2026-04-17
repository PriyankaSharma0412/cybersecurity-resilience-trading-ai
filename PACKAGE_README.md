# Financial AI Robustness Supervised Submission Package

This package contains the complete supervised financial drawdown-risk prediction
project code and generated outputs.

## Project Status

The active pipeline predicts whether an asset will experience a significant
negative 5-day return.

Target definition:

```python
Future_Return_5D = next 5-day return
Target = 1 if Future_Return_5D <= -0.03
Target = 0 otherwise
```

## Folder Layout

```text
Financial_AI_Robustness_Supervised_Submission_Package/
├── 01_documentation/
│   ├── FINANCIAL_AI_ROBUSTNESS_DFD_AND_DOCUMENTATION.md
│   ├── SUPERVISED_CONVERSION_SUMMARY.md
│   └── PACKAGE_README.md
├── 02_code/
│   ├── run_all.py
│   ├── check_syntax.py
│   ├── split_notebook.py
│   ├── extract_*.py / helper scripts
│   └── scripts/
├── 03_notebook_and_report/
│   ├── Updated_Financial_AI_Robustness_Evaluation.ipynb
│   ├── Financial_AI_Robustness_Results.doc
│   └── Info.docx
└── 04_outputs/
    └── dissertation_outputs/
        ├── data/
        ├── results/
        ├── figures/
        ├── models/
        └── state/
```

## How to Run

From the project root after extracting the package, use the project virtual
environment if available:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

If the virtual environment is not included, create/activate an environment and
install the libraries used by the scripts, including:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap
- yfinance
- joblib
- tensorflow
- xgboost, optional

Note: `xgboost` was not installed during the verified run, so the pipeline used
the built-in Gradient Boosting fallback.

## Important Outputs

Key supervised outputs:

- `04_outputs/dissertation_outputs/data/supervised_drawdown_dataset.csv`
- `04_outputs/dissertation_outputs/results/supervised_model_metrics.csv`
- `04_outputs/dissertation_outputs/results/supervised_drawdown_predictions.csv`
- `04_outputs/dissertation_outputs/results/supervised_robustness_results.csv`
- `04_outputs/dissertation_outputs/results/supervised_stress_scenario_results.csv`
- `04_outputs/dissertation_outputs/results/shap_feature_importance_test_sample.csv`
- `04_outputs/dissertation_outputs/results/shap_stability_all_levels_standardized.csv`
- `04_outputs/dissertation_outputs/results/final_model_summary_standardized.csv`
- `04_outputs/dissertation_outputs/figures/supervised_model_test_metrics.png`
- `04_outputs/dissertation_outputs/figures/supervised_model_performance_heatmap.png`
- `04_outputs/dissertation_outputs/figures/chapter4_visuals/01_supervised_framework_pipeline_diagram.png`

## Verified Headline Results

Current test-set results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.478 | 0.235 | 0.745 | 0.358 | 0.612 |
| Random Forest | 0.805 | 0.600 | 0.006 | 0.012 | 0.549 |
| Gradient Boosting Fallback | 0.777 | 0.231 | 0.061 | 0.097 | 0.544 |

## Notes

- The old unsupervised scripts are retained in `02_code/scripts/` as historical
  code, but `run_all.py` now executes the supervised pipeline.
- `.venv`, `tmp`, `.mplconfig`, and `__pycache__` folders are intentionally not
  included in the package.
- The output folders include generated CSVs, figures, trained model files, and
  intermediate state files.
