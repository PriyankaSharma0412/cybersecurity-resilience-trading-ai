# Financial AI Robustness Evaluation — Step-by-Step Scripts

All 18 original notebook steps plus framework-extension and visual-story steps have been split into separate Python scripts in the `scripts/` folder.

---

## 📁 Folder Structure

```
c:\Code\
├── scripts\
│   ├── step_01_environment_setup.py
│   ├── step_02_install_import_libraries.py
│   ├── step_03_data_collection.py
│   ├── step_04_exploratory_data_analysis.py
│   ├── step_05_feature_engineering.py
│   ├── step_06_prepare_model_features.py
│   ├── step_07_time_series_evaluation.py
│   ├── step_08_isolation_forest.py
│   ├── step_09_autoencoder.py
│   ├── step_10_one_class_svm.py
│   ├── step_11_supervised_evaluation.py
│   ├── step_12_robustness_testing.py
│   ├── step_13_stress_scenarios.py
│   ├── step_14_shap_explainability.py
│   ├── step_15_explanation_stability.py
│   ├── step_16_final_summary.py
│   ├── step_17_dissertation_narrative.py
│   ├── step_18_validation_checks.py
│   ├── step_19_framework_extensions.py
│   └── step_20_chapter4_visual_story.py
│
├── dissertation_outputs\
│   ├── data\          ← CSV files saved here
│   ├── figures\       ← All plots/charts saved here (PNG)
│   ├── models\        ← Trained ML models saved here
│   ├── results\       ← Result tables saved here
│   └── state\         ← Intermediate state files (auto-created)
```

---

## ▶️ How to Run

### First time (run all steps in order):

Open a terminal in `c:\Code\` and run each script in order:

```
python scripts\step_01_environment_setup.py
python scripts\step_02_install_import_libraries.py
python scripts\step_03_data_collection.py
python scripts\step_04_exploratory_data_analysis.py
python scripts\step_05_feature_engineering.py
python scripts\step_06_prepare_model_features.py
python scripts\step_07_time_series_evaluation.py
python scripts\step_08_isolation_forest.py
python scripts\step_09_autoencoder.py
python scripts\step_10_one_class_svm.py
python scripts\step_11_supervised_evaluation.py
python scripts\step_12_robustness_testing.py
python scripts\step_13_stress_scenarios.py
python scripts\step_14_shap_explainability.py
python scripts\step_15_explanation_stability.py
python scripts\step_16_final_summary.py
python scripts\step_17_dissertation_narrative.py
python scripts\step_18_validation_checks.py
python scripts\step_19_framework_extensions.py
python scripts\step_20_chapter4_visual_story.py
```

### Re-running a single step:

Each script saves its output and loads from the previous step's saved state.
To re-run e.g. Step 8 (Isolation Forest), you only need Steps 1-7 to have been run once:

```
python scripts\step_08_isolation_forest.py
```

---

## 📸 Taking Screenshots

Each script:
- Prints clear **console output** you can screenshot
- Saves all **figures as PNG** files to `dissertation_outputs\figures\`
- Prints the path of each figure when it's saved

### What each step outputs:

| Step | Script | Output |
|------|--------|--------|
| 1 | `step_01_environment_setup.py` | Directory structure confirmation |
| 2 | `step_02_install_import_libraries.py` | Import success messages |
| 3 | `step_03_data_collection.py` | Data shape `(3522, 60)`, head table, CSV saved |
| 4 | `step_04_exploratory_data_analysis.py` | 3 figures: stock trends, returns distribution, correlation heatmap |
| 5 | `step_05_feature_engineering.py` | Feature table, 2 figures: feature correlation heatmap, rolling volatility |
| 6 | `step_06_prepare_model_features.py` | Scaled feature confirmation |
| 7 | `step_07_time_series_evaluation.py` | Train/Validation/Test split info, labels saved |
| 8 | `step_08_isolation_forest.py` | Anomaly counts table, figure: IF anomalies on AAPL |
| 9 | `step_09_autoencoder.py` | Training progress, reconstruction errors, figure |
| 10 | `step_10_one_class_svm.py` | Training loss figure, multi-model anomaly comparison figure |
| 11 | `step_11_supervised_evaluation.py` | Metrics tables, confusion matrices, overlap analysis |
| 12 | `step_12_robustness_testing.py` | Robustness results table, 2 figures: label flips + Jaccard similarity |
| 13 | `step_13_stress_scenarios.py` | Stress scenario results table |
| 14 | `step_14_shap_explainability.py` | SHAP summary + bar plots |
| 15 | `step_15_explanation_stability.py` | SHAP stability plots, rank comparison |
| 16 | `step_16_final_summary.py` | Final summary table + failure analysis |
| 17 | `step_17_dissertation_narrative.py` | Narrative mapping saved |
| 18 | `step_18_validation_checks.py` | Output snapshot, file paths, research alignment table |
| 19 | `step_19_framework_extensions.py` | Adaptive thresholds, majority-vote ensemble, supervised early-warning classifier, ROC/confusion figures, runtime benchmark, Autoencoder/OC-SVM SHAP, multi-asset SHAP stability, stress timeline, McNemar tests |
| 20 | `step_20_chapter4_visual_story.py` | Granular Chapter 4 visuals: framework diagram, metric heatmaps, precision-recall bubble chart, ensemble lift, runtime-vs-F1, robustness/stress heatmaps, SHAP comparison heatmaps, supervised summary panel, evidence dashboard |

---

## ⚠️ Notes

- **Run Steps 1-7 at least once before jumping ahead** — later steps depend on saved state from earlier steps.
- **Figures are saved to PNG** (not shown interactively) so you can open and screenshot them from `dissertation_outputs\figures\`.
- All state is saved to `dissertation_outputs\state\` as pickle files — these carry variables between steps.
- The `.venv` environment must be active with all packages installed (`yfinance`, `shap`, `tensorflow`, `sklearn`, etc.).
