"""
Supervised Step 18: Validation checks and research-question alignment.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import pandas as pd

from supervised_utils import DATA_DIR, RESULT_DIR, load_state

print("\n============================================================")
print("Supervised Step 18: Validation Checks")
print("============================================================\n")

state = load_state(17)
df = state["df"]
metrics = pd.read_csv(RESULT_DIR / "supervised_model_metrics.csv")

assert set(df["Split"].unique()) == {"Train", "Validation", "Test"}
assert "Target" in df.columns
assert df["Target"].sum() > 0
assert metrics["Model"].nunique() == 3
assert (RESULT_DIR / "supervised_robustness_results.csv").exists()
assert (RESULT_DIR / "supervised_stress_scenario_results.csv").exists()
assert (RESULT_DIR / "shap_feature_importance_test_sample.csv").exists()

train_max = df.loc[df["Split"] == "Train", "Date"].max()
val_min = df.loc[df["Split"] == "Validation", "Date"].min()
val_max = df.loc[df["Split"] == "Validation", "Date"].max()
test_min = df.loc[df["Split"] == "Test", "Date"].min()

print("Validation checks passed.")
print(f"Train ends on: {pd.Timestamp(train_max).date()}")
print(f"Validation starts on: {pd.Timestamp(val_min).date()} and ends on: {pd.Timestamp(val_max).date()}")
print(f"Test starts on: {pd.Timestamp(test_min).date()}")
print(f"Supervised drawdown event rows: {int(df['Target'].sum())}")

rq = pd.DataFrame(
    {
        "Research_Question_Area": [
            "Supervised financial risk prediction",
            "Behaviour under abnormal inputs",
            "Predictive resilience",
            "Explainability stability",
            "Dissertation reporting",
        ],
        "Implemented_Evidence": [
            "Logistic Regression, Random Forest, and XGBoost/fallback classifiers",
            "Gaussian perturbation plus named stress scenarios",
            "Accuracy, Precision, Recall, F1, ROC-AUC, F1 drop, and AUC drop",
            "Tree SHAP feature importance and perturbation stability",
            "Summary tables, figures, and narrative alignment",
        ],
        "Primary_Output": [
            str(RESULT_DIR / "supervised_model_metrics.csv"),
            str(RESULT_DIR / "supervised_stress_scenario_results.csv"),
            str(RESULT_DIR / "supervised_robustness_results.csv"),
            str(RESULT_DIR / "shap_stability_all_levels_standardized.csv"),
            str(RESULT_DIR / "final_model_summary_standardized.csv"),
        ],
    }
)
rq_path = RESULT_DIR / "research_question_alignment.csv"
rq.to_csv(rq_path, index=False)
print(f"Research question alignment saved to: {rq_path}")
print(f"Supervised dataset: {DATA_DIR / 'supervised_drawdown_dataset.csv'}")

print("\nSupervised Step 18 completed successfully!")
