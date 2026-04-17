"""
Supervised Step 16: Final summary and failure analysis.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import pandas as pd

from supervised_utils import RESULT_DIR, load_state, save_state

print("\n============================================================")
print("Supervised Step 16: Final Summary")
print("============================================================\n")

state = load_state(15)
metrics = pd.read_csv(RESULT_DIR / "supervised_model_metrics.csv")
test_metrics = metrics[metrics["Split"] == "Test"].copy()
robustness = state["robustness_df"]
stress = state["stress_df"]

avg_robustness = robustness.groupby("Model")["Robustness_Score"].mean()
worst_stress = stress.groupby("Model")["Stressed_F1"].min()
avg_f1_drop = robustness.groupby("Model")["F1_Drop"].mean()

summary = test_metrics.merge(avg_robustness.rename("Avg_Gaussian_Robustness"), on="Model", how="left")
summary = summary.merge(worst_stress.rename("Worst_Stress_F1"), on="Model", how="left")
summary = summary.merge(avg_f1_drop.rename("Avg_F1_Drop_Under_Noise"), on="Model", how="left")
summary_path = RESULT_DIR / "final_model_summary_standardized.csv"
summary.to_csv(summary_path, index=False)

failure = summary[["Model", "Precision", "Recall", "F1_Score", "ROC_AUC", "Avg_Gaussian_Robustness", "Worst_Stress_F1"]].copy()
failure["Interpretation"] = failure.apply(
    lambda row: "Strong candidate" if row["F1_Score"] == summary["F1_Score"].max() else "Comparator model for trade-off analysis",
    axis=1,
)
failure_path = RESULT_DIR / "model_failure_analysis_standardized.csv"
failure.to_csv(failure_path, index=False)

print(summary.to_string(index=False))
print(f"\nFinal summary saved to: {summary_path}")

save_state(
    16,
    df=state["df"],
    feature_cols=state["feature_cols"],
    models=state["models"],
    final_summary=summary,
    failure_analysis=failure,
    robustness_df=robustness,
    stress_df=stress,
    shap_stability_df=state["shap_stability_df"],
)

print("\nSupervised Step 16 completed successfully!")
