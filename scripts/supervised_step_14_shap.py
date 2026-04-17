"""
Supervised Step 14: SHAP explainability for the strongest tree classifier.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from supervised_utils import FIGURE_DIR, RESULT_DIR, load_state, save_state, split_xy

print("\n============================================================")
print("Supervised Step 14: SHAP Explainability")
print("============================================================\n")

state = load_state(13)
df = state["df"]
feature_cols = state["feature_cols"]
models = state["models"]
metrics_df = pd.read_csv(RESULT_DIR / "supervised_model_metrics.csv")
best_tree_name = metrics_df[(metrics_df["Split"] == "Test") & (metrics_df["Model"] != "Logistic Regression")].sort_values("F1_Score", ascending=False).iloc[0]["Model"]
model = models[best_tree_name]
splits = split_xy(df, feature_cols)
X_test, _, _ = splits["Test"]
X_shap = X_test.sample(min(500, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_values = shap_values[-1]

shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
summary_path = FIGURE_DIR / "shap_summary_plot_test_sample.png"
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()

importance = (
    pd.DataFrame({"Feature": feature_cols, "Mean_Abs_SHAP": np.abs(shap_values).mean(axis=0)})
    .sort_values("Mean_Abs_SHAP", ascending=False)
)
importance["Model"] = best_tree_name
shap_path = RESULT_DIR / "shap_feature_importance_test_sample.csv"
importance.to_csv(shap_path, index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance, x="Mean_Abs_SHAP", y="Feature")
plt.title(f"Mean Absolute SHAP Feature Importance: {best_tree_name}")
plt.tight_layout()
bar_path = FIGURE_DIR / "shap_feature_importance_bar_test_sample.png"
plt.savefig(bar_path, dpi=300)
plt.close()

print(f"SHAP model: {best_tree_name}")
print(f"SHAP importance saved to: {shap_path}")

save_state(
    14,
    df=df,
    feature_cols=feature_cols,
    models=models,
    shap_model_name=best_tree_name,
    shap_model=model,
    X_shap=X_shap,
    base_shap_values=shap_values,
    stress_df=state["stress_df"],
    robustness_df=state["robustness_df"],
)

print("\nSupervised Step 14 completed successfully!")
