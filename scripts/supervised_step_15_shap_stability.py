"""
Supervised Step 15: SHAP explanation stability under perturbation.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

from supervised_utils import FIGURE_DIR, RESULT_DIR, load_state, save_state

print("\n============================================================")
print("Supervised Step 15: SHAP Stability")
print("============================================================\n")

state = load_state(14)
X_shap = state["X_shap"]
model = state["shap_model"]
feature_cols = state["feature_cols"]
base_values = state["base_shap_values"]
base_importance = pd.Series(np.abs(base_values).mean(axis=0), index=feature_cols)
base_rank = base_importance.rank(ascending=False)
base_top = set(base_importance.sort_values(ascending=False).head(5).index)

explainer = shap.TreeExplainer(model)
rows = []
for level in [0.05, 0.10, 0.15, 0.20, 0.30]:
    noise = np.random.normal(0, level, X_shap.shape)
    perturbed = pd.DataFrame(X_shap.values + noise, columns=feature_cols, index=X_shap.index)
    values = explainer.shap_values(perturbed)
    if isinstance(values, list):
        values = values[-1]
    importance = pd.Series(np.abs(values).mean(axis=0), index=feature_cols)
    rank = importance.rank(ascending=False)
    rho, pvalue = spearmanr(base_rank, rank)
    top = set(importance.sort_values(ascending=False).head(5).index)
    rows.append(
        {
            "Perturbation_Level": level,
            "Spearman_Rho": rho,
            "Spearman_pvalue": pvalue,
            "Top_5_Overlap": len(base_top.intersection(top)) / 5,
        }
    )

stability_df = pd.DataFrame(rows)
stability_path = RESULT_DIR / "shap_stability_all_levels_standardized.csv"
stability_df.to_csv(stability_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(stability_df["Perturbation_Level"], stability_df["Spearman_Rho"], marker="o")
plt.title("Supervised SHAP Rank Stability Across Perturbation Levels")
plt.xlabel("Perturbation Level")
plt.ylabel("Spearman Rank Correlation")
plt.tight_layout()
fig_path = FIGURE_DIR / "shap_stability_vs_perturbation_standardized.png"
plt.savefig(fig_path, dpi=300)
plt.close()

single = pd.DataFrame(
    {
        "Metric": ["Spearman_Rank_Correlation", "Top_5_Feature_Overlap"],
        "Value": [
            stability_df.loc[stability_df["Perturbation_Level"] == 0.10, "Spearman_Rho"].iloc[0],
            stability_df.loc[stability_df["Perturbation_Level"] == 0.10, "Top_5_Overlap"].iloc[0],
        ],
    }
)
single.to_csv(RESULT_DIR / "explainability_stability_single_level.csv", index=False)

print(stability_df.to_string(index=False))

save_state(
    15,
    df=state["df"],
    feature_cols=feature_cols,
    models=state["models"],
    robustness_df=state["robustness_df"],
    stress_df=state["stress_df"],
    shap_stability_df=stability_df,
)

print("\nSupervised Step 15 completed successfully!")
