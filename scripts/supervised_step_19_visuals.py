"""
Supervised Step 19: Core supervised result visualisations.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from supervised_utils import FIGURE_DIR, RESULT_DIR

print("\n============================================================")
print("Supervised Step 19: Visualisations")
print("============================================================\n")

metrics = pd.read_csv(RESULT_DIR / "supervised_model_metrics.csv")
test = metrics[metrics["Split"] == "Test"].copy()

plt.figure(figsize=(9, 5))
plot = test.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"], var_name="Metric", value_name="Score")
sns.barplot(data=plot, x="Metric", y="Score", hue="Model")
plt.title("Supervised Drawdown Classifier Test Metrics")
plt.ylim(0, 1)
plt.tight_layout()
metric_fig = FIGURE_DIR / "supervised_model_test_metrics.png"
plt.savefig(metric_fig, dpi=300)
plt.close()

matrix = test.set_index("Model")[["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]]
plt.figure(figsize=(8, 4.8))
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
plt.title("Supervised Performance Heatmap")
plt.tight_layout()
heatmap_fig = FIGURE_DIR / "supervised_model_performance_heatmap.png"
plt.savefig(heatmap_fig, dpi=300)
plt.close()

robust = pd.read_csv(RESULT_DIR / "supervised_robustness_results.csv")
plt.figure(figsize=(9, 5))
sns.lineplot(data=robust, x="Perturbation_Level", y="F1_Drop", hue="Model", marker="o")
plt.title("F1 Drop Under Noise")
plt.tight_layout()
robust_fig = FIGURE_DIR / "supervised_f1_drop_under_noise.png"
plt.savefig(robust_fig, dpi=300)
plt.close()

print(f"Metric figure saved to: {metric_fig}")
print(f"Heatmap saved to: {heatmap_fig}")
print(f"Robustness figure saved to: {robust_fig}")

print("\nSupervised Step 19 completed successfully!")
