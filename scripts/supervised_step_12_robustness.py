"""
Supervised Step 12: Robustness testing under Gaussian perturbation.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score

from supervised_utils import FIGURE_DIR, RESULT_DIR, load_state, predict_proba, save_state, split_xy

print("\n============================================================")
print("Supervised Step 12: Robustness Testing")
print("============================================================\n")

state = load_state(11)
df = state["df"]
feature_cols = state["feature_cols"]
models = state["models"]
splits = split_xy(df, feature_cols)
X_test, y_test, _ = splits["Test"]
levels = [0.05, 0.10, 0.15, 0.20, 0.30]

baseline = {}
for model_name, model in models.items():
    proba = predict_proba(model, X_test)
    pred = (proba >= 0.5).astype(int)
    baseline[model_name] = {
        "pred": pred,
        "proba": proba,
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, proba),
    }

rows = []
for level in levels:
    noise = np.random.normal(0, level, X_test.shape)
    X_perturbed = pd.DataFrame(X_test.values + noise, columns=feature_cols, index=X_test.index)
    for model_name, model in models.items():
        proba = predict_proba(model, X_perturbed)
        pred = (proba >= 0.5).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        flips = int(np.sum(baseline[model_name]["pred"] != pred))
        rows.append(
            {
                "Model": model_name,
                "Perturbation_Level": level,
                "Prediction_Flips": flips,
                "Flip_Rate": flips / len(X_test),
                "Baseline_F1": baseline[model_name]["f1"],
                "Perturbed_F1": f1,
                "F1_Drop": baseline[model_name]["f1"] - f1,
                "Baseline_ROC_AUC": baseline[model_name]["auc"],
                "Perturbed_ROC_AUC": auc,
                "AUC_Drop": baseline[model_name]["auc"] - auc,
                "Robustness_Score": 1 - (flips / len(X_test)),
            }
        )

results_df = pd.DataFrame(rows)
robustness_path = RESULT_DIR / "supervised_robustness_results.csv"
results_df.to_csv(robustness_path, index=False)
results_df.to_csv(RESULT_DIR / "perturbation_robustness_results_standardized.csv", index=False)

plt.figure(figsize=(10, 6))
for model_name, group in results_df.groupby("Model"):
    plt.plot(group["Perturbation_Level"], group["Prediction_Flips"], marker="o", label=model_name)
plt.title("Prediction Flips Under Gaussian Perturbation")
plt.xlabel("Perturbation Level")
plt.ylabel("Prediction Flips")
plt.legend()
plt.tight_layout()
flips_path = FIGURE_DIR / "prediction_flips_vs_gaussian_perturbation_supervised.png"
plt.savefig(flips_path, dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for model_name, group in results_df.groupby("Model"):
    plt.plot(group["Perturbation_Level"], group["F1_Drop"], marker="o", label=model_name)
plt.title("F1 Drop Under Gaussian Perturbation")
plt.xlabel("Perturbation Level")
plt.ylabel("F1 Drop")
plt.legend()
plt.tight_layout()
f1_path = FIGURE_DIR / "f1_drop_vs_gaussian_perturbation_supervised.png"
plt.savefig(f1_path, dpi=300)
plt.close()

print(results_df.to_string(index=False))
print(f"\nRobustness results saved to: {robustness_path}")

save_state(12, df=df, feature_cols=feature_cols, models=models, robustness_df=results_df)

print("\nSupervised Step 12 completed successfully!")
