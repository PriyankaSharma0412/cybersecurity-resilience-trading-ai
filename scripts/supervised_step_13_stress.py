"""
Supervised Step 13: Financial stress scenario testing.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import f1_score, roc_auc_score

from supervised_utils import FIGURE_DIR, RESULT_DIR, load_state, make_stress_scenarios, predict_proba, save_state, split_xy

print("\n============================================================")
print("Supervised Step 13: Stress Scenario Testing")
print("============================================================\n")

state = load_state(12)
df = state["df"]
feature_cols = state["feature_cols"]
models = state["models"]
splits = split_xy(df, feature_cols)
X_test, y_test, _ = splits["Test"]
scenarios = make_stress_scenarios(X_test)

rows = []
for model_name, model in models.items():
    base_proba = predict_proba(model, X_test)
    base_pred = (base_proba >= 0.5).astype(int)
    base_f1 = f1_score(y_test, base_pred, zero_division=0)
    base_auc = roc_auc_score(y_test, base_proba)
    for scenario_name, X_stress in scenarios.items():
        proba = predict_proba(model, X_stress)
        pred = (proba >= 0.5).astype(int)
        stressed_f1 = f1_score(y_test, pred, zero_division=0)
        stressed_auc = roc_auc_score(y_test, proba)
        flips = int((base_pred != pred).sum())
        rows.append(
            {
                "Model": model_name,
                "Scenario": scenario_name,
                "Baseline_F1": base_f1,
                "Stressed_F1": stressed_f1,
                "F1_Drop": base_f1 - stressed_f1,
                "Baseline_ROC_AUC": base_auc,
                "Stressed_ROC_AUC": stressed_auc,
                "AUC_Drop": base_auc - stressed_auc,
                "Prediction_Flips": flips,
                "Robustness": 1 - (flips / len(X_test)),
            }
        )

stress_df = pd.DataFrame(rows)
stress_path = RESULT_DIR / "supervised_stress_scenario_results.csv"
stress_df.to_csv(stress_path, index=False)
stress_df.to_csv(RESULT_DIR / "financial_stress_scenarios_results_standardized.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=stress_df, x="Scenario", y="Stressed_F1", hue="Model")
plt.title("Supervised Model F1 Under Financial Stress Scenarios")
plt.tight_layout()
fig_path = FIGURE_DIR / "supervised_stress_scenario_f1_comparison.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(stress_df.to_string(index=False))
print(f"\nStress scenario results saved to: {stress_path}")

save_state(13, df=df, feature_cols=feature_cols, models=models, stress_df=stress_df, robustness_df=state["robustness_df"])

print("\nSupervised Step 13 completed successfully!")
