"""
Supervised Step 11: Evaluate classifiers on validation and test splits.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import pandas as pd

from supervised_utils import RESULT_DIR, load_state, metric_row, predict_proba, save_state, split_xy

print("\n============================================================")
print("Supervised Step 11: Classifier Evaluation")
print("============================================================\n")

state = load_state(10)
df = state["df"]
feature_cols = state["feature_cols"]
models = {
    "Logistic Regression": state["logistic_model"],
    "Random Forest": state["random_forest_model"],
    state["tree_boost_model_name"]: state["tree_boost_model"],
}
splits = split_xy(df, feature_cols)

metrics = []
prediction_parts = []
for split_name in ["Validation", "Test"]:
    X, y, subset = splits[split_name]
    out = subset[["Date", "Ticker", "Split", "Close", "Future_Return_5D", "Target"]].copy()
    for model_name, model in models.items():
        proba = predict_proba(model, X)
        pred = (proba >= 0.5).astype(int)
        safe_name = model_name.replace(" ", "_").replace("-", "_")
        out[f"Pred_{safe_name}"] = pred
        out[f"Proba_{safe_name}"] = proba
        metrics.append(metric_row(model_name, split_name, y, pred, proba))
    prediction_parts.append(out)

metrics_df = pd.DataFrame(metrics)
predictions_df = pd.concat(prediction_parts, ignore_index=True)
confusion_df = metrics_df[["Model", "Split", "TN", "FP", "FN", "TP"]].copy()

metrics_path = RESULT_DIR / "supervised_model_metrics.csv"
predictions_path = RESULT_DIR / "supervised_drawdown_predictions.csv"
confusion_path = RESULT_DIR / "supervised_confusion_matrices.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

# Compatibility names used by older reporting scripts.
metrics_df.to_csv(RESULT_DIR / "controlled_injection_metrics.csv", index=False)
predictions_df.to_csv(RESULT_DIR / "controlled_injection_predictions.csv", index=False)
confusion_df.to_csv(RESULT_DIR / "controlled_injection_confusion_matrices.csv", index=False)
metrics_df[metrics_df["Split"] == "Test"].to_csv(RESULT_DIR / "baseline_model_comparison_standardized.csv", index=False)

print(metrics_df.to_string(index=False))
print(f"\nMetrics saved to: {metrics_path}")
print(f"Predictions saved to: {predictions_path}")

save_state(11, df=df, feature_cols=feature_cols, models=models, metrics_df=metrics_df, predictions_df=predictions_df)

print("\nSupervised Step 11 completed successfully!")
