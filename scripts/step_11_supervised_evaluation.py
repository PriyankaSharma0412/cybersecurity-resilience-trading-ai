"""
Step 11: 11. Supervised Evaluation on the Controlled Injection Dataset

#### 11.1 Save Evaluation Metrics, Confusion Matrices, Event Detection, and Labeled Data

#### 11.2 Unified Baseline Comparison Using Test-Period Results

#### 11.3 Save Unified Comparison Table

#### 11.4 Overlap Between Baseline Test Predictions

Run: python scripts/step_11_supervised_evaluation.py
"""

# ============================================
# Common Setup
# ============================================
import os
import sys
import pickle
from pathlib import Path

# Ensure working directory is the project root (parent of scripts/)
os.chdir(Path(__file__).resolve().parent.parent)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    jaccard_score,
    precision_recall_fscore_support
)
from scipy.stats import ttest_ind, wilcoxon, spearmanr
from sklearn.svm import OneClassSVM

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

plt.style.use("default")
sns.set_theme(style="whitegrid")

# --- Path Setup ---
project_root = Path.cwd()
base_path = project_root / "dissertation_outputs"
mpl_config_dir = project_root / ".mplconfig"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

folders = ["data", "models", "figures", "results"]
for folder in folders:
    (base_path / folder).mkdir(parents=True, exist_ok=True)

state_dir = base_path / "state"
state_dir.mkdir(parents=True, exist_ok=True)

print(f"Project root: {project_root}")
print(f"Base path: {base_path}")

def save_state(step_num, **kwargs):
    """Save variables to a pickle file for the next step to load."""
    state_file = state_dir / f"state_step_{step_num:02d}.pkl"
    with open(state_file, "wb") as f:
        pickle.dump(kwargs, f)
    print(f"State saved to: {state_file}")

def load_state(step_num):
    """Load variables from a previous step's state file."""
    state_file = state_dir / f"state_step_{step_num:02d}.pkl"
    if not state_file.exists():
        print(f"ERROR: State file not found: {state_file}")
        print(f"Please run step {step_num} first!")
        sys.exit(1)
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    print(f"State loaded from: {state_file}")
    return state

print("\n============================================================")
print("Step 11: 11. Supervised Evaluation on the Controlled Injection Dataset")
print("============================================================\n")

# --- Load state from Step 10 ---
_state = load_state(10)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
eval_predictions_df = eval_df.copy()
eval_predictions_df["Pred_IF"] = 0
eval_predictions_df["Pred_AE"] = 0
eval_predictions_df["Pred_OCSVM"] = 0
eval_predictions_df["Eval_IF_Score"] = np.nan
eval_predictions_df["Eval_AE_Error"] = np.nan
eval_predictions_df["Eval_OCSVM_Score"] = np.nan

def predict_if(frame):
    raw_pred = iso_model.predict(frame)
    labels = pd.Series(raw_pred, index=frame.index).replace({1: 0, -1: 1}).astype(int).values
    scores = iso_model.decision_function(frame)
    return labels, scores

def predict_ae(frame):
    scaled = scaler.transform(frame)
    recon = autoencoder.predict(scaled, verbose=0)
    err = np.mean(np.square(scaled - recon), axis=1)
    labels = (err > ae_threshold).astype(int)
    return labels, err

def predict_ocsvm(frame):
    scaled = scaler.transform(frame)
    raw_pred = ocsvm_model.predict(scaled)
    labels = pd.Series(raw_pred, index=frame.index).replace({1: 0, -1: 1}).astype(int).values
    scores = ocsvm_model.decision_function(scaled)
    return labels, scores

for split_name in ["Validation", "Test"]:
    split_mask = eval_predictions_df["Split"] == split_name
    split_frame = eval_predictions_df.loc[split_mask, feature_cols].copy()
    if_labels, if_scores = predict_if(split_frame)
    ae_labels, ae_scores = predict_ae(split_frame)
    ocsvm_labels, ocsvm_scores = predict_ocsvm(split_frame)
    eval_predictions_df.loc[split_mask, "Pred_IF"] = if_labels
    eval_predictions_df.loc[split_mask, "Pred_AE"] = ae_labels
    eval_predictions_df.loc[split_mask, "Pred_OCSVM"] = ocsvm_labels
    eval_predictions_df.loc[split_mask, "Eval_IF_Score"] = if_scores
    eval_predictions_df.loc[split_mask, "Eval_AE_Error"] = ae_scores
    eval_predictions_df.loc[split_mask, "Eval_OCSVM_Score"] = ocsvm_scores

def build_event_detection_rows(model_name, pred_col):
    rows = []
    event_subset = eval_predictions_df[eval_predictions_df["Evaluation_Event_ID"] != ""].copy()
    for event_id, group in event_subset.groupby("Evaluation_Event_ID"):
        group = group.sort_values("Date")
        detection_positions = np.where(group[pred_col].values == 1)[0]
        detected = len(detection_positions) > 0
        rows.append({"Model": model_name, "Event_ID": event_id, "Split": group["Split"].iloc[0], "Ticker": group["Ticker"].iloc[0], "Scenario": group["Evaluation_Scenario"].iloc[0], "Detected": int(detected), "Time_To_Detection": int(detection_positions[0]) if detected else np.nan})
    return pd.DataFrame(rows)

event_detection_df_all = pd.concat([
    build_event_detection_rows("Isolation Forest", "Pred_IF"),
    build_event_detection_rows("Autoencoder", "Pred_AE"),
    build_event_detection_rows("One-Class SVM", "Pred_OCSVM")
], ignore_index=True)

def build_metric_row(model_name, split_name, pred_col):
    subset = eval_predictions_df[eval_predictions_df["Split"].isin(["Validation", "Test"])].copy() if split_name == "HeldOut" else eval_predictions_df[eval_predictions_df["Split"] == split_name].copy()
    y_true = subset["Evaluation_Label"].astype(int).values
    y_pred = subset[pred_col].astype(int).values
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    detection_subset = event_detection_df_all[event_detection_df_all["Split"].isin(["Validation", "Test"])] if split_name == "HeldOut" else event_detection_df_all[event_detection_df_all["Split"] == split_name]
    detection_subset = detection_subset[detection_subset["Model"] == model_name]
    detected_rows = detection_subset[detection_subset["Detected"] == 1]
    return {"Model": model_name, "Split": split_name, "Precision": precision, "Recall": recall, "F1_Score": f1, "Avg_Time_To_Detection": detected_rows["Time_To_Detection"].mean() if not detected_rows.empty else np.nan, "Event_Detection_Rate": detection_subset["Detected"].mean() if not detection_subset.empty else np.nan, "Predicted_Anomalies": int(y_pred.sum()), "True_Anomalies": int(y_true.sum()), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

metrics_records = []
for model_name in ["Isolation Forest", "Autoencoder", "One-Class SVM"]:
    for split_name in ["Validation", "Test", "HeldOut"]:
        pred_col = {"Isolation Forest": "Pred_IF", "Autoencoder": "Pred_AE", "One-Class SVM": "Pred_OCSVM"}[model_name]
        metrics_records.append(build_metric_row(model_name, split_name, pred_col))

metrics_df = pd.DataFrame(metrics_records)
confusion_df = metrics_df[["Model", "Split", "TN", "FP", "FN", "TP"]].copy()
metrics_df


# --- Code Cell 2 ---
evaluation_predictions_path = base_path / "results" / "controlled_injection_predictions.csv"
metrics_path = base_path / "results" / "controlled_injection_metrics.csv"
confusion_path = base_path / "results" / "controlled_injection_confusion_matrices.csv"
event_detection_path = base_path / "results" / "controlled_injection_event_detection.csv"

eval_predictions_df.to_csv(evaluation_predictions_path, index=False)
metrics_df.to_csv(metrics_path, index=False)
confusion_df.to_csv(confusion_path, index=False)
event_detection_df_all.to_csv(event_detection_path, index=False)

print(f"Prediction table saved to: {evaluation_predictions_path}")
print(f"Metrics table saved to: {metrics_path}")
print(f"Confusion matrices saved to: {confusion_path}")
print(f"Event detection table saved to: {event_detection_path}")


# --- Code Cell 3 ---
test_metrics = metrics_df[metrics_df["Split"] == "Test"].set_index("Model")
comparison = pd.DataFrame({
    "Model": ["Isolation Forest", "Autoencoder", "One-Class SVM"],
    "Baseline_Test_Anomalies": [int(df.loc[df["Split"] == "Test", "Anomaly_IF"].sum()), int(df.loc[df["Split"] == "Test", "Anomaly_AE"].sum()), int(df.loc[df["Split"] == "Test", "Anomaly_OCSVM"].sum())],
    "Precision": [test_metrics.loc["Isolation Forest", "Precision"], test_metrics.loc["Autoencoder", "Precision"], test_metrics.loc["One-Class SVM", "Precision"]],
    "Recall": [test_metrics.loc["Isolation Forest", "Recall"], test_metrics.loc["Autoencoder", "Recall"], test_metrics.loc["One-Class SVM", "Recall"]],
    "F1_Score": [test_metrics.loc["Isolation Forest", "F1_Score"], test_metrics.loc["Autoencoder", "F1_Score"], test_metrics.loc["One-Class SVM", "F1_Score"]],
    "Avg_Time_To_Detection": [test_metrics.loc["Isolation Forest", "Avg_Time_To_Detection"], test_metrics.loc["Autoencoder", "Avg_Time_To_Detection"], test_metrics.loc["One-Class SVM", "Avg_Time_To_Detection"]],
    "Event_Detection_Rate": [test_metrics.loc["Isolation Forest", "Event_Detection_Rate"], test_metrics.loc["Autoencoder", "Event_Detection_Rate"], test_metrics.loc["One-Class SVM", "Event_Detection_Rate"]]
})
comparison


# --- Code Cell 4 ---
comparison_path = base_path / "results" / "baseline_model_comparison_standardized.csv"
comparison.to_csv(comparison_path, index=False)
print(f"Comparison table saved to: {comparison_path}")


# --- Code Cell 5 ---
test_df = df[df["Split"] == "Test"].copy()
overlap_df = pd.DataFrame({
    "Comparison": ["IF vs AE", "IF vs OCSVM", "AE vs OCSVM", "Consensus All Three"],
    "Overlap_Count": [int(((test_df["Anomaly_IF"] == 1) & (test_df["Anomaly_AE"] == 1)).sum()), int(((test_df["Anomaly_IF"] == 1) & (test_df["Anomaly_OCSVM"] == 1)).sum()), int(((test_df["Anomaly_AE"] == 1) & (test_df["Anomaly_OCSVM"] == 1)).sum()), int(((test_df["Anomaly_IF"] == 1) & (test_df["Anomaly_AE"] == 1) & (test_df["Anomaly_OCSVM"] == 1)).sum())]
})
overlap_df


# --- Save state for next step ---
save_state(11, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 11 completed successfully!")
