"""
Step 10: 10. One-Class SVM Baseline and Shared Visual Diagnostics

#### 10.1 Autoencoder Training Loss

#### 10.2 AAPL Test-Period Anomalies Across All Models

Run: python scripts/step_10_one_class_svm.py
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
print("Step 10: 10. One-Class SVM Baseline and Shared Visual Diagnostics")
print("============================================================\n")

# --- Load state from Step 9 ---
_state = load_state(9)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
ocsvm_model = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
ocsvm_model.fit(X_scaled_train)

df["OCSVM_raw"] = np.nan
df["Anomaly_OCSVM"] = 0
df["OCSVM_score"] = np.nan

for split_name, split_scaled in X_scaled_by_split.items():
    split_mask = df["Split"] == split_name
    raw_pred = ocsvm_model.predict(split_scaled)
    score_pred = ocsvm_model.decision_function(split_scaled)
    df.loc[split_mask, "OCSVM_raw"] = raw_pred
    df.loc[split_mask, "Anomaly_OCSVM"] = pd.Series(raw_pred, index=split_feature_frames[split_name].index).replace({1: 0, -1: 1}).astype(int)
    df.loc[split_mask, "OCSVM_score"] = score_pred

ocsvm_model_path = base_path / "models" / "one_class_svm_train_only.pkl"
joblib.dump(ocsvm_model, ocsvm_model_path)
ocsvm_split_counts = df.groupby("Split")["Anomaly_OCSVM"].agg(["sum", "mean"]).rename(columns={"sum": "Anomalies", "mean": "Anomaly_Rate"}).reset_index()
print(f"One-Class SVM model saved to: {ocsvm_model_path}")
ocsvm_split_counts


# --- Code Cell 2 ---
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()

fig_path = base_path / "figures" / "autoencoder_training_loss_train_only.png"
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"Figure saved to: {fig_path}")


# --- Code Cell 3 ---
plot_df = df[(df["Ticker"] == "AAPL") & (df["Split"] == "Test")].copy()
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
model_plot_specs = [("Anomaly_IF", "Isolation Forest", "red"), ("Anomaly_AE", "Autoencoder", "orange"), ("Anomaly_OCSVM", "One-Class SVM", "purple")]
for ax, (label_col, title, color) in zip(axes, model_plot_specs):
    ax.plot(plot_df["Date"], plot_df["Close"], label="AAPL Close", color="steelblue")
    ax.scatter(plot_df.loc[plot_df[label_col] == 1, "Date"], plot_df.loc[plot_df[label_col] == 1, "Close"], color=color, s=18, label=f"{title} Anomalies")
    ax.set_title(f"{title} Test-Period Anomalies on AAPL")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
axes[-1].set_xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()

fig_path = base_path / "figures" / "test_period_anomalies_all_models_aapl.png"
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Figure saved to: {fig_path}")


# --- Save state for next step ---
save_state(10, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 10 completed successfully!")
