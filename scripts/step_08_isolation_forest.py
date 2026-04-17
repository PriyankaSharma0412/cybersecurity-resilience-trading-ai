"""
Step 8: 8. Isolation Forest Model

#### 8.1 Train on the Chronological Training Split Only

#### 8.2 Save Isolation Forest Model

#### 8.3 Isolation Forest Held-Out Anomalies on the Test Split

Run: python scripts/step_08_isolation_forest.py
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
print("Step 8: 8. Isolation Forest Model")
print("============================================================\n")

# --- Load state from Step 7 ---
_state = load_state(7)
df = _state["df"]
feature_cols = _state["feature_cols"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
split_feature_frames = {split_name: df.loc[df["Split"] == split_name, feature_cols].copy() for split_name in ["Train", "Validation", "Test"]}

X_train = split_feature_frames["Train"].copy()
X_val = split_feature_frames["Validation"].copy()
X_test = split_feature_frames["Test"].copy()

iso_model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso_model.fit(X_train)

df["IF_raw"] = np.nan
df["Anomaly_IF"] = 0
df["IF_score"] = np.nan

for split_name, split_frame in split_feature_frames.items():
    split_mask = df["Split"] == split_name
    raw_pred = iso_model.predict(split_frame)
    score_pred = iso_model.decision_function(split_frame)
    df.loc[split_mask, "IF_raw"] = raw_pred
    df.loc[split_mask, "Anomaly_IF"] = pd.Series(raw_pred, index=split_frame.index).replace({1: 0, -1: 1}).astype(int)
    df.loc[split_mask, "IF_score"] = score_pred

if_split_counts = df.groupby("Split")["Anomaly_IF"].agg(["sum", "mean"]).rename(columns={"sum": "Anomalies", "mean": "Anomaly_Rate"}).reset_index()
if_split_counts


# --- Code Cell 2 ---
iso_model_path = base_path / "models" / "isolation_forest_train_only.pkl"
joblib.dump(iso_model, iso_model_path)
print(f"Isolation Forest model saved to: {iso_model_path}")


# --- Code Cell 3 ---
print(if_split_counts)

plot_df = df[(df["Ticker"] == "AAPL") & (df["Split"] == "Test")].copy()

plt.figure(figsize=(14, 6))
plt.plot(plot_df["Date"], plot_df["Close"], label="AAPL Close")
plt.scatter(
    plot_df.loc[plot_df["Anomaly_IF"] == 1, "Date"],
    plot_df.loc[plot_df["Anomaly_IF"] == 1, "Close"],
    color="red",
    s=20,
    label="IF Test Anomalies"
)
plt.title("Isolation Forest Anomalies on AAPL Test Period")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

fig_path = base_path / "figures" / "if_test_period_anomalies_aapl.png"
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"Figure saved to: {fig_path}")


# --- Save state for next step ---
save_state(8, df=df, feature_cols=feature_cols, iso_model=iso_model, if_split_counts=if_split_counts)

print("\nStep 8 completed successfully!")
