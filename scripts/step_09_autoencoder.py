"""
Step 9: 9. Autoencoder Model

#### 9.1 Fit the Feature Scaler on the Training Split Only

#### 9.2 Build Autoencoder Architecture

#### 9.3 Train Autoencoder on Training Data Only

#### 9.4 Save Autoencoder Model

#### 9.5 Autoencoder Reconstruction Errors and Baseline Labels

#### 9.6 Autoencoder Held-Out Results

Run: python scripts/step_09_autoencoder.py
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
print("Step 9: 9. Autoencoder Model")
print("============================================================\n")

# --- Load state from Step 8 ---
_state = load_state(8)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_by_split = {split_name: scaler.transform(split_frame) for split_name, split_frame in split_feature_frames.items()}

scaler_path = base_path / "models" / "standard_scaler_train_only.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")


# --- Code Cell 2 ---
input_dim = X_scaled_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
encoded = Dense(16, activation="relu")(encoded)
encoded = Dense(8, activation="relu")(encoded)
decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(decoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
autoencoder.summary()


# --- Code Cell 3 ---
history = autoencoder.fit(
    X_scaled_train,
    X_scaled_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# --- Code Cell 4 ---
ae_model_path = base_path / "models" / "multi_asset_autoencoder_train_only.h5"
autoencoder.save(ae_model_path)
print(f"Autoencoder model saved to: {ae_model_path}")


# --- Code Cell 5 ---
train_reconstructions = autoencoder.predict(X_scaled_train, verbose=0)
train_mse = np.mean(np.square(X_scaled_train - train_reconstructions), axis=1)
ae_threshold = np.quantile(train_mse, 0.99)

df["AE_error"] = np.nan
df["Anomaly_AE"] = 0

for split_name, split_scaled in X_scaled_by_split.items():
    split_mask = df["Split"] == split_name
    reconstructions = autoencoder.predict(split_scaled, verbose=0)
    mse = np.mean(np.square(split_scaled - reconstructions), axis=1)
    df.loc[split_mask, "AE_error"] = mse
    df.loc[split_mask, "Anomaly_AE"] = (mse > ae_threshold).astype(int)

ae_split_counts = df.groupby("Split")["Anomaly_AE"].agg(["sum", "mean"]).rename(columns={"sum": "Anomalies", "mean": "Anomaly_Rate"}).reset_index()
print(f"Autoencoder threshold (99th percentile of train error): {ae_threshold:.6f}")
ae_split_counts


# --- Code Cell 6 ---
print(ae_split_counts)


# --- Save state for next step ---
save_state(9, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 9 completed successfully!")
