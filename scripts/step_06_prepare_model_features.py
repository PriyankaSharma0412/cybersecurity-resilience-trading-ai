"""
Step 6: 6. Prepare Model Features

Run: python scripts/step_06_prepare_model_features.py
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
print("Step 6: 6. Prepare Model Features")
print("============================================================\n")

# --- Load state from Step 5 ---
_state = load_state(5)
df = _state["df"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
feature_cols = [
    "Return",
    "Volatility_10",
    "Volatility_20",
    "Momentum_10",
    "MA_Ratio_10",
    "MA_Ratio_20",
    "Volume_Change",
    "Rolling_Skew_20",
    "Rolling_Kurt_20",
    "Return_Zscore_20",
    "Drawdown",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_Width"
]

X = df[feature_cols].copy()
print(X.shape)
X.head()

# --- Code Cell 2 ---
model_features_path = os.path.join(base_path, "data", "multi_asset_model_features.csv")
X.to_csv(model_features_path, index=False)
print(f"Model features saved to: {model_features_path}")

# --- Save state for next step ---
save_state(6, df=df, feature_cols=feature_cols)

print("\nStep 6 completed successfully!")
