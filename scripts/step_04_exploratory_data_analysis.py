"""
Step 4: 4. Exploratory Data Analysis

####4.2 Plot Stock Price trends

#### 4.3 Daily Returns

####4.4 Returns Distribution for AAPL

#### 4.5 Correlation Heatmap

Run: python scripts/step_04_exploratory_data_analysis.py
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
print("Step 4: 4. Exploratory Data Analysis")
print("============================================================\n")

# --- Load state from Step 3 ---
_state = load_state(3)
data = _state["data"]
tickers = _state["tickers"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
# Extract closing prices from the wide-format market dataset
close_prices = data["Close"].copy()
close_prices.head()

# --- Code Cell 2 ---
plt.figure(figsize=(12,6))

for ticker in tickers:
    if ticker in close_prices.columns:
        plt.plot(close_prices.index, close_prices[ticker], label=ticker)

plt.title("Multi-Asset Stock Price Trends (2010-2024)")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend(ncol=2)
plt.tight_layout()

fig_path = os.path.join(base_path, "figures", "stock_price_trends_multi_asset.png")
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"Figure saved to: {fig_path}")

# --- Code Cell 3 ---
returns_all = close_prices.pct_change()
print(returns_all.shape)
returns_all.head()

# --- Code Cell 4 ---
plt.figure(figsize=(8,5))
sns.histplot(returns_all["AAPL"].dropna(), bins=50, kde=True)
plt.title("Distribution of Daily Returns (AAPL)")
plt.xlabel("Daily Return")
plt.tight_layout()

fig_path = os.path.join(base_path, "figures", "aapl_returns_distribution.png")
plt.savefig(fig_path,dpi=300)
plt.show()

print(f"Figure saved to: {fig_path}")

# --- Code Cell 5 ---
corr = returns_all

plt.figure(figsize=(10,8))
sns.heatmap(corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Selected Assets")
plt.tight_layout()

fig_path = os.path.join(base_path, "figures", "asset_correlation_heatmap.png")
plt.savefig(fig_path,dpi=300)
plt.show()

print(f"Figure saved to: {fig_path}")

# --- Save state for next step ---
save_state(4, data=data, tickers=tickers, close_prices=close_prices, returns_all=returns_all)

print("\nStep 4 completed successfully!")
