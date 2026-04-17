"""
Step 5: 5. Feature Engineering

#### 5.1 Create Financial Features

#### 5.2 Save Feature Dataset

#### 5.3 Feature Correlation Heatmap

#### 5.4 Rolling Volatility Plot

Run: python scripts/step_05_feature_engineering.py
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
print("Step 5: 5. Feature Engineering")
print("============================================================\n")

# --- Load state from Step 4 ---
_state = load_state(4)
data = _state["data"]
tickers = _state["tickers"]
close_prices = _state["close_prices"]
returns_all = _state["returns_all"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
# Convert the wide-format dataset into long format so that features can be built per asset

data_long = data.stack(level=1).reset_index()
data_long.columns = ["Date", "Ticker", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
data_long = data_long[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data_long = data_long.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Feature engineering for each ticker separately
data_long["Return"] = data_long.groupby("Ticker")["Close"].pct_change()
data_long["Volatility_10"] = data_long.groupby("Ticker")["Return"].rolling(window=10).std().reset_index(level=0, drop=True)
data_long["Volatility_20"] = data_long.groupby("Ticker")["Return"].rolling(window=20).std().reset_index(level=0, drop=True)
data_long["MA_10"] = data_long.groupby("Ticker")["Close"].rolling(window=10).mean().reset_index(level=0, drop=True)
data_long["MA_20"] = data_long.groupby("Ticker")["Close"].rolling(window=20).mean().reset_index(level=0, drop=True)
data_long["Momentum_10"] = data_long.groupby("Ticker")["Close"].diff(10)
data_long["MA_Ratio_10"] = data_long["Close"] / data_long["MA_10"]
data_long["MA_Ratio_20"] = data_long["Close"] / data_long["MA_20"]
data_long["Volume_Change"] = data_long.groupby("Ticker")["Volume"].pct_change()

# Additional dissertation-level technical indicators
data_long["Rolling_Skew_20"] = data_long.groupby("Ticker")["Return"].rolling(window=20).skew().reset_index(level=0, drop=True)
data_long["Rolling_Kurt_20"] = data_long.groupby("Ticker")["Return"].rolling(window=20).kurt().reset_index(level=0, drop=True)
data_long["Return_Zscore_20"] = data_long.groupby("Ticker")["Return"].transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std())
data_long["Drawdown"] = data_long.groupby("Ticker")["Close"].transform(lambda x: x / x.cummax() - 1)

# RSI(14)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data_long["RSI_14"] = data_long.groupby("Ticker")["Close"].transform(lambda x: compute_rsi(x, 14))

# MACD and signal line
ema12 = data_long.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = data_long.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
data_long["MACD"] = ema12 - ema26
data_long["MACD_Signal"] = data_long.groupby("Ticker")["MACD"].transform(lambda x: x.ewm(span=9, adjust=False).mean())

# Bollinger Band width (20-day)
rolling_mean_20 = data_long.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
rolling_std_20 = data_long.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).std())
upper_band = rolling_mean_20 + 2 * rolling_std_20
lower_band = rolling_mean_20 - 2 * rolling_std_20
data_long["BB_Width"] = (upper_band - lower_band) / rolling_mean_20

data_long.dropna(inplace=True)
data_long.reset_index(drop=True, inplace=True)

# Keep df as the main modelling table for downstream compatibility
df = data_long.copy()
print(df.shape)
df.head()

# --- Code Cell 2 ---
feature_data_path = os.path.join(base_path, "data", "multi_asset_financial_features.csv")
df.to_csv(feature_data_path, index=False)
print(f"Feature data saved to: {feature_data_path}")

# --- Code Cell 3 ---
numeric_corr = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12,9))
sns.heatmap(numeric_corr, annot=False, cmap="coolwarm")
plt.title("Correlation Between Engineered Financial Features")
plt.tight_layout()

fig_path = os.path.join(base_path, "figures", "feature_correlation_heatmap_multi_asset.png")
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"Figure saved to: {fig_path}")

# --- Code Cell 4 ---
plot_vol_df = df[df["Ticker"] == "AAPL"].copy()
plt.figure(figsize=(12,5))
plt.plot(plot_vol_df["Date"], plot_vol_df["Volatility_20"])
plt.title("20-Day Rolling Volatility (AAPL)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.xticks(rotation=45)
plt.tight_layout()

fig_path = os.path.join(base_path, "figures", "rolling_volatility_plot_aapl.png")
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"Figure saved to: {fig_path}")

# --- Save state for next step ---
save_state(5, data=data, tickers=tickers, close_prices=close_prices, returns_all=returns_all, df=df)

print("\nStep 5 completed successfully!")
