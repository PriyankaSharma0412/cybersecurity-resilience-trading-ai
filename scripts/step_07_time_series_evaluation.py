"""
Step 7: Supervised Time-Series Evaluation Design

#### 7.1 Save Chronological Split Metadata and Controlled Evaluation Labels

Run: python scripts/step_07_time_series_evaluation.py
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
print("Step 7: 7. Time-Series Evaluation Design")
print("============================================================\n")

# --- Load state from Step 6 ---
_state = load_state(6)
df = _state["df"]
feature_cols = _state["feature_cols"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"])

unique_dates = np.array(sorted(df["Date"].dt.normalize().unique()))
train_end = int(len(unique_dates) * 0.70)
val_end = int(len(unique_dates) * 0.85)

train_dates = set(unique_dates[:train_end])
val_dates = set(unique_dates[train_end:val_end])

def assign_split(date_value):
    normalized = pd.Timestamp(date_value).normalize().to_datetime64()
    if normalized in train_dates:
        return "Train"
    if normalized in val_dates:
        return "Validation"
    return "Test"

df["Split"] = df["Date"].apply(assign_split)

eval_df = df.copy()
eval_df["Evaluation_Label"] = 0
eval_df["Evaluation_Event_ID"] = ""
eval_df["Evaluation_Scenario"] = "Baseline"

def inject_event_block(frame, row_index, scenario_name, event_id, severity):
    frame.loc[row_index, "Evaluation_Label"] = 1
    frame.loc[row_index, "Evaluation_Event_ID"] = event_id
    frame.loc[row_index, "Evaluation_Scenario"] = scenario_name

    if scenario_name == "Price Spike":
        frame.loc[row_index, "Return"] = frame.loc[row_index, "Return"] * (1.0 + 1.20 * severity)
        frame.loc[row_index, "Momentum_10"] = frame.loc[row_index, "Momentum_10"] * (1.0 + 0.80 * severity)
        frame.loc[row_index, "MA_Ratio_10"] = frame.loc[row_index, "MA_Ratio_10"] * (1.0 + 0.10 * severity)
        frame.loc[row_index, "MA_Ratio_20"] = frame.loc[row_index, "MA_Ratio_20"] * (1.0 + 0.12 * severity)
        frame.loc[row_index, "Return_Zscore_20"] = frame.loc[row_index, "Return_Zscore_20"] + (2.0 * severity)
        frame.loc[row_index, "RSI_14"] = np.clip(frame.loc[row_index, "RSI_14"] + (18 * severity), 0, 100)
    elif scenario_name == "Volume Shock":
        frame.loc[row_index, "Volume_Change"] = frame.loc[row_index, "Volume_Change"].fillna(0) + (2.5 * severity)
        frame.loc[row_index, "Volatility_10"] = frame.loc[row_index, "Volatility_10"] * (1.0 + 0.70 * severity)
        frame.loc[row_index, "Volatility_20"] = frame.loc[row_index, "Volatility_20"] * (1.0 + 0.60 * severity)
        frame.loc[row_index, "BB_Width"] = frame.loc[row_index, "BB_Width"] * (1.0 + 0.90 * severity)
        frame.loc[row_index, "Rolling_Skew_20"] = frame.loc[row_index, "Rolling_Skew_20"] + (0.60 * severity)
    elif scenario_name == "Volatility Shock":
        frame.loc[row_index, "Volatility_10"] = frame.loc[row_index, "Volatility_10"] * (1.0 + 1.20 * severity)
        frame.loc[row_index, "Volatility_20"] = frame.loc[row_index, "Volatility_20"] * (1.0 + 1.30 * severity)
        frame.loc[row_index, "Return_Zscore_20"] = frame.loc[row_index, "Return_Zscore_20"] + (3.0 * severity)
        frame.loc[row_index, "Drawdown"] = frame.loc[row_index, "Drawdown"] - (0.12 * severity)
        frame.loc[row_index, "BB_Width"] = frame.loc[row_index, "BB_Width"] * (1.0 + 1.10 * severity)
        frame.loc[row_index, "Rolling_Kurt_20"] = frame.loc[row_index, "Rolling_Kurt_20"] + (1.40 * severity)

event_log = []
scenario_cycle = ["Price Spike", "Volume Shock", "Volatility Shock"]

for split_name, severity, stride in [("Validation", 1.0, 120), ("Test", 1.2, 90)]:
    for ticker in tickers:
        ticker_index = eval_df.index[(eval_df["Split"] == split_name) & (eval_df["Ticker"] == ticker)].tolist()
        if len(ticker_index) < 40:
            continue
        positions = list(range(20, len(ticker_index) - 3, stride))[:3]
        for event_num, pos in enumerate(positions, start=1):
            row_index = ticker_index[pos:pos + 3]
            scenario_name = scenario_cycle[(event_num - 1) % len(scenario_cycle)]
            event_id = f"{split_name}_{ticker}_{event_num}"
            inject_event_block(eval_df, row_index, scenario_name, event_id, severity)
            event_log.append({
                "Event_ID": event_id,
                "Split": split_name,
                "Ticker": ticker,
                "Scenario": scenario_name,
                "Start_Date": eval_df.loc[row_index[0], "Date"],
                "End_Date": eval_df.loc[row_index[-1], "Date"],
                "Injected_Rows": len(row_index)
            })

event_log_df = pd.DataFrame(event_log)
split_summary_df = df.groupby("Split").agg(Rows=("Ticker", "size"), Assets=("Ticker", "nunique"), Start_Date=("Date", "min"), End_Date=("Date", "max")).reset_index()
print(split_summary_df)
print(eval_df["Evaluation_Label"].value_counts())
event_log_df.head()


# --- Code Cell 2 ---
evaluation_dataset_path = base_path / "data" / "evaluation_dataset_with_injected_events.csv"
event_log_path = base_path / "data" / "synthetic_event_log.csv"
split_summary_path = base_path / "results" / "time_series_split_summary.csv"

eval_df.to_csv(evaluation_dataset_path, index=False)
event_log_df.to_csv(event_log_path, index=False)
split_summary_df.to_csv(split_summary_path, index=False)

print(f"Evaluation dataset saved to: {evaluation_dataset_path}")
print(f"Event log saved to: {event_log_path}")
print(f"Split summary saved to: {split_summary_path}")


# --- Save state for next step ---
save_state(7, df=df, feature_cols=feature_cols)

print("\nStep 7 completed successfully!")
