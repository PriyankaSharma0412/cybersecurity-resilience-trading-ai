"""
Step 13: 13. Financial Stress Scenario Evaluation

#### 13.1 Define Test-Split Stress Scenarios with Consistent Feature Names

#### 13.2 Evaluate Stress Scenario Robustness

#### 13.3 Save Stress Results

Run: python scripts/step_13_stress_scenarios.py
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
print("Step 13: 13. Financial Stress Scenario Evaluation")
print("============================================================\n")

# --- Load state from Step 12 ---
_state = load_state(12)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
X_test_scenarios = {}
price_spike = X_test.copy()
price_spike["Momentum_10"] = price_spike["Momentum_10"] * 1.5
price_spike["MA_Ratio_10"] = price_spike["MA_Ratio_10"] * 1.08
price_spike["MA_Ratio_20"] = price_spike["MA_Ratio_20"] * 1.10
price_spike["Return_Zscore_20"] = price_spike["Return_Zscore_20"] + 2.0
X_test_scenarios["Price Spike"] = price_spike

volume_shock = X_test.copy()
volume_shock["Volume_Change"] = volume_shock["Volume_Change"].fillna(0) + 2.0
volume_shock["Volatility_10"] = volume_shock["Volatility_10"] * 1.3
volume_shock["BB_Width"] = volume_shock["BB_Width"] * 1.5
X_test_scenarios["Volume Shock"] = volume_shock

volatility_shock = X_test.copy()
volatility_shock["Volatility_10"] = volatility_shock["Volatility_10"] * 1.8
volatility_shock["Volatility_20"] = volatility_shock["Volatility_20"] * 1.8
volatility_shock["Drawdown"] = volatility_shock["Drawdown"] - 0.10
volatility_shock["Rolling_Kurt_20"] = volatility_shock["Rolling_Kurt_20"] + 1.0
volatility_shock["BB_Width"] = volatility_shock["BB_Width"] * 1.6
X_test_scenarios["Volatility Shock"] = volatility_shock
list(X_test_scenarios.keys())


# --- Code Cell 2 ---
def evaluate_scenario(model_name, baseline_labels, pred_labels):
    flips = int(np.sum(baseline_labels != pred_labels))
    jac = jaccard_score(baseline_labels, pred_labels, zero_division=0)
    robustness = 1 - (flips / len(baseline_labels))
    return {"Model": model_name, "Label_Flips": flips, "Jaccard": jac, "Robustness": robustness}

stress_results = []
for scenario_name, scenario_frame in X_test_scenarios.items():
    if_pred, _ = predict_if(scenario_frame)
    ae_pred, _ = predict_ae(scenario_frame)
    oc_pred, _ = predict_ocsvm(scenario_frame)
    stress_results.append({"Scenario": scenario_name, **evaluate_scenario("Isolation Forest", baseline_test_labels["Isolation Forest"], if_pred)})
    stress_results.append({"Scenario": scenario_name, **evaluate_scenario("Autoencoder", baseline_test_labels["Autoencoder"], ae_pred)})
    stress_results.append({"Scenario": scenario_name, **evaluate_scenario("One-Class SVM", baseline_test_labels["One-Class SVM"], oc_pred)})

stress_results_df = pd.DataFrame(stress_results)
stress_results_df


# --- Code Cell 3 ---
stress_results_path = base_path / "results" / "financial_stress_scenarios_results_standardized.csv"
stress_results_df.to_csv(stress_results_path, index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=stress_results_df, x="Scenario", y="Robustness", hue="Model")
plt.title("Model Robustness Under Financial Stress Scenarios")
plt.tight_layout()
fig_path = base_path / "figures" / "financial_stress_robustness_comparison_standardized.png"
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Stress scenario results saved to: {stress_results_path}")
print(f"Figure saved to: {fig_path}")


# --- Save state for next step ---
save_state(13, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 13 completed successfully!")
