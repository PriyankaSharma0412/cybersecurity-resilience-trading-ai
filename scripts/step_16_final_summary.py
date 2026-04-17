"""
Step 16: 16. Final Summary Tables

#### 16.1 Save Final Summary

#### 16.2 Model Failure Analysis

#### 16.3 Save Failure Analysis

Run: python scripts/step_16_final_summary.py
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
print("Step 16: 16. Final Summary Tables")
print("============================================================\n")

# --- Load state from Step 15 ---
_state = load_state(15)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
avg_robustness = {"Isolation Forest": results_df["IF_Robustness_Score"].mean(), "Autoencoder": results_df["AE_Robustness_Score"].mean(), "One-Class SVM": results_df["OCSVM_Robustness_Score"].mean()}
worst_stress = stress_results_df.groupby("Model")["Robustness"].min().to_dict()
final_summary = comparison.copy()
final_summary["Avg_Gaussian_Robustness"] = final_summary["Model"].map(avg_robustness)
final_summary["Worst_Stress_Robustness"] = final_summary["Model"].map(worst_stress)
final_summary


# --- Code Cell 2 ---
summary_path = base_path / "results" / "final_model_summary_standardized.csv"
final_summary.to_csv(summary_path, index=False)
print(f"Final summary saved to: {summary_path}")


# --- Code Cell 3 ---
gaussian_30 = results_df.set_index("Perturbation_Level")
failure_analysis = pd.DataFrame({
    "Model": ["Isolation Forest", "Autoencoder", "One-Class SVM"],
    "Baseline_Test_Anomalies": [int(df.loc[df["Split"] == "Test", "Anomaly_IF"].sum()), int(df.loc[df["Split"] == "Test", "Anomaly_AE"].sum()), int(df.loc[df["Split"] == "Test", "Anomaly_OCSVM"].sum())],
    "Gaussian_30pct_Label_Flips": [gaussian_30.loc[0.30, "IF_Label_Flips"], gaussian_30.loc[0.30, "AE_Label_Flips"], gaussian_30.loc[0.30, "OCSVM_Label_Flips"]],
    "Worst_Stress_Robustness": [worst_stress["Isolation Forest"], worst_stress["Autoencoder"], worst_stress["One-Class SVM"]],
    "Interpretation": ["Tree-based detector remains comparatively stable under diffuse perturbation but should still be monitored for drift.", "Reconstruction boundary is sensitive to shifted inputs, which is useful for detection but increases perturbation sensitivity.", "Kernel baseline offers a useful comparator but may degrade when scaled feature geometry changes materially."]
})
failure_analysis


# --- Code Cell 4 ---
failure_analysis_path = base_path / "results" / "model_failure_analysis_standardized.csv"
failure_analysis.to_csv(failure_analysis_path, index=False)
print(f"Model failure analysis saved to: {failure_analysis_path}")


# --- Save state for next step ---
save_state(16, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 16 completed successfully!")
