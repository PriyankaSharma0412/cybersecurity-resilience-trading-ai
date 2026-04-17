"""
Step 15: 15. Explanation Stability Under Perturbation

#### 15.1 Evaluate SHAP Stability Across All Gaussian Perturbation Levels

#### 15.2 Plot SHAP Stability

#### 15.3 Example Explanation Shift at 10 Percent Perturbation

#### 15.4 Rank Comparison

#### 15.5 Top-k Overlap and Save Single-Level Explainability Stability

Run: python scripts/step_15_explanation_stability.py
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
print("Step 15: 15. Explanation Stability Under Perturbation")
print("============================================================\n")

# --- Load state from Step 14 ---
_state = load_state(14)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
top_k = 5
base_importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_shap.columns)


# --- Code Cell 2 ---
shap_stability_results = []
for level in perturbation_levels:
    noise = np.random.normal(0, level, X_shap.shape)
    X_perturbed_level = pd.DataFrame(X_shap.values + noise, columns=X_shap.columns, index=X_shap.index)
    shap_values_level = explainer(X_perturbed_level)
    pert_importance = pd.Series(np.abs(shap_values_level.values).mean(axis=0), index=X_shap.columns)
    orig_rank = base_importance.rank(ascending=False)
    pert_rank = pert_importance.rank(ascending=False)
    rho, pval = spearmanr(orig_rank, pert_rank)
    orig_top = set(base_importance.sort_values(ascending=False).head(top_k).index)
    pert_top = set(pert_importance.sort_values(ascending=False).head(top_k).index)
    overlap_ratio = len(orig_top.intersection(pert_top)) / top_k
    shap_stability_results.append({"Perturbation_Level": level, "Spearman_Rho": rho, "Spearman_pvalue": pval, f"Top_{top_k}_Overlap": overlap_ratio})

shap_stability_df = pd.DataFrame(shap_stability_results)
shap_stability_path = base_path / "results" / "shap_stability_all_levels_standardized.csv"
shap_stability_df.to_csv(shap_stability_path, index=False)
print(f"SHAP stability results saved to: {shap_stability_path}")
shap_stability_df


# --- Code Cell 3 ---
plt.figure(figsize=(10, 6))
plt.plot(shap_stability_df["Perturbation_Level"], shap_stability_df["Spearman_Rho"], marker="o")
plt.title("SHAP Rank Stability Across Perturbation Levels")
plt.xlabel("Perturbation Level")
plt.ylabel("Spearman Rank Correlation")
plt.tight_layout()
fig_path = base_path / "figures" / "shap_stability_vs_perturbation_standardized.png"
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Figure saved to: {fig_path}")


# --- Code Cell 4 ---
example_level = 0.10
example_noise = np.random.normal(0, example_level, X_shap.shape)
X_perturbed = pd.DataFrame(X_shap.values + example_noise, columns=X_shap.columns, index=X_shap.index)
shap_values_pert = explainer(X_perturbed)


# --- Code Cell 5 ---
orig_rank = base_importance.rank(ascending=False)
pert_importance = pd.Series(np.abs(shap_values_pert.values).mean(axis=0), index=X_shap.columns)
pert_rank = pert_importance.rank(ascending=False)
rho, pval = spearmanr(orig_rank, pert_rank)
print("Spearman Rank Correlation:", rho)
print("p-value:", pval)


# --- Code Cell 6 ---
orig_top = set(base_importance.sort_values(ascending=False).head(top_k).index)
pert_top = set(pert_importance.sort_values(ascending=False).head(top_k).index)
overlap_ratio = len(orig_top.intersection(pert_top)) / top_k
explain_stability = pd.DataFrame({"Metric": ["Spearman_Rank_Correlation", f"Top_{top_k}_Feature_Overlap"], "Value": [rho, overlap_ratio]})
exp_path = base_path / "results" / "explainability_stability_single_level.csv"
explain_stability.to_csv(exp_path, index=False)
print(f"Top-{top_k} feature overlap: {overlap_ratio:.3f}")
print(f"Explainability stability results saved to: {exp_path}")
explain_stability


# --- Save state for next step ---
save_state(15, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 15 completed successfully!")
