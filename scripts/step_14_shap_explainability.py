"""
Step 14: 14. SHAP Explainability for Isolation Forest

#### 14.1 Compute SHAP Values on a Held-Out Test Sample

#### 14.2 SHAP Summary Plot

#### 14.3 Mean Absolute SHAP Importance

#### 14.4 SHAP Bar Plot

Run: python scripts/step_14_shap_explainability.py
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
print("Step 14: 14. SHAP Explainability for Isolation Forest")
print("============================================================\n")

# --- Load state from Step 13 ---
_state = load_state(13)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
X_shap = X_test.sample(min(500, len(X_test)), random_state=42)
explainer = shap.Explainer(iso_model, X_shap)
shap_values = explainer(X_shap)


# --- Code Cell 2 ---
shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
fig_path = base_path / "figures" / "shap_summary_plot_test_sample.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Figure saved to: {fig_path}")


# --- Code Cell 3 ---
shap_importance = pd.DataFrame({"Feature": X_shap.columns, "Mean_Abs_SHAP": np.abs(shap_values.values).mean(axis=0)}).sort_values(by="Mean_Abs_SHAP", ascending=False)
shap_path = base_path / "results" / "shap_feature_importance_test_sample.csv"
shap_importance.to_csv(shap_path, index=False)
print(f"SHAP importance saved to: {shap_path}")
shap_importance


# --- Code Cell 4 ---
plt.figure(figsize=(8, 5))
sns.barplot(data=shap_importance, x="Mean_Abs_SHAP", y="Feature")
plt.title("Mean Absolute SHAP Feature Importance")
plt.tight_layout()
fig_path = base_path / "figures" / "shap_feature_importance_bar_test_sample.png"
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Figure saved to: {fig_path}")


# --- Save state for next step ---
save_state(14, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 14 completed successfully!")
