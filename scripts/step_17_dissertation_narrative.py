"""
Step 17: 17. Dissertation Narrative Mapping

#### 17.1 Save Narrative Summary

Run: python scripts/step_17_dissertation_narrative.py
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
print("Step 17: 17. Dissertation Narrative Mapping")
print("============================================================\n")

# --- Load state from Step 16 ---
_state = load_state(16)
df = _state["df"]
feature_cols = _state["feature_cols"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
best_f1_model = comparison.sort_values("F1_Score", ascending=False).iloc[0]["Model"]
best_robust_model = final_summary.sort_values("Avg_Gaussian_Robustness", ascending=False).iloc[0]["Model"]
best_detection_model = comparison.sort_values("Event_Detection_Rate", ascending=False).iloc[0]["Model"]
narrative_lines = [
    "# Dissertation Alignment Summary",
    "",
    "## Aim Alignment",
    "The notebook now evaluates anomaly detection, robustness, and explainability within one reproducible time-series pipeline.",
    "",
    "## Research Question Coverage",
    "- Controlled event injection enables Precision, Recall, F1-Score, and Time-to-Detection on held-out data.",
    f"- {best_detection_model} achieved the highest event detection rate on the controlled test set.",
    f"- {best_f1_model} achieved the strongest F1-Score on the controlled test set.",
    f"- {best_robust_model} delivered the strongest average Gaussian robustness across held-out perturbation levels.",
    "",
    "## Explainability Scope",
    "- SHAP is explicitly scoped to Isolation Forest as the primary interpretable model for dissertation analysis.",
    "- Explanation stability is quantified through Spearman rank correlation and top-k feature overlap under perturbation.",
    "",
    "## Methodological Improvements",
    "- Models are trained only on the chronological training split.",
    "- Validation and test splits are reserved for controlled evaluation and robustness assessment.",
    "- Standardized CSV and figure outputs are written to reproducible paths."
]
narrative_text = "`n".join(narrative_lines)
print(narrative_text)


# --- Code Cell 2 ---
narrative_path = base_path / "results" / "dissertation_alignment_summary.md"
with open(narrative_path, "w", encoding="utf-8") as f:
    f.write(narrative_text)
print(f"Narrative summary saved to: {narrative_path}")


# --- Save state for next step ---
save_state(17, df=df, feature_cols=feature_cols)

print("\nStep 17 completed successfully!")
