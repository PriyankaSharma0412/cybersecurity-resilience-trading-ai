"""
Step 18: 18. Validation Checks

#### 18.1 Notebook Output Snapshot

#### 18.2 Key File Paths

#### 18.3 Research Question Alignment Table

Run: python scripts/step_18_validation_checks.py
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
print("Step 18: 18. Validation Checks")
print("============================================================\n")

# --- Load state from Step 17 ---
_state = load_state(17)
df = _state["df"]
feature_cols = _state["feature_cols"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
assert set(["Train", "Validation", "Test"]) == set(df["Split"].unique())
assert eval_df["Evaluation_Label"].sum() > 0
assert all(col in X_test_scenarios["Volume Shock"].columns for col in feature_cols)
assert metrics_df["Model"].nunique() == 3
assert results_df.shape[0] == len(perturbation_levels)
train_max = df.loc[df["Split"] == "Train", "Date"].max()
val_min = df.loc[df["Split"] == "Validation", "Date"].min()
val_max = df.loc[df["Split"] == "Validation", "Date"].max()
test_min = df.loc[df["Split"] == "Test", "Date"].min()
print("Validation checks passed.")
print(f"Train ends on: {train_max.date()}")
print(f"Validation starts on: {val_min.date()} and ends on: {val_max.date()}")
print(f"Test starts on: {test_min.date()}")
print(f"Injected anomaly rows: {int(eval_df['Evaluation_Label'].sum())}")


# --- Code Cell 2 ---
display(metrics_df.head())
display(final_summary)
display(failure_analysis)


# --- Code Cell 3 ---
key_paths = {"Evaluation dataset": evaluation_dataset_path, "Event log": event_log_path, "Metrics": metrics_path, "Robustness": robustness_path, "Stress results": stress_results_path, "SHAP importance": shap_path, "Final summary": summary_path, "Narrative summary": narrative_path}
for label, path_obj in key_paths.items():
    print(f"{label}: {path_obj}")


# --- Code Cell 4 ---
rq_alignment = pd.DataFrame({
    "Research_Question_Area": ["Cybersecurity and anomaly detection", "Behaviour under abnormal inputs", "Predictive resilience without loss of interpretability", "Explainability stability under perturbation", "Integrated monitoring framework"],
    "Implemented_Evidence": ["Isolation Forest, Autoencoder, and One-Class SVM anomaly pipeline", "Gaussian perturbation tests plus named stress scenarios", "Controlled injection metrics with Precision, Recall, F1, and Time-to-Detection", "SHAP feature importance and stability analysis across perturbation levels", "Unified output tables, failure analysis, and narrative synthesis"],
    "Primary_Output": [str(comparison_path), str(robustness_path), str(metrics_path), str(shap_stability_path), str(summary_path)]
})
rq_alignment_path = base_path / "results" / "research_question_alignment.csv"
rq_alignment.to_csv(rq_alignment_path, index=False)
print(f"Research question alignment saved to: {rq_alignment_path}")
rq_alignment


print("\nStep 18 completed successfully!")
