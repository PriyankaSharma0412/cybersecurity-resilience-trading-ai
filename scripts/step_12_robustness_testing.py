"""
Step 12: 12. Robustness Testing on Held-Out Test Data

#### 12.1 Run Gaussian Perturbation Tests on the Test Split

#### 12.2 Robustness Results Table

#### 12.3 Save Robustness Results

#### 12.4 Plot Label Flips vs Perturbation

#### 12.5 Plot Jaccard Similarity vs Perturbation

Run: python scripts/step_12_robustness_testing.py
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
print("Step 12: 12. Robustness Testing on Held-Out Test Data")
print("============================================================\n")

# --- Load state from Step 11 ---
_state = load_state(11)
df = _state["df"]
feature_cols = _state["feature_cols"]
iso_model = _state["iso_model"]
del _state
print("Previous state loaded successfully.\n")

# --- Code Cell 1 ---
perturbation_levels = [0.05, 0.10, 0.15, 0.20, 0.30]
results = []
baseline_test_labels = {"Isolation Forest": df.loc[df["Split"] == "Test", "Anomaly_IF"].astype(int).values, "Autoencoder": df.loc[df["Split"] == "Test", "Anomaly_AE"].astype(int).values, "One-Class SVM": df.loc[df["Split"] == "Test", "Anomaly_OCSVM"].astype(int).values}
baseline_test_scores = {"Isolation Forest": df.loc[df["Split"] == "Test", "IF_score"].values, "Autoencoder": df.loc[df["Split"] == "Test", "AE_error"].values, "One-Class SVM": df.loc[df["Split"] == "Test", "OCSVM_score"].values}


# --- Code Cell 2 ---
for level in perturbation_levels:
    noise = np.random.normal(0, level, X_test.shape)
    X_perturbed = pd.DataFrame(X_test.values + noise, columns=X_test.columns, index=X_test.index)
    if_pred_pert, if_score_pert = predict_if(X_perturbed)
    ae_pred_pert, ae_score_pert = predict_ae(X_perturbed)
    oc_pred_pert, oc_score_pert = predict_ocsvm(X_perturbed)
    model_outputs = {"IF": (baseline_test_labels["Isolation Forest"], if_pred_pert, baseline_test_scores["Isolation Forest"], if_score_pert), "AE": (baseline_test_labels["Autoencoder"], ae_pred_pert, baseline_test_scores["Autoencoder"], ae_score_pert), "OCSVM": (baseline_test_labels["One-Class SVM"], oc_pred_pert, baseline_test_scores["One-Class SVM"], oc_score_pert)}
    result_row = {"Perturbation_Level": level}
    for short_name, (base_labels, pert_labels, base_scores, pert_scores) in model_outputs.items():
        flips = int(np.sum(base_labels != pert_labels))
        jac = jaccard_score(base_labels, pert_labels, zero_division=0)
        ttest_p = ttest_ind(base_scores, pert_scores, equal_var=False).pvalue
        try:
            wilcoxon_p = wilcoxon(base_scores, pert_scores).pvalue
        except Exception:
            wilcoxon_p = np.nan
        result_row[f"{short_name}_Label_Flips"] = flips
        result_row[f"{short_name}_Jaccard"] = jac
        result_row[f"{short_name}_ttest_pvalue"] = ttest_p
        result_row[f"{short_name}_wilcoxon_pvalue"] = wilcoxon_p
    results.append(result_row)


# --- Code Cell 3 ---
results_df = pd.DataFrame(results)
results_df["IF_Robustness_Score"] = 1 - (results_df["IF_Label_Flips"] / len(X_test))
results_df["AE_Robustness_Score"] = 1 - (results_df["AE_Label_Flips"] / len(X_test))
results_df["OCSVM_Robustness_Score"] = 1 - (results_df["OCSVM_Label_Flips"] / len(X_test))
results_df


# --- Code Cell 4 ---
robustness_path = base_path / "results" / "perturbation_robustness_results_standardized.csv"
results_df.to_csv(robustness_path, index=False)
print(f"Robustness results saved to: {robustness_path}")


# --- Code Cell 5 ---
plt.figure(figsize=(10, 6))
plt.plot(results_df["Perturbation_Level"], results_df["IF_Label_Flips"], marker="o", label="Isolation Forest")
plt.plot(results_df["Perturbation_Level"], results_df["AE_Label_Flips"], marker="s", label="Autoencoder")
plt.plot(results_df["Perturbation_Level"], results_df["OCSVM_Label_Flips"], marker="^", label="One-Class SVM")
plt.title("Label Flips Under Increasing Gaussian Perturbation")
plt.xlabel("Perturbation Level")
plt.ylabel("Number of Label Flips")
plt.legend()
plt.tight_layout()
fig_path = base_path / "figures" / "label_flips_vs_gaussian_perturbation_standardized.png"
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Figure saved to: {fig_path}")


# --- Code Cell 6 ---
plt.figure(figsize=(10, 6))
plt.plot(results_df["Perturbation_Level"], results_df["IF_Jaccard"], marker="o", label="Isolation Forest")
plt.plot(results_df["Perturbation_Level"], results_df["AE_Jaccard"], marker="s", label="Autoencoder")
plt.plot(results_df["Perturbation_Level"], results_df["OCSVM_Jaccard"], marker="^", label="One-Class SVM")
plt.title("Jaccard Similarity Under Increasing Gaussian Perturbation")
plt.xlabel("Perturbation Level")
plt.ylabel("Jaccard Similarity")
plt.legend()
plt.tight_layout()
fig_path = base_path / "figures" / "jaccard_vs_gaussian_perturbation_standardized.png"
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Figure saved to: {fig_path}")


# --- Save state for next step ---
save_state(12, df=df, feature_cols=feature_cols, iso_model=iso_model)

print("\nStep 12 completed successfully!")
