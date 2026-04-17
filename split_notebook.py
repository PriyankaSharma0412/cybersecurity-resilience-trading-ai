"""
Script to split the notebook into separate Python files per section.
Each script is fully self-contained with imports, setup, and data loading.
Uses pickle to pass state between steps.
"""
import json
import os

nb = json.load(open('Updated_Financial_AI_Robustness_Evaluation.ipynb', encoding='utf-8'))

# Build a mapping of sections
sections = []
current_section = None

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source']).strip()
        lines = src.split('\n')
        first_line = lines[0].strip()
        if first_line.startswith('## ') or (first_line.startswith('##') and not first_line.startswith('###')):
            current_section = {
                'header': src,
                'first_line': first_line,
                'index': len(sections),
                'cells': [],
                'sub_headers': [],
            }
            sections.append(current_section)
        elif first_line.startswith('###') or first_line.startswith('####'):
            if current_section:
                current_section['sub_headers'].append(src)
    elif cell['cell_type'] == 'code' and current_section is not None:
        code = ''.join(cell['source'])
        current_section['cells'].append(code)

# Create scripts directory
os.makedirs('scripts', exist_ok=True)

# Section name mapping
section_names = {
    1: 'environment_setup',
    2: 'install_import_libraries',
    3: 'data_collection',
    4: 'exploratory_data_analysis',
    5: 'feature_engineering',
    6: 'prepare_model_features',
    7: 'time_series_evaluation',
    8: 'isolation_forest',
    9: 'autoencoder',
    10: 'one_class_svm',
    11: 'supervised_evaluation',
    12: 'robustness_testing',
    13: 'stress_scenarios',
    14: 'shap_explainability',
    15: 'explanation_stability',
    16: 'final_summary',
    17: 'dissertation_narrative',
    18: 'validation_checks',
}

# Common setup code
COMMON_IMPORTS = '''import os
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
'''

# State save/load additions for each step
# This maps step_num -> (load_from_step, vars_to_save)
STEP_STATE_INFO = {
    3: {
        'load': None,
        'save': 'save_state(3, data=data, tickers=tickers, close_prices=data["Close"].copy())',
    },
    4: {
        'load': 3,
        'load_vars': ['data', 'tickers'],
        'save': 'save_state(4, data=data, tickers=tickers, close_prices=close_prices, returns_all=returns_all)',
    },
    5: {
        'load': 4,
        'load_vars': ['data', 'tickers', 'close_prices', 'returns_all'],
        'save': 'save_state(5, data=data, tickers=tickers, close_prices=close_prices, returns_all=returns_all, df=df)',
    },
    6: {
        'load': 5,
        'load_vars': ['df'],
        'save': 'save_state(6, df=df, feature_cols=feature_cols)',
    },
    7: {
        'load': 6,
        'load_vars': ['df', 'feature_cols'],
        'save': 'save_state(7, df=df, feature_cols=feature_cols)',
    },
    8: {
        'load': 7,
        'load_vars': ['df', 'feature_cols'],
        'save': 'save_state(8, df=df, feature_cols=feature_cols, iso_model=iso_model, if_split_counts=if_split_counts)',
    },
    9: {
        'load': 8,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(9, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    10: {
        'load': 9,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(10, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    11: {
        'load': 10,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(11, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    12: {
        'load': 11,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(12, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    13: {
        'load': 12,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(13, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    14: {
        'load': 13,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(14, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    15: {
        'load': 14,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(15, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    16: {
        'load': 15,
        'load_vars': ['df', 'feature_cols', 'iso_model'],
        'save': 'save_state(16, df=df, feature_cols=feature_cols, iso_model=iso_model)',
    },
    17: {
        'load': 16,
        'load_vars': ['df', 'feature_cols'],
        'save': 'save_state(17, df=df, feature_cols=feature_cols)',
    },
    18: {
        'load': 17,
        'load_vars': ['df', 'feature_cols'],
        'save': None,
    },
}


for i, sec in enumerate(sections):
    sec_num = i + 1
    name = section_names.get(sec_num, f'section_{sec_num}')
    filename = f'step_{sec_num:02d}_{name}.py'
    filepath = os.path.join('scripts', filename)
    clean_title = sec["first_line"].replace("#", "").strip()

    with open(filepath, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f'"""\n')
        f.write(f'Step {sec_num}: {clean_title}\n')
        for sh in sec['sub_headers']:
            f.write(f'\n{sh.replace(chr(10), chr(10))}\n')
        f.write(f'\nRun: python scripts/{filename}\n')
        f.write(f'"""\n\n')

        if sec_num <= 2:
            # Steps 1-2: just original code
            for j, code in enumerate(sec['cells']):
                lines = code.split('\n')
                processed = []
                for line in lines:
                    if line.strip().startswith('!pip'):
                        processed.append(f'# {line}  # Run manually if needed')
                    else:
                        processed.append(line)
                f.write(f'# --- Code Cell {j+1} ---\n')
                f.write('\n'.join(processed))
                f.write('\n\n')
        else:
            # Steps 3+: full self-contained script
            f.write('# ============================================\n')
            f.write('# Common Setup\n')
            f.write('# ============================================\n')
            f.write(COMMON_IMPORTS)
            f.write('\n')
            f.write(f'print("\\n{"="*60}")\n')
            f.write(f'print("Step {sec_num}: {clean_title}")\n')
            f.write(f'print("{"="*60}\\n")\n\n')

            # Load state from previous step
            state_info = STEP_STATE_INFO.get(sec_num)
            if state_info and state_info.get('load'):
                load_step = state_info['load']
                load_vars = state_info.get('load_vars', [])
                f.write(f'# --- Load state from Step {load_step} ---\n')
                f.write(f'_state = load_state({load_step})\n')
                for var in load_vars:
                    f.write(f'{var} = _state["{var}"]\n')
                f.write(f'del _state\n')
                f.write(f'print("Previous state loaded successfully.\\n")\n\n')

            # Write code cells
            for j, code in enumerate(sec['cells']):
                lines = code.split('\n')
                processed = []
                for line in lines:
                    if line.strip().startswith('!pip'):
                        processed.append(f'# {line}  # Run manually')
                    else:
                        processed.append(line)
                f.write(f'# --- Code Cell {j+1} ---\n')
                f.write('\n'.join(processed))
                f.write('\n\n')

            # Save state for next step
            if state_info and state_info.get('save'):
                f.write(f'# --- Save state for next step ---\n')
                f.write(f'{state_info["save"]}\n\n')

        f.write(f'print("\\nStep {sec_num} completed successfully!")\n')

    print(f'Created: {filepath} ({len(sec["cells"])} code cells)')

print(f'\nTotal: {len(sections)} scripts created in scripts/')
