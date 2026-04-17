import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = PROJECT_ROOT / "dissertation_outputs"
DATA_DIR = BASE_PATH / "data"
MODEL_DIR = BASE_PATH / "models"
FIGURE_DIR = BASE_PATH / "figures"
RESULT_DIR = BASE_PATH / "results"
STATE_DIR = BASE_PATH / "state"

for folder in [DATA_DIR, MODEL_DIR, FIGURE_DIR, RESULT_DIR, STATE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def save_state(step_num, **kwargs):
    state_file = STATE_DIR / f"state_step_{step_num:02d}.pkl"
    with open(state_file, "wb") as f:
        pickle.dump(kwargs, f)
    print(f"State saved to: {state_file}")


def load_state(step_num):
    state_file = STATE_DIR / f"state_step_{step_num:02d}.pkl"
    if not state_file.exists():
        print(f"ERROR: State file not found: {state_file}")
        print(f"Please run step {step_num} first.")
        sys.exit(1)
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    print(f"State loaded from: {state_file}")
    return state


def add_supervised_target(df):
    frame = df.sort_values(["Ticker", "Date"]).copy()
    frame["Future_Return_5D"] = frame.groupby("Ticker")["Close"].shift(-5) / frame["Close"] - 1
    frame["Target"] = (frame["Future_Return_5D"] <= -0.03).astype(int)
    frame = frame.dropna(subset=["Future_Return_5D"]).reset_index(drop=True)
    return frame


def add_chronological_split(df):
    frame = df.sort_values(["Date", "Ticker"]).reset_index(drop=True).copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    unique_dates = np.array(sorted(frame["Date"].dt.normalize().unique()))
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

    frame["Split"] = frame["Date"].apply(assign_split)
    return frame


def split_xy(df, feature_cols):
    clean = df.copy()
    clean[feature_cols] = clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    clean[feature_cols] = clean[feature_cols].fillna(clean[feature_cols].median(numeric_only=True))
    splits = {}
    for split_name in ["Train", "Validation", "Test"]:
        subset = clean[clean["Split"] == split_name].copy()
        splits[split_name] = (
            subset[feature_cols].copy(),
            subset["Target"].astype(int).copy(),
            subset,
        )
    return splits


def build_logistic_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def build_random_forest_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )


def build_xgb_model():
    try:
        from xgboost import XGBClassifier

        return (
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            ),
            "XGBoost",
        )
    except ModuleNotFoundError:
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
        return model, "Gradient Boosting Fallback"


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def metric_row(model_name, split_name, y_true, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan
    return {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1_Score": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Predicted_Events": int(np.sum(pred)),
        "True_Events": int(np.sum(y_true)),
    }


def save_model(model, filename):
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
    return path


def load_supervised_models(model_names=None):
    candidates = {
        "Logistic Regression": MODEL_DIR / "logistic_regression_drawdown_classifier.pkl",
        "Random Forest": MODEL_DIR / "random_forest_drawdown_classifier.pkl",
        "XGBoost": MODEL_DIR / "xgboost_drawdown_classifier.pkl",
        "Gradient Boosting Fallback": MODEL_DIR / "gradient_boosting_drawdown_classifier.pkl",
    }
    models = {}
    for name, path in candidates.items():
        if model_names is not None and name not in model_names:
            continue
        if path.exists():
            models[name] = joblib.load(path)
    return models


def make_stress_scenarios(X_test):
    scenarios = {}
    price_spike = X_test.copy()
    price_spike["Momentum_10"] = price_spike["Momentum_10"] * 1.5
    price_spike["MA_Ratio_10"] = price_spike["MA_Ratio_10"] * 1.08
    price_spike["MA_Ratio_20"] = price_spike["MA_Ratio_20"] * 1.10
    price_spike["Return_Zscore_20"] = price_spike["Return_Zscore_20"] + 2.0
    scenarios["Price Spike"] = price_spike

    volume_shock = X_test.copy()
    volume_shock["Volume_Change"] = volume_shock["Volume_Change"].fillna(0) + 2.0
    volume_shock["Volatility_10"] = volume_shock["Volatility_10"] * 1.3
    volume_shock["BB_Width"] = volume_shock["BB_Width"] * 1.5
    scenarios["Volume Shock"] = volume_shock

    volatility_shock = X_test.copy()
    volatility_shock["Volatility_10"] = volatility_shock["Volatility_10"] * 1.8
    volatility_shock["Volatility_20"] = volatility_shock["Volatility_20"] * 1.8
    volatility_shock["Drawdown"] = volatility_shock["Drawdown"] - 0.10
    volatility_shock["Rolling_Kurt_20"] = volatility_shock["Rolling_Kurt_20"] + 1.0
    volatility_shock["BB_Width"] = volatility_shock["BB_Width"] * 1.6
    scenarios["Volatility Shock"] = volatility_shock
    return scenarios
