"""
Single-file supervised financial robustness pipeline.

Run from the project root:
    python scripts/supervised_pipeline.py

Pipeline sections:
    1. Imports and global configuration
    2. Environment and persistence helpers
    3. Data loading and feature engineering
    4. Supervised target and chronological split
    5. Exploratory data analysis
    6. Model training and evaluation
    7. Robustness and stress testing
    8. Explainability and SHAP stability
    9. Advanced validation and benchmarking
   10. Final reporting and execution
"""
from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================

import os
import pickle
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
warnings.filterwarnings("ignore")

# =============================================================================
# 2. Third-Party Imports
# =============================================================================

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yfinance as yf
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import binomtest, ks_2samp, spearmanr
from scipy.stats import ttest_rel
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 3. Global Configuration
# =============================================================================

np.random.seed(42)
sns.set_theme(style="whitegrid")

BASE_PATH = PROJECT_ROOT / "dissertation_outputs"
DATA_DIR = BASE_PATH / "data"
MODEL_DIR = BASE_PATH / "models"
FIGURE_DIR = BASE_PATH / "figures"
RESULT_DIR = BASE_PATH / "results"
STATE_DIR = BASE_PATH / "state"
DEPLOYMENT_DIR = BASE_PATH / "deployment"
CHAPTER4_DIR = FIGURE_DIR / "chapter4_visuals"
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig"
YFINANCE_CACHE_DIR = PROJECT_ROOT / ".yfinance_cache"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "GS", "KO", "XOM", "SPY", "QQQ"]
START_DATE = "2010-01-01"
END_DATE = "2024-01-01"
FEATURE_COLS = [
    "Return", "Volatility_10", "Volatility_20", "Momentum_10", "MA_Ratio_10", "MA_Ratio_20",
    "Volume_Change", "Rolling_Skew_20", "Rolling_Kurt_20", "Return_Zscore_20", "Drawdown",
    "RSI_14", "MACD", "MACD_Signal", "BB_Width",
]


# =============================================================================
# 4. Environment and Persistence Helpers
# =============================================================================

def stage(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def setup_environment():
    stage("1. Environment Setup")
    for folder in [DATA_DIR, MODEL_DIR, FIGURE_DIR, RESULT_DIR, STATE_DIR, DEPLOYMENT_DIR, CHAPTER4_DIR, MPL_CONFIG_DIR, YFINANCE_CACHE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))
    install_safe_csv_writer()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output folder: {BASE_PATH}")


def save_state(step_num, **kwargs):
    path = STATE_DIR / f"state_step_{step_num:02d}.pkl"
    with open(path, "wb") as f:
        pickle.dump(kwargs, f)
    print(f"State saved: {path}")


_ORIGINAL_DATAFRAME_TO_CSV = pd.DataFrame.to_csv
_SAFE_CSV_WRITER_INSTALLED = False


def _fallback_csv_path(path):
    path = Path(path)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def _safe_dataframe_to_csv(self, path_or_buf=None, *args, **kwargs):
    try:
        return _ORIGINAL_DATAFRAME_TO_CSV(self, path_or_buf, *args, **kwargs)
    except PermissionError:
        if path_or_buf is None or hasattr(path_or_buf, "write"):
            raise
        fallback = _fallback_csv_path(path_or_buf)
        result = _ORIGINAL_DATAFRAME_TO_CSV(self, fallback, *args, **kwargs)
        print(f"Permission denied for {path_or_buf}; wrote fallback file {fallback}")
        return result


def install_safe_csv_writer():
    global _SAFE_CSV_WRITER_INSTALLED
    if not _SAFE_CSV_WRITER_INSTALLED:
        pd.DataFrame.to_csv = _safe_dataframe_to_csv
        _SAFE_CSV_WRITER_INSTALLED = True


def safe_to_csv(df, path, **kwargs):
    try:
        _ORIGINAL_DATAFRAME_TO_CSV(df, path, **kwargs)
        return path
    except PermissionError:
        fallback = _fallback_csv_path(path)
        _ORIGINAL_DATAFRAME_TO_CSV(df, fallback, **kwargs)
        print(f"Permission denied for {path}; wrote fallback file {fallback}")
        return fallback


# =============================================================================
# 5. Data Loading and Feature Engineering
# =============================================================================

def load_existing_dataset():
    path = DATA_DIR / "supervised_drawdown_dataset.csv"
    if path.exists() and path.stat().st_size > 1000:
        df = pd.read_csv(path, parse_dates=["Date"])
        required = set(FEATURE_COLS + ["Date", "Ticker", "Close", "Future_Return_5D", "Target", "Split"])
        if required.issubset(df.columns) and not df.empty:
            print(f"Loaded existing supervised dataset: {df.shape}")
            return df
    return None


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def download_market_data():
    # Download daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False, group_by="column", threads=False, progress=False)
    if data.empty:
        raise RuntimeError("Yahoo Finance returned empty data and no saved supervised dataset is available.")
    data.to_csv(DATA_DIR / "market_data_raw_wide.csv")
    return data


def show_imported_data(df, rows=5):
    stage("2a. Imported Data Preview")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("Column names:")
    print(", ".join(map(str, df.columns)))
    print(f"\nFirst {rows} rows:")
    print(df.head(rows).to_string(index=False))
    if {"Split", "Target"}.issubset(df.columns):
        print("\nRows by split and target:")
        print(df.groupby(["Split", "Target"]).size().rename("Rows").reset_index().to_string(index=False))


def build_features(market_data):
    frames = []
    for ticker in TICKERS:
        asset = pd.DataFrame({
            "Date": market_data.index,
            "Ticker": ticker,
            "Open": market_data["Open"][ticker],
            "High": market_data["High"][ticker],
            "Low": market_data["Low"][ticker],
            "Close": market_data["Close"][ticker],
            "Volume": market_data["Volume"][ticker],
        }).dropna(subset=["Close"])
        asset["Return"] = asset["Close"].pct_change()
        asset["Volatility_10"] = asset["Return"].rolling(10).std()
        asset["Volatility_20"] = asset["Return"].rolling(20).std()
        asset["Momentum_10"] = asset["Close"].pct_change(10)
        asset["MA_Ratio_10"] = asset["Close"] / asset["Close"].rolling(10).mean() - 1
        asset["MA_Ratio_20"] = asset["Close"] / asset["Close"].rolling(20).mean() - 1
        asset["Volume_Change"] = asset["Volume"].pct_change()
        asset["Rolling_Skew_20"] = asset["Return"].rolling(20).skew()
        asset["Rolling_Kurt_20"] = asset["Return"].rolling(20).kurt()
        mu = asset["Return"].rolling(20).mean()
        sigma = asset["Return"].rolling(20).std()
        asset["Return_Zscore_20"] = (asset["Return"] - mu) / sigma
        asset["Drawdown"] = asset["Close"] / asset["Close"].cummax() - 1
        asset["RSI_14"] = compute_rsi(asset["Close"])
        ema_12 = asset["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = asset["Close"].ewm(span=26, adjust=False).mean()
        asset["MACD"] = ema_12 - ema_26
        asset["MACD_Signal"] = asset["MACD"].ewm(span=9, adjust=False).mean()
        ma20 = asset["Close"].rolling(20).mean()
        std20 = asset["Close"].rolling(20).std()
        asset["BB_Width"] = ((ma20 + 2 * std20) - (ma20 - 2 * std20)) / ma20
        frames.append(asset)
    df = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURE_COLS + ["Close"]).reset_index(drop=True)


# =============================================================================
# 6. Supervised Target, Chronological Split, and Dataset Export
# =============================================================================

def add_target_and_split(df):
    if "Future_Return_5D" not in df.columns or "Target" not in df.columns:
        df = df.sort_values(["Ticker", "Date"]).copy()
        df["Future_Return_5D"] = df.groupby("Ticker")["Close"].shift(-5) / df["Close"] - 1
        df["Target"] = (df["Future_Return_5D"] <= -0.03).astype(int)
        df = df.dropna(subset=["Future_Return_5D"]).reset_index(drop=True)
    if "Split" not in df.columns:
        df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        # Chronological split: earliest 70% of dates = Train, next 15% = Validation, final 15% = Test.
        dates = np.array(sorted(pd.to_datetime(df["Date"]).dt.normalize().unique()))
        train_end, val_end = int(len(dates) * 0.70), int(len(dates) * 0.85)
        train_dates, val_dates = set(dates[:train_end]), set(dates[train_end:val_end])
        df["Split"] = ["Train" if pd.Timestamp(d).normalize().to_datetime64() in train_dates else "Validation" if pd.Timestamp(d).normalize().to_datetime64() in val_dates else "Test" for d in df["Date"]]
    return df


def prepare_dataset():
    stage("2. Data Import, Feature Engineering, Target, and Split")
    df = load_existing_dataset()
    if df is None:
        df = build_features(download_market_data())
        df = add_target_and_split(df)
    show_imported_data(df)
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median(numeric_only=True))
    df["Target"] = df["Target"].astype(int)
    df.to_csv(DATA_DIR / "supervised_drawdown_dataset.csv", index=False)
    df.to_csv(DATA_DIR / "evaluation_dataset_with_injected_events.csv", index=False)
    df.to_csv(DATA_DIR / "multi_asset_financial_features.csv", index=False)
    df[FEATURE_COLS].to_csv(DATA_DIR / "multi_asset_model_features.csv", index=False)
    raw_cols = [c for c in ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df[raw_cols].to_csv(DATA_DIR / "market_data_raw.csv", index=False)
    split_summary = df.groupby("Split").agg(Rows=("Ticker", "size"), Assets=("Ticker", "nunique"), Start_Date=("Date", "min"), End_Date=("Date", "max"), Drawdown_Events=("Target", "sum"), Event_Rate=("Target", "mean")).reset_index()
    split_summary.to_csv(RESULT_DIR / "time_series_split_summary.csv", index=False)
    df.groupby(["Split", "Target"]).size().rename("Rows").reset_index().to_csv(RESULT_DIR / "supervised_target_summary.csv", index=False)
    print(split_summary.to_string(index=False))
    save_state(7, df=df, feature_cols=FEATURE_COLS, target_col="Target")
    return df


# =============================================================================
# 7. Exploratory Data Analysis
# =============================================================================

def create_eda_figures(df):
    stage("3. Exploratory Data Analysis")
    plt.figure(figsize=(12, 6))
    for ticker in df["Ticker"].unique()[:10]:
        sample = df[df["Ticker"] == ticker]
        plt.plot(sample["Date"], sample["Close"], label=ticker, linewidth=1)
    plt.title("Multi-Asset Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "stock_price_trends_multi_asset.png", dpi=300)
    plt.close()
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[FEATURE_COLS].corr(), cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_correlation_heatmap_multi_asset.png", dpi=300)
    plt.close()


# =============================================================================
# 8. Model Input Preparation and Training
# =============================================================================

def split_xy(df):
    clean = df.copy()
    clean[FEATURE_COLS] = clean[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    clean[FEATURE_COLS] = clean[FEATURE_COLS].fillna(clean[FEATURE_COLS].median(numeric_only=True))
    return {s: (g[FEATURE_COLS].copy(), g["Target"].astype(int).copy(), g.copy()) for s, g in clean.groupby("Split")}


def safe_model_name(name):
    return str(name).replace(" ", "_").replace("-", "_").replace("/", "_")


def display_model_name(safe_name):
    display = str(safe_name).replace("_", " ")
    return display


def best_threshold_from_scores(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def safe_auc(y_true, scores):
    return roc_auc_score(y_true, scores) if pd.Series(y_true).nunique() == 2 else np.nan


def build_xgb_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=1), "XGBoost"
    except ModuleNotFoundError:
        return GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42), "Gradient Boosting Fallback"


def train_models(splits):
    stage("4. Supervised Model Training")
    X_train, y_train, _ = splits["Train"]
    # Train baseline model 1: Logistic Regression.
    # Train baseline model 2: Random Forest.
    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=1),
    }
    # Train baseline model 3: XGBoost when installed, otherwise Gradient Boosting fallback.
    boost_model, boost_name = build_xgb_model()
    models[boost_name] = boost_model
    # Fit each of the three supervised classifiers on the chronological training split.
    for name, model in models.items():
        model.fit(X_train, y_train)
        filename = f"{name.lower().replace(' ', '_').replace('-', '_')}_drawdown_classifier.pkl"
        joblib.dump(model, MODEL_DIR / filename)
        print(f"Trained {name}")
    return models


# =============================================================================
# 9. Core Model Evaluation and Performance Visualisations
# =============================================================================

def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def metric_row(model_name, split_name, y_true, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"Model": model_name, "Split": split_name, "Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred, zero_division=0), "Recall": recall_score(y_true, pred, zero_division=0), "F1_Score": f1_score(y_true, pred, zero_division=0), "ROC_AUC": roc_auc_score(y_true, proba), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "Predicted_Events": int(np.sum(pred)), "True_Events": int(np.sum(y_true))}


def evaluate_models(df, splits, models):
    stage("5. Model Evaluation")
    metrics, parts = [], []
    for split_name in ["Validation", "Test"]:
        X, y, subset = splits[split_name]
        out = subset[["Date", "Ticker", "Split", "Close", "Future_Return_5D", "Target"]].copy()
        for name, model in models.items():
            proba = predict_proba(model, X)
            pred = (proba >= 0.5).astype(int)
            safe = safe_model_name(name)
            out[f"Pred_{safe}"] = pred
            out[f"Proba_{safe}"] = proba
            metrics.append(metric_row(name, split_name, y, pred, proba))
        parts.append(out)
    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(parts, ignore_index=True)
    confusion_df = metrics_df[["Model", "Split", "TN", "FP", "FN", "TP"]].copy()
    metrics_df.to_csv(RESULT_DIR / "supervised_model_metrics.csv", index=False)
    predictions_df.to_csv(RESULT_DIR / "supervised_drawdown_predictions.csv", index=False)
    confusion_df.to_csv(RESULT_DIR / "supervised_confusion_matrices.csv", index=False)
    metrics_df.to_csv(RESULT_DIR / "controlled_injection_metrics.csv", index=False)
    predictions_df.to_csv(RESULT_DIR / "controlled_injection_predictions.csv", index=False)
    confusion_df.to_csv(RESULT_DIR / "controlled_injection_confusion_matrices.csv", index=False)
    metrics_df[metrics_df["Split"] == "Test"].to_csv(RESULT_DIR / "baseline_model_comparison_standardized.csv", index=False)
    print(metrics_df.to_string(index=False))
    save_state(11, df=df, feature_cols=FEATURE_COLS, models=models, metrics_df=metrics_df, predictions_df=predictions_df)
    return metrics_df, predictions_df


def create_performance_figures(metrics_df):
    stage("6. Performance Visualisations")
    test = metrics_df[metrics_df["Split"] == "Test"].copy()
    plot = test.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot, x="Metric", y="Score", hue="Model")
    plt.title("Supervised Drawdown Classifier Test Metrics")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "supervised_model_test_metrics.png", dpi=300)
    plt.close()
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(test.set_index("Model")[["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]], annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Supervised Performance Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "supervised_model_performance_heatmap.png", dpi=300)
    plt.close()

    test = metrics_df[metrics_df["Split"] == "Test"].copy()
    fig, axes = plt.subplots(1, len(test), figsize=(4 * len(test), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, test.iterrows()):
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(row["Model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix_heatmaps_supervised.png", dpi=300)
    plt.close()


# =============================================================================
# 10. Additional Supervised Event Detection Subsystem
# =============================================================================

def train_supervised_event_detectors(splits):
    stage("6A. Additional Supervised Event Detector Training")
    X_train, y_train, _ = splits["Train"]
    boost_model, boost_name = build_xgb_model()
    detectors = {
        "Supervised Logistic Event Detector": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Supervised Random Forest Event Detector": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
        f"Supervised {boost_name} Event Detector": boost_model,
    }
    for name, model in detectors.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{safe_model_name(name).lower()}_supervised_detector.pkl")
        print(f"Trained {name}")
    return detectors


def evaluate_supervised_event_detectors(splits, detectors):
    stage("6B. Supervised Event Detector Thresholding and Evaluation")
    X_val, y_val, _ = splits["Validation"]
    threshold_rows, metric_rows, parts = [], [], []
    thresholds = {}

    for name, detector in detectors.items():
        val_scores = predict_proba(detector, X_val)
        threshold, val_best_f1 = best_threshold_from_scores(y_val, val_scores)
        thresholds[name] = threshold
        val_pred = (val_scores >= threshold).astype(int)
        threshold_rows.append({
            "Model": name,
            "Threshold_Source": "Validation",
            "Supervised_Score_Threshold": threshold,
            "Validation_Best_F1": val_best_f1,
            "Validation_Average_Precision": average_precision_score(y_val, val_scores),
            "Validation_ROC_AUC": safe_auc(y_val, val_scores),
            "Validation_Predicted_Events": int(val_pred.sum()),
        })

    for split_name in ["Validation", "Test"]:
        X, y, subset = splits[split_name]
        out = subset[["Date", "Ticker", "Split", "Close", "Future_Return_5D", "Target"]].copy()
        for name, detector in detectors.items():
            scores = predict_proba(detector, X)
            pred = (scores >= thresholds[name]).astype(int)
            row = metric_row(name, split_name, y, pred, scores)
            row["Model_Type"] = "Additional Supervised Event Detector"
            row["Average_Precision"] = average_precision_score(y, scores)
            row["Supervised_Score_Threshold"] = thresholds[name]
            metric_rows.append(row)
            safe = safe_model_name(name)
            out[f"Pred_{safe}"] = pred
            out[f"Score_{safe}"] = scores
        parts.append(out)

    thresholds_df = pd.DataFrame(threshold_rows)
    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(parts, ignore_index=True)
    thresholds_df.to_csv(RESULT_DIR / "supervised_event_detector_thresholds_validation_tuned.csv", index=False)
    metrics_df.to_csv(RESULT_DIR / "supervised_event_detector_metrics.csv", index=False)
    predictions_df.to_csv(RESULT_DIR / "supervised_event_detector_predictions.csv", index=False)
    print(metrics_df.to_string(index=False))
    return metrics_df, predictions_df, thresholds_df


def external_telemetry_gap_assessment(df):
    stage("6C. External Telemetry and Data Realism Gap Assessment")
    required_sources = {
        "order_book_level2.csv": ["Date", "Ticker", "Bid_Price_1", "Ask_Price_1", "Bid_Size_1", "Ask_Size_1", "Depth_Imbalance"],
        "order_flow_messages.csv": ["Timestamp", "Ticker", "Order_ID", "Message_Type", "Side", "Price", "Quantity"],
        "execution_logs.csv": ["Timestamp", "Ticker", "Order_ID", "Execution_Price", "Executed_Quantity", "Venue"],
        "cyber_event_logs.csv": ["Timestamp", "System_ID", "Event_Type", "Severity", "Attack_Label"],
        "market_manipulation_labels.csv": ["Timestamp", "Ticker", "Manipulation_Type", "Spoofing_Label", "Layering_Label"],
    }
    rows = []
    for filename, required_cols in required_sources.items():
        path = DATA_DIR / filename
        present = path.exists() and path.stat().st_size > 0
        found_cols = []
        missing_cols = required_cols
        row_count = 0
        if present:
            try:
                sample = pd.read_csv(path, nrows=5)
                found_cols = list(sample.columns)
                missing_cols = [c for c in required_cols if c not in found_cols]
                row_count = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
            except Exception:
                found_cols = []
        rows.append({
            "Source_File": filename,
            "Present": bool(present),
            "Rows": max(row_count, 0),
            "Required_Columns": ", ".join(required_cols),
            "Found_Columns": ", ".join(found_cols),
            "Missing_Columns": ", ".join(missing_cols),
            "Operational_Coverage": "Available" if present and not missing_cols else "Missing or incomplete",
        })
    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(RESULT_DIR / "external_telemetry_gap_assessment.csv", index=False)

    realism_rows = []
    feature_requirements = {
        "OHLCV daily market data": ["Open", "High", "Low", "Close", "Volume"],
        "Bid-ask spread": ["Best_Bid", "Best_Ask", "Bid_Ask_Spread"],
        "Depth imbalance": ["Bid_Depth", "Ask_Depth", "Depth_Imbalance"],
        "Quote cancellation pressure": ["Cancel_Count", "New_Order_Count", "Cancel_To_Trade_Ratio"],
        "Execution quality": ["Slippage", "Fill_Rate", "Execution_Delay"],
        "Cyber telemetry": ["System_ID", "Event_Type", "Attack_Label"],
        "News/macro/regime labels": ["News_Sentiment", "Macro_Event", "Regime_Label"],
    }
    existing_cols = set(df.columns)
    for category, cols in feature_requirements.items():
        available = [c for c in cols if c in existing_cols]
        realism_rows.append({
            "Feature_Category": category,
            "Available_Columns": ", ".join(available),
            "Missing_Columns": ", ".join([c for c in cols if c not in existing_cols]),
            "Coverage_Ratio": len(available) / len(cols),
        })
    realism_df = pd.DataFrame(realism_rows)
    realism_df.to_csv(RESULT_DIR / "data_realism_feature_gap_assessment.csv", index=False)
    return gap_df, realism_df


def load_optional_csv(filename, date_cols=None):
    path = DATA_DIR / filename
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, parse_dates=date_cols or [])
    except Exception as exc:
        pd.DataFrame([{"Source_File": filename, "Load_Error": str(exc)}]).to_csv(RESULT_DIR / f"{Path(filename).stem}_load_error.csv", index=False)
        return None


def operational_telemetry_ingestion_and_modeling():
    stage("6C2. Real Operational Telemetry Ingestion and Modeling")
    sources = []
    order_book = load_optional_csv("order_book_level2.csv", date_cols=["Date"])
    if order_book is not None and {"Date", "Ticker"}.issubset(order_book.columns):
        work = order_book.copy()
        work["Date"] = pd.to_datetime(work["Date"]).dt.floor("min")
        if {"Bid_Price_1", "Ask_Price_1"}.issubset(work.columns):
            work["Bid_Ask_Spread"] = work["Ask_Price_1"] - work["Bid_Price_1"]
        if {"Bid_Size_1", "Ask_Size_1"}.issubset(work.columns):
            work["Depth_Imbalance"] = (work["Bid_Size_1"] - work["Ask_Size_1"]) / (work["Bid_Size_1"] + work["Ask_Size_1"] + 1e-12)
        numeric_cols = [c for c in ["Bid_Ask_Spread", "Depth_Imbalance", "Bid_Size_1", "Ask_Size_1"] if c in work.columns]
        if numeric_cols:
            windowed = work.groupby(["Date", "Ticker"])[numeric_cols].mean().reset_index()
            windowed["Telemetry_Source"] = "order_book_level2"
            sources.append(windowed)

    order_flow = load_optional_csv("order_flow_messages.csv", date_cols=["Timestamp"])
    if order_flow is not None and {"Timestamp", "Ticker"}.issubset(order_flow.columns):
        work = order_flow.copy()
        work["Date"] = work["Timestamp"].dt.floor("min")
        msg = work.get("Message_Type", pd.Series("", index=work.index)).astype(str).str.lower()
        work["Cancel_Message"] = msg.str.contains("cancel").astype(int)
        work["Trade_Message"] = msg.str.contains("trade|execute|fill").astype(int)
        work["New_Order_Message"] = msg.str.contains("new|add|submit").astype(int)
        daily = work.groupby(["Date", "Ticker"]).agg(
            Cancel_Count=("Cancel_Message", "sum"),
            Trade_Count=("Trade_Message", "sum"),
            New_Order_Count=("New_Order_Message", "sum"),
            Message_Count=("Ticker", "size"),
        ).reset_index()
        daily["Cancel_To_Trade_Ratio"] = daily["Cancel_Count"] / (daily["Trade_Count"] + 1)
        daily["Telemetry_Source"] = "order_flow_messages"
        sources.append(daily)

    executions = load_optional_csv("execution_logs.csv", date_cols=["Timestamp"])
    if executions is not None and {"Timestamp", "Ticker"}.issubset(executions.columns):
        work = executions.copy()
        work["Date"] = work["Timestamp"].dt.floor("min")
        if {"Execution_Price", "Arrival_Price"}.issubset(work.columns):
            work["Execution_Slippage"] = (work["Execution_Price"] - work["Arrival_Price"]) / (work["Arrival_Price"] + 1e-12)
        else:
            work["Execution_Slippage"] = np.nan
        daily = work.groupby(["Date", "Ticker"]).agg(
            Execution_Count=("Ticker", "size"),
            Mean_Execution_Slippage=("Execution_Slippage", "mean"),
        ).reset_index()
        daily["Telemetry_Source"] = "execution_logs"
        sources.append(daily)

    cyber = load_optional_csv("cyber_event_logs.csv", date_cols=["Timestamp"])
    if cyber is not None and "Timestamp" in cyber.columns:
        work = cyber.copy()
        work["Date"] = work["Timestamp"].dt.floor("min")
        if "Ticker" not in work.columns:
            work["Ticker"] = "SYSTEM"
        sev = work.get("Severity", pd.Series(0, index=work.index))
        work["Severity_Numeric"] = pd.to_numeric(sev, errors="coerce").fillna(sev.astype(str).str.extract(r"(\d+)")[0].astype(float)).fillna(0)
        work["Cyber_Attack_Label"] = pd.to_numeric(work.get("Attack_Label", pd.Series(0, index=work.index)), errors="coerce").fillna(0).astype(int)
        daily = work.groupby(["Date", "Ticker"]).agg(
            Cyber_Event_Count=("Ticker", "size"),
            Max_Cyber_Severity=("Severity_Numeric", "max"),
            Cyber_Attack_Label=("Cyber_Attack_Label", "max"),
        ).reset_index()
        daily["Telemetry_Source"] = "cyber_event_logs"
        sources.append(daily)

    if not sources:
        empty = pd.DataFrame([{
            "Status": "No operational telemetry files found",
            "Expected_Location": str(DATA_DIR),
            "Required_Files": "order_book_level2.csv, order_flow_messages.csv, execution_logs.csv, cyber_event_logs.csv, market_manipulation_labels.csv",
        }])
        empty.to_csv(RESULT_DIR / "operational_telemetry_ingestion_summary.csv", index=False)
        pd.DataFrame().to_csv(RESULT_DIR / "operational_telemetry_model_metrics.csv", index=False)
        pd.DataFrame().to_csv(RESULT_DIR / "operational_telemetry_predictions.csv", index=False)
        return empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    operational = sources[0]
    for source in sources[1:]:
        operational = operational.merge(source.drop(columns=["Telemetry_Source"], errors="ignore"), on=["Date", "Ticker"], how="outer")
    labels = load_optional_csv("market_manipulation_labels.csv", date_cols=["Timestamp"])
    if labels is not None and "Timestamp" in labels.columns:
        labels = labels.copy()
        labels["Date"] = labels["Timestamp"].dt.floor("min")
        if "Ticker" not in labels.columns:
            labels["Ticker"] = "SYSTEM"
        label_cols = [c for c in ["Spoofing_Label", "Layering_Label", "Attack_Label"] if c in labels.columns]
        if label_cols:
            labels["Manipulation_Label"] = labels[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1).astype(int)
            label_daily = labels.groupby(["Date", "Ticker"])["Manipulation_Label"].max().reset_index()
            operational = operational.merge(label_daily, on=["Date", "Ticker"], how="left")
    if "Manipulation_Label" not in operational.columns:
        operational["Manipulation_Label"] = operational.get("Cyber_Attack_Label", 0)
    operational["Manipulation_Label"] = pd.to_numeric(operational["Manipulation_Label"], errors="coerce").fillna(0).astype(int)
    numeric = [c for c in operational.select_dtypes(include=[np.number]).columns if c not in ["Manipulation_Label"]]
    operational[numeric] = operational[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    operational.to_csv(RESULT_DIR / "operational_telemetry_features.csv", index=False)

    summary = pd.DataFrame([{
        "Status": "Operational telemetry ingested",
        "Rows": len(operational),
        "Tickers_or_Systems": operational["Ticker"].nunique(),
        "Feature_Count": len(numeric),
        "Positive_Labels": int(operational["Manipulation_Label"].sum()),
    }])
    summary.to_csv(RESULT_DIR / "operational_telemetry_ingestion_summary.csv", index=False)

    metric_rows, prediction_rows = [], []
    if len(operational) >= 30 and operational["Manipulation_Label"].nunique() == 2 and numeric:
        ordered = operational.sort_values("Date").reset_index(drop=True)
        train_end = int(len(ordered) * 0.70)
        val_end = int(len(ordered) * 0.85)
        train, val, test = ordered.iloc[:train_end], ordered.iloc[train_end:val_end], ordered.iloc[val_end:]
        ordered["Split"] = "Train"
        ordered.loc[ordered.index[train_end:val_end], "Split"] = "Validation"
        ordered.loc[ordered.index[val_end:], "Split"] = "Test"
        telemetry_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
        telemetry_model.fit(train[numeric], train["Manipulation_Label"])
        joblib.dump(telemetry_model, MODEL_DIR / "operational_telemetry_logistic_detector.pkl")
        for split_name, split_df in [("Validation", val), ("Test", test)]:
            if split_df.empty:
                continue
            proba = predict_proba(telemetry_model, split_df[numeric])
            pred = (proba >= 0.5).astype(int)
            row = metric_row("Operational Telemetry Logistic", split_name, split_df["Manipulation_Label"], pred, proba)
            row["Feature_Source"] = "Real provided operational telemetry"
            metric_rows.append(row)
            pred_frame = split_df[["Date", "Ticker", "Manipulation_Label"]].copy()
            pred_frame["Split"] = split_name
            pred_frame["Model"] = "Operational Telemetry Logistic"
            pred_frame["Operational_Risk_Probability"] = proba
            pred_frame["Operational_Alert"] = pred
            prediction_rows.append(pred_frame)
        full_proba = predict_proba(telemetry_model, ordered[numeric])
        full_pred = (full_proba >= 0.5).astype(int)
        full_predictions = ordered[["Date", "Ticker", "Split", "Manipulation_Label"]].copy()
        full_predictions["Model"] = "Operational Telemetry Logistic"
        full_predictions["Operational_Risk_Probability"] = full_proba
        full_predictions["Operational_Alert"] = full_pred
        full_predictions.to_csv(RESULT_DIR / "operational_telemetry_predictions.csv", index=False)
    else:
        pd.DataFrame().to_csv(RESULT_DIR / "operational_telemetry_predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULT_DIR / "operational_telemetry_model_metrics.csv", index=False)
    predictions = pd.read_csv(RESULT_DIR / "operational_telemetry_predictions.csv", parse_dates=["Date"]) if (RESULT_DIR / "operational_telemetry_predictions.csv").stat().st_size > 2 else pd.DataFrame()
    return summary, operational, metrics, predictions


def microstructure_attack_generation():
    stage("6C3. Market-Microstructure Attack Generation")
    order_book = load_optional_csv("order_book_level2.csv", date_cols=["Date"])
    order_flow = load_optional_csv("order_flow_messages.csv", date_cols=["Timestamp"])
    rows = []
    if order_book is not None and {"Date", "Ticker", "Bid_Price_1", "Ask_Price_1", "Bid_Size_1", "Ask_Size_1"}.issubset(order_book.columns):
        sample = order_book.sort_values("Date").groupby("Ticker", group_keys=False).head(100).copy()
        for attack_name, size_multiplier, price_shift in [("Spoofing_Depth_Inflation", 8.0, 0.0), ("Layering_Quote_Ladder", 4.0, 0.0005)]:
            attacked = sample.copy()
            attacked["Attack_Type"] = attack_name
            attacked["Original_Bid_Size_1"] = attacked["Bid_Size_1"]
            attacked["Original_Ask_Size_1"] = attacked["Ask_Size_1"]
            attacked["Bid_Size_1"] = attacked["Bid_Size_1"] * size_multiplier
            attacked["Ask_Price_1"] = attacked["Ask_Price_1"] * (1 + price_shift)
            attacked["Depth_Imbalance_Attacked"] = (attacked["Bid_Size_1"] - attacked["Ask_Size_1"]) / (attacked["Bid_Size_1"] + attacked["Ask_Size_1"] + 1e-12)
            rows.append(attacked)
    if rows:
        attacks = pd.concat(rows, ignore_index=True)
        attacks.to_csv(RESULT_DIR / "microstructure_attack_scenarios.csv", index=False)
        summary = pd.DataFrame([{"Status": "Generated from provided order-book telemetry", "Rows": len(attacks), "Attack_Types": attacks["Attack_Type"].nunique()}])
    else:
        attacks = pd.DataFrame()
        summary = pd.DataFrame([{"Status": "Not generated because order_book_level2.csv with level-2 columns is absent", "Rows": 0, "Attack_Types": 0}])
        attacks.to_csv(RESULT_DIR / "microstructure_attack_scenarios.csv", index=False)
    summary.to_csv(RESULT_DIR / "microstructure_attack_generation_summary.csv", index=False)
    return attacks, summary


def operational_incident_time_to_detection(operational_features, operational_predictions):
    stage("6C4. Real Operational Incident Time-to-Detection")
    if operational_features.empty or operational_predictions.empty or "Manipulation_Label" not in operational_features.columns:
        pd.DataFrame().to_csv(RESULT_DIR / "operational_incident_time_to_detection.csv", index=False)
        pd.DataFrame().to_csv(RESULT_DIR / "operational_incident_lead_horizon_metrics.csv", index=False)
        return pd.DataFrame(), pd.DataFrame()

    features = operational_features.copy()
    features["Date"] = pd.to_datetime(features["Date"])
    predictions = operational_predictions.copy()
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    incident_rows = features[features["Manipulation_Label"].astype(int) == 1].sort_values(["Ticker", "Date"])
    events = []
    for ticker, group in incident_rows.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        if group.empty:
            continue
        event_id, start, previous = 1, group.loc[0, "Date"], group.loc[0, "Date"]
        for _, row in group.iloc[1:].iterrows():
            gap_minutes = (row["Date"] - previous).total_seconds() / 60.0
            if gap_minutes > 1.5:
                events.append({"Incident_ID": f"{ticker}_Operational_{event_id:03d}", "Ticker": ticker, "Incident_Start": start, "Incident_End": previous})
                event_id += 1
                start = row["Date"]
            previous = row["Date"]
        events.append({"Incident_ID": f"{ticker}_Operational_{event_id:03d}", "Ticker": ticker, "Incident_Start": start, "Incident_End": previous})
    events_df = pd.DataFrame(events)
    timing_rows, horizon_rows = [], []
    for _, event in events_df.iterrows():
        model_alerts = predictions[(predictions["Ticker"] == event["Ticker"]) & (predictions["Model"] == "Operational Telemetry Logistic")].sort_values("Date")
        if model_alerts.empty:
            continue
        for horizon_minutes in [0, 1, 5, 15, 30]:
            window = model_alerts[
                (model_alerts["Date"] >= event["Incident_Start"] - pd.Timedelta(minutes=horizon_minutes)) &
                (model_alerts["Date"] <= event["Incident_End"])
            ]
            alert_window = window[window["Operational_Alert"].astype(int) == 1]
            horizon_rows.append({
                "Incident_ID": event["Incident_ID"],
                "Ticker": event["Ticker"],
                "Lead_Horizon_Minutes": horizon_minutes,
                "Detected_Within_Horizon": bool(not alert_window.empty),
                "Alerts_Within_Horizon": int(alert_window.shape[0]),
            })
        search = model_alerts[
            (model_alerts["Date"] >= event["Incident_Start"] - pd.Timedelta(minutes=30)) &
            (model_alerts["Date"] <= event["Incident_End"])
        ]
        alert_rows = search[search["Operational_Alert"].astype(int) == 1]
        detected = not alert_rows.empty
        first_alert = alert_rows["Date"].iloc[0] if detected else pd.NaT
        latency_minutes = (first_alert - event["Incident_Start"]).total_seconds() / 60.0 if detected else np.nan
        timing_rows.append({
            "Incident_ID": event["Incident_ID"],
            "Ticker": event["Ticker"],
            "Model": "Operational Telemetry Logistic",
            "Incident_Start": event["Incident_Start"],
            "Incident_End": event["Incident_End"],
            "First_Alert_Timestamp": first_alert,
            "Detected": bool(detected),
            "Alert_Latency_Minutes": latency_minutes,
            "Lead_Time_Minutes": max(-latency_minutes, 0) if detected else np.nan,
            "Detection_Delay_Minutes": max(latency_minutes, 0) if detected else np.nan,
            "Ground_Truth_Source": "Real LOBSTER order-flow heuristic manipulation labels",
        })
    timing_df = pd.DataFrame(timing_rows)
    horizon_detail = pd.DataFrame(horizon_rows)
    horizon_summary = horizon_detail.groupby("Lead_Horizon_Minutes").agg(
        Incidents=("Incident_ID", "count"),
        Detected=("Detected_Within_Horizon", "sum"),
        Recall_At_Lead_Horizon=("Detected_Within_Horizon", "mean"),
        Alerts_Within_Horizon=("Alerts_Within_Horizon", "sum"),
    ).reset_index() if not horizon_detail.empty else pd.DataFrame()
    false_alerts = predictions.merge(events_df, on="Ticker", how="left")
    if not false_alerts.empty:
        in_incident = (
            (false_alerts["Date"] >= false_alerts["Incident_Start"].fillna(pd.Timestamp.max)) &
            (false_alerts["Date"] <= false_alerts["Incident_End"].fillna(pd.Timestamp.min))
        )
        false_alert_count = int(((false_alerts["Operational_Alert"].astype(int) == 1) & ~in_incident).sum())
        total_alert_count = int((false_alerts["Operational_Alert"].astype(int) == 1).sum())
        precision = 1 - false_alert_count / max(total_alert_count, 1)
        horizon_summary["Operational_Alert_Precision_Proxy"] = precision
        horizon_summary["False_Alerts_Outside_Incident_Windows"] = false_alert_count
    timing_df.to_csv(RESULT_DIR / "operational_incident_time_to_detection.csv", index=False)
    horizon_summary.to_csv(RESULT_DIR / "operational_incident_lead_horizon_metrics.csv", index=False)
    return timing_df, horizon_summary


def microstructure_attack_evaluation(attacks, operational_features):
    stage("8C. Order-Book Attack Evaluation")
    if attacks.empty or operational_features.empty:
        pd.DataFrame().to_csv(RESULT_DIR / "microstructure_attack_evaluation.csv", index=False)
        return pd.DataFrame()
    baseline = operational_features.copy()
    rows = []
    imbalance_threshold = baseline.get("Depth_Imbalance", pd.Series([0])).quantile(0.99) if "Depth_Imbalance" in baseline.columns else 0.0
    spread_threshold = baseline.get("Bid_Ask_Spread", pd.Series([0])).quantile(0.99) if "Bid_Ask_Spread" in baseline.columns else 0.0
    attacks = attacks.copy()
    attacks["Bid_Ask_Spread"] = attacks["Ask_Price_1"] - attacks["Bid_Price_1"]
    attacks["Black_Box_Heuristic_Alert"] = ((attacks["Depth_Imbalance_Attacked"].abs() >= abs(imbalance_threshold)) | (attacks["Bid_Ask_Spread"] >= spread_threshold)).astype(int)
    whitebox_available = (MODEL_DIR / "operational_telemetry_logistic_detector.pkl").exists()
    if whitebox_available:
        model = joblib.load(MODEL_DIR / "operational_telemetry_logistic_detector.pkl")
        feature_names = list(model.named_steps["scaler"].feature_names_in_)
        score_frame = attacks.copy()
        for col in feature_names:
            if col not in score_frame.columns:
                score_frame[col] = 0.0
        attacks["White_Box_Logistic_Risk"] = predict_proba(model, score_frame[feature_names])
        attacks["White_Box_Logistic_Alert"] = (attacks["White_Box_Logistic_Risk"] >= 0.5).astype(int)
    else:
        attacks["White_Box_Logistic_Risk"] = np.nan
        attacks["White_Box_Logistic_Alert"] = 0
    for attack_type, group in attacks.groupby("Attack_Type"):
        rows.append({
            "Attack_Type": attack_type,
            "Rows": len(group),
            "Black_Box_Heuristic_Attack_Success_Rate": group["Black_Box_Heuristic_Alert"].mean(),
            "White_Box_Logistic_Attack_Success_Rate": group["White_Box_Logistic_Alert"].mean(),
            "Mean_White_Box_Risk": group["White_Box_Logistic_Risk"].mean(),
            "Business_Impact_Proxy": "Elevated alert workload and potential false liquidity/depth interpretation",
        })
    attacks.to_csv(RESULT_DIR / "microstructure_attack_evaluation_predictions.csv", index=False)
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(RESULT_DIR / "microstructure_attack_evaluation.csv", index=False)
    return evaluation


def population_stability_index(reference, current, bins=10):
    ref = pd.Series(reference).replace([np.inf, -np.inf], np.nan).dropna()
    cur = pd.Series(current).replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or cur.empty:
        return np.nan
    quantiles = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def page_hinkley_alarm(values, threshold=5.0, delta=0.005):
    mean, cumulative, minimum = 0.0, 0.0, 0.0
    for idx, value in enumerate(values, start=1):
        mean += (value - mean) / idx
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
        if cumulative - minimum > threshold:
            return True, idx
    return False, np.nan


def drift_monitoring(df):
    stage("6D. Rolling Drift Monitoring")
    train = df[df["Split"] == "Train"].copy()
    monitored = df[df["Split"].isin(["Validation", "Test"])].sort_values(["Date", "Ticker"]).copy()
    dates = np.array(sorted(pd.to_datetime(monitored["Date"]).dt.normalize().unique()))
    rows = []
    window = 63
    for start_idx in range(0, max(len(dates) - window + 1, 1), window):
        window_dates = set(dates[start_idx:start_idx + window])
        current = monitored[monitored["Date"].dt.normalize().isin(window_dates)]
        if current.empty:
            continue
        split_name = "Validation" if current["Split"].mode().iloc[0] == "Validation" else "Test"
        for feature in FEATURE_COLS:
            psi = population_stability_index(train[feature], current[feature])
            ks_stat, ks_pvalue = ks_2samp(train[feature].dropna(), current[feature].dropna())
            z_values = ((current[feature] - train[feature].mean()) / (train[feature].std() + 1e-12)).fillna(0).abs().values
            ph_alarm, ph_index = page_hinkley_alarm(z_values)
            cumsum_alarm = bool(np.max(np.cumsum(z_values - np.mean(z_values))) > 25) if len(z_values) else False
            rows.append({
                "Split": split_name,
                "Window_Start": current["Date"].min(),
                "Window_End": current["Date"].max(),
                "Feature": feature,
                "PSI": psi,
                "KS_Statistic": ks_stat,
                "KS_pvalue": ks_pvalue,
                "PSI_Drift_Alarm": bool(psi >= 0.20),
                "KS_Drift_Alarm": bool(ks_pvalue < 0.01),
                "Page_Hinkley_Alarm": bool(ph_alarm),
                "Page_Hinkley_First_Index": ph_index,
                "CUSUM_Alarm": cumsum_alarm,
                "Retraining_Trigger": bool((psi >= 0.25) or (ks_pvalue < 0.001) or ph_alarm or cumsum_alarm),
            })
    drift_df = pd.DataFrame(rows)
    alarm_summary = drift_df.groupby(["Split", "Window_Start", "Window_End"], dropna=False).agg(
        Features_Checked=("Feature", "count"),
        PSI_Alarms=("PSI_Drift_Alarm", "sum"),
        KS_Alarms=("KS_Drift_Alarm", "sum"),
        Page_Hinkley_Alarms=("Page_Hinkley_Alarm", "sum"),
        CUSUM_Alarms=("CUSUM_Alarm", "sum"),
        Retraining_Triggers=("Retraining_Trigger", "sum"),
    ).reset_index() if not drift_df.empty else pd.DataFrame()
    retraining = alarm_summary.copy()
    if not retraining.empty:
        retraining["Retraining_Recommended"] = retraining["Retraining_Triggers"] >= 3
        retraining["Policy"] = "Recommend review/retraining when >=3 feature drift triggers occur in a rolling window"
    drift_df.to_csv(RESULT_DIR / "drift_monitoring_feature_windows.csv", index=False)
    alarm_summary.to_csv(RESULT_DIR / "drift_alarm_summary.csv", index=False)
    retraining.to_csv(RESULT_DIR / "drift_retraining_trigger_policy.csv", index=False)
    return drift_df, alarm_summary, retraining


def target_design_variant_report(df):
    stage("6E. Target Design Variant Analysis")
    rows = []
    work = df.sort_values(["Ticker", "Date"]).copy()
    work["Future_Return_5D"] = pd.to_numeric(work["Future_Return_5D"], errors="coerce")
    work["Vol_Adjusted_Return"] = work["Future_Return_5D"] / (work["Volatility_20"] + 1e-12)
    work["Multiclass_Risk_State"] = pd.cut(work["Future_Return_5D"], bins=[-np.inf, -0.05, -0.03, 0.03, np.inf], labels=["Crash", "Drawdown", "Normal", "Positive"])
    variants = {
        "Fixed downside target": work["Future_Return_5D"] <= -0.03,
        "Severe crash target": work["Future_Return_5D"] <= -0.05,
        "Volatility-adjusted target": work["Vol_Adjusted_Return"] <= -1.5,
        "Asset-specific 10pct downside quantile": work.groupby("Ticker")["Future_Return_5D"].transform(lambda s: s <= s.quantile(0.10)),
    }
    for name, labels in variants.items():
        for split_name, group_idx in work.groupby("Split").groups.items():
            split_labels = pd.Series(labels, index=work.index).loc[group_idx].astype(int)
            rows.append({
                "Target_Variant": name,
                "Split": split_name,
                "Rows": len(split_labels),
                "Positive_Events": int(split_labels.sum()),
                "Event_Rate": float(split_labels.mean()),
            })
    multi = work.groupby(["Split", "Multiclass_Risk_State"], observed=False).size().rename("Rows").reset_index()
    variant_df = pd.DataFrame(rows)
    variant_df.to_csv(RESULT_DIR / "target_design_variants.csv", index=False)
    multi.to_csv(RESULT_DIR / "multiclass_risk_state_distribution.csv", index=False)
    return variant_df, multi


def ensure_synthetic_event_log(df):
    path = DATA_DIR / "synthetic_event_log.csv"
    if path.exists() and path.stat().st_size > 0:
        event_log = pd.read_csv(path, parse_dates=["Start_Date", "End_Date"])
        return event_log

    rows = []
    scenarios = ["Price Spike", "Volume Shock", "Volatility Shock"]
    for split_name in ["Validation", "Test"]:
        split_df = df[df["Split"] == split_name].sort_values(["Ticker", "Date"])
        for ticker, group in split_df.groupby("Ticker"):
            dates = sorted(pd.to_datetime(group["Date"]).dt.normalize().unique())
            if len(dates) < 10:
                continue
            positions = [int(len(dates) * 0.25), int(len(dates) * 0.50), int(len(dates) * 0.75)]
            for idx, (scenario, pos) in enumerate(zip(scenarios, positions), start=1):
                start = pd.Timestamp(dates[pos])
                end = pd.Timestamp(dates[min(pos + 2, len(dates) - 1)])
                rows.append({
                    "Event_ID": f"{split_name}_{ticker}_{idx}",
                    "Split": split_name,
                    "Ticker": ticker,
                    "Scenario": scenario,
                    "Start_Date": start,
                    "End_Date": end,
                    "Injected_Rows": 3,
                })
    event_log = pd.DataFrame(rows)
    event_log.to_csv(path, index=False)
    return event_log


def target_event_windows(df):
    rows = []
    for (split_name, ticker), group in df[df["Split"].isin(["Validation", "Test"])].sort_values("Date").groupby(["Split", "Ticker"]):
        group = group.reset_index(drop=True)
        in_event, start_date, end_date, event_num = False, None, None, 0
        for _, row in group.iterrows():
            is_event = int(row["Target"]) == 1
            if is_event and not in_event:
                in_event, start_date, event_num = True, row["Date"], event_num + 1
            if in_event and is_event:
                end_date = row["Date"]
            if in_event and not is_event:
                rows.append({
                    "Event_ID": f"Target_{split_name}_{ticker}_{event_num:03d}",
                    "Event_Source": "Drawdown Target",
                    "Split": split_name,
                    "Ticker": ticker,
                    "Scenario": "Future 5D return <= -3%",
                    "Event_Start": pd.Timestamp(start_date),
                    "Event_End": pd.Timestamp(end_date),
                })
                in_event, start_date, end_date = False, None, None
        if in_event:
            rows.append({
                "Event_ID": f"Target_{split_name}_{ticker}_{event_num:03d}",
                "Event_Source": "Drawdown Target",
                "Split": split_name,
                "Ticker": ticker,
                "Scenario": "Future 5D return <= -3%",
                "Event_Start": pd.Timestamp(start_date),
                "Event_End": pd.Timestamp(end_date),
            })
    return pd.DataFrame(rows)


def synthetic_event_windows(df):
    log = ensure_synthetic_event_log(df)
    return pd.DataFrame({
        "Event_ID": log["Event_ID"],
        "Event_Source": "Synthetic Proxy",
        "Split": log["Split"],
        "Ticker": log["Ticker"],
        "Scenario": log["Scenario"],
        "Event_Start": pd.to_datetime(log["Start_Date"]),
        "Event_End": pd.to_datetime(log["End_Date"]),
    })


def all_event_windows(df):
    return pd.concat([target_event_windows(df), synthetic_event_windows(df)], ignore_index=True)


def alert_model_frames(supervised_predictions, event_detector_predictions):
    frames = []
    for source_name, model_type, predictions in [
        ("Supervised Classifier", "Supervised Classifier", supervised_predictions),
        ("Additional Supervised Event Detector", "Additional Supervised Event Detector", event_detector_predictions),
    ]:
        for col in [c for c in predictions.columns if c.startswith("Pred_")]:
            model_name = display_model_name(col.replace("Pred_", ""))
            frame = predictions[["Date", "Ticker", "Split", col]].copy()
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame["Model"] = model_name
            frame["Model_Type"] = model_type
            frame["Alert"] = frame[col].astype(int)
            frames.append(frame[["Date", "Ticker", "Split", "Model", "Model_Type", "Alert"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score_event_timing(event_windows, alerts, lookback_rows=5):
    rows = []
    for _, event in event_windows.iterrows():
        candidates = alerts[(alerts["Split"] == event["Split"]) & (alerts["Ticker"] == event["Ticker"])]
        for (model, model_type), model_alerts in candidates.groupby(["Model", "Model_Type"]):
            ordered = model_alerts.sort_values("Date").reset_index(drop=True)
            dates = pd.to_datetime(ordered["Date"])
            start_matches = np.where(dates >= pd.Timestamp(event["Event_Start"]))[0]
            end_matches = np.where(dates <= pd.Timestamp(event["Event_End"]))[0]
            if len(start_matches) == 0 or len(end_matches) == 0:
                continue
            start_idx = int(start_matches[0])
            end_idx = int(end_matches[-1])
            window_start = max(0, start_idx - lookback_rows)
            search = ordered.iloc[window_start:end_idx + 1]
            alert_rows = search[search["Alert"] == 1]
            detected = not alert_rows.empty
            first_alert_date = pd.NaT
            latency, lead = np.nan, np.nan
            if detected:
                first_alert_idx = int(alert_rows.index[0])
                first_alert_date = pd.Timestamp(ordered.loc[first_alert_idx, "Date"])
                offset = first_alert_idx - start_idx
                latency = max(offset, 0)
                lead = max(-offset, 0)
            rows.append({
                "Event_ID": event["Event_ID"],
                "Event_Source": event["Event_Source"],
                "Scenario": event["Scenario"],
                "Split": event["Split"],
                "Ticker": event["Ticker"],
                "Model": model,
                "Model_Type": model_type,
                "Event_Start": event["Event_Start"],
                "Event_End": event["Event_End"],
                "First_Alert_Date": first_alert_date,
                "Detected": bool(detected),
                "Missed_Event": bool(not detected),
                "Time_To_Detection_Trading_Days": latency,
                "Lead_Time_Trading_Days": lead,
            })
    return pd.DataFrame(rows)


def time_to_detection_analysis(df, supervised_predictions, event_detector_predictions):
    stage("6C. Time-to-Detection Analysis")
    events = all_event_windows(df)
    alerts = alert_model_frames(supervised_predictions, event_detector_predictions)
    event_level = score_event_timing(events, alerts)
    if event_level.empty:
        summary = pd.DataFrame()
    else:
        summary = event_level.groupby(["Model", "Model_Type", "Event_Source", "Split"], dropna=False).agg(
            Events=("Event_ID", "count"),
            Detected_Events=("Detected", "sum"),
            Detection_Rate=("Detected", "mean"),
            Mean_Time_To_Detection=("Time_To_Detection_Trading_Days", "mean"),
            Median_Time_To_Detection=("Time_To_Detection_Trading_Days", "median"),
            Mean_Lead_Time=("Lead_Time_Trading_Days", "mean"),
            Median_Lead_Time=("Lead_Time_Trading_Days", "median"),
        ).reset_index()
    event_level.to_csv(RESULT_DIR / "time_to_detection_event_level.csv", index=False)
    summary.to_csv(RESULT_DIR / "time_to_detection_summary.csv", index=False)
    return event_level, summary


def synthetic_proxy_detection_analysis(df, supervised_predictions, event_detector_predictions):
    stage("6D. Synthetic Market-Manipulation and Cyber Proxy Evaluation")
    proxy_events = synthetic_event_windows(df)
    combined = alert_model_frames(supervised_predictions, event_detector_predictions)
    base = df[df["Split"].isin(["Validation", "Test"])][["Date", "Ticker", "Split"]].copy()
    base["Date"] = pd.to_datetime(base["Date"])
    base["Synthetic_Proxy_Event"] = 0
    base["Proxy_Scenario"] = ""
    for _, event in proxy_events.iterrows():
        mask = (
            (base["Split"] == event["Split"]) &
            (base["Ticker"] == event["Ticker"]) &
            (base["Date"] >= pd.Timestamp(event["Event_Start"])) &
            (base["Date"] <= pd.Timestamp(event["Event_End"]))
        )
        base.loc[mask, "Synthetic_Proxy_Event"] = 1
        base.loc[mask, "Proxy_Scenario"] = event["Scenario"]

    predictions = combined.merge(base, on=["Date", "Ticker", "Split"], how="left")
    metric_rows = []
    for (model, model_type, split_name), group in predictions.groupby(["Model", "Model_Type", "Split"]):
        y_true = group["Synthetic_Proxy_Event"].fillna(0).astype(int)
        pred = group["Alert"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        metric_rows.append({
            "Model": model,
            "Model_Type": model_type,
            "Split": split_name,
            "Proxy_Label_Source": "Synthetic event windows, not operational cyber/order-book telemetry",
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "F1_Score": f1_score(y_true, pred, zero_division=0),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Predicted_Alerts": int(pred.sum()),
            "Proxy_Event_Rows": int(y_true.sum()),
        })
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(RESULT_DIR / "synthetic_proxy_event_predictions.csv", index=False)
    metrics.to_csv(RESULT_DIR / "synthetic_proxy_event_detection_metrics.csv", index=False)
    return metrics, predictions


def event_membership_frame(df):
    base = df[df["Split"].isin(["Validation", "Test"])][["Date", "Ticker", "Split", "Target"]].copy()
    base["Date"] = pd.to_datetime(base["Date"])
    base["Any_Event_Window"] = base["Target"].astype(int)
    for _, event in synthetic_event_windows(df).iterrows():
        mask = (
            (base["Split"] == event["Split"]) &
            (base["Ticker"] == event["Ticker"]) &
            (base["Date"] >= pd.Timestamp(event["Event_Start"])) &
            (base["Date"] <= pd.Timestamp(event["Event_End"]))
        )
        base.loc[mask, "Any_Event_Window"] = 1
    return base[["Date", "Ticker", "Split", "Any_Event_Window"]]


def alert_latency_operational_metrics(df, supervised_predictions, event_detector_predictions):
    stage("6F. Alert Persistence and False-Alarm Burden")
    alerts = alert_model_frames(supervised_predictions, event_detector_predictions)
    event_rows = event_membership_frame(df)
    alerts = alerts.merge(event_rows, on=["Date", "Ticker", "Split"], how="left").fillna({"Any_Event_Window": 0})
    episode_rows, burden_rows, lead_rows = [], [], []
    for (model, model_type, split_name, ticker), group in alerts.sort_values("Date").groupby(["Model", "Model_Type", "Split", "Ticker"]):
        group = group.reset_index(drop=True)
        alert_idx = np.where(group["Alert"].values == 1)[0]
        if len(alert_idx):
            gaps = np.where(np.diff(alert_idx) > 1)[0] + 1
            episodes = np.split(alert_idx, gaps)
        else:
            episodes = []
        false_alert_dates = group.loc[(group["Alert"] == 1) & (group["Any_Event_Window"] == 0), "Date"].sort_values()
        false_gaps = false_alert_dates.diff().dt.days.dropna()
        burden_rows.append({
            "Model": model,
            "Model_Type": model_type,
            "Split": split_name,
            "Ticker": ticker,
            "Alert_Days": int(group["Alert"].sum()),
            "False_Alert_Days": int(len(false_alert_dates)),
            "False_Alarm_Time_Burden": len(false_alert_dates) / max(len(group), 1),
            "Mean_Time_Between_False_Alerts_Days": float(false_gaps.mean()) if not false_gaps.empty else np.nan,
        })
        for idx, episode in enumerate(episodes, start=1):
            ep = group.iloc[episode]
            episode_rows.append({
                "Model": model,
                "Model_Type": model_type,
                "Split": split_name,
                "Ticker": ticker,
                "Episode_ID": f"{model}_{split_name}_{ticker}_{idx}",
                "Episode_Start": ep["Date"].min(),
                "Episode_End": ep["Date"].max(),
                "Alert_Duration_Trading_Days": len(ep),
                "Overlaps_Event_Window": bool(ep["Any_Event_Window"].max() == 1),
            })
    for max_lead in [0, 1, 3, 5]:
        events = all_event_windows(df)
        timing = score_event_timing(events, alerts, lookback_rows=max_lead)
        if timing.empty:
            continue
        timing["Lead_Bucket_Max_Days"] = max_lead
        lead_rows.append(timing.groupby(["Model", "Model_Type", "Split", "Event_Source", "Lead_Bucket_Max_Days"], dropna=False).agg(
            Events=("Event_ID", "count"),
            Detected_Events=("Detected", "sum"),
            Precision_At_Lead_Proxy=("Detected", "mean"),
        ).reset_index())
    episodes_df = pd.DataFrame(episode_rows)
    burden_df = pd.DataFrame(burden_rows)
    lead_df = pd.concat(lead_rows, ignore_index=True) if lead_rows else pd.DataFrame()
    episodes_df.to_csv(RESULT_DIR / "alert_persistence_metrics.csv", index=False)
    burden_df.to_csv(RESULT_DIR / "false_alarm_time_burden.csv", index=False)
    lead_df.to_csv(RESULT_DIR / "precision_at_lead_time_proxy.csv", index=False)
    return episodes_df, burden_df, lead_df


def rank_normalize(series):
    return pd.Series(series).rank(method="average", pct=True).fillna(0.0).values


def unified_alert_fusion_policy(supervised_predictions, event_detector_predictions):
    stage("6G. Unified Alert Fusion and Severity Policy")
    base_cols = ["Date", "Ticker", "Split", "Target"]
    frame = supervised_predictions[base_cols].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    supervised_pred_cols = [c for c in supervised_predictions.columns if c.startswith("Pred_")]
    supervised_proba_cols = [c for c in supervised_predictions.columns if c.startswith("Proba_")]
    detector_pred_cols = [c for c in event_detector_predictions.columns if c.startswith("Pred_")]
    detector_score_cols = [c for c in event_detector_predictions.columns if c.startswith("Score_")]
    for col in supervised_pred_cols + supervised_proba_cols:
        frame[col] = supervised_predictions[col].values
    detector_small = event_detector_predictions[["Date", "Ticker", "Split"] + detector_pred_cols + detector_score_cols].copy()
    detector_small["Date"] = pd.to_datetime(detector_small["Date"])
    frame = frame.merge(detector_small, on=["Date", "Ticker", "Split"], how="left")
    frame["Supervised_Alert_Count"] = frame[supervised_pred_cols].sum(axis=1)
    frame["Event_Detector_Alert_Count"] = frame[detector_pred_cols].sum(axis=1)
    frame["Fusion_OR_Alert"] = ((frame["Supervised_Alert_Count"] > 0) | (frame["Event_Detector_Alert_Count"] > 0)).astype(int)
    frame["Fusion_Consensus_Alert"] = ((frame["Supervised_Alert_Count"] > 0) & (frame["Event_Detector_Alert_Count"] > 0)).astype(int)
    if supervised_proba_cols:
        frame["Max_Supervised_Risk"] = frame[supervised_proba_cols].max(axis=1)
    else:
        frame["Max_Supervised_Risk"] = 0.0
    for col in detector_score_cols:
        frame[f"Norm_{col}"] = rank_normalize(frame[col])
    norm_score_cols = [c for c in frame.columns if c.startswith("Norm_Score_")]
    frame["Max_Normalized_Event_Detector_Score"] = frame[norm_score_cols].max(axis=1) if norm_score_cols else 0.0
    frame["Fusion_Risk_Score"] = 0.55 * frame["Max_Supervised_Risk"] + 0.45 * frame["Max_Normalized_Event_Detector_Score"]
    frame["Severity"] = pd.cut(frame["Fusion_Risk_Score"], bins=[-np.inf, 0.35, 0.60, 0.80, np.inf], labels=["Low", "Medium", "High", "Critical"])
    frame["Alert_Action"] = np.select(
        [
            frame["Severity"].astype(str).eq("Critical"),
            frame["Severity"].astype(str).eq("High"),
            frame["Fusion_Consensus_Alert"].eq(1),
        ],
        ["Immediate analyst review and safe-mode check", "Review within monitoring cycle", "Queue for consensus investigation"],
        default="Log for audit",
    )
    frame.to_csv(RESULT_DIR / "unified_alert_fusion_policy.csv", index=False)

    metrics = []
    for split_name, group in frame.groupby("Split"):
        y = group["Target"].astype(int)
        for col in ["Fusion_OR_Alert", "Fusion_Consensus_Alert"]:
            pred = group[col].astype(int)
            metrics.append({
                "Policy": col,
                "Split": split_name,
                "Precision": precision_score(y, pred, zero_division=0),
                "Recall": recall_score(y, pred, zero_division=0),
                "F1_Score": f1_score(y, pred, zero_division=0),
                "Predicted_Alerts": int(pred.sum()),
            })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULT_DIR / "unified_alert_fusion_metrics.csv", index=False)
    return frame, metrics_df


def investigator_alert_rationale_outputs(splits, models, fusion_df):
    stage("6H. Investigator Alert Rationale and Audit Trail")
    X_train, _, train_df = splits["Train"]
    train_medians = X_train.median()
    train_stds = X_train.std().replace(0, 1e-6)
    test_df = splits["Test"][2]
    candidates = fusion_df[(fusion_df["Split"] == "Test") & (fusion_df["Fusion_OR_Alert"] == 1)].sort_values("Fusion_Risk_Score", ascending=False).head(250)
    rows = []
    for _, alert in candidates.iterrows():
        match = test_df[(pd.to_datetime(test_df["Date"]) == pd.Timestamp(alert["Date"])) & (test_df["Ticker"] == alert["Ticker"])]
        if match.empty:
            continue
        obs = match.iloc[0][FEATURE_COLS].astype(float)
        deviations = ((obs - train_medians) / train_stds).abs().sort_values(ascending=False)
        top_deviation = "; ".join([f"{idx}={val:.2f}z" for idx, val in deviations.head(3).items()])
        lr_reason = ""
        if "Logistic Regression" in models and isinstance(models["Logistic Regression"], Pipeline):
            coef = pd.Series(models["Logistic Regression"].named_steps["model"].coef_[0], index=FEATURE_COLS)
            lr_contrib = (coef * ((obs - train_medians) / train_stds)).abs().sort_values(ascending=False)
            lr_reason = "; ".join([f"{idx}" for idx in lr_contrib.head(3).index])
        rows.append({
            "Date": alert["Date"],
            "Ticker": alert["Ticker"],
            "Severity": alert["Severity"],
            "Fusion_Risk_Score": alert["Fusion_Risk_Score"],
            "Supervised_Alert_Count": alert["Supervised_Alert_Count"],
            "Event_Detector_Alert_Count": alert["Event_Detector_Alert_Count"],
            "Top_Feature_Deviations": top_deviation,
            "Top_Logistic_Drivers": lr_reason,
            "Audit_Note": "Rationale combines feature deviations and available supervised coefficients; investigator review still required.",
        })
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(RESULT_DIR / "investigator_alert_rationale_audit.csv", index=False)
    return audit_df


def export_deployment_runtime_artifacts():
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOYMENT_DIR / "monitoring_service.py").write_text(
        '''"""
Minimal batch/HTTP monitoring service wrapper for generated model artifacts.
Run:
    python monitoring_service.py --batch ../data/supervised_drawdown_dataset.csv
    python monitoring_service.py --serve
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "logistic_regression_drawdown_classifier.pkl"
FEATURES = ["Return", "Volatility_10", "Volatility_20", "Momentum_10", "MA_Ratio_10", "MA_Ratio_20", "Volume_Change", "Rolling_Skew_20", "Rolling_Kurt_20", "Return_Zscore_20", "Drawdown", "RSI_14", "MACD", "MACD_Signal", "BB_Width"]

def load_model():
    return joblib.load(MODEL_PATH)

def score_frame(df):
    model = load_model()
    scores = model.predict_proba(df[FEATURES])[:, 1]
    out = df.copy()
    out["Risk_Probability"] = scores
    out["Alert"] = (scores >= 0.5).astype(int)
    return out

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        rows = json.loads(self.rfile.read(length) or "[]")
        result = score_frame(pd.DataFrame(rows)).to_dict(orient="records")
        body = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.batch:
        scored = score_frame(pd.read_csv(args.batch))
        scored.to_csv(ROOT / "results" / "deployment_batch_scored_alerts.csv", index=False)
    if args.serve:
        HTTPServer((args.host, args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )
    (DEPLOYMENT_DIR / "streaming_ingestion_loop.py").write_text(
        '''"""
Filesystem streaming adapter: watches an input CSV and appends scored rows on an interval.
This is a lightweight deployment harness, not a managed production stream processor.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from monitoring_service import score_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="../results/streaming_scored_alerts.csv")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    seen = 0
    while True:
        path = Path(args.input)
        if path.exists():
            df = pd.read_csv(path)
            new = df.iloc[seen:]
            if not new.empty:
                scored = score_frame(new)
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                scored.to_csv(out, mode="a", header=not out.exists(), index=False)
                seen = len(df)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )
    (DEPLOYMENT_DIR / "retraining_scheduler.py").write_text(
        '''"""
Retraining scheduler harness that reads drift_retraining_trigger_policy.csv and runs run_all.py when policy breaches.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="../results/drift_retraining_trigger_policy.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    policy = Path(__file__).resolve().parent / args.policy
    df = pd.read_csv(policy)
    trigger = bool(df.get("Retraining_Recommended", pd.Series(dtype=bool)).astype(bool).any())
    if trigger and not args.dry_run:
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / "run_all.py")]))
    print({"retraining_recommended": trigger, "dry_run": args.dry_run})

if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )
    registry = pd.DataFrame([
        {"Model_Name": "Logistic Regression", "Artifact": str(MODEL_DIR / "logistic_regression_drawdown_classifier.pkl"), "Rollback_Rank": 1},
        {"Model_Name": "Random Forest", "Artifact": str(MODEL_DIR / "random_forest_drawdown_classifier.pkl"), "Rollback_Rank": 2},
        {"Model_Name": "Gradient Boosting Fallback", "Artifact": str(MODEL_DIR / "gradient_boosting_fallback_drawdown_classifier.pkl"), "Rollback_Rank": 3},
    ])
    registry.to_csv(DEPLOYMENT_DIR / "model_registry.csv", index=False)
    return pd.DataFrame([
        {"Artifact": "monitoring_service.py", "Purpose": "Batch and HTTP scoring wrapper"},
        {"Artifact": "streaming_ingestion_loop.py", "Purpose": "Filesystem streaming ingestion loop"},
        {"Artifact": "retraining_scheduler.py", "Purpose": "Drift-triggered retraining harness"},
        {"Artifact": "model_registry.csv", "Purpose": "Simple model version/rollback registry"},
    ])


def defence_and_deployment_policy_outputs(drift_summary, fusion_metrics, adversarial_df, defence_training_df):
    stage("6I. Deployment and Defence Policy Artifacts")
    runtime_artifacts = export_deployment_runtime_artifacts()
    runtime_artifacts.to_csv(RESULT_DIR / "deployment_runtime_artifacts.csv", index=False)
    deployment = pd.DataFrame([
        {"Component": "Streaming ingestion", "Implemented": True, "Current_Status": "Filesystem streaming ingestion loop generated", "Required_To_Productionize": "Replace file polling with managed stream processor"},
        {"Component": "Alert queue", "Implemented": True, "Current_Status": "Batch alert fusion table", "Required_To_Productionize": "Queue service with analyst workflow"},
        {"Component": "Alert deduplication", "Implemented": True, "Current_Status": "Alert persistence episode table", "Required_To_Productionize": "Stateful online episode tracker"},
        {"Component": "Drift monitor", "Implemented": True, "Current_Status": "Rolling PSI/KS/Page-Hinkley/CUSUM", "Required_To_Productionize": "Continuous window scheduler"},
        {"Component": "Retraining scheduler", "Implemented": True, "Current_Status": "Retraining scheduler harness generated", "Required_To_Productionize": "Connect to scheduler/orchestrator and approval workflow"},
        {"Component": "Model rollback", "Implemented": True, "Current_Status": "Simple model registry generated", "Required_To_Productionize": "Use versioned registry with signed artifacts"},
        {"Component": "API/service wrapper", "Implemented": True, "Current_Status": "Minimal HTTP scoring wrapper generated", "Required_To_Productionize": "Harden authentication, observability, and deployment platform"},
    ])
    deployment.to_csv(RESULT_DIR / "deployment_readiness_matrix.csv", index=False)

    defence = pd.DataFrame([
        {"Defence": "Drift-triggered retraining review", "Implemented": True, "Trigger": ">=3 feature drift triggers in a rolling window", "Output": "drift_retraining_trigger_policy.csv"},
        {"Defence": "Fusion consensus alerting", "Implemented": True, "Trigger": "Supervised classifier and additional supervised event detectors alert together", "Output": "unified_alert_fusion_policy.csv"},
        {"Defence": "Severity-based safe-mode check", "Implemented": True, "Trigger": "Critical fusion severity", "Output": "unified_alert_fusion_policy.csv"},
        {"Defence": "Uncertainty abstention proxy", "Implemented": True, "Trigger": "Fusion risk score between 0.45 and 0.55", "Output": "active_defence_policy_matrix.csv"},
        {"Defence": "Adversarial training", "Implemented": True, "Trigger": "Bounded feature-space perturbation defence", "Output": "adversarial_training_defence_metrics.csv"},
        {"Defence": "Robust feature filtering", "Implemented": True, "Trigger": "High drift or adversarial sensitivity", "Output": "robust_feature_filter_scores.csv"},
    ])
    if not fusion_metrics.empty:
        consensus = fusion_metrics[fusion_metrics["Policy"] == "Fusion_Consensus_Alert"]
        if not consensus.empty:
            defence.loc[defence["Defence"] == "Fusion consensus alerting", "Observed_Test_F1"] = consensus[consensus["Split"] == "Test"]["F1_Score"].mean()
    if not adversarial_df.empty:
        defence["Max_Observed_Adversarial_F1_Drop"] = adversarial_df["F1_Drop"].max()
    if not defence_training_df.empty:
        defence.loc[defence["Defence"] == "Adversarial training", "Observed_Test_F1"] = defence_training_df[defence_training_df["Defence_Type"] == "Adversarial training"]["F1_Score"].mean()
        defence.loc[defence["Defence"] == "Robust feature filtering", "Observed_Test_F1"] = defence_training_df[defence_training_df["Defence_Type"] == "Robust feature filtering"]["F1_Score"].mean()
    safe_to_csv(defence, RESULT_DIR / "active_defence_policy_matrix.csv", index=False)
    return deployment, defence


# =============================================================================
# 11. Robustness and Stress Scenario Testing
# =============================================================================

def robustness_testing(splits, models):
    stage("7. Robustness Testing")
    X_test, y_test, _ = splits["Test"]
    baseline, rows = {}, []
    for name, model in models.items():
        proba = predict_proba(model, X_test)
        pred = (proba >= 0.5).astype(int)
        baseline[name] = {"pred": pred, "f1": f1_score(y_test, pred, zero_division=0), "auc": roc_auc_score(y_test, proba)}
    for level in [0.05, 0.10, 0.15, 0.20, 0.30]:
        noise = np.random.normal(0, level, X_test.shape)
        X_perturbed = pd.DataFrame(X_test.values + noise, columns=FEATURE_COLS, index=X_test.index)
        for name, model in models.items():
            proba = predict_proba(model, X_perturbed)
            pred = (proba >= 0.5).astype(int)
            f1 = f1_score(y_test, pred, zero_division=0)
            auc = roc_auc_score(y_test, proba)
            flips = int(np.sum(baseline[name]["pred"] != pred))
            rows.append({"Model": name, "Perturbation_Level": level, "Prediction_Flips": flips, "Flip_Rate": flips / len(X_test), "Baseline_F1": baseline[name]["f1"], "Perturbed_F1": f1, "F1_Drop": baseline[name]["f1"] - f1, "Baseline_ROC_AUC": baseline[name]["auc"], "Perturbed_ROC_AUC": auc, "AUC_Drop": baseline[name]["auc"] - auc, "Robustness_Score": 1 - (flips / len(X_test))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "supervised_robustness_results.csv", index=False)
    df.to_csv(RESULT_DIR / "perturbation_robustness_results_standardized.csv", index=False)
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x="Perturbation_Level", y="F1_Drop", hue="Model", marker="o")
    plt.title("F1 Drop Under Noise")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "supervised_f1_drop_under_noise.png", dpi=300)
    plt.close()
    return df


def make_stress_scenarios(X_test):
    scenarios = {}
    price = X_test.copy(); price["Momentum_10"] *= 1.5; price["MA_Ratio_10"] *= 1.08; price["MA_Ratio_20"] *= 1.10; price["Return_Zscore_20"] += 2.0; scenarios["Price Spike"] = price
    volume = X_test.copy(); volume["Volume_Change"] = volume["Volume_Change"].fillna(0) + 2.0; volume["Volatility_10"] *= 1.3; volume["BB_Width"] *= 1.5; scenarios["Volume Shock"] = volume
    vol = X_test.copy(); vol["Volatility_10"] *= 1.8; vol["Volatility_20"] *= 1.8; vol["Drawdown"] -= 0.10; vol["Rolling_Kurt_20"] += 1.0; vol["BB_Width"] *= 1.6; scenarios["Volatility Shock"] = vol
    return scenarios


def stress_testing(splits, models):
    stage("8. Stress Scenario Testing")
    X_test, y_test, _ = splits["Test"]
    rows = []
    for name, model in models.items():
        base_proba = predict_proba(model, X_test)
        base_pred = (base_proba >= 0.5).astype(int)
        base_f1, base_auc = f1_score(y_test, base_pred, zero_division=0), roc_auc_score(y_test, base_proba)
        for scenario, X_stress in make_stress_scenarios(X_test).items():
            proba = predict_proba(model, X_stress)
            pred = (proba >= 0.5).astype(int)
            stressed_f1, stressed_auc = f1_score(y_test, pred, zero_division=0), roc_auc_score(y_test, proba)
            flips = int((base_pred != pred).sum())
            rows.append({"Model": name, "Scenario": scenario, "Baseline_F1": base_f1, "Stressed_F1": stressed_f1, "F1_Drop": base_f1 - stressed_f1, "Baseline_ROC_AUC": base_auc, "Stressed_ROC_AUC": stressed_auc, "AUC_Drop": base_auc - stressed_auc, "Prediction_Flips": flips, "Robustness": 1 - (flips / len(X_test))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "supervised_stress_scenario_results.csv", index=False)
    df.to_csv(RESULT_DIR / "financial_stress_scenarios_results_standardized.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Scenario", y="Stressed_F1", hue="Model")
    plt.title("Supervised Model F1 Under Financial Stress Scenarios")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "supervised_stress_scenario_f1_comparison.png", dpi=300)
    plt.close()
    return df


def adversarial_attack_testing(splits, models):
    stage("8A. Feature-Space Adversarial Robustness Simulations")
    X_train, _, _ = splits["Train"]
    X_val, _, _ = splits["Validation"]
    X_test, y_test, test_df = splits["Test"]
    val_std = X_val.std().replace(0, 1e-6)
    lower = X_train.quantile(0.01)
    upper = X_train.quantile(0.99)
    rows, pred_rows = [], []

    for name, model in models.items():
        base_proba = predict_proba(model, X_test)
        base_pred = (base_proba >= 0.5).astype(int)
        base_f1 = f1_score(y_test, base_pred, zero_division=0)
        base_auc = roc_auc_score(y_test, base_proba)

        if name == "Logistic Regression" and isinstance(model, Pipeline):
            coef = model.named_steps["model"].coef_[0]
            feature_sign = np.sign(coef)
            attack_method = "FGSM-style logistic coefficient sign"
        else:
            val_proba = predict_proba(model, X_val)
            feature_sign = []
            for col in FEATURE_COLS:
                corr = np.corrcoef(X_val[col].values, val_proba)[0, 1]
                feature_sign.append(0.0 if np.isnan(corr) else np.sign(corr))
            feature_sign = np.array(feature_sign)
            attack_method = "Validation score-guided finite-difference proxy"

        label_direction = np.where(y_test.values.reshape(-1, 1) == 1, -1.0, 1.0)
        for budget in [0.05, 0.10, 0.20]:
            perturb = label_direction * feature_sign.reshape(1, -1) * val_std.values.reshape(1, -1) * budget
            X_adv = pd.DataFrame(X_test.values + perturb, columns=FEATURE_COLS, index=X_test.index)
            X_adv = X_adv.clip(lower=lower, upper=upper, axis=1)
            adv_proba = predict_proba(model, X_adv)
            adv_pred = (adv_proba >= 0.5).astype(int)
            adv_f1 = f1_score(y_test, adv_pred, zero_division=0)
            adv_auc = roc_auc_score(y_test, adv_proba)
            flips = int(np.sum(base_pred != adv_pred))
            event_mask = y_test.values == 1
            event_alerts_before = base_pred[event_mask] == 1
            event_missed_after = adv_pred[event_mask] == 0
            targeted_misses = int(np.sum(event_alerts_before & event_missed_after))
            targeted_denominator = int(np.sum(event_alerts_before))
            originally_correct = base_pred == y_test.values
            attack_successes = int(np.sum(originally_correct & (adv_pred != y_test.values)))
            event_attack_successes = int(np.sum(event_mask & (base_pred == 1) & (adv_pred == 0)))
            rows.append({
                "Model": name,
                "Attack_Type": attack_method,
                "Feature_Budget": budget,
                "Feature_Bounds": "budget * validation feature std, clipped to train 1st/99th percentiles",
                "Baseline_F1": base_f1,
                "Attacked_F1": adv_f1,
                "F1_Drop": base_f1 - adv_f1,
                "Baseline_ROC_AUC": base_auc,
                "Attacked_ROC_AUC": adv_auc,
                "AUC_Drop": base_auc - adv_auc,
                "Prediction_Flips": flips,
                "Flip_Rate": flips / len(X_test),
                "Attack_Successes_From_Correct_To_Wrong": attack_successes,
                "Attack_Success_Rate_Among_Originally_Correct": attack_successes / max(int(np.sum(originally_correct)), 1),
                "Event_Attack_Successes_Alert_To_Miss": event_attack_successes,
                "Targeted_Event_Misses": targeted_misses,
                "Baseline_Event_Alerts": targeted_denominator,
                "Targeted_Event_Miss_Rate": targeted_misses / max(targeted_denominator, 1),
                "Scope_Note": "Feature-space adversarial simulation, not market microstructure attack generation",
            })
            out = test_df[["Date", "Ticker", "Split", "Target"]].copy()
            out["Model"] = name
            out["Feature_Budget"] = budget
            out["Baseline_Probability"] = base_proba
            out["Attacked_Probability"] = adv_proba
            out["Baseline_Prediction"] = base_pred
            out["Attacked_Prediction"] = adv_pred
            out["Prediction_Flipped"] = base_pred != adv_pred
            pred_rows.append(out)

    results = pd.DataFrame(rows)
    if not results.empty:
        minimal = results[results["Prediction_Flips"] > 0].groupby("Model")["Feature_Budget"].min().rename("Minimal_Budget_With_Any_Flip")
        results = results.merge(minimal, on="Model", how="left")
    predictions = pd.concat(pred_rows, ignore_index=True)
    results.to_csv(RESULT_DIR / "adversarial_attack_results.csv", index=False)
    predictions.to_csv(RESULT_DIR / "adversarial_attack_predictions.csv", index=False)
    return results, predictions


def adversarial_training_and_robust_feature_filtering(splits, drift_df, adversarial_df):
    stage("8B. Adversarial Training and Robust Feature Filtering Defences")
    X_train, y_train, _ = splits["Train"]
    X_val, y_val, _ = splits["Validation"]
    X_test, y_test, _ = splits["Test"]
    val_std = X_val.std().replace(0, 1e-6)

    base_lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    base_lr.fit(X_train, y_train)
    coef = base_lr.named_steps["model"].coef_[0]
    feature_sign = np.sign(coef)
    label_direction = np.where(y_train.values.reshape(-1, 1) == 1, -1.0, 1.0)
    perturb = label_direction * feature_sign.reshape(1, -1) * val_std.values.reshape(1, -1) * 0.10
    X_train_adv = pd.DataFrame(X_train.values + perturb, columns=FEATURE_COLS, index=X_train.index)
    X_aug = pd.concat([X_train, X_train_adv], ignore_index=True)
    y_aug = pd.concat([y_train, y_train], ignore_index=True)
    adv_trained = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    adv_trained.fit(X_aug, y_aug)
    joblib.dump(adv_trained, MODEL_DIR / "adversarially_trained_logistic_regression.pkl")

    stable_features = FEATURE_COLS.copy()
    if not drift_df.empty:
        drift_scores = drift_df.groupby("Feature").agg(
            Mean_PSI=("PSI", "mean"),
            Drift_Trigger_Rate=("Retraining_Trigger", "mean"),
        ).reset_index()
        stable = drift_scores[(drift_scores["Mean_PSI"] <= drift_scores["Mean_PSI"].median()) & (drift_scores["Drift_Trigger_Rate"] <= drift_scores["Drift_Trigger_Rate"].median())]["Feature"].tolist()
        if len(stable) >= 5:
            stable_features = stable
        else:
            stable_features = drift_scores.sort_values(["Drift_Trigger_Rate", "Mean_PSI"]).head(max(5, min(10, len(drift_scores))))["Feature"].tolist()
        drift_scores["Selected_By_Robust_Filter"] = drift_scores["Feature"].isin(stable_features)
        drift_scores.to_csv(RESULT_DIR / "robust_feature_filter_scores.csv", index=False)
    else:
        pd.DataFrame({"Feature": FEATURE_COLS, "Selected_By_Robust_Filter": True}).to_csv(RESULT_DIR / "robust_feature_filter_scores.csv", index=False)

    filtered_lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    filtered_lr.fit(X_train[stable_features], y_train)
    joblib.dump({"model": filtered_lr, "features": stable_features}, MODEL_DIR / "robust_feature_filtered_logistic_regression.pkl")

    defence_models = {
        "Adversarially Trained Logistic Regression": (adv_trained, FEATURE_COLS),
        "Robust Feature Filtered Logistic Regression": (filtered_lr, stable_features),
    }
    rows = []
    predictions = splits["Test"][2][["Date", "Ticker", "Split", "Target"]].copy()
    for name, (model, features) in defence_models.items():
        proba = predict_proba(model, X_test[features])
        pred = (proba >= 0.5).astype(int)
        row = metric_row(name, "Test", y_test, pred, proba)
        row["Defence_Type"] = "Adversarial training" if "Adversarially" in name else "Robust feature filtering"
        row["Feature_Count"] = len(features)
        row["Features"] = ", ".join(features)
        rows.append(row)
        safe = safe_model_name(name)
        predictions[f"Pred_{safe}"] = pred
        predictions[f"Proba_{safe}"] = proba
    defence_metrics = pd.DataFrame(rows)
    defence_metrics.to_csv(RESULT_DIR / "adversarial_training_defence_metrics.csv", index=False)
    predictions.to_csv(RESULT_DIR / "adversarial_training_defence_predictions.csv", index=False)
    return defence_metrics, predictions


# =============================================================================
# 11. SHAP Explainability and Explanation Stability
# =============================================================================

def shap_analysis(splits, models, metrics_df):
    stage("9. SHAP Explainability and Stability")
    X_test, _, _ = splits["Test"]
    candidates = [n for n in models if n != "Logistic Regression"]
    best = metrics_df[(metrics_df["Split"] == "Test") & (metrics_df["Model"].isin(candidates))].sort_values("F1_Score", ascending=False).iloc[0]["Model"]
    model = models[best]
    X_shap = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_shap)
    if isinstance(values, list): values = values[-1]
    shap.summary_plot(values, X_shap, show=False)
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "shap_summary_plot_test_sample.png", dpi=300, bbox_inches="tight"); plt.close()
    importance = pd.DataFrame({"Feature": FEATURE_COLS, "Mean_Abs_SHAP": np.abs(values).mean(axis=0)}).sort_values("Mean_Abs_SHAP", ascending=False).assign(Model=best)
    importance.to_csv(RESULT_DIR / "shap_feature_importance_test_sample.csv", index=False)
    base_importance = pd.Series(np.abs(values).mean(axis=0), index=FEATURE_COLS)
    base_rank, base_top = base_importance.rank(ascending=False), set(base_importance.sort_values(ascending=False).head(5).index)
    rows = []
    for level in [0.05, 0.10, 0.15, 0.20, 0.30]:
        perturbed = pd.DataFrame(X_shap.values + np.random.normal(0, level, X_shap.shape), columns=FEATURE_COLS, index=X_shap.index)
        pert_values = explainer.shap_values(perturbed)
        if isinstance(pert_values, list): pert_values = pert_values[-1]
        pert_importance = pd.Series(np.abs(pert_values).mean(axis=0), index=FEATURE_COLS)
        rho, pvalue = spearmanr(base_rank, pert_importance.rank(ascending=False))
        top = set(pert_importance.sort_values(ascending=False).head(5).index)
        rows.append({"Perturbation_Level": level, "Spearman_Rho": rho, "Spearman_pvalue": pvalue, "Top_5_Overlap": len(base_top.intersection(top)) / 5})
    stability = pd.DataFrame(rows)
    stability.to_csv(RESULT_DIR / "shap_stability_all_levels_standardized.csv", index=False)
    print(f"SHAP model: {best}")
    return importance, stability


# =============================================================================
# 12. Advanced Probability Diagnostics and Threshold Analysis
# =============================================================================

def optimize_thresholds_and_probability_diagnostics(splits, models):
    stage("10. Threshold Optimisation, PR/ROC, Calibration")
    X_val, y_val, _ = splits["Validation"]
    X_test, y_test, test_df = splits["Test"]
    threshold_rows, metric_rows, prediction_frame = [], [], test_df[["Date", "Ticker", "Close", "Future_Return_5D", "Target"]].copy()

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        val_proba = predict_proba(model, X_val)
        test_proba = predict_proba(model, X_test)
        precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
        f1_values = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
        best_idx = int(np.nanargmax(f1_values[:-1])) if len(thresholds) else 0
        best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
        val_default_pred = (val_proba >= 0.5).astype(int)
        test_default_pred = (test_proba >= 0.5).astype(int)
        test_pred = (test_proba >= best_threshold).astype(int)
        threshold_rows.append({
            "Model": name,
            "Training_Data": "Train only",
            "Refinement_Data": "Validation threshold optimisation",
            "Default_Threshold": 0.5,
            "Optimized_Threshold": best_threshold,
            "Validation_Default_F1": f1_score(y_val, val_default_pred, zero_division=0),
            "Validation_Best_F1": float(f1_values[best_idx]),
            "Test_Default_F1": f1_score(y_test, test_default_pred, zero_division=0),
            "Test_Optimized_F1": f1_score(y_test, test_pred, zero_division=0),
            "Validation_PR_AUC": average_precision_score(y_val, val_proba),
            "Validation_Brier": brier_score_loss(y_val, val_proba),
        })
        metric_rows.append(metric_row(name, "Test_OptimizedThreshold", y_test, test_pred, test_proba))
        safe = safe_model_name(name)
        prediction_frame[f"OptPred_{safe}"] = test_pred
        prediction_frame[f"Proba_{safe}"] = test_proba
        test_precision, test_recall, _ = precision_recall_curve(y_test, test_proba)
        plt.plot(test_recall, test_precision, label=f"{name} AP={average_precision_score(y_test, test_proba):.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on Test Split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "precision_recall_curves_supervised.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        test_proba = predict_proba(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, test_proba):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Test Split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curves_supervised.png", dpi=300)
    plt.close()

    calibration_rows = []
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        test_proba = predict_proba(model, X_test)
        prob_true, prob_pred = calibration_curve(y_test, test_proba, n_bins=10, strategy="quantile")
        for idx, (pred_bin, true_bin) in enumerate(zip(prob_pred, prob_true), start=1):
            calibration_rows.append({"Model": name, "Bin": idx, "Mean_Predicted_Probability": pred_bin, "Observed_Event_Rate": true_bin})
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name} Brier={brier_score_loss(y_test, test_proba):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Event Rate")
    plt.title("Probability Reliability Diagram")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "probability_reliability_diagram.png", dpi=300)
    plt.close()

    threshold_df = pd.DataFrame(threshold_rows)
    optimized_metrics = pd.DataFrame(metric_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    threshold_df.to_csv(RESULT_DIR / "advanced_threshold_optimization.csv", index=False)
    optimized_metrics.to_csv(RESULT_DIR / "advanced_optimized_threshold_metrics.csv", index=False)
    calibration_df.to_csv(RESULT_DIR / "advanced_calibration_curve.csv", index=False)
    prediction_frame.to_csv(RESULT_DIR / "advanced_optimized_threshold_predictions.csv", index=False)
    threshold_df.to_csv(RESULT_DIR / "validation_refinement_summary.csv", index=False)
    return threshold_df, optimized_metrics, prediction_frame


def final_model_selection_and_retraining(splits, models, validation_refinement_df):
    stage("10A. Final Supervised Model Selection and Train+Validation Retraining")
    if validation_refinement_df.empty:
        pd.DataFrame().to_csv(RESULT_DIR / "final_retrained_model_metrics.csv", index=False)
        return pd.DataFrame()
    selected = validation_refinement_df.sort_values("Validation_Best_F1", ascending=False).iloc[0]
    selected_name = selected["Model"]
    threshold = float(selected["Optimized_Threshold"])
    train_val_X = pd.concat([splits["Train"][0], splits["Validation"][0]], ignore_index=True)
    train_val_y = pd.concat([splits["Train"][1], splits["Validation"][1]], ignore_index=True)
    X_test, y_test, test_df = splits["Test"]
    final_model = deepcopy(models[selected_name])
    final_model.fit(train_val_X, train_val_y)
    proba = predict_proba(final_model, X_test)
    pred = (proba >= threshold).astype(int)
    metrics = pd.DataFrame([{
        **metric_row(f"Final Retrained {selected_name}", "Test", y_test, pred, proba),
        "Selected_By": "Highest validation F1 after threshold optimisation",
        "Training_Data": "Train + Validation",
        "Held_Out_Test_Used_For_Selection": False,
        "Applied_Validation_Threshold": threshold,
    }])
    metrics.to_csv(RESULT_DIR / "final_retrained_model_metrics.csv", index=False)
    predictions = test_df[["Date", "Ticker", "Split", "Target"]].copy()
    predictions["Model"] = f"Final Retrained {selected_name}"
    predictions["Probability"] = proba
    predictions["Prediction"] = pred
    predictions.to_csv(RESULT_DIR / "final_retrained_model_predictions.csv", index=False)
    joblib.dump(final_model, MODEL_DIR / f"final_retrained_{safe_model_name(selected_name).lower()}_classifier.pkl")
    return metrics


# =============================================================================
# 13. Baseline Benchmarks, Lift/Gains, and Economic Utility
# =============================================================================

def baseline_benchmarks_and_backtests(df, splits, optimized_predictions):
    stage("11. Baselines, Lift, Deciles, and Economic Backtests")
    _, y_test, test_df = splits["Test"]
    y_test = y_test.values
    rows = []
    baseline_specs = {
        "Naive All Normal": np.zeros(len(test_df), dtype=int),
        "Historical Volatility Benchmark": (test_df["Volatility_20"] >= test_df["Volatility_20"].quantile(0.80)).astype(int).values,
        "Moving Average Rule Benchmark": ((test_df["MA_Ratio_10"] < -0.015) & (test_df["MA_Ratio_20"] < -0.025)).astype(int).values,
    }
    for name, pred in baseline_specs.items():
        rows.append(metric_row(name, "Test", y_test, pred, pred.astype(float)))
    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(RESULT_DIR / "advanced_baseline_benchmarks.csv", index=False)

    proba_cols = [c for c in optimized_predictions.columns if c.startswith("Proba_")]
    decile_rows, utility_rows, transaction_rows = [], [], []
    for col in proba_cols:
        model = col.replace("Proba_", "").replace("_", " ")
        work = optimized_predictions[["Target", "Future_Return_5D", col]].copy()
        work["Decile"] = pd.qcut(work[col].rank(method="first"), 10, labels=False) + 1
        for decile, group in work.groupby("Decile"):
            decile_rows.append({
                "Model": model,
                "Decile": int(decile),
                "Rows": len(group),
                "Mean_Probability": group[col].mean(),
                "Observed_Event_Rate": group["Target"].mean(),
                "Mean_Future_Return_5D": group["Future_Return_5D"].mean(),
            })
        for threshold in np.linspace(0.1, 0.9, 9):
            pred_risk = (work[col] >= threshold).astype(int)
            strategy_returns = np.where(pred_risk == 1, 0.0, work["Future_Return_5D"].values)
            cumulative = np.cumprod(1 + strategy_returns)
            drawdown = cumulative / np.maximum.accumulate(cumulative) - 1
            position = 1 - pred_risk.values
            turnover = np.abs(np.diff(position, prepend=position[0]))
            utility_rows.append({
                "Model": model,
                "Threshold": threshold,
                "Avoided_Risk_Days": int(pred_risk.sum()),
                "Mean_Strategy_Return_5D": float(np.mean(strategy_returns)),
                "Sharpe_Approx": float(np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-12) * np.sqrt(252 / 5)),
                "Max_Drawdown": float(drawdown.min()),
                "Terminal_Growth": float(cumulative[-1]) if len(cumulative) else np.nan,
            })
            for cost_bps in [1, 5, 10]:
                for slippage_bps in [1, 5]:
                    cost = turnover * ((cost_bps + slippage_bps) / 10000.0)
                    net_returns = strategy_returns - cost
                    net_cumulative = np.cumprod(1 + net_returns)
                    net_drawdown = net_cumulative / np.maximum.accumulate(net_cumulative) - 1
                    transaction_rows.append({
                        "Model": model,
                        "Threshold": threshold,
                        "Transaction_Cost_Bps": cost_bps,
                        "Slippage_Bps": slippage_bps,
                        "Turnover_Count": int(turnover.sum()),
                        "Average_Turnover_Per_Row": float(turnover.mean()),
                        "Mean_Net_Strategy_Return_5D": float(np.mean(net_returns)),
                        "Net_Sharpe_Approx": float(np.mean(net_returns) / (np.std(net_returns) + 1e-12) * np.sqrt(252 / 5)),
                        "Net_Max_Drawdown": float(net_drawdown.min()),
                        "Net_Terminal_Growth": float(net_cumulative[-1]) if len(net_cumulative) else np.nan,
                    })
    decile_df = pd.DataFrame(decile_rows)
    utility_df = pd.DataFrame(utility_rows)
    transaction_df = pd.DataFrame(transaction_rows)
    decile_df.to_csv(RESULT_DIR / "advanced_decile_ranking_lift_gains.csv", index=False)
    utility_df.to_csv(RESULT_DIR / "advanced_economic_utility_backtest.csv", index=False)
    transaction_df.to_csv(RESULT_DIR / "advanced_transaction_cost_utility_backtest.csv", index=False)

    if not decile_df.empty:
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=decile_df, x="Decile", y="Observed_Event_Rate", hue="Model", marker="o")
        plt.title("Lift/Gains Style Decile Event Rates")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "lift_gains_decile_chart.png", dpi=300)
        plt.close()
    return baseline_df, decile_df, utility_df, transaction_df


# =============================================================================
# 14. Time-Series Cross-Validation, Tuning, Imbalance, and Ensembles
# =============================================================================

def time_series_cv_tuning_and_ensembles(df, splits, models):
    stage("12. Time-Series CV, Tuning, SMOTE, and Ensembles")
    train_val = df[df["Split"].isin(["Train", "Validation"])].sort_values(["Date", "Ticker"]).copy()
    X_tv = train_val[FEATURE_COLS].copy()
    y_tv = train_val["Target"].astype(int).copy()
    X_test, y_test, _ = splits["Test"]
    if len(X_tv) > 12000:
        sample_idx = np.linspace(0, len(X_tv) - 1, 12000).astype(int)
        X_tv = X_tv.iloc[sample_idx].reset_index(drop=True)
        y_tv = y_tv.iloc[sample_idx].reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=2)
    cv_rows = []
    for name, model in models.items():
        fold = 0
        for train_idx, val_idx in tscv.split(X_tv):
            fold += 1
            candidate = deepcopy(model)
            candidate.fit(X_tv.iloc[train_idx], y_tv.iloc[train_idx])
            proba = predict_proba(candidate, X_tv.iloc[val_idx])
            pred = (proba >= 0.5).astype(int)
            row = metric_row(name, f"TimeSeriesSplit_{fold}", y_tv.iloc[val_idx], pred, proba)
            cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULT_DIR / "advanced_timeseries_cv_metrics.csv", index=False)

    # Lightweight hyperparameter tuning.
    tune_rows = []
    log_grid = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
        {"model__C": [0.1, 1.0, 10.0]},
        scoring="f1",
        cv=tscv,
        n_jobs=1,
    )
    log_grid.fit(X_tv, y_tv)
    tune_rows.append({"Model": "Logistic Regression", "Best_Params": str(log_grid.best_params_), "Best_CV_F1": log_grid.best_score_})

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1),
        {"n_estimators": [50, 100], "max_depth": [4, 8], "min_samples_leaf": [2, 5]},
        n_iter=2,
        scoring="f1",
        cv=tscv,
        random_state=42,
        n_jobs=1,
    )
    rf_search.fit(X_tv, y_tv)
    tune_rows.append({"Model": "Random Forest", "Best_Params": str(rf_search.best_params_), "Best_CV_F1": rf_search.best_score_})

    pos_weight = (y_tv == 0).sum() / max((y_tv == 1).sum(), 1)
    tune_rows.append({"Model": "XGBoost scale_pos_weight", "Best_Params": f"scale_pos_weight={pos_weight:.3f}", "Best_CV_F1": np.nan})
    pd.DataFrame(tune_rows).to_csv(RESULT_DIR / "advanced_hyperparameter_tuning.csv", index=False)

    # SMOTE or deterministic minority oversampling fallback.
    try:
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE(random_state=42).fit_resample(splits["Train"][0], splits["Train"][1])
        smote_method = "SMOTE"
    except ModuleNotFoundError:
        train_X, train_y = splits["Train"][0], splits["Train"][1]
        positives = train_X[train_y == 1]
        positive_y = train_y[train_y == 1]
        repeats = max(int((train_y == 0).sum() / max((train_y == 1).sum(), 1)) - 1, 1)
        X_res = pd.concat([train_X] + [positives] * repeats, ignore_index=True)
        y_res = pd.concat([train_y] + [positive_y] * repeats, ignore_index=True)
        smote_method = "Minority oversampling fallback"
    smote_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    smote_model.fit(X_res, y_res)
    smote_proba = predict_proba(smote_model, X_test)
    smote_pred = (smote_proba >= 0.5).astype(int)
    smote_metrics = pd.DataFrame([metric_row(smote_method, "Test", y_test, smote_pred, smote_proba)])
    smote_metrics.to_csv(RESULT_DIR / "advanced_smote_class_imbalance_metrics.csv", index=False)

    voting = VotingClassifier(
        estimators=[("lr", log_grid.best_estimator_), ("rf", rf_search.best_estimator_), ("gb", models[[m for m in models if m not in ["Logistic Regression", "Random Forest"]][0]])],
        voting="soft",
    )
    stacking = StackingClassifier(
        estimators=[("lr", log_grid.best_estimator_), ("rf", rf_search.best_estimator_)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        stack_method="predict_proba",
        n_jobs=1,
    )
    ensemble_rows = []
    for name, ensemble in {"Voting Ensemble": voting, "Stacking Ensemble": stacking}.items():
        ensemble.fit(X_tv, y_tv)
        proba = predict_proba(ensemble, X_test)
        pred = (proba >= 0.5).astype(int)
        ensemble_rows.append(metric_row(name, "Test", y_test, pred, proba))
        joblib.dump(ensemble, MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl")
    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df.to_csv(RESULT_DIR / "advanced_ensemble_metrics.csv", index=False)
    return cv_df, ensemble_df


# =============================================================================
# 15. Leakage Checks, Feature Selection, and Feature Importance
# =============================================================================

def feature_and_leakage_analysis(df, splits, models):
    stage("13. Leakage Checks, Feature Selection, Permutation Importance")
    X_train, y_train, train_df = splits["Train"]
    X_test, y_test, _ = splits["Test"]
    leakage_rows = []
    blocked_terms = ["future", "target", "label", "split"]
    for feature in FEATURE_COLS:
        leakage_rows.append({
            "Feature": feature,
            "Name_Flag": any(term in feature.lower() for term in blocked_terms),
            "Train_Target_Correlation": train_df[[feature, "Target"]].corr().iloc[0, 1],
            "Missing_Rate": df[feature].isna().mean(),
        })
    leakage_df = pd.DataFrame(leakage_rows)
    leakage_df.to_csv(RESULT_DIR / "advanced_feature_leakage_checks.csv", index=False)

    selector = SelectKBest(mutual_info_classif, k=min(8, len(FEATURE_COLS)))
    selected_train = selector.fit_transform(X_train, y_train)
    selected_features = [feature for feature, keep in zip(FEATURE_COLS, selector.get_support()) if keep]
    selected_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    selected_model.fit(selected_train, y_train)
    selected_test = selector.transform(X_test)
    selected_proba = predict_proba(selected_model, selected_test)
    selected_pred = (selected_proba >= 0.5).astype(int)
    selection_df = pd.DataFrame([metric_row("Selected Feature Logistic", "Test", y_test, selected_pred, selected_proba)])
    selection_df["Selected_Features"] = ", ".join(selected_features)
    selection_df.to_csv(RESULT_DIR / "advanced_feature_selection_pipeline.csv", index=False)

    best_name = max(models, key=lambda name: f1_score(y_test, (predict_proba(models[name], X_test) >= 0.5).astype(int), zero_division=0))
    perm_sample = X_test.sample(min(1200, len(X_test)), random_state=42)
    perm_y = y_test.loc[perm_sample.index]
    perm = permutation_importance(models[best_name], perm_sample, perm_y, scoring="f1", n_repeats=3, random_state=42, n_jobs=1)
    perm_df = pd.DataFrame({
        "Model": best_name,
        "Feature": FEATURE_COLS,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std,
    }).sort_values("Permutation_Importance_Mean", ascending=False)
    perm_df.to_csv(RESULT_DIR / "advanced_permutation_feature_importance.csv", index=False)

    stability_rows = []
    for left, right in [("Train", "Validation"), ("Validation", "Test"), ("Train", "Test")]:
        left_df = splits[left][2]
        right_df = splits[right][2]
        for feature in FEATURE_COLS:
            stability_rows.append({
                "Feature": feature,
                "Comparison": f"{left}_vs_{right}",
                "Mean_Difference": left_df[feature].mean() - right_df[feature].mean(),
                "Std_Ratio": left_df[feature].std() / (right_df[feature].std() + 1e-12),
            })
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(RESULT_DIR / "advanced_feature_stability_across_splits.csv", index=False)
    return leakage_df, selection_df, perm_df


# =============================================================================
# 16. Statistical Significance, Cost Curves, and Calibration
# =============================================================================

def model_significance_and_costs(splits, models, optimized_predictions):
    stage("14. Statistical Tests, Cost Curves, and Calibration After Perturbation")
    X_test, y_test, _ = splits["Test"]
    model_probs = {name: predict_proba(model, X_test) for name, model in models.items()}
    model_preds = {name: (proba >= 0.5).astype(int) for name, proba in model_probs.items()}
    names = list(models)
    stat_rows = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            left_correct = model_preds[left] == y_test.values
            right_correct = model_preds[right] == y_test.values
            b = int(np.sum(left_correct & ~right_correct))
            c = int(np.sum(~left_correct & right_correct))
            n = b + c
            pvalue = binomtest(min(b, c), n=n, p=0.5).pvalue if n else np.nan
            try:
                t_stat, t_pvalue = ttest_rel(np.abs(y_test.values - model_probs[left]), np.abs(y_test.values - model_probs[right]))
            except Exception:
                t_stat, t_pvalue = np.nan, np.nan
            stat_rows.append({
                "Model_A": left,
                "Model_B": right,
                "A_correct_B_wrong": b,
                "A_wrong_B_correct": c,
                "Discordant_Total": n,
                "McNemar_pvalue": pvalue,
                "Paired_Error_tstat": t_stat,
                "Paired_Error_pvalue": t_pvalue,
            })
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(RESULT_DIR / "advanced_model_significance_mcnemar.csv", index=False)

    cost_rows = []
    for name, pred in model_preds.items():
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        for fp_cost in [1, 2, 5, 10]:
            for fn_cost in [1, 5, 10, 25]:
                cost_rows.append({
                    "Model": name,
                    "False_Positive_Cost": fp_cost,
                    "False_Negative_Cost": fn_cost,
                    "Total_Cost": fp * fp_cost + fn * fn_cost,
                    "Average_Cost_Per_Row": (fp * fp_cost + fn * fn_cost) / len(y_test),
                })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(RESULT_DIR / "advanced_cost_sensitive_evaluation.csv", index=False)

    sensitivity_rows = []
    for name, proba in model_probs.items():
        for threshold in np.linspace(0.05, 0.95, 19):
            pred = (proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            sensitivity_rows.append({
                "Model": name,
                "Threshold": threshold,
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1_Score": f1_score(y_test, pred, zero_division=0),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
            })
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(RESULT_DIR / "advanced_threshold_sensitivity_and_fp_fn_costs.csv", index=False)

    calibration_rows = []
    X_train_full, y_train_full, _ = splits["Train"]
    cal_idx = np.linspace(0, len(X_train_full) - 1, min(8000, len(X_train_full))).astype(int)
    X_cal = X_train_full.iloc[cal_idx]
    y_cal = y_train_full.iloc[cal_idx]
    for name, model in models.items():
        calibrated = CalibratedClassifierCV(deepcopy(model), method="sigmoid", cv=2)
        calibrated.fit(X_cal, y_cal)
        base_proba = predict_proba(calibrated, X_test)
        perturbed = pd.DataFrame(X_test.values + np.random.normal(0, 0.10, X_test.shape), columns=FEATURE_COLS, index=X_test.index)
        pert_proba = predict_proba(calibrated, perturbed)
        calibration_rows.append({
            "Model": name,
            "Brier_Calibrated_Base": brier_score_loss(y_test, base_proba),
            "Brier_Calibrated_Perturbed_10pct": brier_score_loss(y_test, pert_proba),
            "PR_AUC_Calibrated_Base": average_precision_score(y_test, base_proba),
            "PR_AUC_Calibrated_Perturbed_10pct": average_precision_score(y_test, pert_proba),
        })
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(RESULT_DIR / "advanced_model_calibration_after_perturbation.csv", index=False)
    return stat_df, cost_df, sensitivity_df


# =============================================================================
# 17. Runtime, Memory, Rolling Backtests, and Stress Extensions
# =============================================================================

def runtime_memory_and_robustness_stress(df, splits, models):
    stage("15. Runtime, Memory, Robustness, and Stress Extensions")
    X_train, y_train, _ = splits["Train"]
    X_test, y_test, test_df = splits["Test"]
    runtime_rows = []
    train_idx = np.linspace(0, len(X_train) - 1, min(8000, len(X_train))).astype(int)
    X_train_bench = X_train.iloc[train_idx]
    y_train_bench = y_train.iloc[train_idx]
    for name, model in models.items():
        candidate = deepcopy(model)
        train_start = time.perf_counter()
        candidate.fit(X_train_bench, y_train_bench)
        train_seconds = time.perf_counter() - train_start
        predict_start = time.perf_counter()
        for _ in range(5):
            predict_proba(candidate, X_test)
        inference_seconds = (time.perf_counter() - predict_start) / 5
        model_size_bytes = len(pickle.dumps(candidate))
        runtime_rows.append({
            "Model": name,
            "Training_Seconds": train_seconds,
            "Inference_Seconds_Per_Test_Run": inference_seconds,
            "Model_File_Size_Bytes": model_size_bytes,
        })
    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(RESULT_DIR / "advanced_runtime_memory_benchmark.csv", index=False)

    robust_rows = []
    scenarios = {
        "Missing 10pct": lambda X: X.mask(np.random.random(X.shape) < 0.10).fillna(X_train.median()),
        "Feature Corruption 20pct": lambda X: X.assign(**{col: X[col].sample(frac=1.0, random_state=42).values for col in FEATURE_COLS[:3]}),
        "Regime Shift High Volatility": lambda X: X.assign(Volatility_10=X["Volatility_10"] * 2.0, Volatility_20=X["Volatility_20"] * 2.0),
        "Correlation Breakdown": lambda X: pd.DataFrame(np.random.permutation(X.values), columns=X.columns, index=X.index),
        "Flash Crash Synthetic": lambda X: X.assign(Return=X["Return"] - 0.08, Drawdown=X["Drawdown"] - 0.15, Return_Zscore_20=X["Return_Zscore_20"] - 3.0),
    }
    for model_name, model in models.items():
        base_proba = predict_proba(model, X_test)
        base_pred = (base_proba >= 0.5).astype(int)
        base_f1 = f1_score(y_test, base_pred, zero_division=0)
        for scenario_name, transform in scenarios.items():
            X_scenario = transform(X_test.copy())
            proba = predict_proba(model, X_scenario)
            pred = (proba >= 0.5).astype(int)
            robust_rows.append({
                "Model": model_name,
                "Scenario": scenario_name,
                "Baseline_F1": base_f1,
                "Scenario_F1": f1_score(y_test, pred, zero_division=0),
                "F1_Drop": base_f1 - f1_score(y_test, pred, zero_division=0),
                "Prediction_Flips": int(np.sum(base_pred != pred)),
            })
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(RESULT_DIR / "advanced_missing_corruption_regime_stress.csv", index=False)

    rolling_rows = []
    ordered = df.sort_values(["Date", "Ticker"]).copy()
    dates = np.array(sorted(pd.to_datetime(ordered["Date"]).dt.normalize().unique()))
    window, horizon = 504, 63
    for start_idx in range(0, max(len(dates) - window - horizon, 1), horizon):
        train_dates = set(dates[start_idx:start_idx + window])
        test_dates = set(dates[start_idx + window:start_idx + window + horizon])
        train = ordered[ordered["Date"].dt.normalize().isin(train_dates)]
        test = ordered[ordered["Date"].dt.normalize().isin(test_dates)]
        if train.empty or test.empty or train["Target"].nunique() < 2:
            continue
        model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
        model.fit(train[FEATURE_COLS], train["Target"].astype(int))
        proba = predict_proba(model, test[FEATURE_COLS])
        pred = (proba >= 0.5).astype(int)
        rolling_rows.append({
            "Window_Start": min(train_dates),
            "Window_End": max(train_dates),
            "Test_Start": min(test_dates),
            "Test_End": max(test_dates),
            "Rows": len(test),
            "F1_Score": f1_score(test["Target"], pred, zero_division=0),
            "ROC_AUC": roc_auc_score(test["Target"], proba) if test["Target"].nunique() == 2 else np.nan,
        })
    rolling_df = pd.DataFrame(rolling_rows)
    rolling_df.to_csv(RESULT_DIR / "advanced_rolling_window_backtest.csv", index=False)
    return runtime_df, robust_df, rolling_df


# =============================================================================
# 18. SHAP Model Comparison and Temporal Drift
# =============================================================================

def shap_comparison_and_drift(df, splits, models):
    stage("16. SHAP Comparison Across Models and Time Drift")
    X_test, _, test_df = splits["Test"]
    X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    comparison_rows = []
    for name, model in models.items():
        if name == "Logistic Regression":
            fitted_lr = model.named_steps["model"]
            values = np.abs(fitted_lr.coef_[0])
            for feature, value in zip(FEATURE_COLS, values):
                comparison_rows.append({"Model": name, "Feature": feature, "Mean_Abs_SHAP_or_Coefficient": value})
        else:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_sample)
            if isinstance(values, list):
                values = values[-1]
            importance = np.abs(values).mean(axis=0)
            for feature, value in zip(FEATURE_COLS, importance):
                comparison_rows.append({"Model": name, "Feature": feature, "Mean_Abs_SHAP_or_Coefficient": value})
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULT_DIR / "advanced_shap_comparison_across_models.csv", index=False)

    tree_names = [name for name in models if name != "Logistic Regression"]
    best_tree = tree_names[0]
    model = models[best_tree]
    dated = test_df.sort_values("Date")
    midpoint = dated["Date"].median()
    drift_rows = []
    explainer = shap.TreeExplainer(model)
    for period, idx in {"Early_Test": dated[dated["Date"] <= midpoint].index, "Late_Test": dated[dated["Date"] > midpoint].index}.items():
        period_X = X_test.loc[idx].sample(min(120, len(idx)), random_state=42)
        values = explainer.shap_values(period_X)
        if isinstance(values, list):
            values = values[-1]
        for feature, value in zip(FEATURE_COLS, np.abs(values).mean(axis=0)):
            drift_rows.append({"Model": best_tree, "Period": period, "Feature": feature, "Mean_Abs_SHAP": value})
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(RESULT_DIR / "advanced_shap_drift_over_time.csv", index=False)
    return comparison_df, drift_df


# =============================================================================
# 19. Final Reports, Dashboard, and Output Manifest
# =============================================================================

def final_outputs(
    df,
    metrics_df,
    robustness_df,
    stress_df,
    shap_stability_df,
    event_detector_metrics_df,
    ttd_summary,
    proxy_metrics_df,
    adversarial_df,
    validation_refinement_df,
    drift_summary,
    telemetry_gap_df,
    fusion_metrics_df,
    defence_df,
):
    stage("17. Final Summaries, Validation, Dashboard, and Manifest")
    CHAPTER4_DIR.mkdir(parents=True, exist_ok=True)
    supervised_test = metrics_df[metrics_df["Split"] == "Test"].copy()
    supervised_test["Model_Type"] = "Supervised Classifier"
    supervised_test["Average_Precision"] = np.nan
    detector_test = event_detector_metrics_df[event_detector_metrics_df["Split"] == "Test"].copy()
    summary = pd.concat([supervised_test, detector_test], ignore_index=True, sort=False)
    summary = summary.merge(robustness_df.groupby("Model")["Robustness_Score"].mean().rename("Avg_Gaussian_Robustness"), on="Model", how="left")
    summary = summary.merge(stress_df.groupby("Model")["Stressed_F1"].min().rename("Worst_Stress_F1"), on="Model", how="left")
    summary = summary.merge(robustness_df.groupby("Model")["F1_Drop"].mean().rename("Avg_F1_Drop_Under_Noise"), on="Model", how="left")
    if not validation_refinement_df.empty:
        refinement = validation_refinement_df[["Model", "Optimized_Threshold", "Validation_Best_F1", "Test_Optimized_F1"]].copy()
        summary = summary.merge(refinement, on="Model", how="left")
    if not ttd_summary.empty:
        target_ttd = ttd_summary[(ttd_summary["Split"] == "Test") & (ttd_summary["Event_Source"] == "Drawdown Target")].groupby("Model").agg(
            Target_Event_Detection_Rate=("Detection_Rate", "mean"),
            Mean_Time_To_Detection=("Mean_Time_To_Detection", "mean"),
            Median_Time_To_Detection=("Median_Time_To_Detection", "mean"),
            Mean_Lead_Time=("Mean_Lead_Time", "mean"),
        ).reset_index()
        proxy_ttd = ttd_summary[(ttd_summary["Split"] == "Test") & (ttd_summary["Event_Source"] == "Synthetic Proxy")].groupby("Model").agg(
            Synthetic_Proxy_Event_Detection_Rate=("Detection_Rate", "mean"),
        ).reset_index()
        summary = summary.merge(target_ttd, on="Model", how="left")
        summary = summary.merge(proxy_ttd, on="Model", how="left")
    if not proxy_metrics_df.empty:
        proxy = proxy_metrics_df[proxy_metrics_df["Split"] == "Test"].groupby("Model").agg(
            Synthetic_Proxy_Row_F1=("F1_Score", "mean"),
            Synthetic_Proxy_Row_Recall=("Recall", "mean"),
        ).reset_index()
        summary = summary.merge(proxy, on="Model", how="left")
    if not adversarial_df.empty:
        adversarial = adversarial_df.groupby("Model").agg(
            Avg_Adversarial_F1_Drop=("F1_Drop", "mean"),
            Worst_Adversarial_F1_Drop=("F1_Drop", "max"),
            Max_Targeted_Event_Miss_Rate=("Targeted_Event_Miss_Rate", "max"),
        ).reset_index()
        summary = summary.merge(adversarial, on="Model", how="left")
    for col in ["Avg_Gaussian_Robustness", "Worst_Stress_F1", "Target_Event_Detection_Rate", "Mean_Time_To_Detection", "Synthetic_Proxy_Event_Detection_Rate", "Avg_Adversarial_F1_Drop"]:
        if col not in summary.columns:
            summary[col] = np.nan
    summary.to_csv(RESULT_DIR / "final_model_summary_standardized.csv", index=False)
    failure = summary[["Model", "Model_Type", "Precision", "Recall", "F1_Score", "ROC_AUC", "Avg_Gaussian_Robustness", "Worst_Stress_F1", "Target_Event_Detection_Rate", "Mean_Time_To_Detection", "Synthetic_Proxy_Event_Detection_Rate", "Avg_Adversarial_F1_Drop"]].copy()
    failure["Interpretation"] = failure.apply(lambda r: "Strong candidate" if r["F1_Score"] == summary["F1_Score"].max() else "Comparator model for trade-off analysis", axis=1)
    failure.to_csv(RESULT_DIR / "model_failure_analysis_standardized.csv", index=False)
    best_f1 = summary.sort_values("F1_Score", ascending=False).iloc[0]
    narrative = "\n".join([
        "# Dissertation Alignment Summary",
        "",
        "## Aim Alignment",
        "The project evaluates supervised financial drawdown-risk prediction, supervised event detection, drift monitoring, event-level time-to-detection, alert fusion, and robustness in a reproducible chronological pipeline.",
        "",
        "## Target Definition",
        "Target = 1 when the next 5-day return is <= -3%, otherwise Target = 0.",
        "",
        "## Research Question Coverage",
        f"- {best_f1['Model']} achieved the strongest test F1 score across supervised classifier and supervised event-detector summaries.",
        "- Time-to-detection is measured from first model alert to event-window onset using ticker-specific trading rows.",
        "- Operational incident time-to-detection is also measured against real external telemetry timestamps when labelled operational files are present.",
        "- Additional event detectors are trained with supervised labels only.",
        "- Validation data refines thresholds and model-selection diagnostics; base supervised model fitting remains train-only.",
        "- Robustness includes Gaussian perturbations, financial stress scenarios, bounded feature-space adversarial simulations, adversarial training, and robust feature filtering.",
        "- Drift monitoring is implemented with rolling PSI, KS-test, Page-Hinkley, CUSUM, and retraining-trigger recommendations.",
        "- Alert fusion combines supervised classifier and supervised event-detector signals into severity levels, analyst actions, persistence metrics, false-alert burden, and audit-rationale outputs.",
        "- Real telemetry outputs and synthetic proxy outputs are written separately; real telemetry is prioritised when order-book, order-flow, execution-log, cyber-log, and manipulation-label files are present.",
    ])
    (RESULT_DIR / "dissertation_alignment_summary.md").write_text(narrative, encoding="utf-8")
    gap_matrix = pd.DataFrame([
        {"Claim_Area": "Real cybersecurity / manipulation telemetry", "Implementation_Status": "Optional real telemetry ingestion and supervised modeling implemented; current run depends on whether external files are supplied", "Primary_Output": "operational_telemetry_model_metrics.csv"},
        {"Claim_Area": "Dedicated drift detection", "Implementation_Status": "Implemented with rolling PSI, KS, Page-Hinkley, CUSUM, and retraining triggers", "Primary_Output": "drift_monitoring_feature_windows.csv"},
        {"Claim_Area": "Production-grade time-to-detection", "Implementation_Status": "Extended with event windows, lead-time proxy, alert persistence, false-alarm burden, and real operational incident timestamp timing when files are present", "Primary_Output": "time_to_detection_event_level.csv"},
        {"Claim_Area": "Operational incident time-to-detection", "Implementation_Status": "Implemented against real external telemetry timestamps when operational labels exist", "Primary_Output": "operational_incident_time_to_detection.csv"},
        {"Claim_Area": "Formal adversarial attacks", "Implementation_Status": "Feature-space attack success, minimal-budget metrics, adversarial training, order-book attack scenario generation, and black-box/white-box microstructure attack evaluation implemented", "Primary_Output": "microstructure_attack_evaluation.csv"},
        {"Claim_Area": "Deployment-grade supervised event policy", "Implementation_Status": "Fusion, severity, score normalization, and alert rationale implemented using supervised signals only", "Primary_Output": "unified_alert_fusion_policy.csv"},
        {"Claim_Area": "Validation-driven refinement", "Implementation_Status": "Validation thresholds and final train+validation retraining policy implemented for supervised models", "Primary_Output": "validation_refinement_summary.csv"},
        {"Claim_Area": "Data realism", "Implementation_Status": "Daily OHLCV coverage plus explicit missing microstructure/news/cyber feature report", "Primary_Output": "data_realism_feature_gap_assessment.csv"},
        {"Claim_Area": "Target design", "Implementation_Status": "Fixed, severe, volatility-adjusted, asset-specific, and multiclass risk-state reports implemented", "Primary_Output": "target_design_variants.csv"},
        {"Claim_Area": "Transaction cost realism", "Implementation_Status": "Turnover, cost, and slippage utility grid implemented", "Primary_Output": "advanced_transaction_cost_utility_backtest.csv"},
        {"Claim_Area": "Deployment and defences", "Implementation_Status": "Minimal HTTP scoring wrapper, file-stream loop, retraining harness, rollback registry, adversarial training, and robust feature filtering implemented", "Primary_Output": "deployment_runtime_artifacts.csv"},
        {"Claim_Area": "Real vs synthetic separation", "Implementation_Status": "Real operational telemetry outputs are separated from synthetic proxy outputs for write-up clarity", "Primary_Output": "operational_telemetry_model_metrics.csv"},
    ])
    gap_matrix.to_csv(RESULT_DIR / "research_gap_closure_matrix.csv", index=False)
    assert set(df["Split"].unique()) == {"Train", "Validation", "Test"}
    assert df["Target"].sum() > 0
    assert metrics_df["Model"].nunique() == 3
    assert len(shap_stability_df) == 5
    fig, ax = plt.subplots(figsize=(13, 4.8)); ax.axis("off")
    boxes = [("Yahoo Finance\nMarket Data", 0.02, "#dceeff"), ("Feature Engineering\nFinancial Indicators", 0.21, "#e7f5df"), ("Supervised Target\nNext 5D Return <= -3%", 0.41, "#fff2cc"), ("Classifiers\nLogistic, RF, XGB", 0.61, "#f8dfdf"), ("Evaluation\nMetrics, Robustness,\nSHAP", 0.80, "#eadcf8")]
    y, w, h = 0.42, 0.16, 0.34
    for text, x, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.3, edgecolor="#444", facecolor=color)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")
    for i in range(len(boxes) - 1):
        ax.add_patch(FancyArrowPatch((boxes[i][1] + w + 0.01, y + h / 2), (boxes[i + 1][1] - 0.01, y + h / 2), arrowstyle="->", mutation_scale=16, linewidth=1.5))
    ax.set_title("Supervised Financial Drawdown-Risk Prediction Pipeline", fontsize=15, weight="bold", pad=14); plt.tight_layout(); plt.savefig(CHAPTER4_DIR / "01_supervised_framework_pipeline_diagram.png", dpi=300, bbox_inches="tight"); plt.close()
    rows = [{"Figure": p.name, "Path": str(p), "Size_Bytes": p.stat().st_size} for p in sorted(FIGURE_DIR.rglob("*.png"))]
    pd.DataFrame(rows).to_csv(RESULT_DIR / "chapter4_visual_story_manifest.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Automated Final Summary Dashboard", fontsize=15, weight="bold")
    axes[0, 0].axis("off")
    best = summary.sort_values("F1_Score", ascending=False).iloc[0]
    axes[0, 0].text(
        0.02,
        0.95,
        f"Best F1 model: {best['Model']}\nF1: {best['F1_Score']:.3f}\nROC-AUC: {best['ROC_AUC']:.3f}\nAvg robustness: {best['Avg_Gaussian_Robustness']:.3f}",
        va="top",
        fontsize=11,
    )
    summary.set_index("Model")[["Precision", "Recall", "F1_Score", "ROC_AUC"]].plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("Core Metrics")
    axes[0, 1].tick_params(axis="x", rotation=20)
    summary.set_index("Model")[["Avg_Gaussian_Robustness", "Worst_Stress_F1"]].plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Robustness and Stress")
    axes[1, 0].tick_params(axis="x", rotation=20)
    summary.set_index("Model")[["FP", "FN"]].plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("False Positive / False Negative Counts")
    axes[1, 1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "automated_final_summary_dashboard.png", dpi=300)
    plt.close()
    manifest = []
    for category, folder in [("data", DATA_DIR), ("results", RESULT_DIR), ("figures", FIGURE_DIR), ("models", MODEL_DIR)]:
        for p in sorted(folder.rglob("*")):
            if p.is_file(): manifest.append({"Category": category, "File": p.name, "Path": str(p), "Size_Bytes": p.stat().st_size})
    pd.DataFrame(manifest).to_csv(RESULT_DIR / "notebook_output_manifest.csv", index=False)
    print(summary.to_string(index=False))
    save_state(17, df=df, feature_cols=FEATURE_COLS, final_summary=summary, failure_analysis=failure)


# =============================================================================
# 20. Cybersecurity Case Studies, Attack Simulation, and Resilience Scorecard
# =============================================================================

def real_world_cyber_attack_case_studies():
    stage("18. Real-World Cyber Attack Case Studies")
    cases = pd.DataFrame([
        {"Year": 2010, "Country": "US", "Attack": "Flash Crash - Sarao spoofing on CME", "Impact": "$1 trillion lost temporarily", "Primary_Threat_Vector": "Spoofing and order-book manipulation", "Framework_Layer": "Layer 5: Operational Response", "Detection_Rationale": "Requires alert fusion, market surveillance, and escalation beyond price-only anomaly detection"},
        {"Year": 2012, "Country": "US", "Attack": "Knight Capital - faulty algo deployment", "Impact": "$440 million loss in 45 mins", "Primary_Threat_Vector": "Defective automated trading deployment", "Framework_Layer": "Layer 2: Model Integrity", "Detection_Rationale": "Model and execution controls should detect abnormal behaviour after deployment"},
        {"Year": 2013, "Country": "US", "Attack": "AP Twitter Hack - fake news moved market", "Impact": "Dow dropped 150 points in 2 mins", "Primary_Threat_Vector": "Compromised news/social signal", "Framework_Layer": "Layer 1: Data Integrity", "Detection_Rationale": "Input validation and source integrity checks should flag compromised external signals"},
        {"Year": 2016, "Country": "UK", "Attack": "Tesco Bank - cyber attack on debit cards", "Impact": "GBP2.26M stolen, FCA fined GBP16.4M", "Primary_Threat_Vector": "Payment card fraud and account compromise", "Framework_Layer": "Layer 5: Operational Response", "Detection_Rationale": "Incident response and alert triage govern customer harm containment"},
        {"Year": 2023, "Country": "UK", "Attack": "ION Trading - LockBit ransomware", "Impact": "Derivatives trading halted for days", "Primary_Threat_Vector": "Ransomware on trading infrastructure", "Framework_Layer": "Layer 5: Operational Response", "Detection_Rationale": "Operational resilience and continuity controls determine recovery from infrastructure outages"},
        {"Year": 2023, "Country": "US", "Attack": "ICBC US - LockBit ransomware", "Impact": "US Treasury bond trading disrupted", "Primary_Threat_Vector": "Ransomware on settlement/trading services", "Framework_Layer": "Layer 5: Operational Response", "Detection_Rationale": "Alert fusion, escalation, and continuity planning mitigate market-service disruption"},
        {"Year": 2024, "Country": "US", "Attack": "EquiLend - LockBit ransomware", "Impact": "Securities lending platform offline 2 weeks", "Primary_Threat_Vector": "Ransomware on securities-lending platform", "Framework_Layer": "Layer 5: Operational Response", "Detection_Rationale": "Business continuity and incident response are central when a market utility is unavailable"},
    ])
    mapping = cases[["Year", "Country", "Attack", "Primary_Threat_Vector", "Framework_Layer", "Detection_Rationale"]].copy()
    cases.to_csv(RESULT_DIR / "real_world_cyber_attack_case_studies.csv", index=False)
    mapping.to_csv(RESULT_DIR / "attack_to_framework_layer_mapping.csv", index=False)

    plt.figure(figsize=(11, 5.5))
    counts = mapping.groupby(["Framework_Layer", "Country"]).size().reset_index(name="Case_Count")
    sns.barplot(data=counts, y="Framework_Layer", x="Case_Count", hue="Country", palette="Set2")
    plt.title("Real Cyber Attack Case Studies Mapped to Defence Framework Layers")
    plt.xlabel("Number of Case Studies")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cyber_attack_case_study_mapping.png", dpi=300, bbox_inches="tight")
    plt.close()
    return cases


def _load_cyber_event_dates():
    candidates = [DATA_DIR / "cyber_event_logs.csv", PROJECT_ROOT / "cyber_event_logs.csv"]
    for path in candidates:
        if path.exists():
            try:
                logs = pd.read_csv(path)
                date_col = next((c for c in logs.columns if "date" in c.lower() or "time" in c.lower()), None)
                if date_col:
                    return set(pd.to_datetime(logs[date_col], errors="coerce").dropna().dt.normalize())
            except Exception:
                return set()
    return set()


def _make_cyber_attack_variant(X_test, test_df, attack_type, rng):
    X_attack = X_test.copy()
    attack_dates = _load_cyber_event_dates()
    if attack_type == "Data Feed Poisoning":
        direction = rng.choice([-1.0, 1.0], size=len(X_attack))
        severity = rng.uniform(0.02, 0.05, size=len(X_attack)) * direction
        for col in ["Return", "Momentum_10", "Return_Zscore_20"]:
            if col in X_attack:
                X_attack[col] = X_attack[col] + severity
        for col in ["MA_Ratio_10", "MA_Ratio_20"]:
            if col in X_attack:
                X_attack[col] = X_attack[col] * (1 + severity)
        if "Volume_Change" in X_attack:
            X_attack["Volume_Change"] = rng.permutation(X_attack["Volume_Change"].fillna(0).values)
    elif attack_type == "Stale Data Injection":
        ordered = test_df.sort_values(["Ticker", "Date"]).index
        stale = X_attack.loc[ordered].copy()
        shifted_parts = []
        for _, idx in test_df.loc[ordered].groupby("Ticker").groups.items():
            lag = int(rng.integers(1, 4))
            shifted_parts.append(stale.loc[idx].shift(lag).bfill())
        X_attack = pd.concat(shifted_parts).loc[X_test.index]
    elif attack_type == "Gradual Drift Attack":
        ramp = np.linspace(0.0, 1.0, len(X_attack)).reshape(-1, 1)
        drift_cols = [c for c in ["Return", "Volatility_10", "Volatility_20", "Momentum_10", "BB_Width", "Drawdown"] if c in X_attack]
        std = X_attack[drift_cols].std().replace(0, 1e-6).values.reshape(1, -1)
        direction = np.array([1 if c != "Drawdown" else -1 for c in drift_cols]).reshape(1, -1)
        X_attack.loc[:, drift_cols] = X_attack[drift_cols].values + ramp * std * direction * 0.75
    elif attack_type == "Coordinated Manipulation":
        if attack_dates:
            mask = pd.to_datetime(test_df["Date"]).dt.normalize().isin(attack_dates).values
            if not mask.any():
                high_risk = test_df["Target"].astype(int).values == 1
                mask = high_risk if high_risk.any() else np.ones(len(X_attack), dtype=bool)
        else:
            score = X_attack[["Volume_Change", "Volatility_10", "Return_Zscore_20"]].abs().sum(axis=1)
            mask = score >= score.quantile(0.75)
        for col, delta in {"Volume_Change": 2.5, "Volatility_10": 1.5, "Volatility_20": 1.3, "Return_Zscore_20": -2.0, "Drawdown": -0.08}.items():
            if col in X_attack:
                X_attack.loc[mask, col] = X_attack.loc[mask, col] + delta
        for col in ["Momentum_10", "MA_Ratio_10", "MA_Ratio_20"]:
            if col in X_attack:
                X_attack.loc[mask, col] = X_attack.loc[mask, col] * 0.92
    return X_attack.replace([np.inf, -np.inf], np.nan).fillna(X_test.median(numeric_only=True))


def cyber_threat_simulation_on_data_feeds(df, splits, models):
    stage("19. Cyber Threat Simulation on Data Feeds")
    X_val, _, _ = splits["Validation"]
    X_test, _, test_df = splits["Test"]
    rng = np.random.default_rng(42)
    rows = []
    attack_types = ["Data Feed Poisoning", "Stale Data Injection", "Gradual Drift Attack", "Coordinated Manipulation"]
    representations = {
        "Data Feed Poisoning": "Corrupted vendor OHLCV and derived feature feed",
        "Stale Data Injection": "Man-in-the-middle latency or replayed stale market rows",
        "Gradual Drift Attack": "Slow compromise that shifts feature distributions over time",
        "Coordinated Manipulation": "Multi-vector manipulation with spoofing-like features and cyber-event timestamps",
    }
    for name, model in models.items():
        baseline_test = predict_proba(model, X_test)
        validation_scores = predict_proba(model, X_val)
        alert_threshold = max(0.5, float(np.quantile(validation_scores, 0.95)))
        baseline_alerts = baseline_test >= alert_threshold
        for attack_type in attack_types:
            X_attack = _make_cyber_attack_variant(X_test, test_df, attack_type, rng)
            attack_scores = predict_proba(model, X_attack)
            score_increase = attack_scores - baseline_test
            attack_alerts = attack_scores >= alert_threshold
            increase_flags = score_increase >= 0.05
            detected = attack_alerts | increase_flags
            rows.append({
                "Model": name,
                "Attack_Type": attack_type,
                "Threat_Representation": representations[attack_type],
                "Rows_Attacked": len(X_test),
                "Baseline_Mean_Anomaly_Score": float(np.mean(baseline_test)),
                "Attacked_Mean_Anomaly_Score": float(np.mean(attack_scores)),
                "Mean_Anomaly_Score_Increase": float(np.mean(score_increase)),
                "Alert_Threshold": alert_threshold,
                "Baseline_Alerts": int(np.sum(baseline_alerts)),
                "Attacked_Alerts": int(np.sum(attack_alerts)),
                "Score_Increase_Detections": int(np.sum(increase_flags)),
                "Attacks_Detected": int(np.sum(detected)),
                "Attacks_Missed": int(len(X_test) - np.sum(detected)),
                "Detection_Rate": float(np.mean(detected)),
                "Alert_Uplift": int(np.sum(attack_alerts) - np.sum(baseline_alerts)),
            })
    cyber_detection = pd.DataFrame(rows)
    cyber_detection.to_csv(RESULT_DIR / "cyber_attack_data_feed_detection_results.csv", index=False)

    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=cyber_detection, x="Attack_Type", y="Detection_Rate", hue="Model")
    plt.title("Cyber Attack Detection Rates by Attack Type")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cyber_attack_detection_rates_by_type.png", dpi=300, bbox_inches="tight")
    plt.close()
    return cyber_detection


def _clip_score(value):
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def cybersecurity_resilience_scorecard(robustness_df, stress_df, adversarial_df, shap_stability_df, drift_summary, cyber_detection_df):
    stage("20. Cybersecurity Resilience Scorecard")
    models = sorted(set(robustness_df.get("Model", pd.Series(dtype=str))).union(stress_df.get("Model", pd.Series(dtype=str))).union(cyber_detection_df.get("Model", pd.Series(dtype=str))))
    if not models:
        models = ["Portfolio Average"]
    framework = pd.DataFrame([
        {"Layer": "Layer 1: Data Integrity", "What_It_Covers": "Cyber attack detection on data feeds", "Source": "cyber_threat_simulation_on_data_feeds()", "Primary_Output": "cyber_attack_data_feed_detection_results.csv"},
        {"Layer": "Layer 2: Model Integrity", "What_It_Covers": "Adversarial robustness and noise tolerance", "Source": "robustness_testing() + adversarial_attack_testing()", "Primary_Output": "supervised_robustness_results.csv; adversarial_attack_results.csv"},
        {"Layer": "Layer 3: Drift Monitoring", "What_It_Covers": "Distribution change detection", "Source": "drift_monitoring()", "Primary_Output": "drift_alarm_summary.csv"},
        {"Layer": "Layer 4: Explainability", "What_It_Covers": "SHAP stability under stress", "Source": "shap_analysis()", "Primary_Output": "shap_stability_all_levels_standardized.csv"},
        {"Layer": "Layer 5: Operational Response", "What_It_Covers": "Alert fusion and deployment policy", "Source": "defence_and_deployment_policy_outputs()", "Primary_Output": "active_defence_policy_matrix.csv"},
    ])
    framework.to_csv(RESULT_DIR / "layered_defence_framework_mapping.csv", index=False)

    cyber_scores = cyber_detection_df.groupby("Model")["Detection_Rate"].mean().to_dict() if not cyber_detection_df.empty else {}
    robust_scores = robustness_df.groupby("Model")["Robustness_Score"].mean().to_dict() if "Robustness_Score" in robustness_df else {}
    stress_scores = stress_df.groupby("Model")["Stressed_F1"].mean().to_dict() if "Stressed_F1" in stress_df else {}
    adv_scores = {}
    if not adversarial_df.empty and "F1_Drop" in adversarial_df:
        adv_scores = (1 - adversarial_df.groupby("Model")["F1_Drop"].mean().clip(lower=0, upper=1)).to_dict()
    drift_score = 1.0
    if isinstance(drift_summary, pd.DataFrame) and not drift_summary.empty and "Retraining_Triggers" in drift_summary:
        checked = drift_summary.get("Features_Checked", pd.Series([len(FEATURE_COLS)])).replace(0, len(FEATURE_COLS))
        drift_score = _clip_score(1 - (drift_summary["Retraining_Triggers"] / checked).mean())
    shap_score = 0.0
    if isinstance(shap_stability_df, pd.DataFrame) and not shap_stability_df.empty:
        rho = shap_stability_df.get("Spearman_Rho", pd.Series(dtype=float)).mean()
        overlap = shap_stability_df.get("Top_5_Overlap", pd.Series(dtype=float)).mean()
        shap_score = _clip_score(np.nanmean([rho, overlap]))

    rows = []
    for model in models:
        model_integrity = np.nanmean([robust_scores.get(model, np.nan), stress_scores.get(model, np.nan), adv_scores.get(model, np.nan)])
        layer_scores = {
            "Layer 1: Data Integrity": cyber_scores.get(model, np.nan),
            "Layer 2: Model Integrity": model_integrity,
            "Layer 3: Drift Monitoring": drift_score,
            "Layer 4: Explainability": shap_score,
            "Layer 5: Operational Response": np.nanmean([cyber_scores.get(model, np.nan), drift_score, model_integrity]),
        }
        for layer, score in layer_scores.items():
            rows.append({"Model": model, "Layer": layer, "Resilience_Score": _clip_score(score), "Score_Percent": round(_clip_score(score) * 100, 1)})
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(RESULT_DIR / "cybersecurity_resilience_scorecard.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    xs = np.linspace(0.04, 0.80, len(framework))
    colors = ["#dceeff", "#e7f5df", "#fff2cc", "#f8dfdf", "#eadcf8"]
    for (_, row), x, color in zip(framework.iterrows(), xs, colors):
        ax.add_patch(Rectangle((x, 0.35), 0.15, 0.32, linewidth=1.2, edgecolor="#444", facecolor=color))
        ax.text(x + 0.075, 0.51, row["Layer"].replace(": ", ":\n"), ha="center", va="center", fontsize=9, weight="bold")
    for i in range(len(xs) - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.16, 0.51), (xs[i + 1] - 0.01, 0.51), arrowstyle="->", mutation_scale=14, linewidth=1.3))
    ax.set_title("Five-Layer Cybersecurity Resilience Framework", fontsize=15, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "layered_defence_framework_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()

    pivot = scorecard.pivot(index="Model", columns="Layer", values="Resilience_Score").fillna(0)
    labels = list(pivot.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for model, values in pivot.iterrows():
        vals = values.tolist() + values.tolist()[:1]
        ax.plot(angles, vals, linewidth=2, label=model)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label.replace(": ", ":\n") for label in labels], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Cybersecurity Resilience Radar Chart", fontsize=14, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cybersecurity_resilience_radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()

    model_summary = scorecard.groupby("Model")["Resilience_Score"].mean().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    model_summary.plot(kind="bar", ax=axes[0], color="#4c78a8")
    axes[0].set_title("Overall Resilience by Model")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Layer Scores")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cybersecurity_resilience_summary_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    return scorecard


# =============================================================================
# 21. Pipeline Orchestration
# =============================================================================

def main():
    start = time.time()
    setup_environment()
    df = prepare_dataset()
    create_eda_figures(df)
    splits = split_xy(df)
    models = train_models(splits)
    metrics_df, supervised_predictions = evaluate_models(df, splits, models)
    create_performance_figures(metrics_df)
    event_detectors = train_supervised_event_detectors(splits)
    event_detector_metrics_df, event_detector_predictions, event_detector_thresholds_df = evaluate_supervised_event_detectors(splits, event_detectors)
    telemetry_gap_df, data_realism_df = external_telemetry_gap_assessment(df)
    operational_summary_df, operational_features_df, operational_metrics_df, operational_predictions_df = operational_telemetry_ingestion_and_modeling()
    microstructure_attacks_df, microstructure_attack_summary_df = microstructure_attack_generation()
    operational_ttd_df, operational_lead_metrics_df = operational_incident_time_to_detection(operational_features_df, operational_predictions_df)
    drift_df, drift_summary, retraining_policy_df = drift_monitoring(df)
    target_variant_df, multiclass_target_df = target_design_variant_report(df)
    ttd_event_level, ttd_summary = time_to_detection_analysis(df, supervised_predictions, event_detector_predictions)
    proxy_metrics_df, proxy_predictions = synthetic_proxy_detection_analysis(df, supervised_predictions, event_detector_predictions)
    alert_episodes_df, false_alarm_burden_df, lead_time_precision_df = alert_latency_operational_metrics(df, supervised_predictions, event_detector_predictions)
    fusion_df, fusion_metrics_df = unified_alert_fusion_policy(supervised_predictions, event_detector_predictions)
    rationale_df = investigator_alert_rationale_outputs(splits, models, fusion_df)
    robustness_df = robustness_testing(splits, models)
    stress_df = stress_testing(splits, models)
    adversarial_df, adversarial_predictions = adversarial_attack_testing(splits, models)
    defence_training_df, defence_predictions = adversarial_training_and_robust_feature_filtering(splits, drift_df, adversarial_df)
    microstructure_attack_eval_df = microstructure_attack_evaluation(microstructure_attacks_df, operational_features_df)
    deployment_df, defence_df = defence_and_deployment_policy_outputs(drift_summary, fusion_metrics_df, adversarial_df, defence_training_df)
    _, shap_stability_df = shap_analysis(splits, models, metrics_df)
    validation_refinement_df, _, optimized_predictions = optimize_thresholds_and_probability_diagnostics(splits, models)
    final_retrained_metrics_df = final_model_selection_and_retraining(splits, models, validation_refinement_df)
    baseline_benchmarks_and_backtests(df, splits, optimized_predictions)
    time_series_cv_tuning_and_ensembles(df, splits, models)
    feature_and_leakage_analysis(df, splits, models)
    model_significance_and_costs(splits, models, optimized_predictions)
    runtime_memory_and_robustness_stress(df, splits, models)
    shap_comparison_and_drift(df, splits, models)
    case_studies_df = real_world_cyber_attack_case_studies()
    cyber_detection_df = cyber_threat_simulation_on_data_feeds(df, splits, models)
    cybersecurity_resilience_scorecard(robustness_df, stress_df, adversarial_df, shap_stability_df, drift_summary, cyber_detection_df)
    final_outputs(df, metrics_df, robustness_df, stress_df, shap_stability_df, event_detector_metrics_df, ttd_summary, proxy_metrics_df, adversarial_df, validation_refinement_df, drift_summary, telemetry_gap_df, fusion_metrics_df, defence_df)
    print("\n" + "=" * 70)
    print(f"Pipeline completed successfully in {time.time() - start:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        raise
