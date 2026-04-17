"""
Step 19: Framework Extensions and Missing Evidence

Adds the dissertation framework pieces that were not present in the original
notebook pipeline:
- validation-tuned adaptive thresholds
- majority-vote ensemble
- supervised early-warning classifier
- ROC curves and confusion-matrix figures
- proper prediction runtime benchmarks
- robustness degradation curve
- SHAP outputs for Autoencoder and One-Class SVM
- SHAP stability for multiple assets
- stress scenario detection timeline
- McNemar pairwise model comparison

Run: python scripts/step_19_framework_extensions.py
"""

import os
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

warnings.filterwarnings("ignore")

from scipy.stats import binomtest
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    jaccard_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model

np.random.seed(42)
plt.style.use("default")
sns.set_theme(style="whitegrid")

project_root = Path.cwd()
base_path = project_root / "dissertation_outputs"
data_dir = base_path / "data"
model_dir = base_path / "models"
figure_dir = base_path / "figures"
result_dir = base_path / "results"

for folder in [data_dir, model_dir, figure_dir, result_dir]:
    folder.mkdir(parents=True, exist_ok=True)

feature_cols = [
    "Return",
    "Volatility_10",
    "Volatility_20",
    "Momentum_10",
    "MA_Ratio_10",
    "MA_Ratio_20",
    "Volume_Change",
    "Rolling_Skew_20",
    "Rolling_Kurt_20",
    "Return_Zscore_20",
    "Drawdown",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_Width",
]


def load_required_artifacts():
    data_path = data_dir / "evaluation_dataset_with_injected_events.csv"
    if not data_path.exists():
        print(f"ERROR: Missing {data_path}. Run steps 1-18 first.")
        sys.exit(1)

    frame = pd.read_csv(data_path, parse_dates=["Date"])
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    frame[feature_cols] = frame[feature_cols].fillna(frame[feature_cols].median(numeric_only=True))

    artifacts = {
        "if": joblib.load(model_dir / "isolation_forest_train_only.pkl"),
        "scaler": joblib.load(model_dir / "standard_scaler_train_only.pkl"),
        "ocsvm": joblib.load(model_dir / "one_class_svm_train_only.pkl"),
        "ae": load_model(model_dir / "multi_asset_autoencoder_train_only.h5", compile=False),
    }
    return frame, artifacts


def as_feature_frame(values, index=None):
    return pd.DataFrame(values, columns=feature_cols, index=index)


def score_if(model, X):
    return -model.decision_function(X)


def score_ae(model, scaler, X):
    scaled = scaler.transform(X)
    recon = model.predict(scaled, verbose=0)
    return np.mean(np.square(scaled - recon), axis=1)


def score_ocsvm(model, scaler, X):
    scaled = scaler.transform(X)
    return -model.decision_function(scaled)


def tune_threshold(y_true, scores, model_name):
    quantiles = np.linspace(0.01, 0.99, 199)
    candidates = np.unique(np.quantile(scores, quantiles))
    best = {"Threshold": candidates[0], "Precision": 0.0, "Recall": 0.0, "F1": -1.0}

    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best["F1"]:
            best = {
                "Model": model_name,
                "Threshold": float(threshold),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
            }
    return best


def metric_row(model_name, split_name, y_true, y_pred, score=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "Model": model_name,
        "Split": split_name,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Predicted_Anomalies": int(y_pred.sum()),
        "True_Anomalies": int(y_true.sum()),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        row["ROC_AUC"] = roc_auc_score(y_true, score)
    else:
        row["ROC_AUC"] = np.nan
    return row


def minmax_from_validation(values, val_values):
    lo = float(np.min(val_values))
    hi = float(np.max(val_values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def future_event_label(group, horizon=3):
    labels = group["Evaluation_Label"].astype(int).to_numpy()
    future = np.zeros_like(labels)
    for offset in range(horizon + 1):
        shifted = np.r_[labels[offset:], np.zeros(offset, dtype=int)]
        future = np.maximum(future, shifted)
    return pd.Series(future, index=group.index)


def plot_confusion_matrices(metrics_df, output_path):
    plot_df = metrics_df[metrics_df["Split"] == "Test"].copy()
    n = len(plot_df)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.7))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, plot_df.iterrows()):
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
        )
        ax.set_title(row["Model"])
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Test Confusion Matrices", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(roc_inputs, output_path):
    plt.figure(figsize=(8, 6))
    for model_name, y_true, score in roc_inputs:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Test Split")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_adaptive_and_ensemble_outputs(df, artifacts):
    X_val = df.loc[df["Split"] == "Validation", feature_cols].copy()
    X_test = df.loc[df["Split"] == "Test", feature_cols].copy()
    y_val = df.loc[df["Split"] == "Validation", "Evaluation_Label"].astype(int).values
    y_test = df.loc[df["Split"] == "Test", "Evaluation_Label"].astype(int).values

    scores = {
        "Isolation Forest": {
            "Validation": score_if(artifacts["if"], X_val),
            "Test": score_if(artifacts["if"], X_test),
        },
        "Autoencoder": {
            "Validation": score_ae(artifacts["ae"], artifacts["scaler"], X_val),
            "Test": score_ae(artifacts["ae"], artifacts["scaler"], X_test),
        },
        "One-Class SVM": {
            "Validation": score_ocsvm(artifacts["ocsvm"], artifacts["scaler"], X_val),
            "Test": score_ocsvm(artifacts["ocsvm"], artifacts["scaler"], X_test),
        },
    }

    thresholds = [tune_threshold(y_val, values["Validation"], name) for name, values in scores.items()]
    thresholds_df = pd.DataFrame(thresholds)

    predictions = df[["Date", "Ticker", "Split", "Evaluation_Label", "Evaluation_Event_ID", "Evaluation_Scenario"]].copy()
    predictions["Pred_IF_Adaptive"] = 0
    predictions["Pred_AE_Adaptive"] = 0
    predictions["Pred_OCSVM_Adaptive"] = 0
    predictions["Ensemble_MajorityVote"] = 0
    predictions["Ensemble_Score"] = np.nan

    pred_cols = {
        "Isolation Forest": "Pred_IF_Adaptive",
        "Autoencoder": "Pred_AE_Adaptive",
        "One-Class SVM": "Pred_OCSVM_Adaptive",
    }

    metrics = []
    roc_inputs = []
    for split_name, X_split, y_split in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        split_mask = df["Split"] == split_name
        vote_matrix = []
        normalized_scores = []
        for _, threshold_row in thresholds_df.iterrows():
            model_name = threshold_row["Model"]
            model_scores = scores[model_name][split_name]
            pred = (model_scores >= threshold_row["Threshold"]).astype(int)
            predictions.loc[split_mask, pred_cols[model_name]] = pred
            vote_matrix.append(pred)
            normalized_scores.append(minmax_from_validation(model_scores, scores[model_name]["Validation"]))
            metrics.append(metric_row(f"{model_name} Adaptive", split_name, y_split, pred, model_scores))
            if split_name == "Test":
                roc_inputs.append((f"{model_name} Adaptive", y_split, model_scores))

        vote_matrix = np.vstack(vote_matrix)
        ensemble_pred = (vote_matrix.sum(axis=0) >= 2).astype(int)
        ensemble_score = np.mean(np.vstack(normalized_scores), axis=0)
        predictions.loc[split_mask, "Ensemble_MajorityVote"] = ensemble_pred
        predictions.loc[split_mask, "Ensemble_Score"] = ensemble_score
        metrics.append(metric_row("Ensemble Majority Vote", split_name, y_split, ensemble_pred, ensemble_score))
        if split_name == "Test":
            roc_inputs.append(("Ensemble Mean Score", y_split, ensemble_score))

    metrics_df = pd.DataFrame(metrics)
    thresholds_df.to_csv(result_dir / "adaptive_thresholds_validation_tuned.csv", index=False)
    predictions.to_csv(result_dir / "adaptive_ensemble_predictions.csv", index=False)
    metrics_df.to_csv(result_dir / "adaptive_ensemble_metrics.csv", index=False)

    return predictions, metrics_df, thresholds_df, roc_inputs


def build_supervised_early_warning(df, adaptive_predictions):
    enriched = df.merge(
        adaptive_predictions[
            [
                "Date",
                "Ticker",
                "Split",
                "Pred_IF_Adaptive",
                "Pred_AE_Adaptive",
                "Pred_OCSVM_Adaptive",
                "Ensemble_MajorityVote",
                "Ensemble_Score",
            ]
        ],
        on=["Date", "Ticker", "Split"],
        how="left",
    )
    enriched = enriched.sort_values(["Ticker", "Date"]).copy()
    enriched["Early_Warning_Label"] = (
        enriched.groupby("Ticker", group_keys=False).apply(future_event_label).astype(int)
    )

    model_features = feature_cols + [
        "Pred_IF_Adaptive",
        "Pred_AE_Adaptive",
        "Pred_OCSVM_Adaptive",
        "Ensemble_MajorityVote",
        "Ensemble_Score",
    ]
    train_df = enriched[enriched["Split"] == "Validation"].copy()
    test_df = enriched[enriched["Split"] == "Test"].copy()
    X_train = train_df[model_features].fillna(0)
    y_train = train_df["Early_Warning_Label"].astype(int)
    X_test = test_df[model_features].fillna(0)
    y_test = test_df["Early_Warning_Label"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_train, y_train)
    val_prob = clf.predict_proba(X_train)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]
    threshold = tune_threshold(y_train.values, val_prob, "Supervised Early Warning")["Threshold"]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = pd.DataFrame(
        [
            metric_row(
                "Supervised Early Warning",
                "Test",
                y_test.values,
                test_pred,
                test_prob,
            )
        ]
    )
    metrics["Threshold"] = threshold
    metrics["Training_Split"] = "Validation"
    metrics["Warning_Horizon_Days"] = 3

    out = test_df[["Date", "Ticker", "Evaluation_Label", "Early_Warning_Label"]].copy()
    out["Early_Warning_Probability"] = test_prob
    out["Early_Warning_Prediction"] = test_pred

    metrics.to_csv(result_dir / "supervised_early_warning_metrics.csv", index=False)
    out.to_csv(result_dir / "supervised_early_warning_predictions.csv", index=False)
    joblib.dump(clf, model_dir / "supervised_early_warning_random_forest.pkl")

    feature_importance = (
        pd.DataFrame({"Feature": model_features, "Importance": clf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(result_dir / "supervised_early_warning_feature_importance.csv", index=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_importance.head(12), x="Importance", y="Feature", color="steelblue")
    plt.title("Supervised Early-Warning Feature Importance")
    plt.tight_layout()
    plt.savefig(figure_dir / "supervised_early_warning_feature_importance.png", dpi=300)
    plt.close()
    return metrics, y_test.values, test_prob


def benchmark_prediction_runtime(df, artifacts, n_runs=7):
    X_test = df.loc[df["Split"] == "Test", feature_cols].copy()
    benchmark_fns = {
        "Isolation Forest": lambda: artifacts["if"].predict(X_test),
        "Autoencoder": lambda: artifacts["ae"].predict(artifacts["scaler"].transform(X_test), verbose=0),
        "One-Class SVM": lambda: artifacts["ocsvm"].predict(artifacts["scaler"].transform(X_test)),
    }
    rows = []
    for model_name, fn in benchmark_fns.items():
        fn()
        timings = []
        for _ in range(n_runs):
            start = time.perf_counter()
            fn()
            timings.append(time.perf_counter() - start)
        rows.append(
            {
                "Model": model_name,
                "Benchmark_Type": "timeit_repeat_prediction_on_test_split",
                "Rows": len(X_test),
                "Runs": n_runs,
                "Mean_Seconds": float(np.mean(timings)),
                "Std_Seconds": float(np.std(timings, ddof=1)),
                "Min_Seconds": float(np.min(timings)),
                "Max_Seconds": float(np.max(timings)),
            }
        )
    runtime_df = pd.DataFrame(rows)
    runtime_df.to_csv(result_dir / "runtime_benchmark_prediction.csv", index=False)
    return runtime_df


def plot_robustness_degradation():
    path = result_dir / "perturbation_robustness_results_standardized.csv"
    if not path.exists():
        return None
    robust = pd.read_csv(path)
    plot_df = pd.DataFrame(
        {
            "Perturbation_Level": np.tile(robust["Perturbation_Level"], 3),
            "Model": np.repeat(["Isolation Forest", "Autoencoder", "One-Class SVM"], len(robust)),
            "Degradation": np.r_[
                1 - robust["IF_Robustness_Score"],
                1 - robust["AE_Robustness_Score"],
                1 - robust["OCSVM_Robustness_Score"],
            ],
        }
    )
    out_csv = result_dir / "robustness_degradation_curve.csv"
    out_png = figure_dir / "robustness_degradation_curve.png"
    plot_df.to_csv(out_csv, index=False)

    plt.figure(figsize=(9, 6))
    sns.lineplot(data=plot_df, x="Perturbation_Level", y="Degradation", hue="Model", marker="o")
    plt.title("Robustness Degradation Under Gaussian Perturbation")
    plt.xlabel("Perturbation Level")
    plt.ylabel("Label-Flip Rate")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_png


def predict_adaptive_ensemble(X, artifacts, thresholds_df):
    if_scores = score_if(artifacts["if"], X)
    ae_scores = score_ae(artifacts["ae"], artifacts["scaler"], X)
    oc_scores = score_ocsvm(artifacts["ocsvm"], artifacts["scaler"], X)
    threshold_map = thresholds_df.set_index("Model")["Threshold"].to_dict()
    pred_if = (if_scores >= threshold_map["Isolation Forest"]).astype(int)
    pred_ae = (ae_scores >= threshold_map["Autoencoder"]).astype(int)
    pred_oc = (oc_scores >= threshold_map["One-Class SVM"]).astype(int)
    ensemble_pred = ((pred_if + pred_ae + pred_oc) >= 2).astype(int)
    ensemble_score = np.mean(
        np.vstack(
            [
                minmax_from_validation(if_scores, if_scores),
                minmax_from_validation(ae_scores, ae_scores),
                minmax_from_validation(oc_scores, oc_scores),
            ]
        ),
        axis=0,
    )
    return ensemble_pred, ensemble_score


def build_ensemble_gaussian_robustness(df, artifacts, thresholds_df, adaptive_predictions):
    X_test = df.loc[df["Split"] == "Test", feature_cols].copy()
    baseline = adaptive_predictions.loc[
        adaptive_predictions["Split"] == "Test", "Ensemble_MajorityVote"
    ].astype(int).values
    rows = []
    for level in [0.05, 0.10, 0.20, 0.30]:
        noise = np.random.normal(0, level, X_test.shape)
        X_perturbed = pd.DataFrame(X_test.values + noise, columns=feature_cols, index=X_test.index)
        pred, _ = predict_adaptive_ensemble(X_perturbed, artifacts, thresholds_df)
        flips = int(np.sum(baseline != pred))
        rows.append(
            {
                "Perturbation_Level": level,
                "Ensemble_Label_Flips": flips,
                "Ensemble_Jaccard": jaccard_score(baseline, pred, zero_division=0),
                "Ensemble_Robustness_Score": 1 - (flips / len(baseline)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "ensemble_gaussian_robustness_results.csv", index=False)

    existing_path = result_dir / "perturbation_robustness_results_standardized.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        combined = pd.DataFrame(
            {
                "Perturbation_Level": np.r_[
                    existing["Perturbation_Level"],
                    existing["Perturbation_Level"],
                    existing["Perturbation_Level"],
                    out["Perturbation_Level"],
                ],
                "Model": np.r_[
                    np.repeat("Isolation Forest", len(existing)),
                    np.repeat("Autoencoder", len(existing)),
                    np.repeat("One-Class SVM", len(existing)),
                    np.repeat("Ensemble Majority Vote", len(out)),
                ],
                "Jaccard": np.r_[
                    existing["IF_Jaccard"],
                    existing["AE_Jaccard"],
                    existing["OCSVM_Jaccard"],
                    out["Ensemble_Jaccard"],
                ],
            }
        )
        combined.to_csv(result_dir / "jaccard_vs_noise_with_ensemble.csv", index=False)
        plt.figure(figsize=(9, 6))
        sns.lineplot(data=combined, x="Perturbation_Level", y="Jaccard", hue="Model", marker="o")
        plt.title("Jaccard Robustness vs Gaussian Noise")
        plt.xlabel("Perturbation Level")
        plt.ylabel("Jaccard Similarity")
        plt.tight_layout()
        plt.savefig(figure_dir / "jaccard_vs_noise_with_ensemble.png", dpi=300)
        plt.close()
    return out


def build_ensemble_stress_scenarios(df, artifacts, thresholds_df, adaptive_predictions):
    X_test = df.loc[df["Split"] == "Test", feature_cols].copy()
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

    baseline = adaptive_predictions.loc[
        adaptive_predictions["Split"] == "Test", "Ensemble_MajorityVote"
    ].astype(int).values
    rows = []
    for scenario_name, scenario_X in scenarios.items():
        pred, _ = predict_adaptive_ensemble(scenario_X, artifacts, thresholds_df)
        flips = int(np.sum(baseline != pred))
        rows.append(
            {
                "Scenario": scenario_name,
                "Model": "Ensemble Majority Vote",
                "Label_Flips": flips,
                "Jaccard": jaccard_score(baseline, pred, zero_division=0),
                "Robustness": 1 - (flips / len(baseline)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "ensemble_stress_scenarios_results.csv", index=False)

    existing_path = result_dir / "financial_stress_scenarios_results_standardized.csv"
    if existing_path.exists():
        combined = pd.concat([pd.read_csv(existing_path), out], ignore_index=True)
        combined.to_csv(result_dir / "stress_scenarios_results_with_ensemble.csv", index=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=combined, x="Scenario", y="Robustness", hue="Model")
        plt.title("Stress Scenario Robustness with Ensemble")
        plt.xlabel("Scenario")
        plt.ylabel("Robustness Score")
        plt.tight_layout()
        plt.savefig(figure_dir / "stress_scenario_robustness_with_ensemble.png", dpi=300)
        plt.close()
    return out


def build_chapter4_eda_figures(df):
    returns = df[["Ticker", "Date", "Return", "Volatility_20"]].dropna().copy()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=returns, x="Ticker", y="Return")
    plt.title("Daily Return Distribution by Asset")
    plt.xlabel("Asset")
    plt.ylabel("Daily Return")
    plt.tight_layout()
    plt.savefig(figure_dir / "daily_returns_boxplot_by_asset.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    aapl = returns[returns["Ticker"] == "AAPL"].copy()
    plt.plot(aapl["Date"], aapl["Volatility_20"], color="steelblue", linewidth=1.3)
    plt.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"), color="crimson", alpha=0.16)
    max_row = aapl.loc[aapl["Volatility_20"].idxmax()]
    plt.scatter([max_row["Date"]], [max_row["Volatility_20"]], color="crimson", zorder=3)
    plt.title("AAPL 20-Day Rolling Volatility with COVID-19 Spike Highlight")
    plt.xlabel("Date")
    plt.ylabel("20-Day Rolling Volatility")
    plt.tight_layout()
    plt.savefig(figure_dir / "rolling_volatility_aapl_covid_highlight.png", dpi=300)
    plt.close()


def build_chapter4_model_figures(adaptive_metrics):
    baseline_path = result_dir / "controlled_injection_metrics.csv"
    if not baseline_path.exists():
        return
    baseline = pd.read_csv(baseline_path)
    baseline_test = baseline[baseline["Split"] == "Test"].copy()
    ensemble_test = adaptive_metrics[
        (adaptive_metrics["Split"] == "Test") & (adaptive_metrics["Model"] == "Ensemble Majority Vote")
    ].copy()
    performance = pd.concat(
        [
            baseline_test[["Model", "Precision", "Recall", "F1_Score", "TN", "FP", "FN", "TP"]],
            ensemble_test[["Model", "Precision", "Recall", "F1_Score", "TN", "FP", "FN", "TP"]],
        ],
        ignore_index=True,
    )
    performance.to_csv(result_dir / "chapter4_model_performance_with_ensemble.csv", index=False)

    long_perf = performance.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1_Score"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=long_perf, x="Model", y="Score", hue="Metric")
    plt.title("Precision, Recall, and F1 by Model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "model_precision_recall_f1_with_ensemble.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    before_after = performance[performance["Model"].isin(["Isolation Forest", "Ensemble Majority Vote"])]
    before_after_long = before_after.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1_Score"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(data=before_after_long, x="Metric", y="Score", hue="Model")
    plt.title("Before vs After: Baseline IF vs Ensemble")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(figure_dir / "before_after_if_vs_ensemble.png", dpi=300)
    plt.close()

    plot_confusion_matrices(
        performance.assign(Split="Test"),
        figure_dir / "confusion_matrices_baseline_models_plus_ensemble.png",
    )


def shap_for_model_outputs(df, artifacts, max_rows=80, background_rows=40):
    test = df[df["Split"] == "Test"].copy()
    X_sample = test[feature_cols].sample(min(max_rows, len(test)), random_state=42)
    background = test[feature_cols].sample(min(background_rows, len(test)), random_state=7)

    def ae_score_fn(values):
        frame = as_feature_frame(values)
        return score_ae(artifacts["ae"], artifacts["scaler"], frame)

    def ocsvm_score_fn(values):
        frame = as_feature_frame(values)
        return score_ocsvm(artifacts["ocsvm"], artifacts["scaler"], frame)

    specs = [
        ("Autoencoder", shap.KernelExplainer(ae_score_fn, background), X_sample),
        ("One-Class SVM", shap.KernelExplainer(ocsvm_score_fn, background), X_sample),
    ]
    all_importance = []
    for model_name, explainer, sample in specs:
        values = explainer.shap_values(sample, nsamples=100)
        values = np.asarray(values)
        importance = (
            pd.DataFrame({"Feature": feature_cols, "Mean_Abs_SHAP": np.abs(values).mean(axis=0)})
            .sort_values("Mean_Abs_SHAP", ascending=False)
            .assign(Model=model_name)
        )
        importance.to_csv(
            result_dir / f"shap_feature_importance_{model_name.lower().replace(' ', '_').replace('-', '')}.csv",
            index=False,
        )
        all_importance.append(importance)

        shap.summary_plot(values, sample, show=False)
        plt.tight_layout()
        fig_path = figure_dir / f"shap_summary_{model_name.lower().replace(' ', '_').replace('-', '')}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

    combined = pd.concat(all_importance, ignore_index=True)
    combined.to_csv(result_dir / "shap_feature_importance_autoencoder_ocsvm.csv", index=False)
    return combined


def shap_stability_multi_asset(df, artifacts, tickers=("AAPL", "MSFT", "NVDA"), levels=(0.05, 0.10, 0.20, 0.30)):
    rows = []
    top_k = 5
    for ticker in tickers:
        ticker_df = df[(df["Split"] == "Test") & (df["Ticker"] == ticker)].copy()
        if ticker_df.empty:
            continue
        X_sample = ticker_df[feature_cols].sample(min(120, len(ticker_df)), random_state=42)
        explainer = shap.Explainer(artifacts["if"], X_sample)
        base_values = explainer(X_sample)
        base_importance = pd.Series(np.abs(base_values.values).mean(axis=0), index=feature_cols)
        base_rank = base_importance.rank(ascending=False)
        base_top = set(base_importance.sort_values(ascending=False).head(top_k).index)

        for level in levels:
            noise = np.random.normal(0, level, X_sample.shape)
            perturbed = pd.DataFrame(X_sample.values + noise, columns=feature_cols, index=X_sample.index)
            pert_values = explainer(perturbed)
            pert_importance = pd.Series(np.abs(pert_values.values).mean(axis=0), index=feature_cols)
            rho, pval = spearmanr(base_rank, pert_importance.rank(ascending=False))
            pert_top = set(pert_importance.sort_values(ascending=False).head(top_k).index)
            rows.append(
                {
                    "Ticker": ticker,
                    "Perturbation_Level": level,
                    "Spearman_Rho": rho,
                    "Spearman_pvalue": pval,
                    "Top_5_Overlap": len(base_top.intersection(pert_top)) / top_k,
                }
            )

    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(result_dir / "shap_stability_multi_asset.csv", index=False)
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=stability_df, x="Perturbation_Level", y="Spearman_Rho", hue="Ticker", marker="o")
    plt.title("Multi-Asset SHAP Rank Stability")
    plt.xlabel("Perturbation Level")
    plt.ylabel("Spearman Rank Correlation")
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_stability_multi_asset.png", dpi=300)
    plt.close()
    return stability_df


def shap_stability_autoencoder_ocsvm(df, artifacts, max_rows=35, background_rows=25):
    test = df[df["Split"] == "Test"].copy()
    X_sample = test[feature_cols].sample(min(max_rows, len(test)), random_state=123)
    background = test[feature_cols].sample(min(background_rows, len(test)), random_state=321)
    levels = [0.05, 0.10, 0.20, 0.30]
    top_k = 5

    def ae_score_fn(values):
        return score_ae(artifacts["ae"], artifacts["scaler"], as_feature_frame(values))

    def ocsvm_score_fn(values):
        return score_ocsvm(artifacts["ocsvm"], artifacts["scaler"], as_feature_frame(values))

    rows = []
    for model_name, score_fn in [
        ("Autoencoder", ae_score_fn),
        ("One-Class SVM", ocsvm_score_fn),
    ]:
        explainer = shap.KernelExplainer(score_fn, background)
        base_values = np.asarray(explainer.shap_values(X_sample, nsamples=80))
        base_importance = pd.Series(np.abs(base_values).mean(axis=0), index=feature_cols)
        base_rank = base_importance.rank(ascending=False)
        base_top = set(base_importance.sort_values(ascending=False).head(top_k).index)

        for level in levels:
            noise = np.random.normal(0, level, X_sample.shape)
            perturbed = pd.DataFrame(X_sample.values + noise, columns=feature_cols, index=X_sample.index)
            pert_values = np.asarray(explainer.shap_values(perturbed, nsamples=80))
            pert_importance = pd.Series(np.abs(pert_values).mean(axis=0), index=feature_cols)
            rho, pval = spearmanr(base_rank, pert_importance.rank(ascending=False))
            pert_top = set(pert_importance.sort_values(ascending=False).head(top_k).index)
            rows.append(
                {
                    "Model": model_name,
                    "Perturbation_Level": level,
                    "Spearman_Rho": rho,
                    "Spearman_pvalue": pval,
                    "Top_5_Overlap": len(base_top.intersection(pert_top)) / top_k,
                }
            )

    stability = pd.DataFrame(rows)
    stability.to_csv(result_dir / "shap_stability_autoencoder_ocsvm.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=stability, x="Perturbation_Level", y="Spearman_Rho", hue="Model", marker="o")
    plt.title("Autoencoder and OC-SVM SHAP Stability")
    plt.xlabel("Perturbation Level")
    plt.ylabel("Spearman Rank Correlation")
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_stability_autoencoder_ocsvm.png", dpi=300)
    plt.close()
    return stability


def shap_rank_shift_heatmap(df, artifacts):
    test = df[df["Split"] == "Test"].copy()
    X_sample = test[feature_cols].sample(min(140, len(test)), random_state=456)
    explainer = shap.Explainer(artifacts["if"], X_sample)
    base_values = explainer(X_sample)
    base_importance = pd.Series(np.abs(base_values.values).mean(axis=0), index=feature_cols)
    top_features = base_importance.sort_values(ascending=False).head(10).index.tolist()
    rank_table = pd.DataFrame(index=top_features)
    rank_table["0%"] = base_importance.rank(ascending=False).loc[top_features]

    for level in [0.05, 0.10, 0.20, 0.30]:
        noise = np.random.normal(0, level, X_sample.shape)
        perturbed = pd.DataFrame(X_sample.values + noise, columns=feature_cols, index=X_sample.index)
        pert_values = explainer(perturbed)
        pert_importance = pd.Series(np.abs(pert_values.values).mean(axis=0), index=feature_cols)
        rank_table[f"{int(level * 100)}%"] = pert_importance.rank(ascending=False).loc[top_features]

    rank_table.to_csv(result_dir / "shap_rank_shift_heatmap_data.csv")
    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_table, annot=True, fmt=".0f", cmap="viridis_r", cbar_kws={"label": "Feature Rank"})
    plt.title("SHAP Feature Rank Shifts Across Noise Levels")
    plt.xlabel("Noise Level")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_rank_shift_heatmap.png", dpi=300)
    plt.close()
    return rank_table


def plot_stress_detection_timeline(df, adaptive_predictions):
    merged = df.merge(
        adaptive_predictions[["Date", "Ticker", "Split", "Ensemble_MajorityVote"]],
        on=["Date", "Ticker", "Split"],
        how="left",
    )
    focus = merged[
        (merged["Split"] == "Test")
        & (merged["Ticker"] == "AAPL")
        & (
            (merged["Evaluation_Label"] == 1)
            | (merged["Date"].between(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")))
        )
    ].copy()
    if focus.empty:
        focus = merged[(merged["Split"] == "Test") & (merged["Ticker"] == "AAPL")].copy()

    plt.figure(figsize=(12, 5))
    plt.plot(focus["Date"], focus["Close"], color="steelblue", linewidth=1.4, label="AAPL Close")
    detections = focus[focus["Ensemble_MajorityVote"] == 1]
    injected = focus[focus["Evaluation_Label"] == 1]
    plt.scatter(detections["Date"], detections["Close"], color="darkorange", s=28, label="Ensemble Detection")
    plt.scatter(injected["Date"], injected["Close"], color="crimson", s=48, marker="x", label="Injected Stress Row")
    plt.title("AAPL Stress Scenario Detection Timeline")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path = figure_dir / "stress_scenario_detection_timeline_aapl.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def mcnemar_tests(predictions):
    test = predictions[predictions["Split"] == "Test"].copy()
    y = test["Evaluation_Label"].astype(int).values
    cols = {
        "Isolation Forest Adaptive": "Pred_IF_Adaptive",
        "Autoencoder Adaptive": "Pred_AE_Adaptive",
        "One-Class SVM Adaptive": "Pred_OCSVM_Adaptive",
        "Ensemble Majority Vote": "Ensemble_MajorityVote",
    }
    rows = []
    names = list(cols.keys())
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_correct = test[cols[left]].astype(int).values == y
            right_correct = test[cols[right]].astype(int).values == y
            b = int(np.sum(left_correct & ~right_correct))
            c = int(np.sum(~left_correct & right_correct))
            n = b + c
            pvalue = binomtest(min(b, c), n=n, p=0.5).pvalue if n > 0 else np.nan
            rows.append(
                {
                    "Model_A": left,
                    "Model_B": right,
                    "A_correct_B_wrong": b,
                    "A_wrong_B_correct": c,
                    "Discordant_Total": n,
                    "Exact_McNemar_pvalue": pvalue,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "mcnemar_pairwise_model_comparison.csv", index=False)
    return out


def main():
    print("\n============================================================")
    print("Step 19: Framework Extensions and Missing Evidence")
    print("============================================================\n")

    df, artifacts = load_required_artifacts()

    adaptive_predictions, adaptive_metrics, thresholds, roc_inputs = build_adaptive_and_ensemble_outputs(df, artifacts)
    supervised_metrics, supervised_y, supervised_score = build_supervised_early_warning(df, adaptive_predictions)
    runtime_df = benchmark_prediction_runtime(df, artifacts)
    mcnemar_df = mcnemar_tests(adaptive_predictions)
    ensemble_gaussian = build_ensemble_gaussian_robustness(df, artifacts, thresholds, adaptive_predictions)
    ensemble_stress = build_ensemble_stress_scenarios(df, artifacts, thresholds, adaptive_predictions)

    combined_metrics = pd.concat([adaptive_metrics, supervised_metrics], ignore_index=True, sort=False)
    combined_metrics.to_csv(result_dir / "framework_extension_metrics.csv", index=False)

    build_chapter4_eda_figures(df)
    build_chapter4_model_figures(adaptive_metrics)
    roc_inputs.append(("Supervised Early Warning", supervised_y, supervised_score))
    plot_roc_curves(roc_inputs, figure_dir / "roc_curves_framework_models.png")
    plot_confusion_matrices(combined_metrics, figure_dir / "confusion_matrices_framework_models.png")
    degradation_path = plot_robustness_degradation()
    shap_importance = shap_for_model_outputs(df, artifacts)
    multi_asset_stability = shap_stability_multi_asset(df, artifacts)
    ae_ocsvm_stability = shap_stability_autoencoder_ocsvm(df, artifacts)
    shap_rank_table = shap_rank_shift_heatmap(df, artifacts)
    stress_timeline_path = plot_stress_detection_timeline(df, adaptive_predictions)

    summary = pd.DataFrame(
        [
            {"Artifact": "Adaptive thresholds", "Path": str(result_dir / "adaptive_thresholds_validation_tuned.csv")},
            {"Artifact": "Adaptive ensemble metrics", "Path": str(result_dir / "adaptive_ensemble_metrics.csv")},
            {"Artifact": "Supervised early warning metrics", "Path": str(result_dir / "supervised_early_warning_metrics.csv")},
            {"Artifact": "ROC curves", "Path": str(figure_dir / "roc_curves_framework_models.png")},
            {"Artifact": "Confusion matrix figure", "Path": str(figure_dir / "confusion_matrices_framework_models.png")},
            {"Artifact": "Runtime benchmark", "Path": str(result_dir / "runtime_benchmark_prediction.csv")},
            {"Artifact": "Ensemble Gaussian robustness", "Path": str(result_dir / "ensemble_gaussian_robustness_results.csv")},
            {"Artifact": "Ensemble stress scenarios", "Path": str(result_dir / "ensemble_stress_scenarios_results.csv")},
            {"Artifact": "Chapter 4 model performance figure", "Path": str(figure_dir / "model_precision_recall_f1_with_ensemble.png")},
            {"Artifact": "Before/after IF vs ensemble", "Path": str(figure_dir / "before_after_if_vs_ensemble.png")},
            {"Artifact": "Daily returns boxplot", "Path": str(figure_dir / "daily_returns_boxplot_by_asset.png")},
            {"Artifact": "COVID volatility highlight", "Path": str(figure_dir / "rolling_volatility_aapl_covid_highlight.png")},
            {"Artifact": "Robustness degradation curve", "Path": str(degradation_path)},
            {"Artifact": "Jaccard robustness with ensemble", "Path": str(figure_dir / "jaccard_vs_noise_with_ensemble.png")},
            {"Artifact": "Stress robustness with ensemble", "Path": str(figure_dir / "stress_scenario_robustness_with_ensemble.png")},
            {"Artifact": "Autoencoder and OC-SVM SHAP", "Path": str(result_dir / "shap_feature_importance_autoencoder_ocsvm.csv")},
            {"Artifact": "Autoencoder and OC-SVM SHAP stability", "Path": str(result_dir / "shap_stability_autoencoder_ocsvm.csv")},
            {"Artifact": "SHAP rank shift heatmap", "Path": str(figure_dir / "shap_rank_shift_heatmap.png")},
            {"Artifact": "Multi-asset SHAP stability", "Path": str(result_dir / "shap_stability_multi_asset.csv")},
            {"Artifact": "Stress detection timeline", "Path": str(stress_timeline_path)},
            {"Artifact": "McNemar tests", "Path": str(result_dir / "mcnemar_pairwise_model_comparison.csv")},
        ]
    )
    summary.to_csv(result_dir / "framework_extension_artifact_manifest.csv", index=False)

    print("Adaptive thresholds:")
    print(thresholds.to_string(index=False))
    print("\nFramework metrics:")
    print(combined_metrics[["Model", "Split", "Precision", "Recall", "F1_Score", "ROC_AUC"]].to_string(index=False))
    print("\nRuntime benchmark:")
    print(runtime_df.to_string(index=False))
    print("\nEnsemble Gaussian robustness:")
    print(ensemble_gaussian.to_string(index=False))
    print("\nEnsemble stress scenarios:")
    print(ensemble_stress.to_string(index=False))
    print("\nAutoencoder / OC-SVM SHAP stability:")
    print(ae_ocsvm_stability.to_string(index=False))
    print("\nMcNemar comparison:")
    print(mcnemar_df.to_string(index=False))
    print("\nStep 19 completed successfully!")


if __name__ == "__main__":
    main()
