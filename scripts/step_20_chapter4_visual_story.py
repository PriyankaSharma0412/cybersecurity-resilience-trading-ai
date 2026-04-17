"""
Step 20: Chapter 4 Visual Storytelling Layer

Creates granular, comparative figures and companion tables that make the
methodology and findings easier to explain in Chapter 4. This step reads the
existing result CSVs and produces export-ready PNGs in dissertation_outputs.

Run: python scripts/step_20_chapter4_visual_story.py
"""

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import seaborn as sns

plt.style.use("default")
sns.set_theme(style="whitegrid")

project_root = Path.cwd()
base_path = project_root / "dissertation_outputs"
result_dir = base_path / "results"
figure_dir = base_path / "figures"
chapter4_dir = figure_dir / "chapter4_visuals"
chapter4_dir.mkdir(parents=True, exist_ok=True)


def require_csv(filename):
    path = result_dir / filename
    if not path.exists():
        print(f"ERROR: Missing {path}. Run steps 1-19 first.")
        sys.exit(1)
    return pd.read_csv(path)


def save_current(name):
    path = chapter4_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")
    return path


def annotate_bars(ax, fmt="{:.2f}", offset=0.01):
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if np.isnan(value):
                labels.append("")
            elif abs(value) >= 10:
                labels.append(f"{value:.0f}")
            else:
                labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def visual_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    boxes = [
        ("Market Data\n10 Assets", 0.02, "#dceeff"),
        ("Feature Engineering\nReturns, Volatility,\nMomentum, Technicals", 0.20, "#e7f5df"),
        ("Chronological Split\nTrain / Validation / Test", 0.40, "#fff2cc"),
        ("Base Detectors\nIF, Autoencoder,\nOC-SVM", 0.60, "#f8dfdf"),
        ("Improved Framework\nAdaptive Thresholds +\nMajority Vote", 0.78, "#eadcf8"),
    ]
    y = 0.42
    w = 0.16
    h = 0.34
    for text, x, color in boxes:
        patch = Rectangle((x, y), w, h, linewidth=1.3, edgecolor="#444", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + w + 0.01
        x1 = boxes[i + 1][1] - 0.01
        arrow = FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2), arrowstyle="->", mutation_scale=16, linewidth=1.5)
        ax.add_patch(arrow)
    ax.text(
        0.5,
        0.14,
        "Evaluation outputs: performance metrics, ROC-AUC, confusion matrices, robustness curves, stress scenarios, SHAP explanations, runtime benchmarks",
        ha="center",
        fontsize=10,
    )
    ax.set_title("Financial AI Robustness Evaluation Pipeline", fontsize=15, weight="bold", pad=14)
    save_current("01_framework_pipeline_diagram.png")


def model_metric_heatmap(performance):
    heatmap_df = performance.set_index("Model")[["Precision", "Recall", "F1_Score"]]
    plt.figure(figsize=(8.8, 4.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Score"})
    plt.title("Model Performance Heatmap: Precision, Recall, F1")
    plt.xlabel("Metric")
    plt.ylabel("Model")
    save_current("02_model_performance_metric_heatmap.png")


def precision_recall_tradeoff(performance):
    plot_df = performance.copy()
    plot_df["False_Positives"] = plot_df["FP"]
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=plot_df,
        x="Recall",
        y="Precision",
        size="False_Positives",
        hue="Model",
        sizes=(80, 600),
        alpha=0.8,
    )
    for _, row in plot_df.iterrows():
        ax.text(row["Recall"] + 0.012, row["Precision"] + 0.004, row["Model"], fontsize=8)
    plt.title("Precision-Recall Trade-Off by Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.05)
    plt.ylim(0, max(0.35, plot_df["Precision"].max() + 0.08))
    save_current("03_precision_recall_tradeoff_bubble.png")


def ensemble_lift_table_and_chart(performance):
    baseline = performance[performance["Model"] == "Isolation Forest"].iloc[0]
    ensemble = performance[performance["Model"] == "Ensemble Majority Vote"].iloc[0]
    rows = []
    for metric in ["Precision", "Recall", "F1_Score", "FP", "FN"]:
        before = baseline[metric]
        after = ensemble[metric]
        change = after - before
        pct = (change / before * 100) if before != 0 else np.nan
        rows.append({"Metric": metric, "Isolation_Forest": before, "Ensemble": after, "Absolute_Change": change, "Percent_Change": pct})
    lift = pd.DataFrame(rows)
    lift.to_csv(result_dir / "chapter4_ensemble_lift_vs_isolation_forest.csv", index=False)

    plot_df = lift[lift["Metric"].isin(["Precision", "Recall", "F1_Score"])].melt(
        id_vars="Metric",
        value_vars=["Isolation_Forest", "Ensemble"],
        var_name="Model",
        value_name="Score",
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model")
    annotate_bars(ax)
    plt.title("Before vs After: Isolation Forest Baseline vs Ensemble")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    save_current("04_ensemble_lift_vs_isolation_forest.png")


def anomaly_error_breakdown(performance):
    plot_df = performance[["Model", "TP", "FP", "FN"]].melt(id_vars="Model", var_name="Outcome", value_name="Count")
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=plot_df, x="Model", y="Count", hue="Outcome")
    annotate_bars(ax, fmt="{:.0f}", offset=1)
    plt.title("Detection Outcome Breakdown: True Positives, False Positives, False Negatives")
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.xticks(rotation=18, ha="right")
    save_current("05_detection_outcome_breakdown.png")


def runtime_vs_quality(performance, runtime):
    quality = performance[["Model", "F1_Score"]].copy()
    name_map = {"Ensemble Majority Vote": "Ensemble Majority Vote"}
    runtime_df = runtime[["Model", "Mean_Seconds"]].copy()
    runtime_quality = quality.merge(runtime_df, on="Model", how="left")
    ensemble_time = runtime_df["Mean_Seconds"].sum()
    runtime_quality.loc[runtime_quality["Model"] == "Ensemble Majority Vote", "Mean_Seconds"] = ensemble_time
    runtime_quality.to_csv(result_dir / "chapter4_runtime_vs_f1.csv", index=False)

    plt.figure(figsize=(8, 5.5))
    ax = sns.scatterplot(data=runtime_quality, x="Mean_Seconds", y="F1_Score", hue="Model", s=160)
    for _, row in runtime_quality.iterrows():
        ax.text(row["Mean_Seconds"] + 0.005, row["F1_Score"] + 0.003, row["Model"], fontsize=8)
    plt.title("Runtime vs F1 Score")
    plt.xlabel("Mean Prediction Runtime on Test Split (seconds)")
    plt.ylabel("F1 Score")
    save_current("06_runtime_vs_f1_tradeoff.png")


def robustness_comparison_heatmap(jaccard_with_ensemble):
    pivot = jaccard_with_ensemble.pivot(index="Model", columns="Perturbation_Level", values="Jaccard")
    plt.figure(figsize=(8.8, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", linewidths=0.5, cbar_kws={"label": "Jaccard"})
    plt.title("Gaussian Robustness Heatmap: Jaccard Similarity by Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Model")
    save_current("07_gaussian_robustness_jaccard_heatmap.png")


def stress_robustness_heatmap(stress):
    pivot = stress.pivot(index="Model", columns="Scenario", values="Robustness")
    plt.figure(figsize=(8.8, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="BuGn", linewidths=0.5, cbar_kws={"label": "Robustness"})
    plt.title("Stress Scenario Robustness Heatmap")
    plt.xlabel("Scenario")
    plt.ylabel("Model")
    save_current("08_stress_scenario_robustness_heatmap.png")


def shap_model_feature_heatmap():
    files = {
        "Isolation Forest": "shap_feature_importance_test_sample.csv",
        "Autoencoder": "shap_feature_importance_autoencoder.csv",
        "One-Class SVM": "shap_feature_importance_oneclass_svm.csv",
    }
    parts = []
    for model, filename in files.items():
        frame = require_csv(filename)
        frame["Model"] = model
        parts.append(frame[["Model", "Feature", "Mean_Abs_SHAP"]])
    shap_all = pd.concat(parts, ignore_index=True)
    top_features = (
        shap_all.groupby("Feature")["Mean_Abs_SHAP"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    pivot = shap_all[shap_all["Feature"].isin(top_features)].pivot(index="Feature", columns="Model", values="Mean_Abs_SHAP")
    normalized = pivot.apply(lambda col: col / col.max() if col.max() else col, axis=0)
    normalized.to_csv(result_dir / "chapter4_normalized_shap_feature_comparison.csv")

    plt.figure(figsize=(8.5, 6.2))
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Purples", linewidths=0.5, cbar_kws={"label": "Normalized Importance"})
    plt.title("Cross-Model SHAP Feature Importance Heatmap")
    plt.xlabel("Model")
    plt.ylabel("Feature")
    save_current("09_cross_model_shap_feature_heatmap.png")


def shap_stability_heatmap(if_stability, ae_ocsvm_stability):
    if_frame = if_stability.rename(columns={"Spearman_Rho": "Spearman_Rho"}).copy()
    if_frame["Model"] = "Isolation Forest"
    combined = pd.concat(
        [
            if_frame[["Model", "Perturbation_Level", "Spearman_Rho"]],
            ae_ocsvm_stability[["Model", "Perturbation_Level", "Spearman_Rho"]],
        ],
        ignore_index=True,
    )
    pivot = combined.pivot(index="Model", columns="Perturbation_Level", values="Spearman_Rho")
    pivot.to_csv(result_dir / "chapter4_shap_stability_spearman_heatmap.csv")

    plt.figure(figsize=(8.5, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"label": "Spearman Rho"})
    plt.title("SHAP Explanation Stability Across Noise Levels")
    plt.xlabel("Noise Level")
    plt.ylabel("Model")
    save_current("10_shap_stability_spearman_heatmap.png")


def supervised_early_warning_panel(supervised_metrics, feature_importance):
    metrics = supervised_metrics.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})

    metric_names = ["Precision", "Recall", "F1_Score", "ROC_AUC"]
    values = [metrics[m] for m in metric_names]
    axes[0].bar(metric_names, values, color=["#3568a8", "#5b9f73", "#c58b31", "#8b63a7"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Supervised Early-Warning Metrics")
    axes[0].set_ylabel("Score")
    for idx, value in enumerate(values):
        axes[0].text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    top = feature_importance.head(10).sort_values("Importance")
    axes[1].barh(top["Feature"], top["Importance"], color="#527aa3")
    axes[1].set_title("Top Early-Warning Features")
    axes[1].set_xlabel("Importance")

    fig.suptitle("Supervised Early-Warning Model Summary", fontsize=14, weight="bold")
    path = chapter4_dir / "11_supervised_early_warning_summary_panel.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")


def chapter4_evidence_dashboard(performance, runtime):
    ensemble = performance[performance["Model"] == "Ensemble Majority Vote"].iloc[0]
    best_base = performance[performance["Model"] != "Ensemble Majority Vote"].sort_values("F1_Score", ascending=False).iloc[0]
    runtime_fastest = runtime.sort_values("Mean_Seconds").iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Chapter 4 Evidence Dashboard", fontsize=16, weight="bold")

    axes[0, 0].axis("off")
    text = (
        f"Best baseline F1: {best_base['Model']} = {best_base['F1_Score']:.3f}\n"
        f"Ensemble F1: {ensemble['F1_Score']:.3f}\n"
        f"Ensemble recall: {ensemble['Recall']:.3f}\n"
        f"Fastest detector: {runtime_fastest['Model']} ({runtime_fastest['Mean_Seconds']:.3f}s)"
    )
    axes[0, 0].text(0.04, 0.72, text, fontsize=12, va="top", bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f1f4f8", "edgecolor": "#9aa7b8"})
    axes[0, 0].set_title("Headline Results")

    perf_long = performance.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1_Score"], var_name="Metric", value_name="Score")
    sns.barplot(data=perf_long, x="Metric", y="Score", hue="Model", ax=axes[0, 1])
    axes[0, 1].set_title("Core Metrics")
    axes[0, 1].legend(fontsize=7, loc="upper right")

    outcome = performance[["Model", "FP", "FN"]].melt(id_vars="Model", var_name="Error Type", value_name="Count")
    sns.barplot(data=outcome, x="Model", y="Count", hue="Error Type", ax=axes[1, 0])
    axes[1, 0].set_title("Error Profile")
    axes[1, 0].tick_params(axis="x", rotation=20)

    sns.scatterplot(data=performance, x="Recall", y="Precision", hue="Model", s=120, ax=axes[1, 1])
    axes[1, 1].set_title("Precision vs Recall")
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_ylim(0, max(0.35, performance["Precision"].max() + 0.08))
    axes[1, 1].legend(fontsize=7)

    path = chapter4_dir / "12_chapter4_evidence_dashboard.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")


def build_visual_manifest():
    rows = []
    for path in sorted(chapter4_dir.glob("*.png")):
        rows.append({"Figure": path.name, "Path": str(path), "Size_Bytes": path.stat().st_size})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(result_dir / "chapter4_visual_story_manifest.csv", index=False)
    return manifest


def main():
    print("\n============================================================")
    print("Step 20: Chapter 4 Visual Storytelling Layer")
    print("============================================================\n")

    performance = require_csv("chapter4_model_performance_with_ensemble.csv")
    runtime = require_csv("runtime_benchmark_prediction.csv")
    jaccard = require_csv("jaccard_vs_noise_with_ensemble.csv")
    stress = require_csv("stress_scenarios_results_with_ensemble.csv")
    if_stability = require_csv("shap_stability_all_levels_standardized.csv")
    ae_ocsvm_stability = require_csv("shap_stability_autoencoder_ocsvm.csv")
    supervised_metrics = require_csv("supervised_early_warning_metrics.csv")
    supervised_importance = require_csv("supervised_early_warning_feature_importance.csv")

    visual_pipeline_diagram()
    model_metric_heatmap(performance)
    precision_recall_tradeoff(performance)
    ensemble_lift_table_and_chart(performance)
    anomaly_error_breakdown(performance)
    runtime_vs_quality(performance, runtime)
    robustness_comparison_heatmap(jaccard)
    stress_robustness_heatmap(stress)
    shap_model_feature_heatmap()
    shap_stability_heatmap(if_stability, ae_ocsvm_stability)
    supervised_early_warning_panel(supervised_metrics, supervised_importance)
    chapter4_evidence_dashboard(performance, runtime)

    manifest = build_visual_manifest()
    print("\nChapter 4 visual manifest:")
    print(manifest.to_string(index=False))
    print("\nStep 20 completed successfully!")


if __name__ == "__main__":
    main()
