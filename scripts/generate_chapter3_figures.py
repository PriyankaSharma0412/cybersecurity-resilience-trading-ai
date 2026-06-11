from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dissertation_outputs" / "figures" / "chapter3_visuals"
OUT.mkdir(parents=True, exist_ok=True)


COLORS = {
    "ink": "#172033",
    "muted": "#5B6475",
    "line": "#CAD3DF",
    "blue": "#2F6C9E",
    "teal": "#2F8F83",
    "green": "#5B8C5A",
    "amber": "#C7892B",
    "red": "#B95D5D",
    "purple": "#7557A3",
    "bg": "#F7F9FC",
    "white": "#FFFFFF",
}


def save(fig, filename: str):
    path = OUT / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def add_title(ax, title: str, subtitle: str | None = None):
    ax.text(0.02, 0.965, title, transform=ax.transAxes, fontsize=17, fontweight="bold",
            color=COLORS["ink"], va="top")
    if subtitle:
        ax.text(0.02, 0.915, subtitle, transform=ax.transAxes, fontsize=10.5,
                color=COLORS["muted"], va="top")


def box(ax, xy, wh, title, body="", color="#2F6C9E", fontsize=10.5):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor=COLORS["white"],
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top",
            fontsize=fontsize, fontweight="bold", color=color)
    if body:
        wrapped_lines = []
        for line in body.splitlines():
            wrapped_lines.extend(textwrap.wrap(line, width=max(18, int(w * 75))) or [""])
        wrapped = "\n".join(wrapped_lines)
        ax.text(x + w / 2, y + h / 2 - 0.015, wrapped, ha="center", va="center",
                fontsize=fontsize - 1.3, color=COLORS["ink"], linespacing=1.25)


def arrow(ax, start, end, color="#8793A5"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color, shrinkA=4, shrinkB=4),
    )


def base_fig(width=14, height=8):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def make_table(filename, title, subtitle, columns, rows, widths=None, font_size=9.2):
    fig, ax = base_fig(14, max(4.8, 1.1 + 0.52 * len(rows)))
    add_title(ax, title, subtitle)
    widths = widths or [1 / len(columns)] * len(columns)
    x0, y_top = 0.035, 0.82
    table_w = 0.93
    row_h = min(0.105, 0.66 / (len(rows) + 1))
    header_h = row_h * 1.12
    xs = [x0]
    for width in widths:
        xs.append(xs[-1] + table_w * width)

    ax.add_patch(FancyBboxPatch((x0, y_top - header_h), table_w, header_h,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor=COLORS["blue"], edgecolor=COLORS["blue"]))
    for i, col in enumerate(columns):
        ax.text((xs[i] + xs[i + 1]) / 2, y_top - header_h / 2, col,
                ha="center", va="center", fontsize=font_size, fontweight="bold",
                color=COLORS["white"])

    y = y_top - header_h
    for r, row in enumerate(rows):
        y -= row_h
        fill = COLORS["white"] if r % 2 == 0 else "#EEF3F7"
        ax.add_patch(FancyBboxPatch((x0, y), table_w, row_h,
                                    boxstyle="square,pad=0",
                                    facecolor=fill, edgecolor=COLORS["line"], linewidth=0.7))
        for i, cell in enumerate(row):
            text = "\n".join(textwrap.wrap(str(cell), width=max(10, int(widths[i] * 72))))
            ax.text(xs[i] + 0.008, y + row_h / 2, text,
                    ha="left", va="center", fontsize=font_size - 0.8,
                    color=COLORS["ink"], linespacing=1.15)
            if i > 0:
                ax.plot([xs[i], xs[i]], [y, y + row_h], color=COLORS["line"], lw=0.7)

    ax.text(0.035, 0.045, "Source: Methodology design extracted from Chapter3_v3.docx",
            fontsize=8.5, color=COLORS["muted"])
    return save(fig, filename)


def figure_3_1():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.1: Overall Integrated Methodology Pipeline",
              "One connected workflow from market data to monitored decision support.")
    items = [
        ("Data collection", "Yahoo Finance OHLCV\n2010-2024\n10 liquid assets", COLORS["blue"]),
        ("Preprocessing", "Chronological ordering\nmissing-value handling\nStandardScaler", COLORS["teal"]),
        ("Target design", "Target = 1 when\nfuture 5-day return <= -3%", COLORS["green"]),
        ("Model training", "Logistic Regression\nRandom Forest\nGradient Boosting", COLORS["amber"]),
        ("Evaluation", "Precision, Recall, F1\nROC-AUC\nTime-to-Detection", COLORS["purple"]),
        ("Robustness", "Noise, stress, adversarial\nmissing data, drift\nfeature corruption", COLORS["red"]),
        ("Explainability", "SHAP importance\nSHAP stability\nalert rationale", COLORS["blue"]),
        ("Operational outputs", "Alert fusion\nmonitoring, retraining\nregistry and rollback", COLORS["teal"]),
    ]
    xs = [0.04, 0.28, 0.52, 0.76, 0.76, 0.52, 0.28, 0.04]
    ys = [0.66, 0.66, 0.66, 0.66, 0.31, 0.31, 0.31, 0.31]
    for (title, body, color), x, y in zip(items, xs, ys):
        box(ax, (x, y), (0.19, 0.19), title, body, color, 10.3)
    for i in range(3):
        arrow(ax, (xs[i] + 0.19, ys[i] + 0.095), (xs[i + 1], ys[i + 1] + 0.095))
    arrow(ax, (0.855, 0.66), (0.855, 0.50))
    arrow(ax, (0.855, 0.50), (0.855, 0.31))
    for i in range(4, 7):
        arrow(ax, (xs[i], ys[i] + 0.095), (xs[i + 1] + 0.19, ys[i + 1] + 0.095))
    ax.text(0.5, 0.13,
            "Methodological logic: prediction quality is tested together with robustness, explanation stability, and operational controls.",
            ha="center", fontsize=11.2, color=COLORS["ink"], fontweight="bold")
    return save(fig, "figure_3_1_integrated_methodology_pipeline.png")


def figure_3_2():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.2: Data Preprocessing and Feature Engineering Workflow",
              "Raw market data is converted into stable, model-ready financial indicators.")
    stages = [
        ("Raw OHLCV", "Open, High, Low,\nClose, Volume\nby ticker and date", COLORS["blue"]),
        ("Clean and align", "sort timestamps\nremove duplicates\nforward-fill gaps", COLORS["teal"]),
        ("Scale features", "fit StandardScaler\non training split\napply to later splits", COLORS["green"]),
        ("Engineer features", "returns, volatility,\nmomentum, drawdown,\nRSI, MACD, BB width", COLORS["amber"]),
        ("Create target", "future 5-day return\n<= -3% becomes\nrisk event label", COLORS["red"]),
        ("Split chronologically", "train -> validation -> test\nno shuffled k-fold\nno look-ahead bias", COLORS["purple"]),
    ]
    xs = [0.04, 0.20, 0.36, 0.52, 0.68, 0.84]
    for i, (title, body, color) in enumerate(stages):
        box(ax, (xs[i], 0.54), (0.12, 0.23), title, body, color, 9.4)
        if i < len(stages) - 1:
            arrow(ax, (xs[i] + 0.12, 0.655), (xs[i + 1], 0.655))
    groups = [
        ("Return-based", "log returns\nmomentum\nreturn z-score", 0.12),
        ("Risk-based", "rolling volatility\ndrawdown depth", 0.30),
        ("Distribution", "rolling skewness\nrolling kurtosis", 0.48),
        ("Technical", "RSI\nMACD\nBollinger width", 0.66),
        ("Volume", "volume change\nactivity shifts", 0.84),
    ]
    for title, body, x in groups:
        box(ax, (x - 0.07, 0.18), (0.14, 0.18), title, body, COLORS["blue"], 9.2)
    ax.plot([0.58, 0.58], [0.54, 0.41], color=COLORS["line"], lw=1.5)
    ax.plot([0.12, 0.84], [0.41, 0.41], color=COLORS["line"], lw=1.5)
    for _, _, x in groups:
        arrow(ax, (x, 0.41), (x, 0.36), COLORS["line"])
    return save(fig, "figure_3_2_preprocessing_feature_engineering_workflow.png")


def figure_3_3():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.3: Supervised Classifier Mechanisms and Detection Approaches",
              "Model families are compared because each captures a different warning pattern.")
    models = [
        ("Logistic Regression", "Linear boundary\ntransparent coefficients\nfast probability scores", COLORS["blue"]),
        ("Random Forest", "Many decision trees\nnon-linear interactions\nstrong noise tolerance", COLORS["green"]),
        ("Gradient Boosting", "Sequential trees\nfocuses on previous errors\ncaptures weak warning signals", COLORS["amber"]),
    ]
    for x, item in zip([0.07, 0.39, 0.71], models):
        box(ax, (x, 0.60), (0.22, 0.20), item[0], item[1], item[2], 10.5)
        arrow(ax, (x + 0.11, 0.60), (0.50, 0.48))
    box(ax, (0.38, 0.34), (0.24, 0.14), "Shared supervised target",
        "Target = 1 when future 5-day return <= -3%; otherwise normal", COLORS["red"], 10.5)
    arrow(ax, (0.50, 0.34), (0.50, 0.25))
    outputs = [
        ("Baseline classifier", "default predictions\nand risk scores", 0.18),
        ("Event detector", "validation-tuned\nalert threshold", 0.50),
        ("Operational telemetry", "optional external or\nproxy incident stream", 0.82),
    ]
    for title, body, x in outputs:
        box(ax, (x - 0.12, 0.08), (0.24, 0.15), title, body, COLORS["purple"], 9.8)
        arrow(ax, (0.50, 0.34), (x, 0.23), COLORS["line"])
    return save(fig, "figure_3_3_detection_mechanism_comparison.png")


def figure_3_4():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.4: Robustness, Adversarial Testing, and Drift-Monitoring Framework",
              "The clean test set is deliberately disturbed to reveal model fragility.")
    box(ax, (0.04, 0.55), (0.18, 0.20), "Baseline test set", "chronological unseen data\nclean engineered features\nbaseline scores", COLORS["blue"], 10.2)
    tests = [
        ("Gaussian noise", "random feature perturbation\nprediction flips and F1 drop", COLORS["teal"], 0.30, 0.66),
        ("Financial stress", "price, volume, volatility shocks\nstressed F1 and recall", COLORS["amber"], 0.30, 0.43),
        ("Adversarial attack", "feature-space manipulation\nmiss rate and F1 degradation", COLORS["red"], 0.53, 0.66),
        ("Drift monitoring", "PSI, KS, Page-Hinkley,\nCUSUM retraining triggers", COLORS["purple"], 0.53, 0.43),
        ("Missing/corruption", "missing values, corrupt columns,\nregime and correlation breaks", COLORS["green"], 0.76, 0.55),
    ]
    for title, body, color, x, y in tests:
        box(ax, (x, y), (0.18, 0.16), title, body, color, 9.4)
        arrow(ax, (0.22, 0.65), (x, y + 0.08))
    box(ax, (0.31, 0.14), (0.38, 0.16), "Validation evidence",
        "label consistency, F1, recall, ROC-AUC, Time-to-Detection, McNemar tests", COLORS["blue"], 10.2)
    for _, _, _, x, y in tests:
        arrow(ax, (x + 0.09, y), (0.50, 0.30), COLORS["line"])
    box(ax, (0.74, 0.14), (0.20, 0.16), "Control response",
        "threshold review\nretraining trigger\nrollback or analyst review", COLORS["red"], 9.8)
    arrow(ax, (0.69, 0.22), (0.74, 0.22))
    return save(fig, "figure_3_4_robustness_adversarial_drift_framework.png")


def figure_3_5():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.5: SHAP-Based Explainability and Stability Analysis Workflow",
              "The method checks both what drives alerts and whether those explanations remain consistent.")
    steps = [
        ("Model score", "trained supervised model\nproduces risk probability", COLORS["blue"]),
        ("SHAP values", "feature-level contribution\nfor sample observations", COLORS["teal"]),
        ("Global explanation", "mean absolute SHAP\nfeature ranking", COLORS["green"]),
        ("Alert rationale", "top drivers attached\nto reviewed alerts", COLORS["amber"]),
    ]
    for i, (title, body, color) in enumerate(steps):
        x = 0.06 + i * 0.23
        box(ax, (x, 0.60), (0.17, 0.16), title, body, color, 9.6)
        if i < len(steps) - 1:
            arrow(ax, (x + 0.17, 0.68), (x + 0.23, 0.68))
    box(ax, (0.08, 0.28), (0.24, 0.17), "Original explanations",
        "SHAP ranking from clean test sample", COLORS["blue"], 10)
    box(ax, (0.38, 0.28), (0.24, 0.17), "Perturbed explanations",
        "SHAP ranking after noise or adversarial change", COLORS["red"], 10)
    box(ax, (0.68, 0.28), (0.24, 0.17), "Stability metrics",
        "Spearman rank correlation\ntop-k feature overlap", COLORS["purple"], 10)
    arrow(ax, (0.32, 0.365), (0.38, 0.365))
    arrow(ax, (0.62, 0.365), (0.68, 0.365))
    ax.text(0.5, 0.12,
            "Stable explanations support analyst confidence; unstable explanations require caution, drift checks, and data-quality review.",
            ha="center", fontsize=11.0, color=COLORS["ink"], fontweight="bold")
    return save(fig, "figure_3_5_shap_explainability_stability_workflow.png")


def figure_3_6():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.6: System Integration, Governance, and Deployment Artefact Flow",
              "The prototype produces operational artefacts while preserving human review and governance boundaries.")
    top = [
        ("Scoring wrapper", "model probability\nand alert label", COLORS["blue"]),
        ("Streaming loop", "file-based ingestion\nand monitoring cycle", COLORS["teal"]),
        ("Drift policy", "review or retrain\nwhen alarms trigger", COLORS["purple"]),
        ("Model registry", "versioning, rollback,\nthreshold record", COLORS["green"]),
    ]
    for i, (title, body, color) in enumerate(top):
        x = 0.06 + i * 0.23
        box(ax, (x, 0.62), (0.17, 0.16), title, body, color, 9.6)
        if i < len(top) - 1:
            arrow(ax, (x + 0.17, 0.70), (x + 0.23, 0.70))
    box(ax, (0.12, 0.31), (0.22, 0.16), "Analyst review",
        "alerts remain decision-support\nnot automatic trading actions", COLORS["amber"], 10)
    box(ax, (0.39, 0.31), (0.22, 0.16), "Ethical boundary",
        "public data only\nresearch prototype framing", COLORS["red"], 10)
    box(ax, (0.66, 0.31), (0.22, 0.16), "Method summary",
        "prediction + robustness\n+ explainability + controls", COLORS["blue"], 10)
    for x in [0.145, 0.375, 0.605, 0.835]:
        arrow(ax, (x, 0.62), (0.50, 0.47), COLORS["line"])
    for x in [0.23, 0.50, 0.77]:
        arrow(ax, (0.50, 0.47), (x, 0.47))
        arrow(ax, (x, 0.47), (x, 0.31), COLORS["line"])
    ax.text(0.5, 0.15,
            "Deployment artefacts demonstrate operational design, but live use would require independent validation and formal approval.",
            ha="center", fontsize=11.0, color=COLORS["ink"], fontweight="bold")
    return save(fig, "figure_3_6_system_integration_governance_flow.png")


def figure_3_4a():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.4a: Six-Part Robustness Evaluation Design",
              "Section 3.5 tests whether model usefulness continues when data, markets, or adversaries disturb the input stream.")
    tests = [
        ("1. Gaussian\nperturbation", "15 features\n5 noise levels\n0.05 to 0.30 SD", COLORS["teal"], 0.06, 0.60),
        ("2. Financial\nstress scenarios", "Price Spike\nVolume Shock\nVolatility Shock", COLORS["amber"], 0.37, 0.60),
        ("3. Adversarial\nsimulation", "FGSM-style logistic\nfinite-difference proxy\nfor tree models", COLORS["red"], 0.68, 0.60),
        ("4. Microstructure\nattacks", "Layering Quote Ladder\nSpoofing Depth Inflation\nblack-box and white-box", COLORS["purple"], 0.06, 0.28),
        ("5. Drift\nmonitoring", "PSI, KS tests\nPage-Hinkley\nCUSUM-like alarms", COLORS["blue"], 0.37, 0.28),
        ("6. Missing and\ncorruption tests", "missing values\nfeature corruption\nregime breaks", COLORS["green"], 0.68, 0.28),
    ]
    for title, body, color, x, y in tests:
        box(ax, (x, y), (0.24, 0.20), title, body, color, 10.0)
    ax.text(0.5, 0.13,
            "Common outputs: prediction flips, F1 drop, ROC-AUC drop, recall change, detection delay, and retraining signals.",
            ha="center", fontsize=11.0, color=COLORS["ink"], fontweight="bold")
    return save(fig, "figure_3_4a_six_part_robustness_evaluation_design.png")


def figure_3_4b():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.4b: Noise, Stress, Adversarial, and Defence Workflow",
              "The first three robustness tests move from random disturbance to targeted manipulation and active defence.")
    box(ax, (0.05, 0.57), (0.17, 0.18), "Clean baseline", "test split\nbaseline predictions\nF1 and ROC-AUC", COLORS["blue"], 9.8)
    branches = [
        ("Gaussian noise", "scaled random noise\n0.05, 0.10, 0.15,\n0.20, 0.30 SD", COLORS["teal"], 0.31, 0.68),
        ("Market stress", "Price Spike\nVolume Shock\nVolatility Shock", COLORS["amber"], 0.31, 0.44),
        ("Adversarial attack", "FGSM-style logistic\nfinite-difference proxy\nbudgets 0.05, 0.10, 0.20", COLORS["red"], 0.31, 0.20),
    ]
    for title, body, color, x, y in branches:
        box(ax, (x, y), (0.22, 0.16), title, body, color, 9.4)
        arrow(ax, (0.22, 0.66), (x, y + 0.08))
    box(ax, (0.62, 0.54), (0.18, 0.16), "Metric comparison", "prediction flips\nF1 degradation\nROC-AUC degradation", COLORS["purple"], 9.6)
    for _, _, _, x, y in branches:
        arrow(ax, (x + 0.22, y + 0.08), (0.62, 0.62), COLORS["line"])
    box(ax, (0.62, 0.24), (0.18, 0.16), "Active defences", "adversarial training\nrobust feature filtering\n8 least-sensitive features", COLORS["green"], 9.6)
    arrow(ax, (0.71, 0.54), (0.71, 0.40))
    box(ax, (0.84, 0.39), (0.12, 0.16), "Post-defence\ncomparison", "before vs after\nprotection", COLORS["blue"], 9.0)
    arrow(ax, (0.80, 0.32), (0.84, 0.47))
    return save(fig, "figure_3_4b_noise_stress_adversarial_defence_workflow.png")


def figure_3_4c():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.4c: Cybersecurity Resilience Scorecard Layers",
              "Robustness outputs are aggregated into five cybersecurity-resilience layers.")
    layers = [
        ("Layer 1\nData Integrity", "cyber feed detection\npoisoning, stale data,\ndrift, manipulation", COLORS["blue"]),
        ("Layer 2\nModel Integrity", "adversarial robustness\nnoise tolerance\nstress response", COLORS["red"]),
        ("Layer 3\nDrift Monitoring", "PSI, KS,\nPage-Hinkley,\nCUSUM-like alarms", COLORS["purple"]),
        ("Layer 4\nExplainability", "SHAP stability\nfeature-rank consistency\nalert rationale", COLORS["teal"]),
        ("Layer 5\nOperational Response", "alert fusion\npolicy matrix\nretraining and rollback", COLORS["green"]),
    ]
    for i, (title, body, color) in enumerate(layers):
        x = 0.05 + i * 0.185
        box(ax, (x, 0.50), (0.15, 0.22), title, body, color, 8.8)
        if i < len(layers) - 1:
            arrow(ax, (x + 0.15, 0.61), (x + 0.185, 0.61), COLORS["line"])
    box(ax, (0.22, 0.20), (0.56, 0.15), "Model-level resilience profile",
        "Each model receives a score per layer, allowing direct comparison of strengths, weaknesses, and deployment-readiness gaps.",
        COLORS["amber"], 10.2)
    for i in range(len(layers)):
        x = 0.05 + i * 0.185 + 0.075
        arrow(ax, (x, 0.50), (0.50, 0.35), COLORS["line"])
    return save(fig, "figure_3_4c_cybersecurity_resilience_scorecard_layers.png")


def figure_3_4d():
    fig, ax = base_fig(15, 8)
    add_title(ax, "Figure 3.4d: Telemetry Time-to-Detection Measurement",
              "Operational tests measure when an incident begins, when the first alert appears, and whether the alert arrives early enough.")
    y = 0.50
    ax.plot([0.10, 0.90], [y, y], color=COLORS["line"], lw=4)
    events = [
        ("Incident start", "first affected timestamp", 0.18, COLORS["red"]),
        ("First model alert", "earliest alert timestamp", 0.42, COLORS["blue"]),
        ("Escalation/review", "analyst or policy action", 0.66, COLORS["amber"]),
        ("Incident end", "episode closes", 0.84, COLORS["green"]),
    ]
    for title, body, x, color in events:
        ax.scatter([x], [y], s=230, color=color, edgecolor=COLORS["white"], linewidth=2, zorder=3)
        box(ax, (x - 0.09, 0.61), (0.18, 0.14), title, body, color, 9.0)
        ax.plot([x, x], [y, 0.61], color=color, lw=1.4)
    ax.annotate("Detection delay", xy=(0.30, 0.39), xytext=(0.30, 0.39),
                ha="center", fontsize=10.5, color=COLORS["ink"], fontweight="bold")
    arrow(ax, (0.19, 0.42), (0.41, 0.42), COLORS["red"])
    ax.annotate("Lead time before full escalation", xy=(0.54, 0.31), xytext=(0.54, 0.31),
                ha="center", fontsize=10.5, color=COLORS["ink"], fontweight="bold")
    arrow(ax, (0.43, 0.34), (0.65, 0.34), COLORS["blue"])
    box(ax, (0.26, 0.12), (0.48, 0.12), "Recorded telemetry outputs",
        "true incident finding time, first alert timestamp, detection lateness, lead time, and missed-incident count",
        COLORS["purple"], 9.8)
    return save(fig, "figure_3_4d_telemetry_time_to_detection_measurement.png")


def main():
    generated = []
    generated.append(figure_3_1())
    generated.append(make_table(
        "table_3_1_experimental_stages.png",
        "Table 3.1: Experimental Stages of the Research Design",
        "Seven connected stages used to evaluate prediction, robustness, explainability, and operational readiness.",
        ["Stage", "Purpose", "Main output"],
        [
            ["1. Data collection", "Acquire daily Yahoo Finance OHLCV data from January 2010 to January 2024.", "Raw multi-asset market dataset"],
            ["2. Cleaning and structuring", "Sort timestamps, remove duplicates, handle gaps, and preserve extreme observations.", "Clean chronological feature base"],
            ["3. Target creation", "Label downside-risk events when the forward five-day return is <= -3%.", "Binary supervised target"],
            ["4. Model training", "Train Logistic Regression, Random Forest, and Gradient Boosting under consistent splits.", "Comparable trained classifiers"],
            ["5. Controlled testing", "Evaluate baseline metrics, threshold tuning, stress tests, adversarial pressure, and drift.", "Robustness and validation tables"],
            ["6. Explainability", "Use SHAP to identify feature drivers and test explanation stability under perturbation.", "Feature importance and stability evidence"],
            ["7. Operational artefacts", "Generate alert, monitoring, retraining, registry, and rollback outputs.", "Decision-support framework evidence"],
        ],
        widths=[0.18, 0.52, 0.30],
    ))
    generated.append(make_table(
        "table_3_2_dataset_overview.png",
        "Table 3.2: Dataset Overview",
        "Market data scope and preprocessing choices used for the supervised financial-risk study.",
        ["Element", "Methodological choice", "Reason"],
        [
            ["Source", "Yahoo Finance daily OHLCV data", "Public, reproducible source suitable for dissertation research"],
            ["Period", "January 2010 to January 2024", "Captures stable periods, crises, recoveries, and regime changes"],
            ["Assets", "AAPL, MSFT, NVDA, AMZN, JPM, GS, KO, XOM, SPY, QQQ", "Mix of technology, finance, defensive, energy, and index exposures"],
            ["Target", "Forward five-day return <= -3%", "Connects detection to a financially meaningful drawdown event"],
            ["Splitting", "Chronological train, validation, and test windows", "Prevents look-ahead bias in time-series evaluation"],
            ["Scaling", "StandardScaler fitted on training data", "Keeps feature magnitudes comparable without using future information"],
        ],
        widths=[0.18, 0.42, 0.40],
    ))
    generated.append(figure_3_2())
    generated.append(make_table(
        "table_3_3_dual_data_stream_structure.png",
        "Table 3.3: Dual Data-Stream Structure Used in the Study",
        "The framework separates verified market data from optional operational or proxy telemetry.",
        ["Stream", "Inputs", "Use in methodology", "Interpretation boundary"],
        [
            ["Market data stream", "OHLCV prices, volume, engineered technical indicators", "Main supervised drawdown-risk modelling and robustness testing", "Primary empirical evidence"],
            ["Operational telemetry stream", "External incident files when supplied; otherwise synthetic or proxy incident outputs", "Demonstrates monitoring, alerting, and cyber-resilience workflow", "Indicative unless real telemetry is supplied"],
        ],
        widths=[0.18, 0.30, 0.32, 0.20],
    ))
    generated.append(make_table(
        "table_3_4_model_families_and_configurations.png",
        "Table 3.4: Model Families and Configurations Used in the Framework",
        "Multiple supervised models are used to expose trade-offs in interpretability, non-linearity, and alert behaviour.",
        ["Model family", "Detection logic", "Methodological role", "Main limitation"],
        [
            ["Logistic Regression", "Linear probability boundary with class weighting", "Transparent baseline and broad warning model", "May create many false positives"],
            ["Random Forest", "Ensemble of decision trees using majority voting", "Non-linear comparator with strong prediction stability", "Can be too conservative on rare events"],
            ["Gradient Boosting fallback", "Sequential trees correcting previous errors", "Captures weaker non-linear warning patterns", "Sensitive to thresholds and scenario design"],
            ["Event detectors", "Validation-tuned probability thresholds", "Convert risk scores into operational alerts", "Alert volume must be controlled"],
        ],
        widths=[0.22, 0.28, 0.30, 0.20],
    ))
    generated.append(figure_3_3())
    generated.append(figure_3_4())
    generated.append(figure_3_4a())
    generated.append(make_table(
        "table_3_5a_robustness_test_matrix.png",
        "Table 3.5a: Robustness Tests Implemented in Section 3.5",
        "Detailed mapping of each robustness test to its disturbance type and validation output.",
        ["Test", "Disturbance applied", "Models or data stream", "Measured outputs"],
        [
            ["Gaussian perturbation", "Scaled random noise across all 15 input features at 0.05, 0.10, 0.15, 0.20, and 0.30 standard deviations", "Logistic Regression, Random Forest, Gradient Boosting", "prediction flips, flip rate, F1 drop, ROC-AUC drop"],
            ["Financial stress scenarios", "Feature-group amplification for Price Spike, Volume Shock, and Volatility Shock", "All baseline supervised classifiers", "stressed F1, stressed ROC-AUC, prediction flips"],
            ["Adversarial attack simulation", "FGSM-style Logistic Regression attack and finite-difference proxy attacks for tree models", "Logistic Regression, Random Forest, Gradient Boosting", "attacked F1, attacked ROC-AUC, event miss rate"],
            ["Microstructure attack scenarios", "Layering Quote Ladder and Spoofing Depth Inflation patterns", "black-box and white-box order-book simulations", "attack success rate and detection outcomes"],
            ["Drift monitoring", "Rolling 63-day PSI, KS, Page-Hinkley, and CUSUM-like checks", "feature distributions over time", "drift alarms and retraining triggers"],
            ["Missing/corruption testing", "missing values, feature corruption, regime breaks, and correlation breakdowns", "degraded test inputs", "fragility under imperfect data conditions"],
        ],
        widths=[0.20, 0.33, 0.24, 0.23],
        font_size=8.5,
    ))
    generated.append(figure_3_4b())
    generated.append(make_table(
        "table_3_5b_cyber_threat_scenario_matrix.png",
        "Table 3.5b: Cyber Threat Scenarios and Detection Logic",
        "Cybersecurity-resilience tests extend robustness evaluation from market noise to data-feed compromise.",
        ["Scenario", "What it represents", "Detection evidence"],
        [
            ["Data Feed Poisoning", "incorrect or malicious values entering the live model input stream", "alert-score increase, attacked-row detection rate, missed attacked rows"],
            ["Stale Data Injection", "old prices or features being replayed as if current", "score deviation and stale-row alert rate"],
            ["Gradual Drift Attack", "slow distributional shift designed to avoid obvious one-day alarms", "rolling drift alarms and model alert response"],
            ["Coordinated Manipulation", "multiple features shifted together to mimic organised market manipulation", "combined score movement and detection coverage"],
            ["Real incident mapping", "historical cases such as Flash Crash spoofing, Knight Capital, AP Twitter, Tesco Bank, and ION Trading", "mapping to the most responsible framework layers"],
        ],
        widths=[0.24, 0.43, 0.33],
        font_size=8.8,
    ))
    generated.append(figure_3_4c())
    generated.append(make_table(
        "table_3_5c_statistical_validation_and_significance_tests.png",
        "Table 3.5c: Statistical Validation and Significance Tests",
        "Statistical checks support the comparison of model predictions, ranking performance, and distributional drift.",
        ["Method", "Applied to", "Purpose"],
        [
            ["Paired McNemar test", "paired model predictions on the same test observations", "tests whether classification disagreement is statistically meaningful"],
            ["DeLong ROC-AUC comparison", "model probability rankings and ROC-AUC values", "tests whether AUC differences are significant at alpha = 0.05"],
            ["Population Stability Index", "rolling feature distributions versus training baseline", "flags large population shifts"],
            ["Kolmogorov-Smirnov test", "feature distribution differences over time", "tests whether two feature samples likely come from different distributions"],
            ["Page-Hinkley alarm", "sequential feature or score behaviour", "detects persistent mean shifts"],
            ["CUSUM-like signal", "rolling cumulative deviations", "detects sustained directional change"],
        ],
        widths=[0.25, 0.37, 0.38],
        font_size=8.8,
    ))
    generated.append(figure_3_4d())
    generated.append(make_table(
        "table_3_5_explainability_stability_metrics.png",
        "Table 3.5: Metrics Used for Explainability Stability Analysis",
        "SHAP explanations are evaluated for both importance and consistency under perturbation.",
        ["Metric", "What it measures", "Why it matters"],
        [
            ["Mean absolute SHAP", "Average magnitude of each feature contribution", "Identifies the most influential global risk drivers"],
            ["Spearman rank correlation", "Agreement between clean and perturbed feature rankings", "Tests whether explanation order remains stable"],
            ["Top-k feature overlap", "Share of top features retained after perturbation", "Shows whether analysts see the same main drivers"],
            ["Alert rationale audit", "Top feature deviations attached to alert records", "Supports review, traceability, and governance"],
        ],
        widths=[0.24, 0.38, 0.38],
    ))
    generated.append(figure_3_5())
    generated.append(figure_3_6())
    generated.append(make_table(
        "table_3_6_methodological_controls_and_ethics.png",
        "Table 3.6: Methodological Controls and Ethical Safeguards",
        "Controls used to keep the experimental design realistic, auditable, and clearly bounded.",
        ["Control area", "Implementation", "Purpose"],
        [
            ["Time-series integrity", "Chronological splitting and validation-only threshold tuning", "Avoids future leakage and inflated results"],
            ["Class imbalance", "Class weighting rather than random resampling", "Preserves natural financial time order"],
            ["Human oversight", "Outputs treated as probability scores and analyst alerts", "Prevents framing prototype outputs as automatic decisions"],
            ["Data ethics", "Public market data; no private or personal information", "Keeps the study within dissertation research boundaries"],
            ["Deployment boundary", "Prototype scoring, monitoring, retraining, and registry artefacts", "Shows operational design without claiming production readiness"],
        ],
        widths=[0.24, 0.42, 0.34],
    ))

    manifest = OUT / "chapter3_visual_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["caption", "filename", "path", "bytes"])
        captions = [
            "Figure 3.1: Overall integrated methodology pipeline",
            "Table 3.1: Experimental stages of the research design",
            "Table 3.2: Dataset overview",
            "Figure 3.2: Data preprocessing and feature engineering workflow",
            "Table 3.3: Dual data-stream structure used in the study",
            "Table 3.4: Model families and configurations used in the framework",
            "Figure 3.3: Supervised classifier mechanisms and detection approaches",
            "Figure 3.4: Robustness, adversarial testing, and drift-monitoring framework",
            "Figure 3.4a: Six-part robustness evaluation design",
            "Table 3.5a: Robustness tests implemented in Section 3.5",
            "Figure 3.4b: Noise, stress, adversarial, and defence workflow",
            "Table 3.5b: Cyber threat scenarios and detection logic",
            "Figure 3.4c: Cybersecurity resilience scorecard layers",
            "Table 3.5c: Statistical validation and significance tests",
            "Figure 3.4d: Telemetry Time-to-Detection measurement",
            "Table 3.5: Metrics used for explainability stability analysis",
            "Figure 3.5: SHAP-based explainability and stability analysis workflow",
            "Figure 3.6: System integration, governance, and deployment artefact flow",
            "Table 3.6: Methodological controls and ethical safeguards",
        ]
        for caption, path in zip(captions, generated):
            writer.writerow([caption, path.name, str(path), path.stat().st_size])
    print(f"Generated {len(generated)} Chapter 3 visuals in {OUT}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
