from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import panel as pn


pn.extension("tabulator", sizing_mode="stretch_width")

ROOT = Path(__file__).resolve().parent
DOC_PATH = ROOT / "docs" / "beginner_pipeline_explanation.md"
RESULT_DIR = ROOT / "dissertation_outputs" / "results"
FIGURE_DIR = ROOT / "dissertation_outputs" / "figures"


SECTION_OUTPUTS = {
    "Section 5: Data Loading and Feature Engineering": {
        "tables": [],
        "figures": [
            "stock_price_trends_multi_asset.png",
            "feature_correlation_heatmap_multi_asset.png",
        ],
    },
    "Section 6: Supervised Target, Chronological Split, and Dataset Export": {
        "tables": ["time_series_split_summary.csv", "supervised_target_summary.csv"],
        "figures": [],
    },
    "Section 7: Exploratory Data Analysis": {
        "tables": [],
        "figures": [
            "stock_price_trends_multi_asset.png",
            "feature_correlation_heatmap_multi_asset.png",
        ],
    },
    "Section 9: Core Model Evaluation and Performance Visualisations": {
        "tables": ["supervised_model_metrics.csv", "supervised_confusion_matrices.csv"],
        "figures": [
            "supervised_model_test_metrics.png",
            "supervised_model_performance_heatmap.png",
            "confusion_matrix_heatmaps_supervised.png",
        ],
    },
    "Section 10: Additional Supervised Event Detection Subsystem": {
        "tables": [
            "supervised_event_detector_metrics.csv",
            "supervised_event_detector_thresholds_validation_tuned.csv",
            "drift_alarm_summary.csv",
            "unified_alert_fusion_metrics.csv",
        ],
        "figures": [],
    },
    "Section 11: Robustness and Stress Scenario Testing": {
        "tables": [
            "supervised_robustness_results.csv",
            "supervised_stress_scenario_results.csv",
            "adversarial_attack_results.csv",
            "adversarial_training_defence_metrics.csv",
        ],
        "figures": [
            "supervised_f1_drop_under_noise.png",
            "supervised_stress_scenario_f1_comparison.png",
        ],
    },
    "Section 12: SHAP Explainability and Explanation Stability": {
        "tables": [
            "shap_feature_importance_test_sample.csv",
            "shap_stability_all_levels_standardized.csv",
        ],
        "figures": ["shap_summary_plot_test_sample.png"],
    },
    "Section 13: Advanced Probability Diagnostics and Threshold Analysis": {
        "tables": [
            "advanced_threshold_optimization.csv",
            "advanced_optimized_threshold_metrics.csv",
            "advanced_calibration_curve.csv",
        ],
        "figures": [
            "precision_recall_curves_supervised.png",
            "roc_curves_supervised.png",
            "probability_reliability_diagram.png",
        ],
    },
    "Section 14: Baseline Benchmarks, Lift/Gains, and Economic Utility": {
        "tables": [
            "advanced_baseline_benchmarks.csv",
            "advanced_decile_ranking_lift_gains.csv",
            "advanced_economic_utility_backtest.csv",
        ],
        "figures": ["lift_gains_decile_chart.png"],
    },
    "Section 15: Time-Series Cross-Validation, Tuning, Imbalance, and Ensembles": {
        "tables": [
            "advanced_timeseries_cv_metrics.csv",
            "advanced_hyperparameter_tuning.csv",
            "advanced_ensemble_metrics.csv",
            "advanced_smote_class_imbalance_metrics.csv",
        ],
        "figures": [],
    },
    "Section 16: Leakage Checks, Feature Selection, and Feature Importance": {
        "tables": [
            "advanced_feature_leakage_checks.csv",
            "advanced_feature_selection_pipeline.csv",
            "advanced_permutation_feature_importance.csv",
        ],
        "figures": [],
    },
    "Section 17: Statistical Significance, Cost Curves, and Calibration": {
        "tables": [
            "advanced_model_significance_mcnemar.csv",
            "advanced_threshold_sensitivity_and_fp_fn_costs.csv",
            "advanced_cost_sensitive_evaluation.csv",
        ],
        "figures": [],
    },
    "Section 18: Runtime, Memory, Rolling Backtests, and Stress Extensions": {
        "tables": [
            "advanced_runtime_memory_benchmark.csv",
            "advanced_rolling_window_backtest.csv",
            "advanced_missing_corruption_regime_stress.csv",
        ],
        "figures": [],
    },
    "Section 19: SHAP Model Comparison and Temporal Drift": {
        "tables": [
            "advanced_shap_comparison_across_models.csv",
            "advanced_shap_drift_over_time.csv",
        ],
        "figures": [],
    },
    "Section 20: Final Reports, Dashboard, and Output Manifest": {
        "tables": [
            "final_model_summary_standardized.csv",
            "deployment_readiness_matrix.csv",
            "notebook_output_manifest.csv",
        ],
        "figures": ["automated_final_summary_dashboard.png"],
    },
    "Section 21: Cybersecurity Case Studies, Attack Simulation, and Resilience Scorecard": {
        "tables": [
            "real_world_cyber_attack_case_studies.csv",
            "cyber_attack_data_feed_detection_results.csv",
            "cybersecurity_resilience_scorecard.csv",
            "layered_defence_framework_mapping.csv",
        ],
        "figures": [
            "cyber_attack_case_study_mapping.png",
            "cyber_attack_detection_rates_by_type.png",
            "layered_defence_framework_diagram.png",
            "cybersecurity_resilience_radar_chart.png",
            "cybersecurity_resilience_summary_dashboard.png",
        ],
    },
}


CUSTOM_CSS = """
:root {
  --paper: #f4f7fb;
  --ink: #122033;
  --muted: #5b6675;
  --line: #d9e1ea;
  --accent: #0d6b78;
  --accent-2: #c45434;
  --accent-soft: #dff3f5;
}
body {
  background: var(--paper);
}
.bk-root, body, .markdown {
  color: var(--ink);
}
.fade-in {
  animation: rise 420ms ease-out both;
}
@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.step-kicker {
  color: var(--accent);
  font-weight: 700;
  letter-spacing: 0;
  margin-bottom: 0.2rem;
}
.hero-title {
  font-size: 2rem;
  line-height: 1.15;
  font-weight: 700;
  margin-bottom: 0.35rem;
}
.hero-copy {
  color: var(--muted);
  font-size: 1.02rem;
  max-width: 760px;
}
.metric-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0 8px;
}
.metric {
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  transition: transform 180ms ease, box-shadow 180ms ease;
}
.metric:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 24px rgba(18, 32, 51, 0.08);
}
.metric-value {
  font-size: 1.35rem;
  font-weight: 700;
}
.metric-label {
  color: var(--muted);
  font-size: 0.88rem;
}
.block-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 1rem 0 0.35rem;
}
.progress-shell {
  width: 100%;
  height: 8px;
  border-radius: 999px;
  background: #dfe8ef;
  overflow: hidden;
  margin: 14px 0 8px;
}
.progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), #2d96a3);
  transition: width 300ms ease;
}
.story-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 14px 0 18px;
}
.story-band {
  border-top: 3px solid var(--accent);
  background: white;
  border-radius: 8px;
  padding: 14px 16px;
  border-left: 1px solid var(--line);
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.story-band:nth-child(2) { border-top-color: var(--accent-2); }
.story-band:nth-child(3) { border-top-color: #6a7f3f; }
.story-label {
  color: var(--muted);
  font-size: 0.82rem;
  text-transform: uppercase;
}
.story-copy {
  font-size: 1rem;
  line-height: 1.45;
  margin-top: 4px;
}
.figure-frame img {
  border-radius: 8px;
  border: 1px solid var(--line);
}
@media (max-width: 900px) {
  .metric-strip, .story-grid {
    grid-template-columns: 1fr;
  }
}
"""


def parse_sections(markdown: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## (Section \d+: .+)$", markdown, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[match.group(1)] = markdown[start:end].strip()
    return sections


def clean_section_markdown(text: str) -> str:
    stop_markers = [
        "## How To Explain The Outputs During A Presentation",
        "## Key Terms For Beginners",
        "## Short Summary You Can Say Aloud",
        "## How To Answer Common Questions",
        "## Detailed Presentation Script",
    ]
    for marker in stop_markers:
        text = text.split(marker)[0]
    return text.strip()


STEP_SUMMARIES = {
    "Section 1: Standard Library Imports": (
        "Prepare Python utilities.",
        "The pipeline needs paths, timing, saving, and clean execution before any analysis can begin.",
        "A stable working environment for the rest of the script.",
    ),
    "Section 2: Third-Party Imports": (
        "Load the data-science toolset.",
        "These libraries provide data handling, modelling, plotting, finance downloads, and explainability.",
        "All external tools required by later sections.",
    ),
    "Section 3: Global Configuration": (
        "Define the experiment setup.",
        "Keeping tickers, dates, folders, and features in one place makes the pipeline reproducible.",
        "One source of truth for settings and model inputs.",
    ),
    "Section 4: Environment and Persistence Helpers": (
        "Create folders and save checkpoints.",
        "Long pipelines need reliable output locations and save points for recovery and inspection.",
        "Reusable helper functions and state files.",
    ),
    "Section 5: Data Loading and Feature Engineering": (
        "Turn market data into learning signals.",
        "Raw prices alone are not enough; the model needs volatility, momentum, trend, and drawdown features.",
        "A feature-rich supervised dataset.",
    ),
    "Section 6: Supervised Target, Chronological Split, and Dataset Export": (
        "Define the prediction task.",
        "The model must know what counts as an event and learn from the past before being tested on the future.",
        "Target labels plus Train, Validation, and Test splits.",
    ),
    "Section 7: Exploratory Data Analysis": (
        "Inspect the prepared dataset visually.",
        "Before training, we check whether the data looks sensible and how features relate to each other.",
        "Price-trend and correlation figures.",
    ),
    "Section 8: Model Input Preparation and Training": (
        "Train the core supervised models.",
        "The prepared dataset becomes X inputs and y labels, then multiple model families are fitted for comparison.",
        "Saved Logistic Regression, Random Forest, and boosting models.",
    ),
    "Section 9: Core Model Evaluation and Performance Visualisations": (
        "Measure model performance.",
        "Predictions only matter once we know how many events are found, missed, or falsely flagged.",
        "Metrics, confusion matrices, and comparison charts.",
    ),
    "Section 10: Additional Supervised Event Detection Subsystem": (
        "Extend the classifier into an alerting system.",
        "Operational use needs thresholds, timing, drift monitoring, alert fusion, and deployment support.",
        "Event detectors, telemetry outputs, drift results, and alert policies.",
    ),
    "Section 11: Robustness and Stress Scenario Testing": (
        "Test whether the models stay reliable under pressure.",
        "Good clean-data scores are not enough if noise, stress, or manipulation breaks the model.",
        "Noise, stress, attack, and defence results.",
    ),
    "Section 12: SHAP Explainability and Explanation Stability": (
        "Explain what drives predictions.",
        "A model is more trustworthy when we can see which features matter and whether explanations stay stable.",
        "SHAP rankings, stability metrics, and a summary plot.",
    ),
    "Section 13: Advanced Probability Diagnostics and Threshold Analysis": (
        "Refine thresholds and probability quality.",
        "The default cutoff is not always best, and probability scores should be trustworthy.",
        "Optimised thresholds, PR/ROC curves, and calibration outputs.",
    ),
    "Section 14: Baseline Benchmarks, Lift/Gains, and Economic Utility": (
        "Check whether the models add practical value.",
        "A complex model should beat simpler baselines and rank high-risk cases usefully.",
        "Benchmark, lift, gains, and utility analyses.",
    ),
    "Section 15: Time-Series Cross-Validation, Tuning, Imbalance, and Ensembles": (
        "Strengthen validation.",
        "One split is not enough; we also test across time windows, tune settings, and handle rare events.",
        "Cross-validation, tuning, imbalance, and ensemble results.",
    ),
    "Section 16: Leakage Checks, Feature Selection, and Feature Importance": (
        "Protect experiment credibility.",
        "Financial models can look excellent for the wrong reasons if future information leaks in.",
        "Leakage checks and feature-importance outputs.",
    ),
    "Section 17: Statistical Significance, Cost Curves, and Calibration": (
        "Judge trade-offs more carefully.",
        "Different errors have different costs, and score differences may or may not be meaningful.",
        "Significance, cost, and calibration tables.",
    ),
    "Section 18: Runtime, Memory, Rolling Backtests, and Stress Extensions": (
        "Check practical reliability.",
        "A model should remain useful across time and be feasible to run in practice.",
        "Runtime, memory, rolling-window, and corruption-stress outputs.",
    ),
    "Section 19: SHAP Model Comparison and Temporal Drift": (
        "See whether reasoning changes over time.",
        "Feature importance can drift even when headline metrics look acceptable.",
        "Cross-model and temporal SHAP comparisons.",
    ),
    "Section 20: Final Reports, Dashboard, and Output Manifest": (
        "Package the experiment.",
        "A strong project should end with reusable summaries, not scattered files.",
        "Final tables, dashboard, manifest, and readiness outputs.",
    ),
    "Section 21: Cybersecurity Case Studies, Attack Simulation, and Resilience Scorecard": (
        "Connect the model to cyber resilience.",
        "Trading systems depend on data feeds, so we test simulated cyber threats and summarise layered defence.",
        "Attack results, case studies, and resilience scorecards.",
    ),
    "Section 22: Pipeline Orchestration": (
        "Show the full execution order.",
        "This is the recipe that runs every earlier section in the right sequence.",
        "A single reproducible end-to-end pipeline.",
    ),
}


def dataframe_pane(filename: str) -> pn.Column:
    path = RESULT_DIR / filename
    if not path.exists():
        return pn.Column(
            pn.pane.Markdown(f"#### `{filename}`"),
            pn.pane.Alert("This output has not been generated yet.", alert_type="warning"),
        )
    df = pd.read_csv(path)
    preview = df.head(12)
    return pn.Column(
        pn.pane.Markdown(f"#### `{filename}`"),
        pn.widgets.Tabulator(
            preview,
            pagination="local",
            page_size=min(12, max(len(preview), 1)),
            disabled=True,
            show_index=False,
            height=280,
        ),
    )


def figure_pane(filename: str) -> pn.Column:
    path = FIGURE_DIR / filename
    if not path.exists():
        return pn.Column(
            pn.pane.Markdown(f"#### `{filename}`"),
            pn.pane.Alert("This figure has not been generated yet.", alert_type="warning"),
        )
    return pn.Column(
        pn.pane.Markdown(f"#### `{filename}`"),
        pn.pane.PNG(path, sizing_mode="stretch_width", max_width=960, css_classes=["figure-frame"]),
    )


def outputs_for(section_title: str) -> pn.Column:
    spec = SECTION_OUTPUTS.get(section_title, {"tables": [], "figures": []})
    blocks: list[object] = []
    if spec["tables"]:
        blocks.append(pn.pane.Markdown("### Output Tables"))
        blocks.append(pn.Tabs(*[(name, dataframe_pane(name)) for name in spec["tables"]], dynamic=True))
    if spec["figures"]:
        blocks.append(pn.pane.Markdown("### Figures"))
        blocks.append(pn.Tabs(*[(name, figure_pane(name)) for name in spec["figures"]], dynamic=True))
    if not blocks:
        blocks.append(
            pn.pane.Alert(
                "This section mainly defines helpers or configuration; it does not produce a standalone result artifact.",
                alert_type="info",
            )
        )
    return pn.Column(*blocks, sizing_mode="stretch_width")


def overview_panel() -> pn.Column:
    metrics = pn.pane.HTML(
        """
        <div class="metric-strip">
          <div class="metric"><div class="metric-value">22</div><div class="metric-label">Pipeline sections</div></div>
          <div class="metric"><div class="metric-value">15</div><div class="metric-label">Model features</div></div>
          <div class="metric"><div class="metric-value">3+</div><div class="metric-label">Core model families</div></div>
          <div class="metric"><div class="metric-value">18</div><div class="metric-label">Saved figures</div></div>
        </div>
        """
    )
    return pn.Column(
        pn.pane.HTML(
            """
            <div class="step-kicker">Presentation Overview</div>
            <div class="hero-title">Supervised Financial AI Robustness Evaluation</div>
            <div class="hero-copy">
              A guided presentation of the full pipeline: data preparation, supervised learning,
              robustness testing, explainability, advanced validation, and cybersecurity resilience.
            </div>
            """
        ),
        metrics,
        pn.Row(
            figure_pane("automated_final_summary_dashboard.png"),
            figure_pane("cybersecurity_resilience_summary_dashboard.png"),
            sizing_mode="stretch_width",
        ),
        pn.pane.Markdown(
            """
            ### How to use this presentation

            Use the sidebar to move through the steps in order, or jump directly to a section when a question comes up.
            Each section includes a beginner-friendly explanation and any result tables or figures that section produced.

            ### Recommended presentation route

            1. Start with the project overview.
            2. Explain data preparation and target design.
            3. Show model evaluation and robustness.
            4. Use SHAP and advanced validation to discuss trustworthiness.
            5. Finish with the cybersecurity resilience outputs and final dashboard.
            """
        ),
        sizing_mode="stretch_width",
    )


markdown_text = DOC_PATH.read_text(encoding="utf-8")
sections = parse_sections(markdown_text)
section_titles = list(sections)
page_options = ["Overview", *section_titles]

selector = pn.widgets.Select(
    name="Jump to section",
    options=page_options,
    value="Overview",
    sizing_mode="stretch_width",
)

previous_button = pn.widgets.Button(name="Previous", button_type="default", width=110)
next_button = pn.widgets.Button(name="Next", button_type="primary", width=110)


def move_selection(delta: int) -> None:
    current = page_options.index(selector.value)
    selector.value = page_options[max(0, min(len(page_options) - 1, current + delta))]


previous_button.on_click(lambda _event: move_selection(-1))
next_button.on_click(lambda _event: move_selection(1))


def render_page(selection: str) -> pn.Column:
    if selection == "Overview":
        return overview_panel()
    explanation = clean_section_markdown(sections[selection])
    what, why, achieved = STEP_SUMMARIES[selection]
    section_index = page_options.index(selection)
    progress = (section_index / (len(page_options) - 1)) * 100
    return pn.Column(
        pn.pane.HTML(
            f"""
            <div class="fade-in">
              <div class="step-kicker">{selection.split(":")[0]} of 22</div>
              <div class="hero-title">{selection.split(': ', 1)[1]}</div>
              <div class="progress-shell"><div class="progress-fill" style="width:{progress:.1f}%"></div></div>
              <div class="hero-copy">{what}</div>
            </div>
            """
        ),
        pn.pane.HTML(
            f"""
            <div class="story-grid fade-in">
              <div class="story-band"><div class="story-label">What happens</div><div class="story-copy">{what}</div></div>
              <div class="story-band"><div class="story-label">Why it matters</div><div class="story-copy">{why}</div></div>
              <div class="story-band"><div class="story-label">What we achieved</div><div class="story-copy">{achieved}</div></div>
            </div>
            """
        ),
        outputs_for(selection),
        pn.Accordion(("Detailed explanation", pn.pane.Markdown(explanation)), active=[]),
        sizing_mode="stretch_width",
    )


content = pn.bind(render_page, selector)

template = pn.template.FastListTemplate(
    title="Financial AI Robustness Presentation",
    sidebar=[
        pn.pane.Markdown("## Navigation"),
        selector,
        pn.Row(previous_button, next_button),
        pn.layout.Divider(),
        pn.pane.Markdown(
            """
            **Files**

            - `docs/beginner_pipeline_explanation.md`
            - `scripts/supervised_pipeline.py`
            - `dissertation_outputs/`
            """
        ),
    ],
    main=[content],
    main_max_width="1100px",
    accent_base_color="#146c94",
    header_background="#123447",
    raw_css=[CUSTOM_CSS],
)

template.servable()
