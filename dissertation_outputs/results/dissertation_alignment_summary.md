# Dissertation Alignment Summary

## Aim Alignment
The project evaluates supervised financial drawdown-risk prediction, supervised event detection, drift monitoring, event-level time-to-detection, alert fusion, and robustness in a reproducible chronological pipeline.

## Target Definition
Target = 1 when the next 5-day return is <= -3%, otherwise Target = 0.

## Research Question Coverage
- Logistic Regression achieved the strongest test F1 score across supervised classifier and supervised event-detector summaries.
- Time-to-detection is measured from first model alert to event-window onset using ticker-specific trading rows.
- Operational incident time-to-detection is also measured against real external telemetry timestamps when labelled operational files are present.
- Additional event detectors are trained with supervised labels only.
- Validation data refines thresholds and model-selection diagnostics; base supervised model fitting remains train-only.
- Robustness includes Gaussian perturbations, financial stress scenarios, bounded feature-space adversarial simulations, adversarial training, and robust feature filtering.
- Drift monitoring is implemented with rolling PSI, KS-test, Page-Hinkley, CUSUM, and retraining-trigger recommendations.
- Alert fusion combines supervised classifier and supervised event-detector signals into severity levels, analyst actions, persistence metrics, false-alert burden, and audit-rationale outputs.
- Real telemetry outputs and synthetic proxy outputs are written separately; real telemetry is prioritised when order-book, order-flow, execution-log, cyber-log, and manipulation-label files are present.