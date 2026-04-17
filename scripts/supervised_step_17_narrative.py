"""
Supervised Step 17: Dissertation narrative mapping.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from supervised_utils import RESULT_DIR, load_state, save_state

print("\n============================================================")
print("Supervised Step 17: Dissertation Narrative")
print("============================================================\n")

state = load_state(16)
summary = state["final_summary"]
best_f1 = summary.sort_values("F1_Score", ascending=False).iloc[0]
best_auc = summary.sort_values("ROC_AUC", ascending=False).iloc[0]
best_robust = summary.sort_values("Avg_Gaussian_Robustness", ascending=False).iloc[0]

narrative = "\n".join(
    [
        "# Dissertation Alignment Summary",
        "",
        "## Aim Alignment",
        "The project now evaluates supervised financial drawdown-risk prediction in a reproducible chronological pipeline.",
        "",
        "## Target Definition",
        "The supervised label is Target = 1 when the next 5-day return is <= -3%, otherwise Target = 0.",
        "",
        "## Research Question Coverage",
        f"- {best_f1['Model']} achieved the strongest test F1 score.",
        f"- {best_auc['Model']} achieved the strongest test ROC-AUC.",
        f"- {best_robust['Model']} achieved the strongest average robustness under Gaussian perturbation.",
        "- Robustness is measured using prediction flips, F1 drop, and ROC-AUC drop under noisy inputs.",
        "- Stress scenarios compare baseline F1 with stressed F1 across models.",
        "",
        "## Explainability Scope",
        "- SHAP is now applied to the strongest tree-based supervised classifier.",
        "- Explanation stability is measured with Spearman rank correlation and top-5 feature overlap.",
        "",
        "## Methodological Improvements",
        "- The previous unsupervised anomaly detectors were replaced with labelled classifiers.",
        "- Training, validation, and test splits remain chronological to reduce leakage risk.",
        "- CSV and PNG outputs continue to be exported to reproducible dissertation paths.",
    ]
)

narrative_path = RESULT_DIR / "dissertation_alignment_summary.md"
narrative_path.write_text(narrative, encoding="utf-8")
print(narrative)
print(f"\nNarrative summary saved to: {narrative_path}")

save_state(17, **state, narrative_path=narrative_path)

print("\nSupervised Step 17 completed successfully!")
