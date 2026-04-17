"""
Supervised Step 20: Chapter 4 visual story manifest.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd

from supervised_utils import FIGURE_DIR, RESULT_DIR

print("\n============================================================")
print("Supervised Step 20: Chapter 4 Story")
print("============================================================\n")

chapter4_dir = FIGURE_DIR / "chapter4_visuals"
chapter4_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.axis("off")
boxes = [
    ("Yahoo Finance\nMarket Data", 0.02, "#dceeff"),
    ("Feature Engineering\nFinancial Indicators", 0.21, "#e7f5df"),
    ("Supervised Target\nNext 5D Return <= -3%", 0.41, "#fff2cc"),
    ("Classifiers\nLogistic, RF, XGB", 0.61, "#f8dfdf"),
    ("Evaluation\nMetrics, Robustness,\nSHAP", 0.80, "#eadcf8"),
]
y = 0.42
w = 0.16
h = 0.34
for text, x, color in boxes:
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.3, edgecolor="#444", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")
for i in range(len(boxes) - 1):
    x0 = boxes[i][1] + w + 0.01
    x1 = boxes[i + 1][1] - 0.01
    ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2), arrowstyle="->", mutation_scale=16, linewidth=1.5))
ax.set_title("Supervised Financial Drawdown-Risk Prediction Pipeline", fontsize=15, weight="bold", pad=14)
pipeline_fig = chapter4_dir / "01_supervised_framework_pipeline_diagram.png"
plt.tight_layout()
plt.savefig(pipeline_fig, dpi=300, bbox_inches="tight")
plt.close()

rows = []
for path in sorted(FIGURE_DIR.glob("*.png")) + sorted(chapter4_dir.glob("*.png")):
    rows.append({"Figure": path.name, "Path": str(path), "Size_Bytes": path.stat().st_size})
manifest = pd.DataFrame(rows)
manifest_path = RESULT_DIR / "chapter4_visual_story_manifest.csv"
manifest.to_csv(manifest_path, index=False)

print(f"Pipeline diagram saved to: {pipeline_fig}")
print(f"Visual manifest saved to: {manifest_path}")

print("\nSupervised Step 20 completed successfully!")
