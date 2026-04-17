"""
Supervised Step 7: Target Label and Chronological Split.

Target = 1 when the next 5-day return is <= -3%, else 0.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from supervised_utils import (
    DATA_DIR,
    RESULT_DIR,
    add_chronological_split,
    add_supervised_target,
    load_state,
    save_state,
)

print("\n============================================================")
print("Supervised Step 7: Target Label and Chronological Split")
print("============================================================\n")

state = load_state(6)
df = add_chronological_split(add_supervised_target(state["df"]))
feature_cols = state["feature_cols"]

split_summary = (
    df.groupby("Split")
    .agg(
        Rows=("Ticker", "size"),
        Assets=("Ticker", "nunique"),
        Start_Date=("Date", "min"),
        End_Date=("Date", "max"),
        Drawdown_Events=("Target", "sum"),
        Event_Rate=("Target", "mean"),
    )
    .reset_index()
)
target_summary = df.groupby(["Split", "Target"]).size().rename("Rows").reset_index()

dataset_path = DATA_DIR / "supervised_drawdown_dataset.csv"
compat_path = DATA_DIR / "evaluation_dataset_with_injected_events.csv"
split_path = RESULT_DIR / "time_series_split_summary.csv"
target_path = RESULT_DIR / "supervised_target_summary.csv"

df.to_csv(dataset_path, index=False)
df.to_csv(compat_path, index=False)
split_summary.to_csv(split_path, index=False)
target_summary.to_csv(target_path, index=False)

print(split_summary.to_string(index=False))
print(f"\nTarget: 1 means next 5-day return <= -3%.")
print(f"Supervised dataset saved to: {dataset_path}")
print(f"Split summary saved to: {split_path}")

save_state(7, df=df, feature_cols=feature_cols, target_col="Target")

print("\nSupervised Step 7 completed successfully!")
