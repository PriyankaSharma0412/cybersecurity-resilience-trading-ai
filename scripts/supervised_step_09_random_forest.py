"""
Supervised Step 9: Random Forest classifier.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from supervised_utils import build_random_forest_model, load_state, save_model, save_state, split_xy

print("\n============================================================")
print("Supervised Step 9: Random Forest")
print("============================================================\n")

state = load_state(8)
df = state["df"]
feature_cols = state["feature_cols"]
logistic_model = state["logistic_model"]
splits = split_xy(df, feature_cols)
X_train, y_train, _ = splits["Train"]

model = build_random_forest_model()
model.fit(X_train, y_train)
save_model(model, "random_forest_drawdown_classifier.pkl")

print(f"Training rows: {len(X_train)}")
print(f"Training event rate: {y_train.mean():.4f}")

save_state(9, df=df, feature_cols=feature_cols, logistic_model=logistic_model, random_forest_model=model)

print("\nSupervised Step 9 completed successfully!")
