"""
Supervised Step 8: Logistic Regression classifier.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from supervised_utils import build_logistic_model, load_state, save_model, save_state, split_xy

print("\n============================================================")
print("Supervised Step 8: Logistic Regression")
print("============================================================\n")

state = load_state(7)
df = state["df"]
feature_cols = state["feature_cols"]
splits = split_xy(df, feature_cols)
X_train, y_train, _ = splits["Train"]

model = build_logistic_model()
model.fit(X_train, y_train)
save_model(model, "logistic_regression_drawdown_classifier.pkl")

print(f"Training rows: {len(X_train)}")
print(f"Training event rate: {y_train.mean():.4f}")

save_state(8, df=df, feature_cols=feature_cols, logistic_model=model)

print("\nSupervised Step 8 completed successfully!")
