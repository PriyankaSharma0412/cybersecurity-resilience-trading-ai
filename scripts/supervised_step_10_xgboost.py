"""
Supervised Step 10: XGBoost classifier.

Uses XGBClassifier when available. If xgboost is not installed, uses a
GradientBoostingClassifier fallback so the project still runs.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from supervised_utils import build_xgb_model, load_state, save_model, save_state, split_xy

print("\n============================================================")
print("Supervised Step 10: XGBoost / Gradient Boosting")
print("============================================================\n")

state = load_state(9)
df = state["df"]
feature_cols = state["feature_cols"]
logistic_model = state["logistic_model"]
random_forest_model = state["random_forest_model"]
splits = split_xy(df, feature_cols)
X_train, y_train, _ = splits["Train"]

model, model_name = build_xgb_model()
model.fit(X_train, y_train)
filename = "xgboost_drawdown_classifier.pkl" if model_name == "XGBoost" else "gradient_boosting_drawdown_classifier.pkl"
save_model(model, filename)

print(f"Trained model: {model_name}")
print(f"Training rows: {len(X_train)}")
print(f"Training event rate: {y_train.mean():.4f}")

save_state(
    10,
    df=df,
    feature_cols=feature_cols,
    logistic_model=logistic_model,
    random_forest_model=random_forest_model,
    tree_boost_model=model,
    tree_boost_model_name=model_name,
)

print("\nSupervised Step 10 completed successfully!")
