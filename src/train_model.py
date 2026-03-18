import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_splits
from feature_engineering import build_features

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "xgb_model.pkl")


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    print("\n[train] Loading data splits …")
    X_train, X_test, y_train, y_test = load_splits()

    # 2. Build features
    print("[train] Building features for train set …")
    X_train_feat = build_features(X_train)

    print("[train] Building features for test set …")
    X_test_feat  = build_features(X_test)

    # 3. Train XGBoost
    print("\n[train] Training XGBoost classifier …")
    model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = 42,
    )
    model.fit(
        X_train_feat, y_train,
        eval_set=[(X_test_feat, y_test)],
        verbose=50,
    )

    # 4. Evaluate
    y_pred = model.predict(X_test_feat)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[train] Test Accuracy: {acc:.4f}")
    print("\n[train] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    # 5. Feature importance
    print("[train] Feature importances:")
    feat_names = X_train_feat.columns.tolist()
    for name, score in sorted(
        zip(feat_names, model.feature_importances_), key=lambda x: -x[1]
    ):
        bar = "█" * int(score * 40)
        print(f"  {name:<20} {bar} {score:.4f}")

    # 6. Save
    joblib.dump(model, MODEL_PATH)
    print(f"\n[train] Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()