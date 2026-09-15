"""
train_url_model.py

Trains a phishing-URL classifier on the public "Phishing-Dataset"
(Vrbancic et al., 2020 -- https://github.com/GregaVrbancic/Phishing-Dataset).

Only the lexical/structural columns are used (see url_features.py for why),
so the exact same feature extractor can be applied at inference time to a
brand-new URL with no network calls.

Run:
    python src/train_url_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from url_features import FEATURE_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "url_dataset_small.csv")
MODEL_OUT = os.path.join(ROOT, "models", "url_model.pkl")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    X = df[FEATURE_ORDER].copy()
    y = df["phishing"].copy()

    print(f"  Class balance -> legitimate: {(y == 0).sum():,} | phishing: {(y == 1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    print("\nEvaluating on held-out test split...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Feature importances -- useful for the app's "why" explanation
    importances = dict(zip(FEATURE_ORDER, clf.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\nTop 15 most important features:")
    for name, imp in top_features:
        print(f"  {name:30s} {imp:.4f}")

    joblib.dump(
        {"model": clf, "feature_order": FEATURE_ORDER, "importances": importances},
        MODEL_OUT,
    )
    print(f"\nSaved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
