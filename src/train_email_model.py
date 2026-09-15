"""
train_email_model.py

Trains a phishing-email classifier on a public compiled dataset of
labeled phishing and legitimate ("ham") emails, drawn from Enron employee
emails plus several phishing-corpus sources
(https://github.com/angelfonsecar/phishing-compilation).

Uses TF-IDF word features + Logistic Regression, a strong and fast
baseline for text classification that also gives interpretable
per-word weights (useful for explaining a prediction in the app).

Run:
    python src/train_email_model.py
"""

import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "data", "email_train.csv")
TEST_PATH = os.path.join(ROOT, "data", "email_test.csv")
MODEL_OUT = os.path.join(ROOT, "models", "email_model.pkl")


def clean_text(text: str) -> str:
    """Light cleaning: lowercase, strip 'Subject:' boilerplate, collapse whitespace."""
    text = str(text)
    text = re.sub(r"^\s*subject\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["mensaje", "tipo"])
    df["label"] = (df["tipo"].str.lower() == "phishing").astype(int)
    df["text"] = df["mensaje"].apply(clean_text)
    return df[df["text"].str.len() > 0]


def main():
    print("Loading dataset...")
    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)
    print(f"  train: {len(train_df):,} rows | test: {len(test_df):,} rows")
    print(f"  train class balance -> ham: {(train_df['label'] == 0).sum():,} "
          f"| phishing: {(train_df['label'] == 1).sum():,}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=5.0,
        )),
    ])

    print("Training TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(train_df["text"], train_df["label"])

    print("\nEvaluating on held-out test split...")
    y_pred = pipeline.predict(test_df["text"])
    y_proba = pipeline.predict_proba(test_df["text"])[:, 1]
    print(classification_report(test_df["label"], y_pred, target_names=["legitimate", "phishing"]))
    auc = roc_auc_score(test_df["label"], y_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Top predictive words for each class -- useful for the app's "why" explanation
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    top_phishing_idx = coefs.argsort()[-25:][::-1]
    top_legit_idx = coefs.argsort()[:25]
    top_phishing_words = [feature_names[i] for i in top_phishing_idx]
    top_legit_words = [feature_names[i] for i in top_legit_idx]

    print("\nTop words associated with PHISHING:", ", ".join(top_phishing_words[:15]))
    print("Top words associated with LEGITIMATE:", ", ".join(top_legit_words[:15]))

    joblib.dump(
        {
            "pipeline": pipeline,
            "top_phishing_words": set(top_phishing_words),
            "top_legit_words": set(top_legit_words),
        },
        MODEL_OUT,
    )
    print(f"\nSaved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
