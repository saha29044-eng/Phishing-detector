"""
app.py

Flask web app for the phishing detection system. Serves a single page with
two tools:
  1. URL checker  -- paste a link, get a phishing/legitimate verdict
  2. Email checker -- paste email text, get a phishing/legitimate verdict

Both tools run entirely locally using the models trained by
src/train_url_model.py and src/train_email_model.py. Run those training
scripts first (see README.md) so that models/url_model.pkl and
models/email_model.pkl exist.
"""

import os
import sys
import joblib
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from url_features import extract_url_features, FEATURE_ORDER  # noqa: E402
from train_email_model import clean_text  # noqa: E402
from explain import explain_url, explain_email  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
URL_MODEL_PATH = os.path.join(ROOT, "models", "url_model.pkl")
EMAIL_MODEL_PATH = os.path.join(ROOT, "models", "email_model.pkl")

app = Flask(__name__)

_url_bundle = None
_email_bundle = None


def get_url_bundle():
    global _url_bundle
    if _url_bundle is None:
        if not os.path.exists(URL_MODEL_PATH):
            return None
        _url_bundle = joblib.load(URL_MODEL_PATH)
    return _url_bundle


def get_email_bundle():
    global _email_bundle
    if _email_bundle is None:
        if not os.path.exists(EMAIL_MODEL_PATH):
            return None
        _email_bundle = joblib.load(EMAIL_MODEL_PATH)
    return _email_bundle


@app.route("/")
def index():
    return render_template(
        "index.html",
        url_model_ready=os.path.exists(URL_MODEL_PATH),
        email_model_ready=os.path.exists(EMAIL_MODEL_PATH),
    )


@app.route("/api/check-url", methods=["POST"])
def check_url():
    data = request.get_json(force=True, silent=True) or {}
    raw_url = (data.get("url") or "").strip()
    if not raw_url:
        return jsonify({"error": "Please enter a URL."}), 400

    bundle = get_url_bundle()
    if bundle is None:
        return jsonify({"error": "URL model not found. Run 'python src/train_url_model.py' first."}), 503

    try:
        features = extract_url_features(raw_url)
        vector = [[features[name] for name in FEATURE_ORDER]]
    except Exception as e:
        return jsonify({"error": f"Could not parse that URL ({e})."}), 400

    model = bundle["model"]
    proba = float(model.predict_proba(vector)[0][1])
    label = "phishing" if proba >= 0.5 else "legitimate"
    reasons = explain_url(features, proba)

    return jsonify({
        "label": label,
        "probability": round(proba, 4),
        "reasons": reasons,
    })


@app.route("/api/check-email", methods=["POST"])
def check_email():
    data = request.get_json(force=True, silent=True) or {}
    raw_text = (data.get("text") or "").strip()
    if not raw_text:
        return jsonify({"error": "Please paste some email text."}), 400

    bundle = get_email_bundle()
    if bundle is None:
        return jsonify({"error": "Email model not found. Run 'python src/train_email_model.py' first."}), 503

    cleaned = clean_text(raw_text)
    pipeline = bundle["pipeline"]
    proba = float(pipeline.predict_proba([cleaned])[0][1])
    label = "phishing" if proba >= 0.5 else "legitimate"
    signals = explain_email(cleaned, bundle["top_phishing_words"], bundle["top_legit_words"], proba)

    return jsonify({
        "label": label,
        "probability": round(proba, 4),
        "phishing_signals": signals["phishing_signals"],
        "legitimate_signals": signals["legitimate_signals"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
