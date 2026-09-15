# Phishing Detection System

A machine learning system that detects phishing **URLs** and **emails**, with a simple local web app to try it out interactively.

- **URL model**: RandomForestClassifier on 98 lexical/structural URL features (no live network lookups needed)
- **Email model**: TF-IDF + Logistic Regression on email text
- **Web app**: Flask + vanilla JS, single page, two tools ("Check a URL" / "Check an Email")

---

## 1. Setup

```bash
# (recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Train the models

The datasets are already included in `data/`, so you just need to run the two training scripts once. This creates `models/url_model.pkl` and `models/email_model.pkl`.

```bash
python src/train_url_model.py
python src/train_email_model.py
```

Each script prints accuracy, precision/recall, ROC AUC, and the most predictive features/words so you can include them in your project report.

## 3. Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Paste a URL or email text into the relevant tab and click Analyze.

---

## How it works

### URL detection

The URL model is trained on the public **["Phishing Dataset" by Vrbančič et al.](https://github.com/GregaVrbancic/Phishing-Dataset)** (Data in Brief, 2020) — 58,645 real URLs labeled legitimate/phishing, with 111 engineered features.

The full dataset includes some features that require **live lookups** (domain age via WHOIS, DNS records, SSL certificate validity, Google indexing, server response time, etc.). Those are great for accuracy but make a live demo slow, network-dependent, and fragile (WHOIS servers rate-limit, DNS can time out). This project instead trains on the **98 purely lexical/structural** columns — counts of special characters, string lengths, IP-address detection, URL-shortener detection, etc. — all of which can be recomputed instantly from a raw URL string with zero network calls. See `src/url_features.py` for the exact feature definitions; it's written to exactly reproduce the original dataset's column semantics (including its `-1` "not applicable" convention for missing path/query components) so the trained model sees consistent input at both training and inference time.

**Result on held-out test data:** ~90% accuracy, 0.97 ROC AUC.

### Email detection

The email model is trained on a public compiled dataset of ~54,000 labeled emails (Enron employee emails as "legitimate", combined with several phishing-email corpora) — see [`angelfonsecar/phishing-compilation`](https://github.com/angelfonsecar/phishing-compilation). Text is lightly cleaned (lowercased, "Subject:" prefix stripped) and fed through a TF-IDF vectorizer (unigrams + bigrams) followed by Logistic Regression.

**Result on held-out test data:** ~98% accuracy, 0.998 ROC AUC.

### Explanations

Rather than a black-box score, each result includes **why** it was flagged:
- URL: which specific red flags fired (IP address as domain, URL shortener, embedded `@`, excessive hyphens/subdomains, unusually long URL, etc.), based on the model's top important features.
- Email: which words in the message are most strongly associated with phishing vs. legitimate email in the training data.

---

## Project structure

```
phishing-detector/
├── app.py                    # Flask web app
├── requirements.txt
├── data/
│   ├── url_dataset_small.csv     # Vrbančič et al. URL dataset
│   ├── email_train.csv           # Email dataset (train split)
│   └── email_test.csv            # Email dataset (test split)
├── src/
│   ├── url_features.py       # URL feature extraction (no network calls)
│   ├── train_url_model.py    # Trains + evaluates the URL model
│   ├── train_email_model.py  # Trains + evaluates the email model
│   └── explain.py            # Turns model output into plain-English reasons
├── models/                    # Created by the training scripts
│   ├── url_model.pkl
│   └── email_model.pkl
├── templates/
│   └── index.html
└── static/
    └── style.css
```

---

## Limitations & honest caveats (worth discussing in your write-up)

- **Lexical-only URL features mean no domain reputation/age signal.** A brand-new but perfectly legitimate URL with a longer path or a few hyphens (e.g. a deep link on a large e-commerce or documentation site) can occasionally be flagged, because the model only sees string structure, not who owns the domain or how old it is. The full 111-feature dataset (including WHOIS/DNS/SSL features) would reduce this, at the cost of needing live network calls per prediction.
- **The "legitimate" email class leans heavily on the Enron corpus**, so the model partly learns "sounds like a 2001-2002 corporate email" rather than a fully general notion of legitimate email. Expect it to be very good at classic phishing patterns (urgency, "click here", credential requests) but less reliable on, say, modern marketing emails or transactional receipts.
- **This is a demo/learning project, not a production security tool.** Real anti-phishing systems combine many more signals: sender authentication (SPF/DKIM/DMARC), real-time domain reputation feeds, visual similarity detection for spoofed login pages, browser/email-client telemetry, and human reporting — not just a single ML model.
- Probabilities near 50% should be read as "uncertain," not as a confident verdict either way.

## Ideas to extend this project

- Add the live WHOIS/DNS/SSL features back in (with graceful timeouts/fallbacks) for a "quick check" vs. "deep check" mode.
- Add a browser extension front-end that calls the same Flask API.
- Try a gradient-boosted model (XGBoost/LightGBM) or a small transformer for the email text and compare against the baselines here.
- Collect a more modern, more diverse "legitimate" email set to reduce the Enron-era bias.
- Add SHAP values for per-prediction feature attribution instead of the current rule-based explanations.

## Data sources & citation

- G. Vrbančič, I. Fister Jr., V. Podgorelec. "Datasets for Phishing Websites Detection." *Data in Brief*, Vol. 33, 2020. [DOI: 10.1016/j.dib.2020.106438](http://dx.doi.org/10.1016/j.dib.2020.106438) — [GitHub](https://github.com/GregaVrbancic/Phishing-Dataset)
- Compiled email dataset (Enron + phishing corpora): [angelfonsecar/phishing-compilation](https://github.com/angelfonsecar/phishing-compilation)
