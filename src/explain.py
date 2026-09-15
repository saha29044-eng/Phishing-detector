"""
explain.py

Turns raw model output into short, human-readable reasons a non-technical
user can understand -- e.g. "Domain is a raw IP address" instead of
"domain_in_ip = 1".
"""

FRIENDLY_URL_FLAGS = [
    # (feature_name, condition_fn, message)
    ("domain_in_ip", lambda v: v == 1, "The domain is a raw IP address instead of a normal domain name"),
    ("url_shortened", lambda v: v == 1, "The link uses a known URL-shortening service, which hides the real destination"),
    ("email_in_url", lambda v: v == 1, "The URL contains an embedded email address"),
    ("qty_at_url", lambda v: v > 0, "The URL contains an '@' symbol, which can be used to disguise the real destination"),
    ("server_client_domain", lambda v: v == 1, "The domain name contains the word 'server' or 'client', a common phishing trick"),
    ("length_url", lambda v: v > 75, "The URL is unusually long"),
    ("qty_hyphen_domain", lambda v: v >= 2, "The domain name contains multiple hyphens, often used to mimic a real brand"),
    ("qty_dot_domain", lambda v: v >= 4, "The domain has an unusually high number of subdomains"),
    ("qty_percent_url", lambda v: v > 0, "The URL contains encoded characters ('%'), sometimes used to obscure content"),
    ("directory_length", lambda v: v > 40, "The URL path is unusually long and complex"),
]


def explain_url(features: dict, probability: float) -> list:
    reasons = [msg for name, cond, msg in FRIENDLY_URL_FLAGS if cond(features.get(name, 0))]
    if not reasons and probability >= 0.5:
        reasons.append("The overall pattern of characters in this URL statistically resembles known phishing URLs")
    if not reasons and probability < 0.5:
        reasons.append("No common phishing red flags were detected in the URL structure")
    return reasons


def explain_email(text: str, top_phishing_words: set, top_legit_words: set, probability: float) -> dict:
    text_lower = text.lower()
    matched_phishing = sorted({w for w in top_phishing_words if w in text_lower})
    matched_legit = sorted({w for w in top_legit_words if w in text_lower})
    return {
        "phishing_signals": matched_phishing[:10],
        "legitimate_signals": matched_legit[:10],
    }
