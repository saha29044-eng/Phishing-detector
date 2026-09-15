"""
url_features.py

Extracts lexical / structural features from a raw URL string.

These features mirror the schema of the public "Phishing-Dataset" by
G. Vrbancic et al. (Data in Brief, 2020) so that a model trained on that
dataset can be applied to brand-new, never-before-seen URLs typed in by a
user -- with NO network calls (no DNS/WHOIS/HTTP requests). This keeps the
detector fast, offline-capable, and free of flaky external dependencies.

We deliberately drop the ~13 columns of the original dataset that require
live lookups (domain age, DNS records, SSL certificate validity, Google
indexing, response time, etc.) since those can't be reproduced instantly
and reliably for an arbitrary URL a user pastes into a web form.
"""

import re
from urllib.parse import urlparse

SPECIAL_CHARS = {
    "dot": ".",
    "hyphen": "-",
    "underline": "_",
    "slash": "/",
    "questionmark": "?",
    "equal": "=",
    "at": "@",
    "and": "&",
    "exclamation": "!",
    "space": " ",
    "tilde": "~",
    "comma": ",",
    "plus": "+",
    "asterisk": "*",
    "hashtag": "#",
    "dollar": "$",
    "percent": "%",
}

VOWELS = set("aeiouAEIOU")

# A small, well-known set of URL-shortening services. This is checked purely
# as a string match against the hostname -- no network call required.
KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "lnkd.in", "rebrand.ly", "cutt.ly", "shorte.st",
    "tiny.cc", "soo.gd", "s2r.co", "clck.ru", "v.gd", "qr.ae", "rb.gy",
}


def _char_counts(prefix: str, text: str) -> dict:
    """Count each special character's occurrences within a URL component."""
    return {f"qty_{name}_{prefix}": text.count(char) for name, char in SPECIAL_CHARS.items()}


def _is_ip_address(host: str) -> int:
    ipv4 = re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host or "")
    ipv6 = ":" in (host or "") and re.fullmatch(r"[0-9a-fA-F:]+", host or "")
    return 1 if (ipv4 or ipv6) else 0


def extract_url_features(raw_url: str) -> dict:
    """
    Given a raw URL string (with or without scheme), return a dict of
    features matching (a subset of) the training dataset's columns.

    IMPORTANT: the original dataset uses -1 as a "not applicable" sentinel
    for an entire block of columns whenever a URL component (directory,
    file, or query/params) is absent -- e.g. a bare domain like
    "https://example.com" has no path at all, so EVERY qty_*_directory and
    qty_*_file column is -1, not 0. Filling those with 0 instead of -1 would
    make ordinary URLs look statistically alien to the trained model (0
    never occurs there in training data), so we reproduce that convention
    exactly.
    """
    url = raw_url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        # No scheme given -- assume http/https so urlparse works correctly
        url = "http://" + url

    parsed = urlparse(url)
    domain = parsed.netloc.split("@")[-1]  # drop userinfo like user:pass@
    domain = domain.split(":")[0]  # drop port
    path = parsed.path or ""
    query = parsed.query or ""
    tld = domain.split(".")[-1] if "." in domain else ""

    features = {}

    # --- whole URL ---
    features.update(_char_counts("url", url))
    # qty_tld_url: how many times the URL's own TLD string reappears in the
    # full URL. Phishing URLs often embed a brand's real TLD elsewhere
    # (e.g. "paypal.com.verify-secure.tk") to look legitimate at a glance.
    features["qty_tld_url"] = url.lower().count(tld.lower()) if tld else 0
    features["length_url"] = len(url)

    # --- domain ---
    features.update(_char_counts("domain", domain))
    features["qty_vowels_domain"] = sum(1 for c in domain if c in VOWELS)
    features["domain_length"] = len(domain)
    features["domain_in_ip"] = _is_ip_address(domain)
    features["server_client_domain"] = 1 if ("server" in domain.lower() or "client" in domain.lower()) else 0

    # --- directory & file: split path at its last "/" ---
    if path == "":
        dir_str, file_str = None, None
    else:
        idx = path.rfind("/")
        dir_str = path[: idx + 1]
        file_str = path[idx + 1:]

    if dir_str is None:
        for name in SPECIAL_CHARS:
            features[f"qty_{name}_directory"] = -1
        features["directory_length"] = -1
    else:
        features.update(_char_counts("directory", dir_str))
        features["directory_length"] = len(dir_str)

    if file_str is None:
        for name in SPECIAL_CHARS:
            features[f"qty_{name}_file"] = -1
        features["file_length"] = -1
    else:
        features.update(_char_counts("file", file_str))
        features["file_length"] = len(file_str)

    # --- params (query string) ---
    if query == "":
        for name in SPECIAL_CHARS:
            features[f"qty_{name}_params"] = -1
        features["params_length"] = -1
        features["tld_present_params"] = -1
        features["qty_params"] = -1
    else:
        features.update(_char_counts("params", query))
        features["params_length"] = len(query)
        features["tld_present_params"] = 1 if tld and tld in query else 0
        features["qty_params"] = query.count("=")

    # --- misc ---
    features["email_in_url"] = 1 if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", url) else 0
    features["url_shortened"] = 1 if domain.lower() in KNOWN_SHORTENERS else 0

    return features


# The exact ordered list of feature names our model is trained on.
# (All columns from the public dataset EXCEPT the ~13 that require live
# DNS/WHOIS/HTTP lookups: time_response, domain_spf, asn_ip,
# time_domain_activation, time_domain_expiration, qty_ip_resolved,
# qty_nameservers, qty_mx_servers, ttl_hostname, tls_ssl_certificate,
# qty_redirects, url_google_index, domain_google_index.)
FEATURE_ORDER = [
    "qty_dot_url", "qty_hyphen_url", "qty_underline_url", "qty_slash_url",
    "qty_questionmark_url", "qty_equal_url", "qty_at_url", "qty_and_url",
    "qty_exclamation_url", "qty_space_url", "qty_tilde_url", "qty_comma_url",
    "qty_plus_url", "qty_asterisk_url", "qty_hashtag_url", "qty_dollar_url",
    "qty_percent_url", "qty_tld_url", "length_url",
    "qty_dot_domain", "qty_hyphen_domain", "qty_underline_domain",
    "qty_slash_domain", "qty_questionmark_domain", "qty_equal_domain",
    "qty_at_domain", "qty_and_domain", "qty_exclamation_domain",
    "qty_space_domain", "qty_tilde_domain", "qty_comma_domain",
    "qty_plus_domain", "qty_asterisk_domain", "qty_hashtag_domain",
    "qty_dollar_domain", "qty_percent_domain", "qty_vowels_domain",
    "domain_length", "domain_in_ip", "server_client_domain",
    "qty_dot_directory", "qty_hyphen_directory", "qty_underline_directory",
    "qty_slash_directory", "qty_questionmark_directory", "qty_equal_directory",
    "qty_at_directory", "qty_and_directory", "qty_exclamation_directory",
    "qty_space_directory", "qty_tilde_directory", "qty_comma_directory",
    "qty_plus_directory", "qty_asterisk_directory", "qty_hashtag_directory",
    "qty_dollar_directory", "qty_percent_directory", "directory_length",
    "qty_dot_file", "qty_hyphen_file", "qty_underline_file", "qty_slash_file",
    "qty_questionmark_file", "qty_equal_file", "qty_at_file", "qty_and_file",
    "qty_exclamation_file", "qty_space_file", "qty_tilde_file",
    "qty_comma_file", "qty_plus_file", "qty_asterisk_file",
    "qty_hashtag_file", "qty_dollar_file", "qty_percent_file", "file_length",
    "qty_dot_params", "qty_hyphen_params", "qty_underline_params",
    "qty_slash_params", "qty_questionmark_params", "qty_equal_params",
    "qty_at_params", "qty_and_params", "qty_exclamation_params",
    "qty_space_params", "qty_tilde_params", "qty_comma_params",
    "qty_plus_params", "qty_asterisk_params", "qty_hashtag_params",
    "qty_dollar_params", "qty_percent_params", "params_length",
    "tld_present_params", "qty_params", "email_in_url", "url_shortened",
]


def featurize(raw_url: str) -> list:
    """Return the feature vector (list of numbers) in FEATURE_ORDER."""
    feats = extract_url_features(raw_url)
    return [feats[name] for name in FEATURE_ORDER]


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/login.php?user=admin&pass=1234",
        "http://paypal-secure-update.tk/account/verify@confirm.php",
        "bit.ly/3xYzAbC",
    ]
    for u in test_urls:
        print(u, "->", extract_url_features(u))
