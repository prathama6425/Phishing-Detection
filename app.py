"""
app.py
------
A small Flask web application. This is what you'll actually run and show
in interviews / demo videos.

It gives you a simple webpage with a text box. You paste in a URL,
click "Check", and it tells you whether the model thinks it's phishing
or legitimate, plus a confidence score and which red flags it found.

Run with: python app.py
Then open: http://127.0.0.1:5000 in your browser
"""

import json
import joblib
import pandas as pd
from flask import Flask, render_template, request
from features import extract_features, FEATURE_NAMES

app = Flask(__name__)

# Load the trained model once, when the server starts (not on every request)
model = joblib.load("model/phishing_model.pkl")

with open("model/metrics.json") as f:
    METRICS = json.load(f)

# Human-readable explanations shown when a feature looks suspicious
FEATURE_EXPLANATIONS = {
    "has_ip_address": "Uses a raw IP address instead of a domain name",
    "is_https": "Not using HTTPS (no valid encryption indicator)",
    "num_hyphens": "Contains multiple hyphens (often used to fake brand names)",
    "suspicious_word_count": "Contains suspicious words like 'login', 'verify', 'secure'",
    "is_shortened": "Uses a URL-shortening service to hide the real destination",
    "num_subdomains": "Has an unusually high number of subdomains",
    "url_length": "Unusually long URL",
    "num_at_symbols": "Contains an '@' symbol (can be used to disguise the real domain)",
    "double_slash_redirect": "Contains '//' inside the path (possible redirect trick)",
    "brand_keyword_count": "Mentions a well-known brand name in the URL text",
}


def get_flags(feats: dict) -> list:
    """Look at the extracted features and return a plain-English list of
    red flags found in this specific URL, for display in the UI."""
    flags = []
    if feats["has_ip_address"]:
        flags.append(FEATURE_EXPLANATIONS["has_ip_address"])
    if not feats["is_https"]:
        flags.append(FEATURE_EXPLANATIONS["is_https"])
    if feats["num_hyphens"] >= 3:
        flags.append(FEATURE_EXPLANATIONS["num_hyphens"])
    if feats["suspicious_word_count"] >= 2:
        flags.append(FEATURE_EXPLANATIONS["suspicious_word_count"])
    if feats["is_shortened"]:
        flags.append(FEATURE_EXPLANATIONS["is_shortened"])
    if feats["num_subdomains"] >= 3:
        flags.append(FEATURE_EXPLANATIONS["num_subdomains"])
    if feats["url_length"] > 75:
        flags.append(FEATURE_EXPLANATIONS["url_length"])
    if feats["num_at_symbols"] > 0:
        flags.append(FEATURE_EXPLANATIONS["num_at_symbols"])
    if feats["double_slash_redirect"]:
        flags.append(FEATURE_EXPLANATIONS["double_slash_redirect"])
    if feats["brand_keyword_count"] > 0 and feats["has_hyphen_in_domain"]:
        flags.append(FEATURE_EXPLANATIONS["brand_keyword_count"])
    return flags


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            feats = extract_features(url)
            X = pd.DataFrame([feats], columns=FEATURE_NAMES)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]  # [prob_legit, prob_phishing]

            result = {
                "url": url,
                "is_phishing": bool(pred == 1),
                "confidence": round(max(proba) * 100, 1),
                "phishing_probability": round(proba[1] * 100, 1),
                "flags": get_flags(feats),
                "features": feats,
            }
    return render_template("index.html", result=result, metrics=METRICS)


if __name__ == "__main__":
    app.run(debug=True)
