"""
train_model.py
---------------
This ONE script does everything needed to produce a trained phishing
detection model. Run it with:

    python train_model.py

It has 3 clearly labeled steps:
  STEP 1: Build a list of real legitimate + real phishing URLs
  STEP 2: Convert every URL into numeric features and save as dataset.csv
  STEP 3: Train a Random Forest model on those features and save it

Sources used:
  - Legitimate URLs: a curated list of well-known real-world domains
    (below), plus a public "benign URL" list used in phishing research
    (included as legit.csv).
  - Phishing URLs: PhishTank.com, a public phishing-reporting feed
    (included as phish.csv).
"""

import os
import csv
import json
import random
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from features import extract_features, FEATURE_NAMES

random.seed(42)
SAMPLE_SIZE_PER_CLASS = 4000


# ============================================================
# STEP 1 — Build the list of URLs to train on
# ============================================================

# A curated list of well-known, real-world domains (tech, finance,
# government, education, shopping, social media, news). We use this to
# build realistic LEGITIMATE example URLs. Without this, a model trained
# only on a public "benign URL" academic dataset (which is mostly old
# torrent/media links) ends up thinking "short URL = phishing" and wrongly
# flags totally normal sites like google.com. This list fixes that.
TOP_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "instagram.com", "twitter.com", "x.com", "linkedin.com", "reddit.com",
    "netflix.com", "microsoft.com", "apple.com", "yahoo.com", "bing.com",
    "whatsapp.com", "office.com", "live.com", "zoom.us", "github.com",
    "gitlab.com", "stackoverflow.com", "wordpress.com", "adobe.com", "salesforce.com",
    "dropbox.com", "paypal.com", "ebay.com", "aliexpress.com", "walmart.com",
    "target.com", "bestbuy.com", "etsy.com", "shopify.com", "spotify.com",
    "pinterest.com", "tumblr.com", "quora.com", "medium.com", "nytimes.com",
    "cnn.com", "bbc.com", "theguardian.com", "forbes.com", "bloomberg.com",
    "reuters.com", "wsj.com", "espn.com", "imdb.com", "twitch.tv",
    "discord.com", "slack.com", "notion.so", "trello.com", "figma.com",
    "canva.com", "airbnb.com", "booking.com", "expedia.com", "uber.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "americanexpress.com", "irs.gov",
    "usa.gov", "nasa.gov", "cdc.gov", "harvard.edu", "mit.edu",
    "stanford.edu", "coursera.org", "udemy.com", "khanacademy.org", "archive.org",
    "indeed.com", "glassdoor.com", "hubspot.com", "cloudflare.com", "digitalocean.com",
    "aws.amazon.com", "oracle.com", "ibm.com", "intel.com", "nvidia.com",
    "samsung.com", "sony.com", "dell.com", "nike.com", "ikea.com",
    "costco.com", "starbucks.com", "flipkart.com", "zomato.com", "paytm.com",
    "hdfcbank.com", "coinbase.com", "robinhood.com", "fidelity.com", "nasdaq.com",
    "weather.com", "duckduckgo.com", "gmail.com", "outlook.com", "icloud.com",
    "steamcommunity.com", "epicgames.com", "playstation.com", "roblox.com",
]

# Real websites are reached through a mix of URL styles - bare domains
# (github.com), "www." (www.amazon.com), other subdomains (mail.google.com,
# docs.python.org), short paths, and long paths (article/product pages).
# We generate a realistic MIX of all of these so the model doesn't latch
# onto one narrow pattern (like "always has www.") as a rule.
SUBDOMAINS = ["", "www.", "www.", "mail.", "docs.", "en.", "support.", "app."]
PATHS = [
    "", "", "/", "/about", "/login", "/help", "/search",                     # short
    "/wiki/Python", "/pricing", "/settings", "/questions/1234",              # medium
    "/in/john-smith", "/watch?v=abc123xy", "/reset-password",                # medium
    "/mail/u/0/", "/3/library/urllib.html", "/docs/tutorial/index",          # medium
    "/wiki/Machine_learning", "/blog/2024/best-practices-for-web-dev",       # long
    "/gp/product/B08N5WRWNW/ref=sr_1_3", "/support/articles/how-do-i-reset", # long
]


def generate_legit_urls():
    """Combine every domain with a random subdomain + path to create
    realistic legitimate URL examples."""
    urls = []
    for domain in TOP_DOMAINS:
        for sub in random.sample(SUBDOMAINS, 3):
            for path in PATHS:
                urls.append(f"https://{sub}{domain}{path}")
    return list(set(urls))


def load_legit_urls_from_file(path="legit.csv"):
    """Loads a supplementary public list of benign URLs for extra variety."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def load_phish_urls_from_file(path="phish.csv"):
    """Loads real, verified phishing URLs reported to PhishTank."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row["url"].strip() for row in reader if row.get("url", "").strip()]


def build_url_lists():
    generated_legit = generate_legit_urls()
    file_legit = load_legit_urls_from_file()
    phish_urls = load_phish_urls_from_file()

    random.shuffle(generated_legit)
    random.shuffle(file_legit)
    random.shuffle(phish_urls)

    # Mostly our curated realistic domains (85%), topped up with the public
    # benign list for extra lexical variety (15%).
    num_generated = min(len(generated_legit), int(SAMPLE_SIZE_PER_CLASS * 0.85))
    num_from_file = SAMPLE_SIZE_PER_CLASS - num_generated
    legit_urls = generated_legit[:num_generated] + file_legit[:num_from_file]
    phish_sample = phish_urls[:SAMPLE_SIZE_PER_CLASS]

    print(f"Legitimate URLs: {len(legit_urls)}  |  Phishing URLs: {len(phish_sample)}")
    return legit_urls, phish_sample


# ============================================================
# STEP 2 — Turn URLs into numeric features -> dataset.csv
# ============================================================

def build_dataset(legit_urls, phish_urls):
    rows = []
    for u in legit_urls:
        f = extract_features(u)
        f["label"] = 0  # 0 = legitimate
        rows.append(f)
    for u in phish_urls:
        f = extract_features(u)
        f["label"] = 1  # 1 = phishing
        rows.append(f)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv("dataset.csv", index=False)
    print(f"Saved dataset.csv with {len(df)} rows "
          f"({(df['label']==0).sum()} legit, {(df['label']==1).sum()} phishing)")
    return df


# ============================================================
# STEP 3 — Train the model on dataset.csv
# ============================================================

def train_and_save_model(df):
    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n=== Model Performance on Held-Out Test Set ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))

    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("Top 10 most important features:")
    for name, score in importances[:10]:
        print(f"  {name}: {score:.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/phishing_model.pkl")

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "confusion_matrix": cm,
        "top_features": [{"name": n, "importance": float(s)} for n, s in importances[:10]],
        "train_size": len(X_train), "test_size": len(X_test),
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved to model/phishing_model.pkl")
    print("Metrics saved to model/metrics.json")


if __name__ == "__main__":
    print("STEP 1/3 - Gathering URLs...")
    legit_urls, phish_urls = build_url_lists()

    print("\nSTEP 2/3 - Extracting features and building dataset.csv...")
    df = build_dataset(legit_urls, phish_urls)

    print("\nSTEP 3/3 - Training the model...")
    train_and_save_model(df)
