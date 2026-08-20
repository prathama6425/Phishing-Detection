"""
features.py
------------
This file takes a raw URL (as text, e.g. "http://paypal-login.secure-verify.com/account")
and turns it into a set of NUMBERS that describe suspicious patterns.

Machine learning models can't understand text directly - they need numbers.
So for every URL we calculate things like:
    - How long is it?
    - Does it use an IP address instead of a real domain name?
    - Does it contain words like "login", "verify", "secure", "bank"?
    - How many dots, hyphens, @ symbols does it have?
    - Is it using HTTPS or not?

These numbers are called "features". A phishing URL usually LOOKS different
from a real one even before you visit it - that's what we're teaching the
model to notice.

Every function here is deliberately simple and commented so you can explain
each one in an interview.
"""

import re
from urllib.parse import urlparse

# Common words attackers stuff into fake URLs to look trustworthy
SUSPICIOUS_WORDS = [
    "login", "secure", "account", "update", "verify", "signin", "banking",
    "confirm", "password", "pay", "billing", "webscr", "ebayisapi",
    "suspend", "unlock", "wallet", "security", "alert", "urgent"
]

# Well-known brands that phishing sites often impersonate in the URL text
BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "netflix", "bank", "chase", "wellsfargo", "instagram", "whatsapp"
]

# Free URL-shortening services (phishers use these to hide the real destination)
SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "rebrand.ly", "cutt.ly"
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def _safe_parse(url: str):
    """Parses a URL safely, adding http:// if the scheme is missing."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z]+://", url):
        url = "http://" + url
    return urlparse(url)


def extract_features(url: str) -> dict:
    """
    Takes one URL string and returns a dictionary of ~23 numeric features.
    This SAME function is used both when training the model (on thousands
    of URLs) and when a user pastes a single URL into the web app - so the
    model always sees data in the exact same shape.
    """
    url = url.strip()
    parsed = _safe_parse(url)
    domain = parsed.netloc.lower()
    path = parsed.path or ""
    full = url.lower()

    # Strip port number if present (e.g. example.com:8080 -> example.com)
    domain_no_port = domain.split(":")[0]

    features = {
        # --- basic length / structure ---
        "url_length": len(url),
        "domain_length": len(domain_no_port),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdomains": max(domain_no_port.count(".") - 1, 0),

        # --- red-flag structural signals ---
        "has_ip_address": 1 if IP_PATTERN.match(domain_no_port) else 0,
        "is_https": 1 if parsed.scheme == "https" else 0,
        "has_port": 1 if ":" in domain else 0,
        "double_slash_redirect": 1 if "//" in path else 0,
        "has_hyphen_in_domain": 1 if "-" in domain_no_port else 0,

        # --- keyword / word-based signals ---
        "suspicious_word_count": sum(w in full for w in SUSPICIOUS_WORDS),
        "brand_keyword_count": sum(b in full for b in BRAND_KEYWORDS),
        "is_shortened": 1 if any(
            domain_no_port == s or domain_no_port.endswith("." + s) for s in SHORTENERS
        ) else 0,

        # --- ratios (help the model generalize across URL lengths) ---
        "digit_ratio": (sum(c.isdigit() for c in url) / len(url)) if len(url) > 0 else 0,
        "special_char_ratio": (
            sum(not c.isalnum() for c in url) / len(url)
        ) if len(url) > 0 else 0,

        # --- TLD sanity check ---
        "tld_length": len(domain_no_port.split(".")[-1]) if "." in domain_no_port else 0,
    }
    return features


FEATURE_NAMES = list(extract_features("http://example.com/test").keys())


if __name__ == "__main__":
    # Quick manual test - run: python features.py
    sample_urls = [
        "https://www.google.com",
        "http://paypal-login.secure-verify-account.com/signin?user=1",
        "http://192.168.1.1/wp-admin/update.php",
    ]
    for u in sample_urls:
        print(u)
        for k, v in extract_features(u).items():
            print(f"   {k}: {v}")
        print()
