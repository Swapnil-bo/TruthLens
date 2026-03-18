import re
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import clean_text, count_sentences, count_words, safe_divide, truncate

# ── Load spaCy model once at import time ──────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1 — Headline / Body Consistency  (0.0 → 1.0)
# How many content words in the headline also appear in the body?
# Low overlap = clickbait / misleading headline.
# ─────────────────────────────────────────────────────────────────────────────
def headline_body_consistency(headline: str, body: str) -> float:
    h_tokens = set(clean_text(headline).split())
    b_tokens = set(clean_text(body).split())
    if not h_tokens or not b_tokens:
        return 0.5          # neutral when body is missing
    overlap = h_tokens & b_tokens
    return safe_divide(len(overlap), len(h_tokens))


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2 — Emotional Language Score  (0.0 → 1.0)
# Ratio of emotionally charged words (from a curated seed list) to total words.
# High ratio = sensational / manipulative tone.
# ─────────────────────────────────────────────────────────────────────────────
EMOTIONAL_WORDS = {
    "shocking", "outrageous", "unbelievable", "horrifying", "scandalous",
    "explosive", "bombshell", "terrifying", "disgusting", "alarming",
    "breaking", "urgent", "exclusive", "revealed", "secret", "exposed",
    "destroy", "crush", "obliterate", "slam", "attack", "blast", "hits",
    "crisis", "chaos", "disaster", "catastrophe", "panic", "threat",
    "conspiracy", "hoax", "fake", "fraud", "lie", "cheat", "corrupt",
    "miracle", "amazing", "incredible", "stunning", "mind-blowing",
}

def emotional_language_score(text: str) -> float:
    words  = clean_text(text).split()
    if not words:
        return 0.0
    hits   = sum(1 for w in words if w in EMOTIONAL_WORDS)
    return min(safe_divide(hits, len(words)) * 10, 1.0)   # scale & cap at 1


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3 — Quote Density  (0.0 → 1.0)
# Fraction of sentences that contain a direct quotation.
# Real journalism tends to cite sources via quotes; low density = opinion piece.
# ─────────────────────────────────────────────────────────────────────────────
QUOTE_PATTERN = re.compile(r'["\u201c\u201d].{10,200}["\u201c\u201d]')

def quote_density(text: str) -> float:
    sentences  = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    quoted     = sum(1 for s in sentences if QUOTE_PATTERN.search(s))
    return min(safe_divide(quoted, len(sentences)), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Signal 4 — Factual Noun Ratio  (0.0 → 1.0)
# Ratio of proper nouns (NNP / NNPS) to all tokens (via spaCy POS tags).
# Articles with real facts reference specific people, places, organisations.
# ─────────────────────────────────────────────────────────────────────────────
def factual_noun_ratio(text: str) -> float:
    doc    = nlp(truncate(text, max_words=300))
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return 0.0
    proper = sum(1 for t in tokens if t.tag_ in ("NNP", "NNPS"))
    return min(safe_divide(proper, len(tokens)) * 3, 1.0)   # scale & cap


# ─────────────────────────────────────────────────────────────────────────────
# Signal 5 — Source Credibility Proxy  (0.0 → 1.0)
# Simple URL / domain heuristic:
#   - Known credible TLDs / domains push score up
#   - Known low-credibility patterns push score down
#   - No URL → neutral 0.5
# ─────────────────────────────────────────────────────────────────────────────
CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "bloomberg.com", "economist.com", "ft.com", "scientificamerican.com",
    "nature.com", "who.int", "cdc.gov", "nih.gov",
}
LOW_CRED_PATTERNS = [
    r"\.xyz$", r"\.info$", r"daily\w+news", r"truth\w+media",
    r"freedom\w+press", r"real\w+news\w+", r"\d{3,}news",
]

def source_credibility(url: str) -> float:
    if not isinstance(url, str) or url.strip() == "":
        return 0.5
    url_lower = url.lower()
    domain    = re.sub(r"https?://(www\.)?", "", url_lower).split("/")[0]

    if domain in CREDIBLE_DOMAINS:
        return 0.9
    if any(re.search(p, domain) for p in LOW_CRED_PATTERNS):
        return 0.1
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.85
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Master feature builder
# ─────────────────────────────────────────────────────────────────────────────
def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with columns [headline, body, url]
    Output : DataFrame with 5 credibility signal columns
    """
    print("[feature_engineering] Computing credibility signals …")

    # Sanitise — fill NaN with empty strings so string functions never see float
    X = X.copy()
    X["headline"] = X["headline"].fillna("").astype(str)
    X["body"]     = X["body"].fillna("").astype(str)
    X["url"]      = X["url"].fillna("").astype(str)

    features = pd.DataFrame()

    features["consistency"]   = X.apply(
        lambda r: headline_body_consistency(r["headline"], r["body"]), axis=1)
    features["emotional"]     = X["headline"].apply(emotional_language_score)
    features["quote_density"] = X["body"].apply(quote_density)
    features["factual_nouns"] = X["headline"].apply(factual_noun_ratio)
    features["source_cred"]   = X["url"].apply(source_credibility)

    print(f"[feature_engineering] Done. Shape: {features.shape}")
    return features


if __name__ == "__main__":
    # Quick smoke-test
    sample = pd.DataFrame([{
        "headline": "SHOCKING: Politicians secretly destroy the economy!",
        "body"    : "",
        "url"     : "https://randomtruth123news.xyz/article",
    }, {
        "headline": "Federal Reserve raises interest rates by 0.25 points",
        "body"    : 'Fed Chair stated "inflation remains the primary concern" at the press briefing.',
        "url"     : "https://reuters.com/markets/fed-rates-2024",
    }])
    print(build_features(sample))