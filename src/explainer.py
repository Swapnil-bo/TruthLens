import os
import sys
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    headline_body_consistency,
    emotional_language_score,
    quote_density,
    factual_noun_ratio,
    source_credibility,
)

# ── LIME explainer (reuse across calls) ───────────────────────────────────────
_explainer = LimeTextExplainer(class_names=["Fake", "Real"])


def _text_to_features(texts: list, url: str = "") -> np.ndarray:
    """
    LIME perturbs the input text and calls this function many times.
    We treat the full article text as both headline proxy and body.
    Returns an (N, 5) array of credibility signals.
    """
    rows = []
    for text in texts:
        row = [
            headline_body_consistency(text, text),   # consistency proxy
            emotional_language_score(text),
            quote_density(text),
            factual_noun_ratio(text),
            source_credibility(url),                 # fixed — URL doesn't change
        ]
        rows.append(row)
    return np.array(rows)


def explain_prediction(model, text: str, url: str = "", num_features: int = 8):
    """
    Run LIME on the combined article text and return:
        - exp_list : list of (phrase, weight) tuples
        - html     : LIME's built-in HTML visualisation string
    """
    def predict_proba(texts):
        features = _text_to_features(texts, url=url)
        return model.predict_proba(features)

    explanation = _explainer.explain_instance(
        text_instance  = text,
        classifier_fn  = predict_proba,
        num_features   = num_features,
        num_samples    = 300,
    )

    exp_list = explanation.as_list()          # [(word, weight), ...]
    html     = explanation.as_html()

    return exp_list, html


def get_signal_scores(text: str, url: str = "") -> dict:
    """
    Return the raw 5 credibility signal scores for a given article.
    Used by app.py to draw the radar chart.
    """
    return {
        "Headline–Body Consistency" : headline_body_consistency(text, text),
        "Emotional Language"        : emotional_language_score(text),
        "Quote Density"             : quote_density(text),
        "Factual Noun Ratio"        : factual_noun_ratio(text),
        "Source Credibility"        : source_credibility(url),
    }


if __name__ == "__main__":
    import joblib

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.pkl")
    model = joblib.load(MODEL_PATH)

    sample_text = (
        'Federal Reserve Chair Jerome Powell stated "inflation remains the '
        'primary concern" at the press briefing on Wednesday. The central bank '
        'raised interest rates by 0.25 percentage points, citing persistent '
        'price pressures across housing and energy sectors.'
    )
    sample_url = "https://reuters.com/markets/fed-rates-2024"

    print("\n── Signal Scores ──────────────────────────────")
    scores = get_signal_scores(sample_text, sample_url)
    for k, v in scores.items():
        print(f"  {k:<30} {v:.3f}")

    print("\n── LIME Phrase Highlights ─────────────────────")
    exp_list, _ = explain_prediction(model, sample_text, sample_url)
    for phrase, weight in exp_list:
        direction = "→ Real" if weight > 0 else "→ Fake"
        print(f"  {phrase:<25} {weight:+.4f}  {direction}")