import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_features
from explainer import get_signal_scores, explain_prediction

# ── Load model once at import time ────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.pkl")

def _load_model():
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. "
            "Please run `python src/train_model.py` first."
        )
    return joblib.load(_MODEL_PATH)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, url: str = "") -> dict:
    """
    Full prediction pipeline for a single article.

    Parameters
    ----------
    text : str   — article body (or headline if body unavailable)
    url  : str   — article URL (optional, improves source_cred signal)

    Returns
    -------
    dict with keys:
        label        : "Real" or "Fake"
        confidence   : float 0–1 (probability of predicted class)
        trust_score  : float 0–100 (overall credibility score)
        signals      : dict of 5 signal scores (0–1 each)
        lime_phrases : list of (phrase, weight) tuples
    """
    model = get_model()

    # Build feature row
    X = pd.DataFrame([{"headline": text, "body": text, "url": url}])
    features = build_features(X)

    # Predict
    proba      = model.predict_proba(features)[0]   # [p_fake, p_real]
    pred_class = int(np.argmax(proba))
    label      = "Real" if pred_class == 1 else "Fake"
    confidence = float(proba[pred_class])

    # Trust score: weighted blend of all 5 signals (higher = more credible)
    signals     = get_signal_scores(text, url)
    trust_score = (
        signals["Headline–Body Consistency"] * 15 +
        (1 - signals["Emotional Language"])  * 20 +
        signals["Quote Density"]             * 20 +
        signals["Factual Noun Ratio"]        * 20 +
        signals["Source Credibility"]        * 25
    )  # max = 100

    # LIME explanations
    lime_phrases, _ = explain_prediction(model, text, url)

    return {
        "label"       : label,
        "confidence"  : confidence,
        "trust_score" : round(trust_score, 1),
        "signals"     : signals,
        "lime_phrases": lime_phrases,
    }


if __name__ == "__main__":
    # ── Test 1: likely real ───────────────────────────────────────────────────
    real_text = (
        'Federal Reserve Chair Jerome Powell stated "inflation remains the '
        'primary concern" at Wednesday\'s press briefing. The central bank '
        'raised interest rates by 0.25 percentage points, citing persistent '
        'price pressures across housing and energy sectors. Analysts at Goldman '
        'Sachs described the move as "measured and data-driven".'
    )
    result = predict(real_text, url="https://reuters.com/markets/fed-2024")
    print("\n── Test 1: Reuters article ────────────────────────────────")
    print(f"  Label       : {result['label']} ({result['confidence']:.0%} confidence)")
    print(f"  Trust Score : {result['trust_score']} / 100")
    for k, v in result["signals"].items():
        print(f"    {k:<30} {v:.3f}")

    # ── Test 2: likely fake ───────────────────────────────────────────────────
    fake_text = (
        "SHOCKING bombshell exposed! Secret deep-state cabal DESTROYS economy "
        "in horrifying conspiracy that mainstream media refuses to cover. "
        "Unbelievable truth revealed — share before they delete this!"
    )
    result2 = predict(fake_text, url="https://dailytruth247.xyz/exposed")
    print("\n── Test 2: Clickbait article ───────────────────────────────")
    print(f"  Label       : {result2['label']} ({result2['confidence']:.0%} confidence)")
    print(f"  Trust Score : {result2['trust_score']} / 100")
    for k, v in result2["signals"].items():
        print(f"    {k:<30} {v:.3f}")