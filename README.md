# 🔍 TruthLens — Multi-Dimensional Credibility Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-FF6600?style=flat-square)
![spaCy](https://img.shields.io/badge/spaCy-3.8+-09A3D5?style=flat-square&logo=spacy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![LIME](https://img.shields.io/badge/LIME-Explainability-6C3483?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Paste any article. Get a trust score, 5-signal radar breakdown, and phrase-level LIME highlights — instantly.**

[Live Demo](https://truthlens-b94mdhrr5aivpntlnzpju6.streamlit.app/) · [Report Bug](https://github.com/swapnil-hazra/TruthLens/issues) · [Request Feature](https://github.com/swapnil-hazra/TruthLens/issues)

</div>

---

## 🧠 Why TruthLens Is Different

Most fake news detectors are binary black boxes — they spit out "Fake" or "Real" with no explanation of *why*.

TruthLens goes further:

| Approach | Binary Classifier | **TruthLens** |
|---|---|---|
| Output | Fake / Real | Trust score 0–100 + 5-dimensional radar |
| Explainability | ❌ None | ✅ LIME phrase-level highlights |
| Signals | 1 (text only) | 5 (headline, emotion, quotes, nouns, source) |
| Transparency | Black box | Every score is inspectable |
| Interview value | Low | **High — shows PM + ML thinking** |

> Binary classifiers are overdone. The differentiation here is **multi-dimensional credibility scoring** — five independent signals combined into a trust index, with LIME highlighting exactly which phrases tanked or boosted the score.

---

## ⚡ Demo

```
Paste this → get this:
```

**Input:** *"SHOCKING bombshell EXPOSED! Secret cabal DESTROYS economy in horrifying conspiracy..."*

| Signal | Score | Interpretation |
|---|---|---|
| Headline–Body Consistency | 1.00 | Same words used repeatedly — no real body |
| Emotional Language | 1.00 | Maxed out on charged words |
| Quote Density | 0.00 | Zero source citations |
| Factual Noun Ratio | 0.00 | No proper nouns, names, or places |
| Source Credibility | 0.10 | `.xyz` domain — known low-cred pattern |
| **Trust Score** | **17.5 / 100** | ⚠️ Suspicious |

**Input:** *"Federal Reserve Chair Jerome Powell stated 'inflation remains the primary concern'..."*

| Signal | Score | Interpretation |
|---|---|---|
| Headline–Body Consistency | 1.00 | Dense overlap between headline and body |
| Emotional Language | 0.00 | Zero sensational language |
| Quote Density | 0.50 | Direct source quotes present |
| Factual Noun Ratio | 0.43 | Named people, institutions referenced |
| Source Credibility | 0.90 | `reuters.com` — verified credible domain |
| **Trust Score** | **76.1 / 100** | ✅ Credible |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TruthLens Pipeline                   │
└─────────────────────────────────────────────────────────┘

  Article Text + URL
        │
        ▼
┌──────────────────┐
│  utils.py        │  clean_text · count_words · safe_divide · truncate
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  feature_engineering.py  — 5 Credibility Signals         │
│                                                          │
│  ① Headline–Body Consistency   (token overlap ratio)    │
│  ② Emotional Language Score    (seed lexicon match)     │
│  ③ Quote Density               (regex + sentence split) │
│  ④ Factual Noun Ratio          (spaCy POS: NNP/NNPS)   │
│  ⑤ Source Credibility          (domain heuristic)       │
└──────┬───────────────────────────────────────────────────┘
       │  (1 × 5 feature vector)
       ▼
┌──────────────────┐         ┌───────────────────────────┐
│  train_model.py  │ ──pkl──▶│  models/xgb_model.pkl     │
│  XGBClassifier   │         │  Trained on FakeNewsNet   │
└──────────────────┘         │  PolitiFact split (983)   │
                             └──────────────┬────────────┘
                                            │
       ┌────────────────────────────────────┘
       ▼
┌──────────────────┐
│  predict.py      │  → label · confidence · trust_score · signals
└──────┬───────────┘
       │
       ├──▶ ┌──────────────────┐
       │    │  explainer.py    │  LIME → phrase weights
       │    └──────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  app.py  — Streamlit UI                                  │
│                                                          │
│  • Trust score (0–100)                                   │
│  • Verdict badge (✦ Credible / ✕ Suspicious)            │
│  • Plotly radar chart (5 signals)                        │
│  • LIME pill highlights (green ↑ / red ↓)               │
└──────────────────────────────────────────────────────────┘
```

---

## 🔬 The 5 Credibility Signals — Deep Dive

### ① Headline–Body Consistency
**What:** Token overlap ratio between the headline and article body.
**Why:** Clickbait headlines use emotionally charged words that don't appear in the actual article. Low overlap = misleading headline.
**How:** `len(headline_tokens ∩ body_tokens) / len(headline_tokens)`

### ② Emotional Language Score
**What:** Proportion of words from a curated seed set of charged/sensational terms.
**Why:** Fake news relies on fear, outrage, and shock to drive clicks. Real journalism uses neutral language.
**How:** 50+ seed words (shocking, bombshell, exposed, destroy…), scaled and capped at 1.0.

### ③ Quote Density
**What:** Fraction of sentences containing a direct quotation (regex for `"..."` patterns).
**Why:** Credible journalism cites sources through direct quotes. Opinion pieces and fabricated stories rarely include them.
**How:** `quoted_sentences / total_sentences`

### ④ Factual Noun Ratio
**What:** Ratio of proper nouns (NNP/NNPS POS tags via spaCy) to total tokens.
**Why:** Real reporting references specific people, places, and organisations. Vague articles avoid proper nouns because they avoid verifiable facts.
**How:** spaCy `en_core_web_sm`, `t.tag_ in ("NNP", "NNPS")`, scaled ×3 and capped.

### ⑤ Source Credibility
**What:** Domain reputation heuristic based on URL.
**Why:** The source is the fastest credibility signal available.
**How:** Whitelist of 18 verified credible domains (Reuters, BBC, NPR…), regex patterns for low-cred TLDs (`.xyz`, `dailytruth*`), `.gov`/`.edu` bonus.

---

## 📁 Project Structure

```
TruthLens/
│
├── data/
│   ├── raw_news.csv              # FakeNewsNet PolitiFact split (auto-downloaded)
│   └── processed_features.csv   # Engineered feature matrix
│
├── models/
│   └── xgb_model.pkl            # Trained XGBoost classifier
│
├── src/
│   ├── utils.py                  # Shared helpers (clean_text, truncate, safe_divide)
│   ├── data_loader.py            # FakeNewsNet download + train/test split
│   ├── feature_engineering.py   # 5 credibility signal extractors
│   ├── train_model.py            # XGBoost training + evaluation + feature importance
│   ├── explainer.py              # LIME phrase-level explanations
│   └── predict.py                # End-to-end prediction pipeline
│
├── app.py                        # Streamlit UI (entry point)
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone & setup

```bash
git clone https://github.com/swapnil-hazra/TruthLens.git
cd TruthLens
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate        # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download data & train model

```bash
# Downloads FakeNewsNet automatically, trains XGBoost, saves .pkl
python src/train_model.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` — paste any article and hit **Run Analysis →**

---

## 📊 Model Performance

Trained on **FakeNewsNet PolitiFact** subset (983 articles, 80/20 split):

| Metric | Score |
|---|---|
| Test Accuracy | 60.4% |
| Real News F1 | 0.70 |
| Fake News F1 | 0.41 |

> **Note on accuracy:** The FakeNewsNet dataset contains only headlines + URLs — no full article body text. Two of the five signals (`consistency`, `quote_density`) require body text and score neutral (0.5 / 0.0) across the entire training set. When users paste full articles in the app, all 5 signals activate and the trust score becomes substantially more meaningful. The 60.4% reflects a 3-signal model effectively.

**Feature importance (from trained model):**

```
source_cred          ████████████████████  0.4758
factual_nouns        ████████████          0.2622
emotional            ████████████          0.2619
consistency          ░                     0.0000
quote_density        ░                     0.0000
```

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|---|---|---|
| ML Model | XGBoost 2.1+ | Multi-signal classification |
| NLP | spaCy 3.8 (en_core_web_sm) | POS tagging for factual noun ratio |
| Explainability | LIME 0.2.0.1 | Phrase-level credibility attribution |
| Vectorisation | scikit-learn TF-IDF | Text feature baseline |
| UI | Streamlit 1.35+ | Interactive web app |
| Visualisation | Plotly (Scatterpolar) | Radar chart |
| Data | FakeNewsNet (PolitiFact) | Training corpus |
| Fonts | Syne · Space Mono · DM Sans | UI typography |

---

## 🗺️ 15-Day Build Roadmap

| Days | Milestone | Status |
|---|---|---|
| 1–3 | FakeNewsNet ingestion, utils, data loader | ✅ Done |
| 4–6 | 5-signal feature engineering + spaCy integration | ✅ Done |
| 7–9 | XGBoost training, evaluation, feature importance | ✅ Done |
| 10–12 | LIME explainability pipeline | ✅ Done |
| 13–15 | Streamlit UI — radar chart + phrase highlights | ✅ Done |

---

## 💡 PM Signal: Why This Project

This project was designed to demonstrate **product + ML thinking together**:

- **The insight:** Real fake news is multi-dimensional. A binary classifier doesn't answer *why* an article is suspicious — an AI PM working on trust & safety needs that explainability.
- **The trade-off:** 5 hand-crafted signals vs. a fine-tuned LLM. The signals are interpretable, fast, and don't require GPU inference — deployable at scale.
- **The feature prioritisation:** Source credibility (47% importance) >> emotional language (26%) >> factual nouns (26%). This tells a product story: the domain is the fastest trust signal, but phrase-level signals matter for articles from unknown sources.
- **The user insight:** Journalists, fact-checkers, and media-literate users don't want a verdict — they want to understand *what* makes an article suspicious. LIME highlights give them that.

---

## 🔭 Future Roadmap

- [ ] Newspaper3k integration — auto-fetch full article body from URL
- [ ] Cross-source validation — compare same story across 3 outlets
- [ ] Calibration layer — Platt scaling for better probability estimates
- [ ] REST API — FastAPI wrapper for `predict.py`
- [ ] Fine-tuned BERT signal — replace TF-IDF baseline with sentence embeddings
- [ ] Browser extension — run TruthLens on any article in-tab

---

## 👤 Author

**Swapnil Hazra**
- GitHub: [@swapnil-hazra](https://github.com/swapnil-hazra)
- Part of the **100 Days of Vibe Coding** challenge — building and shipping one AI project every few days.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with XGBoost · spaCy · LIME · Streamlit · FakeNewsNet</sub>
</div>
