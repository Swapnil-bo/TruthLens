import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import plotly.graph_objects as go
from predict import predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens · Credibility Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Master CSS + Google Fonts ─────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #07090f !important;
    color: #e8edf5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"],
[data-testid="stToolbar"] { display: none !important; }

[data-testid="stHeader"] { background: transparent !important; }

.block-container {
    max-width: 980px !important;
    padding: 0 2.5rem 5rem !important;
    margin: 0 auto !important;
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.022) 1px, transparent 1px);
    background-size: 64px 64px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f16; }
::-webkit-scrollbar-thumb { background: #1a2030; border-radius: 2px; }

.tl-hero { padding: 4.5rem 0 2.5rem; position: relative; }

.tl-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.28em;
    color: #00d4ff;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.tl-eyebrow::before {
    content: '';
    width: 28px; height: 1px;
    background: #00d4ff;
    display: inline-block;
}
.tl-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3.5rem, 7vw, 5.5rem);
    font-weight: 800;
    line-height: 0.92;
    color: #f0f4ff;
    letter-spacing: -0.035em;
    margin-bottom: 1.4rem;
}
.tl-title .accent { color: #00d4ff; }
.tl-subtitle {
    font-size: 1rem;
    font-weight: 300;
    color: #505a72;
    line-height: 1.65;
    max-width: 460px;
}
.tl-scan {
    position: absolute;
    top: 0; left: -2.5rem; right: -2.5rem;
    height: 1px;
    background: linear-gradient(90deg,transparent,rgba(0,212,255,0.35),rgba(0,212,255,0.9),rgba(0,212,255,0.35),transparent);
    animation: scanDown 4s ease-in-out infinite;
}
@keyframes scanDown {
    0%   { transform: translateY(0);     opacity: 0; }
    8%   { opacity: 1; }
    92%  { opacity: 1; }
    100% { transform: translateY(140px); opacity: 0; }
}

label[data-testid="stWidgetLabel"] p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color: #00d4ff !important;
    margin-bottom: 0.4rem !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e8edf5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(0,212,255,.45) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,.07) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: #2c3348 !important; }

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
    color: #07090f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.07em !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 0 28px rgba(0,212,255,.2) !important;
    transition: transform .18s, box-shadow .18s !important;
    cursor: pointer !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 44px rgba(0,212,255,.35) !important;
}

[data-testid="stSpinner"] p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: #00d4ff !important;
}

hr { border-color: rgba(255,255,255,0.05) !important; }

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    color: #3a4560 !important;
}

[data-testid="stAlert"] {
    background: rgba(255,71,87,.07) !important;
    border: 1px solid rgba(255,71,87,.2) !important;
    border-radius: 10px !important;
    color: #ff4757 !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.au { animation: fadeUp .5s ease both; }
.d1 { animation-delay: .08s; }
.d2 { animation-delay: .16s; }
</style>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tl-hero au">
<div class="tl-scan"></div>
<div class="tl-eyebrow">Credibility Intelligence System v1.0</div>
<div class="tl-title">Truth<span class="accent">Lens</span></div>
<div class="tl-subtitle">
Five-signal credibility analysis with LIME explainability.
Paste any article — get a trust score, radar breakdown, and phrase-level highlights.
</div>
</div>
""", unsafe_allow_html=True)


# ── Inputs ────────────────────────────────────────────────────────────────────
url  = st.text_input(
    "Article URL (optional — improves source credibility signal)",
    placeholder="https://reuters.com/…"
)
text = st.text_area(
    "Article body",
    height=210,
    placeholder="Paste the full article text here …"
)
run  = st.button("Run Analysis →", type="primary", use_container_width=True)


# ── Helper: build a signal bar row with 100% inline styles ───────────────────
def signal_row_html(label, score, accent):
    fill = accent if score >= 0.5 else "#ff4757"
    pct  = f"{score * 100:.1f}"
    return (
        f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;">'
        f'<span style="font-size:0.78rem;color:#5a6680;min-width:128px;flex-shrink:0;">{label}</span>'
        f'<div style="flex:1;height:3px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct}%;background:{fill};box-shadow:0 0 6px {fill}99;border-radius:2px;"></div>'
        f'</div>'
        f'<span style="font-family:Space Mono,monospace;font-size:0.68rem;color:{fill};min-width:34px;text-align:right;">{score:.2f}</span>'
        f'</div>'
    )


# ── Helper: build a LIME pill with 100% inline styles ────────────────────────
def pill_html(phrase, weight, is_positive):
    bg      = "rgba(46,213,115,.08)"  if is_positive else "rgba(255,71,87,.08)"
    color   = "#2ed573"               if is_positive else "#ff4757"
    border  = "rgba(46,213,115,.18)"  if is_positive else "rgba(255,71,87,.18)"
    dot_bg  = "#2ed573"               if is_positive else "#ff4757"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
        f'font-family:Space Mono,monospace;font-size:0.64rem;letter-spacing:0.02em;'
        f'padding:0.28rem 0.7rem;border-radius:999px;margin:0.2rem 0.1rem;'
        f'background:{bg};color:{color};border:1px solid {border};">'
        f'<span style="width:5px;height:5px;border-radius:50%;background:{dot_bg};flex-shrink:0;display:inline-block;"></span>'
        f'{phrase} {weight:+.3f}</span>'
    )


# ── Helper: glass card wrapper ────────────────────────────────────────────────
def card_html(inner, extra_style=""):
    return (
        f'<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:18px;padding:1.8rem;position:relative;overflow:hidden;{extra_style}">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:1px;'
        f'background:linear-gradient(90deg,transparent,rgba(0,212,255,0.35),transparent);"></div>'
        f'{inner}'
        f'</div>'
    )


# ── Helper: section label ─────────────────────────────────────────────────────
def section_label(text_label):
    return (
        f'<div style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.25em;'
        f'text-transform:uppercase;color:#2e3a52;margin-bottom:1rem;'
        f'border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:0.6rem;">'
        f'{text_label}</div>'
    )


# ── Analysis ──────────────────────────────────────────────────────────────────
if run:
    if not text.strip():
        st.warning("Please paste some article text to analyse.")
        st.stop()

    with st.spinner("Scanning credibility signals …"):
        result = predict(text.strip(), url=url.strip())

    label      = result["label"]
    confidence = result["confidence"]
    trust      = result["trust_score"]
    signals    = result["signals"]
    phrases    = result["lime_phrases"]

    is_real = label == "Real"
    accent  = "#2ed573" if is_real else "#ff4757"
    rgb     = "46,213,115" if is_real else "255,71,87"
    blabel  = "✦ Credible" if is_real else "✕ Suspicious"

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Verdict bar ───────────────────────────────────────────────────────────
    badge_bg     = f"rgba({rgb},.1)"
    badge_border = f"rgba({rgb},.25)"
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">'
        f'<span style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.25em;text-transform:uppercase;color:#2e3a52;">Analysis result</span>'
        f'<span style="font-family:Syne,sans-serif;font-weight:700;font-size:0.78rem;letter-spacing:0.1em;text-transform:uppercase;'
        f'padding:0.4rem 1.1rem;border-radius:999px;color:{accent};background:{badge_bg};border:1px solid {badge_border};">'
        f'{blabel}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Two-column layout ─────────────────────────────────────────────────────
    c1, c2 = st.columns([1, 1.65], gap="large")

    with c1:
        short_names = {
            "Headline\u2013Body Consistency": "H-B Consistency",
            "Emotional Language":             "Emotional Lang.",
            "Quote Density":                  "Quote Density",
            "Factual Noun Ratio":             "Factual Nouns",
            "Source Credibility":             "Source Cred.",
        }

        # Build every signal row as a single-line string (no indentation)
        rows = "".join(
            signal_row_html(short_names.get(k, k), v, accent)
            for k, v in signals.items()
        )

        inner = (
            section_label("Trust Score") +
            f'<div style="text-align:center;padding:1.2rem 0 1rem;">'
            f'<div style="font-family:Syne,sans-serif;font-size:6rem;font-weight:800;line-height:1;letter-spacing:-0.04em;color:{accent};">{trust}</div>'
            f'<div style="font-family:Space Mono,monospace;font-size:1rem;color:#2e3a52;">/ 100</div>'
            f'<div style="font-family:Space Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:#2e3a52;margin-top:0.5rem;">Credibility Index</div>'
            f'<div style="font-size:0.82rem;color:#3a4560;margin-top:0.2rem;">Model confidence: {confidence:.0%}</div>'
            f'</div>'
            f'<hr style="border:none;border-top:1px solid rgba(255,255,255,0.05);margin:0.8rem 0 1.2rem;">' +
            section_label("Signals") +
            rows
        )
        st.markdown(card_html(inner, "height:100%;"), unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    with c2:
        signal_names  = list(signals.keys())
        signal_values = list(signals.values())

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[1]*len(signal_names)+[1],
            theta=signal_names+[signal_names[0]],
            fill="toself",
            fillcolor="rgba(255,255,255,0.018)",
            line=dict(color="rgba(255,255,255,0.05)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

        fig.add_trace(go.Scatterpolar(
            r=[0.5]*len(signal_names)+[0.5],
            theta=signal_names+[signal_names[0]],
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="rgba(255,255,255,0.04)", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

        fig.add_trace(go.Scatterpolar(
            r=signal_values+[signal_values[0]],
            theta=signal_names+[signal_names[0]],
            fill="toself",
            fillcolor=f"rgba({rgb},0.1)",
            line=dict(color=accent, width=2.5),
            marker=dict(size=8, color=accent, line=dict(color="#07090f", width=2.5)),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>",
            showlegend=False,
        ))

        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0,1],
                    tickvals=[0.25, 0.5, 0.75],
                    ticktext=["", "0.5", ""],
                    tickfont=dict(size=8, color="#2a3045", family="Space Mono"),
                    gridcolor="rgba(255,255,255,0.04)",
                    linecolor="rgba(255,255,255,0.04)",
                ),
                angularaxis=dict(
                    tickfont=dict(size=10.5, color="#5a6680", family="DM Sans"),
                    gridcolor="rgba(255,255,255,0.035)",
                    linecolor="rgba(255,255,255,0.035)",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=50, b=50, l=70, r=70),
            height=420,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── LIME Highlights card ──────────────────────────────────────────────────
    real_ph = sorted([(p,w) for p,w in phrases if w > 0], key=lambda x: -x[1])
    fake_ph = sorted([(p,w) for p,w in phrases if w < 0], key=lambda x:  x[1])

    real_pills = "".join(pill_html(p, w, True)  for p,w in real_ph) \
        or '<span style="font-size:0.78rem;color:#2a3045;">None detected</span>'
    fake_pills = "".join(pill_html(p, w, False) for p,w in fake_ph) \
        or '<span style="font-size:0.78rem;color:#2a3045;">None detected</span>'

    lime_inner = (
        section_label("Phrase Highlights — LIME Explainability") +
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;">'
        f'<div>'
        f'<div style="font-family:Space Mono,monospace;font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;color:#2ed573;margin-bottom:0.7rem;">↑ Credibility signals</div>'
        f'<div style="display:flex;flex-wrap:wrap;">{real_pills}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-family:Space Mono,monospace;font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;color:#ff4757;margin-bottom:0.7rem;">↓ Suspicion signals</div>'
        f'<div style="display:flex;flex-wrap:wrap;">{fake_pills}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html(lime_inner, "margin-top:1.5rem;"), unsafe_allow_html=True)

    # ── Raw signal values ─────────────────────────────────────────────────────
    with st.expander("Show raw signal values"):
        import pandas as pd
        df = pd.DataFrame([
            {
                "Signal": k,
                "Score":  round(v, 4),
                "Level":  "High" if v >= 0.65 else "Medium" if v >= 0.35 else "Low",
            }
            for k, v in signals.items()
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-family:Space Mono,monospace;font-size:0.55rem;letter-spacing:0.2em;'
    'color:#1a2030;text-align:center;padding:3rem 0 1rem;text-transform:uppercase;">'
    'TruthLens &nbsp;·&nbsp; XGBoost + spaCy + LIME &nbsp;·&nbsp;'
    '5-Signal Credibility Engine &nbsp;·&nbsp; Swapnil Hazra · 2026'
    '</div>',
    unsafe_allow_html=True,
)