"""Global CSS + decorative HTML for Streamlit.

Professional refresh:
  - Calm dark-glass surface (single subtle aurora, no orbiting decorations).
  - Reusable primitives: status pill, confidence ring (SVG donut), badge,
    agent status strip, color-coded LIME chip, comparison panel.
  - Hero stays branded but is information-dense (chip + title + subtitle +
    one stat row — no redundant KPI grid).
"""
from __future__ import annotations

import html
import math


def _html(body: str) -> None:
    """Inject raw HTML; uses st.html when available to avoid markdown reflow."""
    import streamlit as st
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


# =====================================================================
# THEME
# =====================================================================
def inject_theme() -> None:
    """Inject fonts + responsive theme (call once per run, after set_page_config)."""
    _html(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --mh-bg0: #070912;
  --mh-bg1: #0c1326;
  --mh-bg2: #14143a;
  --mh-card: rgba(255,255,255,0.045);
  --mh-card-strong: rgba(255,255,255,0.075);
  --mh-card-border: rgba(148,163,184,0.14);
  --mh-card-border-strong: rgba(148,163,184,0.26);
  --mh-text: #f1f5f9;
  --mh-text-soft: #cbd5e1;
  --mh-muted: #94a3b8;
  --mh-dim: #64748b;
  --mh-indigo: #818cf8;
  --mh-indigo-strong: #6366f1;
  --mh-violet: #a855f7;
  --mh-emerald: #34d399;
  --mh-amber: #fbbf24;
  --mh-rose: #fb7185;
  --mh-sky: #38bdf8;
  --mh-grad-primary: linear-gradient(135deg,#6366f1 0%,#8b5cf6 60%,#ec4899 100%);
  --mh-shadow: 0 12px 36px rgba(2,6,23,0.45), 0 2px 8px rgba(99,102,241,0.10);
  --mh-shadow-strong: 0 22px 70px rgba(2,6,23,0.55), 0 12px 32px rgba(99,102,241,0.14);
}

/* ===== Page ===== */
html, body, [data-testid="stAppViewContainer"], .stApp {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.stApp {
  background:
    radial-gradient(1100px 540px at 8% -10%, rgba(99,102,241,0.22), transparent 60%),
    radial-gradient(900px 480px at 100% 8%, rgba(168,85,247,0.16), transparent 60%),
    linear-gradient(165deg, var(--mh-bg0) 0%, var(--mh-bg1) 50%, var(--mh-bg2) 100%) fixed !important;
  background-size: cover !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { background: rgba(15,23,42,0.55) !important; border-radius: 0 0 12px 12px; }
section.main { position: relative; z-index: 1; }
section.main > div.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 3rem !important;
  max-width: min(1240px, 100%) !important;
}

/* Markdown text */
[data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
  color: var(--mh-text);
}
.stMarkdown p { line-height: 1.65; }
h1, h2, h3, h4 {
  font-family: 'Inter', sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--mh-text);
}
h2 { font-weight: 700; }
h3 { font-weight: 650; }

/* ===== Inputs ===== */
[data-testid="stTextArea"] textarea {
  background: rgba(8, 13, 28, 0.85) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(148,163,184,0.20) !important;
  border-radius: 12px !important;
  font-size: 0.95rem !important;
  padding: 0.8rem 1rem !important;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: rgba(129,140,248,0.65) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.20) !important;
}
[data-testid="stFileUploader"] section {
  background: rgba(8,13,28,0.55) !important;
  border: 1.5px dashed rgba(148,163,184,0.30) !important;
  border-radius: 12px !important;
  transition: border-color 160ms ease, background 160ms ease;
}
[data-testid="stFileUploader"] section:hover {
  border-color: rgba(129,140,248,0.65) !important;
  background: rgba(15,23,42,0.85) !important;
}
[data-testid="stFileUploader"] small { color: var(--mh-muted) !important; }
[data-testid="stCameraInput"] > div { border-radius: 12px !important; overflow: hidden; }

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: rgba(255,255,255,0.04);
  padding: 5px;
  border-radius: 12px;
  border: 1px solid var(--mh-card-border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  padding: 0.5rem 0.95rem !important;
  color: var(--mh-text-soft) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  transition: background 160ms ease, color 160ms ease;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(129,140,248,0.10) !important; color: #fff !important; }
.stTabs [aria-selected="true"] {
  background: var(--mh-grad-primary) !important;
  color: #fff !important;
  box-shadow: 0 6px 18px rgba(99,102,241,0.30);
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

/* ===== Expanders ===== */
div[data-testid="stExpander"] {
  background: var(--mh-card) !important;
  border: 1px solid var(--mh-card-border) !important;
  border-radius: 14px !important;
  transition: border-color 160ms ease, background 160ms ease;
}
div[data-testid="stExpander"]:hover {
  border-color: rgba(129,140,248,0.32) !important;
  background: var(--mh-card-strong) !important;
}
div[data-testid="stExpander"] summary { font-weight: 600 !important; color: var(--mh-text) !important; }

/* ===== Metrics (kept neutral/professional) ===== */
.stMetric, [data-testid="stMetric"] {
  background: rgba(15,23,42,0.55);
  border: 1px solid var(--mh-card-border);
  border-radius: 14px;
  padding: 0.95rem 1.1rem;
}
[data-testid="stMetricValue"] {
  color: var(--mh-text) !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
  color: var(--mh-muted) !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.7rem !important;
}
[data-testid="stMetricDelta"] { color: var(--mh-emerald) !important; font-weight: 600 !important; }

/* ===== Alerts ===== */
.stAlert { border-radius: 12px !important; }

/* ===== Buttons ===== */
.stButton > button {
  font-weight: 600 !important;
  border-radius: 10px !important;
  transition: transform 140ms ease, box-shadow 200ms ease, filter 200ms ease;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

.stButton > button[kind="primary"] {
  background: var(--mh-grad-primary) !important;
  border: none !important;
  color: #fff !important;
  padding: 0.7rem 1.4rem !important;
  letter-spacing: 0.01em;
  box-shadow: 0 6px 22px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.18) !important;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.07); }
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--mh-card-border) !important;
  color: var(--mh-text-soft) !important;
  padding: 0.7rem 1.2rem !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: rgba(129,140,248,0.55) !important;
  background: rgba(129,140,248,0.10) !important;
  color: #fff !important;
}
.stDownloadButton > button {
  background: rgba(129,140,248,0.12) !important;
  border: 1px solid rgba(129,140,248,0.40) !important;
  color: #e0e7ff !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}
.stDownloadButton > button:hover {
  background: rgba(129,140,248,0.22) !important;
  border-color: rgba(129,140,248,0.7) !important;
}

/* ===== Code blocks ===== */
[data-testid="stCodeBlock"] pre, code {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  background: rgba(7,11,26,0.7) !important;
  border-radius: 10px !important;
  border: 1px solid rgba(148,163,184,0.14) !important;
  font-size: 0.82rem !important;
}

/* ===== Captions ===== */
.stCaption, [data-testid="stCaptionContainer"] { color: var(--mh-muted) !important; }

/* =====================================================================
   HERO
===================================================================== */
.mh-hero {
  position: relative;
  padding: 1.6rem 1.75rem 1.4rem;
  margin: 0.25rem 0 1rem;
  border-radius: 18px;
  background:
    radial-gradient(120% 90% at 0% 0%, rgba(99,102,241,0.20) 0%, transparent 60%),
    radial-gradient(120% 90% at 100% 0%, rgba(236,72,153,0.13) 0%, transparent 60%),
    linear-gradient(160deg, rgba(15,23,42,0.55) 0%, rgba(30,27,75,0.40) 100%);
  border: 1px solid rgba(148,163,184,0.18);
  box-shadow: var(--mh-shadow);
  overflow: hidden;
}
.mh-hero-chip {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.28rem 0.7rem; border-radius: 999px;
  background: rgba(52,211,153,0.10);
  border: 1px solid rgba(52,211,153,0.36);
  color: #6ee7b7; font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.10em; text-transform: uppercase;
  margin-bottom: 0.85rem;
}
.mh-hero-chip .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #34d399; box-shadow: 0 0 0 0 rgba(52,211,153,0.55);
  animation: mh-pulse 2.4s ease-out infinite;
}
@keyframes mh-pulse {
  0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
  70% { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
  100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}
.mh-hero-title {
  font-size: clamp(1.6rem, 4vw, 2.2rem);
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.025em;
  background: linear-gradient(120deg, #f8fafc 0%, #c7d2fe 40%, #a5b4fc 70%, #f0abfc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.55rem 0;
}
.mh-hero-sub {
  color: var(--mh-text-soft);
  font-size: 0.93rem;
  max-width: 60rem;
  line-height: 1.6;
  margin: 0;
}
.mh-stat-strip {
  display: flex; flex-wrap: wrap;
  gap: 0.5rem 0.75rem;
  margin-top: 1rem;
}
.mh-stat {
  display: inline-flex; align-items: center; gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  border-radius: 8px;
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(148,163,184,0.14);
  color: var(--mh-text-soft);
  font-size: 0.76rem; font-weight: 600;
}
.mh-stat .icon {
  width: 20px; height: 20px;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.8rem;
  background: rgba(129,140,248,0.18);
  border: 1px solid rgba(129,140,248,0.30);
}
.mh-stat .val { color: var(--mh-text); font-weight: 700; }

/* =====================================================================
   CRISIS BANNER (SafetyAgent override)
===================================================================== */
.mh-crisis-banner {
  position: relative;
  margin: 0.4rem 0 1rem;
  padding: 1.1rem 1.25rem 1rem;
  border-radius: 16px;
  background:
    radial-gradient(120% 80% at 100% 0%, rgba(239,68,68,0.22) 0%, transparent 60%),
    linear-gradient(135deg, rgba(190,18,60,0.28) 0%, rgba(127,29,29,0.55) 100%);
  border: 1.5px solid rgba(248,113,133,0.55);
  box-shadow:
    0 14px 40px rgba(190,18,60,0.32),
    inset 0 1px 0 rgba(255,255,255,0.06);
  color: #fecaca;
  animation: mh-crisis-pulse 2.6s ease-in-out infinite;
  overflow: hidden;
}
@keyframes mh-crisis-pulse {
  0%, 100% { box-shadow: 0 14px 40px rgba(190,18,60,0.30), inset 0 1px 0 rgba(255,255,255,0.06); }
  50%      { box-shadow: 0 18px 52px rgba(248,113,133,0.55), inset 0 1px 0 rgba(255,255,255,0.10); }
}
.mh-crisis-head {
  display: flex; align-items: center; gap: 0.6rem;
  margin-bottom: 0.45rem;
}
.mh-crisis-head .ico { font-size: 1.45rem; line-height: 1; }
.mh-crisis-head .title {
  color: #fff; font-weight: 800;
  font-size: 1.08rem; letter-spacing: -0.01em;
}
.mh-crisis-head .badge {
  margin-left: auto;
  font-size: 0.66rem; font-weight: 800; letter-spacing: 0.16em;
  text-transform: uppercase;
  background: rgba(255,255,255,0.10);
  color: #fff;
  padding: 0.25rem 0.6rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
}
.mh-crisis-body {
  color: #fee2e2; font-size: 0.92rem; line-height: 1.55;
  margin-bottom: 0.7rem;
}
.mh-crisis-body strong { color: #fff; }
.mh-crisis-detected {
  display: flex; flex-wrap: wrap; gap: 0.35rem;
  margin-bottom: 0.85rem;
}
.mh-crisis-detected .mh-pill { color: #fff !important; background: rgba(255,255,255,0.10) !important; border-color: rgba(255,255,255,0.22) !important; }
.mh-crisis-hotlines {
  display: grid; grid-template-columns: 1fr; gap: 0.35rem;
  border-top: 1px dashed rgba(248,113,133,0.45);
  padding-top: 0.75rem;
}
.mh-hotline-row {
  display: grid;
  grid-template-columns: 100px 1fr auto;
  gap: 0.7rem; align-items: center;
  font-size: 0.85rem; color: #fee2e2;
}
.mh-hotline-row .region {
  font-size: 0.66rem; font-weight: 800;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: #fda4af;
}
.mh-hotline-row .name { color: #fff; font-weight: 600; }
.mh-hotline-row .num {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  background: rgba(255,255,255,0.12);
  padding: 0.18rem 0.55rem; border-radius: 6px;
  color: #fff; font-weight: 700; font-size: 0.78rem;
  white-space: nowrap;
}

/* =====================================================================
   AGENT STATUS STRIP (replaces decorative orbits)
===================================================================== */
.mh-agent-strip {
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  gap: 0.5rem;
  margin: 0.65rem 0 1rem;
}
.mh-agent-card {
  padding: 0.65rem 0.7rem 0.6rem;
  border-radius: 12px;
  background: rgba(15,23,42,0.55);
  border: 1px solid var(--mh-card-border);
  text-align: left;
  transition: border-color 160ms ease, transform 160ms ease;
}
.mh-agent-card:hover { transform: translateY(-1px); border-color: rgba(129,140,248,0.34); }
.mh-agent-card .head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 0.3rem;
}
.mh-agent-card .num {
  font-size: 0.62rem; font-weight: 800; letter-spacing: 0.14em;
  color: var(--mh-muted); text-transform: uppercase;
}
.mh-agent-card .ico { font-size: 0.95rem; opacity: 0.95; }
.mh-agent-card .name {
  color: var(--mh-text); font-weight: 700; font-size: 0.82rem;
  letter-spacing: -0.01em;
}
.mh-agent-card .sub {
  color: var(--mh-muted); font-size: 0.7rem; margin-top: 0.15rem;
}
.mh-agent-card.ok { border-left: 2px solid var(--mh-emerald); }
.mh-agent-card.warn { border-left: 2px solid var(--mh-amber); }
.mh-agent-card.off { border-left: 2px solid var(--mh-rose); opacity: 0.78; }

/* =====================================================================
   SECTION LABELS
===================================================================== */
.mh-section-label {
  color: var(--mh-muted) !important;
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  margin: 1rem 0 0.6rem 0 !important;
  display: flex; align-items: center; gap: 0.5rem;
}
.mh-section-label::before {
  content: ""; display: inline-block;
  width: 16px; height: 2px; border-radius: 2px;
  background: var(--mh-grad-primary);
}

/* =====================================================================
   INPUT SHELL (panel around the two-column input)
===================================================================== */
.mh-input-shell {
  position: relative;
  border-radius: 16px;
  padding: 0.85rem 0.95rem 0.7rem;
  margin: 0.4rem 0 1rem;
  background: linear-gradient(160deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.012) 100%);
  border: 1px solid rgba(148,163,184,0.18);
  box-shadow: var(--mh-shadow);
}

/* =====================================================================
   DECISION SNAPSHOT (executive summary card)
===================================================================== */
.mh-decision {
  position: relative;
  padding: 1.1rem 1.25rem 1rem;
  border-radius: 18px;
  background: linear-gradient(160deg, rgba(15,23,42,0.78) 0%, rgba(30,27,75,0.55) 100%);
  border: 1px solid rgba(148,163,184,0.20);
  box-shadow: var(--mh-shadow-strong);
  overflow: hidden;
  margin-bottom: 0.9rem;
}
.mh-decision::after {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(60% 50% at 100% 0%, rgba(168,85,247,0.10), transparent 70%);
  pointer-events: none;
}
.mh-decision-grid {
  position: relative; z-index: 1;
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 1.25rem;
  align-items: center;
}
.mh-decision-headline {
  display: flex; flex-direction: column; gap: 0.45rem;
}
.mh-decision-status {
  font-size: clamp(1.4rem, 3vw, 1.8rem);
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.02em;
  margin: 0;
}
.mh-decision-status.stable { color: #6ee7b7; }
.mh-decision-status.high { color: #fb7185; }
.mh-decision-status.likely { color: #fbbf24; }
.mh-decision-status.conflict { color: #fbbf24; }
.mh-decision-status.mixed { color: #38bdf8; }
.mh-decision-explanation {
  color: var(--mh-text-soft); font-size: 0.92rem; line-height: 1.55; margin: 0;
}
.mh-decision-badges {
  display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.35rem;
}

/* =====================================================================
   STATUS PILLS (reusable)
===================================================================== */
.mh-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  border: 1px solid var(--mh-card-border);
  background: rgba(255,255,255,0.04);
  color: var(--mh-text-soft);
  white-space: nowrap;
}
.mh-pill strong { color: #f8fafc; font-weight: 700; }
.mh-pill.success { border-color: rgba(52,211,153,0.40); background: rgba(52,211,153,0.10); color: #6ee7b7; }
.mh-pill.warn { border-color: rgba(251,191,36,0.40); background: rgba(251,191,36,0.10); color: #fcd34d; }
.mh-pill.danger { border-color: rgba(251,113,133,0.40); background: rgba(251,113,133,0.10); color: #fda4af; }
.mh-pill.info { border-color: rgba(56,189,248,0.40); background: rgba(56,189,248,0.10); color: #7dd3fc; }
.mh-pill.muted { background: rgba(148,163,184,0.08); color: var(--mh-muted); }

/* =====================================================================
   CONFIDENCE RING (SVG donut wrapper)
===================================================================== */
.mh-ring-wrap {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center;
}
.mh-ring-wrap .ring-label {
  margin-top: 0.3rem;
  color: var(--mh-muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.mh-ring-wrap .ring-value {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  pointer-events: none;
}
.mh-ring-wrap .ring-value .pct {
  font-size: 1.45rem; font-weight: 800;
  color: var(--mh-text); letter-spacing: -0.02em;
}
.mh-ring-wrap .ring-value .lbl {
  font-size: 0.62rem; color: var(--mh-muted);
  letter-spacing: 0.16em; text-transform: uppercase; font-weight: 700;
}

/* =====================================================================
   MINI METRIC CARDS (text/face/fusion summaries)
===================================================================== */
.mh-mini-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
}
.mh-mini-card {
  padding: 0.85rem 0.95rem 0.75rem;
  border-radius: 14px;
  background: rgba(15,23,42,0.55);
  border: 1px solid var(--mh-card-border);
  transition: border-color 160ms ease, transform 160ms ease;
}
.mh-mini-card:hover { transform: translateY(-1px); border-color: rgba(129,140,248,0.30); }
.mh-mini-card .label {
  color: var(--mh-muted);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 0.66rem;
  font-weight: 700;
}
.mh-mini-card .value {
  margin-top: 0.35rem;
  color: var(--mh-text);
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.mh-mini-card .note {
  margin-top: 0.35rem;
  color: var(--mh-text-soft);
  font-size: 0.78rem;
  line-height: 1.5;
}
.mh-mini-card .meter {
  margin-top: 0.7rem;
  height: 6px; border-radius: 999px;
  background: rgba(255,255,255,0.07);
  overflow: hidden;
}
.mh-mini-card .meter > span {
  display: block; height: 100%; border-radius: inherit;
  background: var(--mh-grad-primary);
}
.mh-mini-card.text .meter > span { background: linear-gradient(90deg,#f59e0b 0%,#ec4899 100%); }
.mh-mini-card.face .meter > span { background: linear-gradient(90deg,#38bdf8 0%,#818cf8 100%); }
.mh-mini-card.fusion .meter > span { background: linear-gradient(90deg,#34d399 0%,#22c55e 100%); }

/* =====================================================================
   COMPARISON PANEL (CNN vs DeepFace verdict)
===================================================================== */
.mh-compare {
  border: 1px solid var(--mh-card-border);
  background: rgba(15,23,42,0.55);
  border-radius: 14px;
  padding: 1rem 1.1rem;
}
.mh-compare-title {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--mh-muted);
  margin: 0 0 0.7rem 0;
}
.mh-compare-row {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 0.85rem;
  align-items: stretch;
}
.mh-compare-side {
  padding: 0.7rem 0.85rem;
  border-radius: 10px;
  background: rgba(8,13,28,0.55);
  border: 1px solid var(--mh-card-border);
}
.mh-compare-side.win { border-color: rgba(52,211,153,0.45); background: rgba(52,211,153,0.06); }
.mh-compare-side .src {
  font-size: 0.65rem; color: var(--mh-muted); font-weight: 700;
  letter-spacing: 0.16em; text-transform: uppercase;
}
.mh-compare-side .emo {
  margin-top: 0.25rem; color: var(--mh-text);
  font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em;
}
.mh-compare-side .conf {
  margin-top: 0.18rem; color: var(--mh-text-soft); font-size: 0.78rem;
}
.mh-compare-vs {
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.18em;
  color: var(--mh-muted); text-transform: uppercase;
}
.mh-compare-verdict {
  margin-top: 0.85rem; padding: 0.55rem 0.7rem;
  border-radius: 9px;
  background: rgba(129,140,248,0.10);
  border: 1px solid rgba(129,140,248,0.30);
  color: var(--mh-text-soft); font-size: 0.82rem;
}
.mh-compare-verdict strong { color: #c7d2fe; }

/* =====================================================================
   AGENT DETAIL CARDS (used in tab 2)
===================================================================== */
.mh-detail-card {
  padding: 0.95rem 1.1rem;
  border-radius: 14px;
  background: rgba(15,23,42,0.55);
  border: 1px solid var(--mh-card-border);
  margin-bottom: 0.7rem;
}
.mh-detail-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.5rem;
}
.mh-detail-title {
  display: flex; align-items: center; gap: 0.55rem;
}
.mh-detail-title .ix {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px;
  border-radius: 8px;
  background: rgba(129,140,248,0.16);
  border: 1px solid rgba(129,140,248,0.30);
  color: #c7d2fe;
  font-size: 0.78rem; font-weight: 800;
}
.mh-detail-title .name {
  color: var(--mh-text); font-weight: 700; font-size: 0.96rem;
}
.mh-detail-desc {
  margin: 0.5rem 0 0.65rem;
  color: var(--mh-text-soft); font-size: 0.85rem; line-height: 1.55;
}
.mh-kv {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.4rem 0.8rem;
}
.mh-kv .k {
  color: var(--mh-muted); font-size: 0.66rem;
  letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700;
}
.mh-kv .v {
  color: var(--mh-text); font-size: 0.88rem; font-weight: 600; margin-top: 0.05rem;
}

/* =====================================================================
   LIME CHIPS (color-coded) + highlighted text
===================================================================== */
.mh-lime-row {
  display: flex; flex-wrap: wrap; gap: 0.4rem;
  padding: 0.5rem 0.1rem 0.2rem;
}
.mh-lime-chip {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.32rem 0.65rem;
  border-radius: 999px;
  font-weight: 700;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.78rem;
  border: 1px solid rgba(255,255,255,0.10);
  color: #f8fafc;
  letter-spacing: 0.01em;
}
.mh-lime-chip .w {
  font-size: 0.66rem; opacity: 0.85; font-weight: 600;
}
.mh-lime-legend {
  display: flex; gap: 0.55rem; flex-wrap: wrap;
  font-size: 0.74rem; color: var(--mh-muted); margin-bottom: 0.5rem;
}
.mh-lime-legend .swatch {
  display: inline-block; width: 14px; height: 14px;
  border-radius: 4px; vertical-align: -2px; margin-right: 0.3rem;
}

.mh-lime-text {
  margin-top: 0.7rem;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  background: rgba(8,13,28,0.6);
  border: 1px solid var(--mh-card-border);
  color: var(--mh-text);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.84rem;
  line-height: 1.85;
  word-spacing: 0.04em;
  white-space: pre-wrap;
}
.mh-lime-text .hl {
  padding: 0.05rem 0.3rem; border-radius: 5px; margin: 0 1px;
  font-weight: 700;
}

/* =====================================================================
   SIDEBAR
===================================================================== */
[data-testid="stSidebar"] {
  background:
    radial-gradient(120% 60% at 0% 0%, rgba(99,102,241,0.18), transparent 60%),
    linear-gradient(180deg, rgba(7,11,26,0.96) 0%, rgba(20,12,50,0.92) 100%) !important;
  border-right: 1px solid var(--mh-card-border) !important;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
  color: #e2e8f0 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: #e0e7ff !important;
}
.mh-sb-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--mh-card-border);
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
  margin-bottom: 0.7rem;
}
.mh-sb-card .head {
  font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--mh-muted); font-weight: 700; margin-bottom: 0.5rem;
}
.mh-sb-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.28rem 0;
  border-bottom: 1px dashed rgba(148,163,184,0.10);
}
.mh-sb-row:last-child { border-bottom: none; }
.mh-sb-row .k { color: var(--mh-text-soft); font-size: 0.78rem; }
.mh-sb-row .v { color: #f8fafc; font-weight: 700; font-size: 0.78rem; }

/* =====================================================================
   EMPTY HINT + FOOTER
===================================================================== */
.mh-empty-hint {
  margin: 1rem 0;
  padding: 1.5rem 1.5rem;
  text-align: center;
  border-radius: 16px;
  background:
    radial-gradient(120% 80% at 50% 0%, rgba(99,102,241,0.16) 0%, transparent 60%),
    linear-gradient(145deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.05) 50%, rgba(15,23,42,0.30) 100%);
  border: 1px solid rgba(129,140,248,0.28);
  color: var(--mh-text-soft);
  font-size: 0.95rem;
  line-height: 1.6;
}
.mh-empty-hint .icon {
  font-size: 1.6rem; line-height: 1;
  display: inline-block; margin-bottom: 0.5rem;
}
.mh-empty-hint strong { color: #f1f5f9; font-weight: 700; }
.mh-footer {
  margin-top: 1.4rem;
  padding: 0.95rem 1.1rem;
  border-radius: 12px;
  text-align: center;
  background: rgba(7,11,26,0.55);
  border: 1px solid rgba(148,163,184,0.14);
  color: var(--mh-muted);
  font-size: 0.78rem;
  line-height: 1.65;
}
.mh-footer strong { color: #e2e8f0; }

/* =====================================================================
   RESPONSIVE
===================================================================== */
@media (max-width: 960px) {
  .mh-decision-grid { grid-template-columns: 1fr; }
  .mh-mini-grid { grid-template-columns: 1fr; }
  .mh-agent-strip { grid-template-columns: repeat(3, 1fr); }
  .mh-compare-row { grid-template-columns: 1fr; }
  .mh-compare-vs { display: none; }
}
@media (max-width: 600px) {
  .mh-agent-strip { grid-template-columns: repeat(2, 1fr); }
  .mh-stat { font-size: 0.7rem; padding: 0.32rem 0.65rem; }
}

/* =====================================================================
   STREAMLIT ⋮ MENU — Light theme (System / Light / Dark)
   Base UI variables stay dark on :root; Light selection sets html[data-theme="light"]
   on the document so these overrides actually take effect.
===================================================================== */
html[data-theme="light"] {
  --mh-bg0: #f8fafc;
  --mh-bg1: #f1f5f9;
  --mh-bg2: #e8eef5;
  --mh-card: rgba(15, 23, 42, 0.045);
  --mh-card-strong: rgba(15, 23, 42, 0.08);
  --mh-card-border: rgba(51, 65, 85, 0.16);
  --mh-card-border-strong: rgba(51, 65, 85, 0.26);
  --mh-text: #0f172a;
  --mh-text-soft: #334155;
  --mh-muted: #64748b;
  --mh-dim: #94a3b8;
  --mh-shadow: 0 12px 36px rgba(15, 23, 42, 0.08), 0 2px 8px rgba(99, 102, 241, 0.06);
  --mh-shadow-strong: 0 22px 70px rgba(15, 23, 42, 0.10), 0 12px 32px rgba(99, 102, 241, 0.08);
}
html[data-theme="light"] .stApp {
  background:
    radial-gradient(1100px 540px at 8% -10%, rgba(99, 102, 241, 0.14), transparent 60%),
    radial-gradient(900px 480px at 100% 8%, rgba(168, 85, 247, 0.10), transparent 60%),
    linear-gradient(165deg, var(--mh-bg0) 0%, var(--mh-bg1) 50%, var(--mh-bg2) 100%) fixed !important;
}
html[data-theme="light"] [data-testid="stToolbar"] {
  background: rgba(241, 245, 249, 0.92) !important;
  border: 1px solid rgba(148, 163, 184, 0.25);
}
html[data-theme="light"] [data-testid="stTextArea"] textarea {
  background: rgba(255, 255, 255, 0.98) !important;
  color: #0f172a !important;
  border: 1px solid rgba(51, 65, 85, 0.22) !important;
}
html[data-theme="light"] [data-testid="stFileUploader"] section {
  background: rgba(255, 255, 255, 0.75) !important;
  border: 1.5px dashed rgba(100, 116, 139, 0.35) !important;
}
html[data-theme="light"] [data-testid="stFileUploader"] section:hover {
  background: rgba(248, 250, 252, 0.95) !important;
}
html[data-theme="light"] .stMetric,
html[data-theme="light"] [data-testid="stMetric"] {
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid var(--mh-card-border);
}
html[data-theme="light"] .stTabs [data-baseweb="tab"]:hover {
  color: #0f172a !important;
}
html[data-theme="light"] .stButton > button[kind="secondary"] {
  color: var(--mh-text-soft) !important;
}
html[data-theme="light"] .stButton > button[kind="secondary"]:hover {
  color: #0f172a !important;
}
html[data-theme="light"] [data-testid="stCodeBlock"] pre,
html[data-theme="light"] [data-testid="stCodeBlock"] code {
  background: rgba(241, 245, 249, 0.95) !important;
  border: 1px solid rgba(148, 163, 184, 0.35) !important;
  color: #0f172a !important;
}
html[data-theme="light"] .mh-hero {
  background:
    radial-gradient(120% 90% at 0% 0%, rgba(99, 102, 241, 0.12) 0%, transparent 60%),
    radial-gradient(120% 90% at 100% 0%, rgba(236, 72, 153, 0.08) 0%, transparent 60%),
    linear-gradient(160deg, rgba(255, 255, 255, 0.9) 0%, rgba(241, 245, 249, 0.85) 100%);
  border: 1px solid rgba(148, 163, 184, 0.28);
}
html[data-theme="light"] .mh-hero-title {
  background: linear-gradient(120deg, #0f172a 0%, #4338ca 38%, #7c3aed 72%, #be185d 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
html[data-theme="light"] .mh-stat {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(148, 163, 184, 0.24);
}
html[data-theme="light"] .mh-empty-hint {
  background:
    radial-gradient(120% 80% at 50% 0%, rgba(99, 102, 241, 0.10) 0%, transparent 60%),
    linear-gradient(145deg, rgba(99, 102, 241, 0.06) 0%, rgba(241, 245, 249, 0.95) 100%);
  border: 1px solid rgba(129, 140, 248, 0.22);
}
html[data-theme="light"] .mh-empty-hint strong {
  color: #0f172a;
}
html[data-theme="light"] .mh-footer {
  background: rgba(241, 245, 249, 0.85);
  border: 1px solid rgba(148, 163, 184, 0.28);
}
html[data-theme="light"] .mh-footer strong {
  color: #1e293b;
}
html[data-theme="light"] .mh-sb-row .v {
  color: #0f172a;
}
</style>
        """
    )


# =====================================================================
# HERO
# =====================================================================
def render_hero() -> None:
    """Compact, professional hero — one chip, gradient title, one stat row."""
    _html(
        """
<div class="mh-hero">
  <div class="mh-hero-chip"><span class="dot"></span> Live · Multi-agent pipeline</div>
  <h1 class="mh-hero-title">Multimodal mental-health signal assistant</h1>
  <p class="mh-hero-sub">
    Three pathways run in parallel — language understanding, face emotion (CNN),
    and a pretrained benchmark — then comparison, fusion, supervision, and
    explainability. Multimodal AI demo — experimental results for educational analysis only.
  </p>
  <div class="mh-stat-strip">
    <span class="mh-stat"><span class="icon">🧠</span>Agents <span class="val">9</span></span>
    <span class="mh-stat"><span class="icon">⚡</span>Parallel inference</span>
    <span class="mh-stat"><span class="icon">🧪</span>HPT at train-time</span>
    <span class="mh-stat"><span class="icon">🔍</span>LIME · XAI</span>
  </div>
</div>
        """
    )


# =====================================================================
# AGENT STATUS STRIP (replaces decorative orbits)
# =====================================================================
def render_agent_strip(df_backend: str) -> None:
    """Compact strip showing the seven agents and their runtime status."""
    backend_safe = html.escape(str(df_backend))
    df_state = "ok" if df_backend in ("deepface", "hf-vit") else "off"
    df_sub = backend_safe if df_backend != "none" else "offline"

    cards = [
        ("01", "📝", "NLP", "TF-IDF + LR", "ok"),
        ("02", "🧪", "HPT", "train-time", "ok"),
        ("03", "👁️", "CV (CNN)", "PyTorch · 48×48", "ok"),
        ("04", "🔮", "DeepFace", df_sub, df_state),
        ("05", "⚖️", "Comparison", "CNN vs DF", "ok"),
        ("06", "🧬", "Fusion", "text + face", "ok"),
        ("07", "🚨", "Safety", "crisis lexicon", "ok"),
        ("08", "🛡️", "Supervisor", "+ XAI / LIME", "ok"),
    ]
    parts = ['<div class="mh-agent-strip">']
    for ix, ico, name, sub, state in cards:
        parts.append(
            f"""
<div class="mh-agent-card {state}">
  <div class="head"><span class="num">{ix}</span><span class="ico">{ico}</span></div>
  <div class="name">{name}</div>
  <div class="sub">{sub}</div>
</div>"""
        )
    parts.append("</div>")
    _html("".join(parts))


# =====================================================================
# REUSABLE PRIMITIVES (used by streamlit_ui.py)
# =====================================================================
def status_pill(text: str, kind: str = "muted") -> str:
    """Return HTML for an inline pill. `kind` ∈ success/warn/danger/info/muted."""
    kind = kind if kind in {"success", "warn", "danger", "info", "muted"} else "muted"
    return f'<span class="mh-pill {kind}">{html.escape(text)}</span>'


def confidence_ring(pct: float, label: str = "Confidence", color: str = "#818cf8",
                    size: int = 138, stroke: int = 12) -> str:
    """SVG donut showing a percentage 0..100 with a centered numeric label."""
    pct = max(0.0, min(100.0, float(pct)))
    radius = (size / 2) - (stroke / 2) - 2
    circumference = 2 * math.pi * radius
    dash = (pct / 100.0) * circumference
    gap = circumference - dash
    cx = cy = size / 2
    return f"""
<div class="mh-ring-wrap" style="position:relative;width:{size}px;height:{size}px;">
  <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
    <defs>
      <linearGradient id="mh-ring-g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="{color}" stop-opacity="0.95"/>
        <stop offset="100%" stop-color="#ec4899" stop-opacity="0.95"/>
      </linearGradient>
    </defs>
    <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none"
            stroke="rgba(148,163,184,0.18)" stroke-width="{stroke}"/>
    <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none"
            stroke="url(#mh-ring-g)" stroke-width="{stroke}"
            stroke-dasharray="{dash:.2f} {gap:.2f}"
            stroke-linecap="round"
            transform="rotate(-90 {cx} {cy})"/>
  </svg>
  <div class="ring-value">
    <div class="pct">{pct:.0f}%</div>
    <div class="lbl">{html.escape(label)}</div>
  </div>
</div>
"""


def lime_chip(word: str, weight: float) -> str:
    """Color-coded LIME chip: positive weight → red (Negative cue), negative → green."""
    a = max(0.05, min(1.0, abs(weight) * 4.0))
    if weight > 0:
        bg = f"rgba(251,113,133,{a:.2f})"
        bd = "rgba(251,113,133,0.55)"
    else:
        bg = f"rgba(52,211,153,{a:.2f})"
        bd = "rgba(52,211,153,0.55)"
    return (
        f'<span class="mh-lime-chip" '
        f'style="background:{bg};border-color:{bd};">'
        f'{html.escape(word)}'
        f'<span class="w">{weight:+.2f}</span></span>'
    )


def lime_highlighted_text(cleaned_text: str, weights: dict[str, float]) -> str:
    """Render the cleaned text with words tinted by LIME weight."""
    if not cleaned_text:
        return '<div class="mh-lime-text"><em style="color:#94a3b8">No text after cleaning.</em></div>'
    out_words: list[str] = []
    for tok in cleaned_text.split():
        w = weights.get(tok.lower())
        if w is None:
            out_words.append(html.escape(tok))
            continue
        a = max(0.10, min(0.85, abs(w) * 4.0))
        if w > 0:
            color = f"rgba(251,113,133,{a:.2f})"
        else:
            color = f"rgba(52,211,153,{a:.2f})"
        out_words.append(
            f'<span class="hl" style="background:{color};">{html.escape(tok)}</span>'
        )
    return '<div class="mh-lime-text">' + " ".join(out_words) + "</div>"


def lime_legend() -> str:
    return (
        '<div class="mh-lime-legend">'
        '<span><span class="swatch" style="background:rgba(251,113,133,0.55)"></span>pushes toward <strong style="color:#fb7185">Negative</strong></span>'
        '<span><span class="swatch" style="background:rgba(52,211,153,0.55)"></span>pushes toward <strong style="color:#34d399">Positive</strong></span>'
        '</div>'
    )


def comparison_panel(cnn_emotion: str, cnn_conf: float,
                     df_emotion: str | None, df_conf: float | None,
                     winner: str, agree: bool, note: str) -> str:
    """Side-by-side CNN vs DeepFace verdict card."""
    cnn_win = winner in {"agreement", "cnn"}
    df_win = winner in {"agreement", "deepface"}
    df_e = (df_emotion or "n/a").title()
    df_c = f"{df_conf * 100:.0f}%" if df_conf is not None else "n/a"
    verdict = (
        f"Models <strong>agree</strong> — high reliability."
        if agree else
        f"Models <strong>disagree</strong> — using <strong>{html.escape(winner)}</strong> by policy."
    )
    return f"""
<div class="mh-compare">
  <div class="mh-compare-title">Image-side comparison</div>
  <div class="mh-compare-row">
    <div class="mh-compare-side {'win' if cnn_win else ''}">
      <div class="src">CV · CNN (ours)</div>
      <div class="emo">{html.escape(cnn_emotion.title())}</div>
      <div class="conf">Confidence {cnn_conf * 100:.0f}%</div>
    </div>
    <div class="mh-compare-vs">vs</div>
    <div class="mh-compare-side {'win' if df_win else ''}">
      <div class="src">DeepFace · benchmark</div>
      <div class="emo">{html.escape(df_e)}</div>
      <div class="conf">Confidence {df_c}</div>
    </div>
  </div>
  <div class="mh-compare-verdict">{verdict} <span style="color:#94a3b8">— {html.escape(note)}</span></div>
</div>
"""


def crisis_banner(detected_phrases: list[tuple[str, str]],
                  hotlines: list[dict[str, str]]) -> str:
    """Top-of-results crisis banner with detected phrases + hotline list.

    `detected_phrases` is a list of (category_label, phrase) tuples.
    `hotlines` is a list of {region, name, number} dicts.
    """
    chips = "".join(
        f'<span class="mh-pill" title="{html.escape(cat)}">{html.escape(phrase)}</span>'
        for cat, phrase in detected_phrases[:8]
    )
    chips_html = (
        f'<div class="mh-crisis-detected">{chips}</div>' if chips else ''
    )

    hotline_rows = "".join(
        f'<div class="mh-hotline-row">'
        f'<span class="region">{html.escape(h["region"])}</span>'
        f'<span class="name">{html.escape(h["name"])}</span>'
        f'<span class="num">{html.escape(h["number"])}</span>'
        f'</div>'
        for h in hotlines
    )

    return f"""
<div class="mh-crisis-banner">
  <div class="mh-crisis-head">
    <span class="ico">🚨</span>
    <span class="title">Crisis language detected · safety override</span>
    <span class="badge">SafetyAgent</span>
  </div>
  <div class="mh-crisis-body">
    Your text contains language indicating possible crisis. The SafetyAgent
    has overridden the probabilistic models. <strong>This is an automated tool — please reach out to a real person now.</strong>
  </div>
  {chips_html}
  <div class="mh-crisis-hotlines">{hotline_rows}</div>
</div>
"""


def detail_card(ix: str, name: str, desc: str, kv: dict[str, str],
                pill_text: str | None = None, pill_kind: str = "muted") -> str:
    """Structured agent-detail card (replaces st.json dumps)."""
    pill_html = ""
    if pill_text:
        pill_html = status_pill(pill_text, pill_kind)
    rows = "".join(
        f'<div><div class="k">{html.escape(k)}</div><div class="v">{html.escape(str(v))}</div></div>'
        for k, v in kv.items()
    )
    return f"""
<div class="mh-detail-card">
  <div class="mh-detail-head">
    <div class="mh-detail-title">
      <span class="ix">{ix}</span><span class="name">{html.escape(name)}</span>
    </div>
    {pill_html}
  </div>
  <div class="mh-detail-desc">{desc}</div>
  <div class="mh-kv">{rows}</div>
</div>
"""
