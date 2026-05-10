"""Streamlit layout helpers — presentation only (no model logic).

Aligned with `system.py` and the agents in `agents/`. Keeps the page
information-dense without dumping raw JSON: structured agent cards,
real charts, color-coded XAI, side-by-side model comparison.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import (
    CNN_CLASS_NAMES_JSON,
    CNN_CONFUSION_PNG,
    CNN_EVAL_JSON,
    CNN_MODEL_PATH,
    CNN_TRAINING_META_JSON,
    LOW_CONFIDENCE_THRESHOLD,
    NLP_CONFUSION_PNG,
    NLP_EVAL_JSON,
    NLP_MODEL_PATH,
    NLP_THRESHOLD_JSON,
    NLP_VECTORIZER_PATH,
)
from agents.safety_lexicon import REGION_LABELS, get_hotlines_for_region
from agents.suggestions_agent import mode_label, suggestions_for
from streamlit_theme import (
    comparison_panel,
    confidence_ring,
    crisis_banner,
    detail_card,
    lime_chip,
    lime_highlighted_text,
    lime_legend,
    status_pill,
)
from system import (
    AnalysisResult,
    System,
    analysis_to_markdown,
    build_printable_report_html,
    format_report,
)


_CRISIS_CATEGORY_LABELS = {
    "suicidal_ideation": "suicide",
    "imminent_action": "imminent",
    "self_harm": "self-harm",
    "hopelessness": "hopelessness",
}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _html(body: str) -> None:
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def _sidebar_html(body: str) -> None:
    with st.sidebar:
        if hasattr(st, "html"):
            st.html(body)
        else:
            st.markdown(body, unsafe_allow_html=True)


def _status_class(name: str) -> str:
    """Map supervisor status → decision-card CSS modifier."""
    n = name.lower()
    if "crisis" in n:
        return "high"
    if "stable" in n:
        return "stable"
    if "likely" in n:
        return "likely"
    if "high" in n:
        return "high"
    if "conflict" in n:
        return "conflict"
    return "mixed"


def _status_kind(name: str) -> str:
    """Map supervisor status → pill kind."""
    n = name.lower()
    if "crisis" in n or "high" in n:
        return "danger"
    if "stable" in n:
        return "success"
    if "likely" in n or "conflict" in n:
        return "warn"
    return "info"


def _band_kind(band: str) -> str:
    return {"high": "success", "medium": "info", "low": "warn"}.get(band, "muted")


def _bytes_kb(p: Path) -> str:
    try:
        return f"{p.stat().st_size / 1024:.0f} KB"
    except Exception:  # noqa: BLE001
        return "—"


def _read_json_safe(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@st.cache_data(show_spinner=False)
def _load_cnn_meta() -> dict:
    return _read_json_safe(Path(CNN_TRAINING_META_JSON))


@st.cache_data(show_spinner=False)
def _load_classes() -> list[str]:
    data = _read_json_safe(Path(CNN_CLASS_NAMES_JSON))
    if isinstance(data, list):
        return [str(c) for c in data]
    try:
        # Older fallback: file contained a JSON list
        return list(json.loads(Path(CNN_CLASS_NAMES_JSON).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(show_spinner=False)
def _load_cnn_eval() -> dict:
    return _read_json_safe(Path(CNN_EVAL_JSON))


@st.cache_data(show_spinner=False)
def _load_nlp_eval() -> dict:
    return _read_json_safe(Path(NLP_EVAL_JSON))


@st.cache_data(show_spinner=False)
def _load_nlp_threshold() -> dict:
    return _read_json_safe(Path(NLP_THRESHOLD_JSON))


# ---------------------------------------------------------------------
# Sidebar — real model status
# ---------------------------------------------------------------------
def render_sidebar(system: System) -> None:
    meta = _load_cnn_meta()
    classes = _load_classes()
    backend = html.escape(str(system.deepface.backend))

    st.sidebar.markdown("### ⚙️ System status")

    val_acc = meta.get("best_val_acc")
    test_acc = meta.get("test_set_acc")
    val_acc_s = f"{val_acc * 100:.1f}%" if isinstance(val_acc, (int, float)) else "—"
    test_acc_s = f"{test_acc * 100:.1f}%" if isinstance(test_acc, (int, float)) else "—"
    epochs = meta.get("epochs_run", "—")
    n_train = meta.get("n_train_images", "—")

    pipeline_html = """
<div class="mh-sb-card">
  <div class="head">Pipeline</div>
  <div style="font-size:0.8rem;line-height:1.85;color:#e2e8f0;">
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">NLP</span>
    →
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">HPT</span>
    <span style="color:#64748b;font-size:0.72rem;">(train)</span>
    <br/>
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">CV</span>
    +
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">DeepFace</span>
    <br/>
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">Compare</span>
    →
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">Fusion</span>
    <br/>
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">Supervisor</span>
    →
    <span style="background:rgba(129,140,248,0.18);color:#c7d2fe;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.74rem;">XAI</span>
  </div>
</div>
"""
    _sidebar_html(pipeline_html)

    nlp_status = "loaded" if Path(NLP_MODEL_PATH).is_file() else "missing"
    nlp_kind = "success" if nlp_status == "loaded" else "danger"
    cnn_status = "loaded" if Path(CNN_MODEL_PATH).is_file() else "missing"
    cnn_kind = "success" if cnn_status == "loaded" else "danger"
    df_kind = "success" if backend in ("deepface", "hf-vit") else "warn"

    models_html = f"""
<div class="mh-sb-card">
  <div class="head">Models</div>
  <div class="mh-sb-row"><span class="k">NLP · LR</span><span class="v">{status_pill(nlp_status, nlp_kind)}</span></div>
  <div class="mh-sb-row"><span class="k">TF-IDF</span><span class="v">{_bytes_kb(Path(NLP_VECTORIZER_PATH))}</span></div>
  <div class="mh-sb-row"><span class="k">CNN weights</span><span class="v">{status_pill(cnn_status, cnn_kind)}</span></div>
  <div class="mh-sb-row"><span class="k">CNN file</span><span class="v">{_bytes_kb(Path(CNN_MODEL_PATH))}</span></div>
  <div class="mh-sb-row"><span class="k">DeepFace</span><span class="v">{status_pill(backend, df_kind)}</span></div>
</div>
"""
    _sidebar_html(models_html)

    quality_html = f"""
<div class="mh-sb-card">
  <div class="head">CNN training</div>
  <div class="mh-sb-row"><span class="k">Val accuracy</span><span class="v">{val_acc_s}</span></div>
  <div class="mh-sb-row"><span class="k">Test accuracy</span><span class="v">{test_acc_s}</span></div>
  <div class="mh-sb-row"><span class="k">Epochs</span><span class="v">{epochs}</span></div>
  <div class="mh-sb-row"><span class="k">Train images</span><span class="v">{n_train:,}</span> </div>
  <div class="mh-sb-row"><span class="k">Classes</span><span class="v">{len(classes) or '—'}</span></div>
</div>
""" if isinstance(n_train, int) else f"""
<div class="mh-sb-card">
  <div class="head">CNN training</div>
  <div class="mh-sb-row"><span class="k">Val accuracy</span><span class="v">{val_acc_s}</span></div>
  <div class="mh-sb-row"><span class="k">Test accuracy</span><span class="v">{test_acc_s}</span></div>
  <div class="mh-sb-row"><span class="k">Epochs</span><span class="v">{epochs}</span></div>
  <div class="mh-sb-row"><span class="k">Classes</span><span class="v">{len(classes) or '—'}</span></div>
</div>
"""
    _sidebar_html(quality_html)

    if system.deepface.backend == "none":
        st.sidebar.warning(
            "Pretrained face benchmark is offline — see `agents/deepface_agent.py`."
        )

    nlp_eval = _load_nlp_eval()
    cnn_eval = _load_cnn_eval()
    nlp_thr = _load_nlp_threshold()

    if nlp_eval or cnn_eval:
        nlp_acc = nlp_eval.get("accuracy")
        nlp_macro = nlp_eval.get("macro_f1")
        nlp_roc = nlp_eval.get("roc_auc")
        cnn_acc = cnn_eval.get("accuracy")
        cnn_macro = cnn_eval.get("macro_f1")
        cnn_tta = cnn_eval.get("tta")

        def _pct(v: object) -> str:
            return f"{float(v) * 100:.1f}%" if isinstance(v, (int, float)) else "—"

        eval_html = f"""
<div class="mh-sb-card">
  <div class="head">Held-out evaluation</div>
  <div class="mh-sb-row"><span class="k">NLP accuracy</span><span class="v">{_pct(nlp_acc)}</span></div>
  <div class="mh-sb-row"><span class="k">NLP macro-F1</span><span class="v">{_pct(nlp_macro)}</span></div>
  <div class="mh-sb-row"><span class="k">NLP ROC-AUC</span><span class="v">{_pct(nlp_roc)}</span></div>
  <div class="mh-sb-row"><span class="k">CNN accuracy</span><span class="v">{_pct(cnn_acc)}</span></div>
  <div class="mh-sb-row"><span class="k">CNN macro-F1</span><span class="v">{_pct(cnn_macro)}</span></div>
  <div class="mh-sb-row"><span class="k">CNN TTA</span><span class="v">{'on' if cnn_tta else 'off / unknown'}</span></div>
</div>
"""
        _sidebar_html(eval_html)

    if nlp_thr:
        thr_val = nlp_thr.get("threshold", 0.5)
        delta = nlp_thr.get("delta_macro_f1")
        delta_s = f" (+{float(delta) * 100:.2f}% macro-F1)" if isinstance(delta, (int, float)) and delta > 0 else ""
        st.sidebar.caption(
            f"NLP threshold calibrated to **{thr_val:.2f}** (vs. 0.50 default){delta_s} — "
            "see `scripts/calibrate_nlp_threshold.py`."
        )

    st.sidebar.markdown("### 📞 Crisis hotlines")
    region_keys = [k for k, _ in REGION_LABELS]
    region_default = region_keys.index("auto")
    sel_region = st.sidebar.selectbox(
        "Region",
        options=region_keys,
        index=region_default,
        format_func=lambda k: dict(REGION_LABELS).get(k, k),
        help="Filter the hotline list shown when the safety net fires.",
        key="mh_region",
    )
    st.session_state["mh_region_value"] = sel_region

    st.sidebar.caption(
        "Train in Jupyter; enable HPT with `train_and_save_nlp(run_grid_search=True)` "
        "or **02_train_nlp.ipynb**. Re-run `python -m scripts.eval_cnn --tta` and "
        "`python -m scripts.eval_nlp` after retraining to refresh metrics."
    )


# ---------------------------------------------------------------------
# Mode-aware wellbeing suggestions (offline, deterministic)
# ---------------------------------------------------------------------
_SUGGESTION_MODE_KIND = {
    "Crisis support": "danger",
    "Active support": "warn",
    "Reflection": "info",
    "Light self-care": "info",
    "Maintain": "success",
}


def _render_suggestions(sup) -> None:
    """Render a mode-aware wellbeing-suggestions card.

    Hidden when nothing useful would be shown (defensive — ``suggestions_for``
    always returns at least one item, but stay safe in case of future edits).
    """
    has_hopelessness = any(
        getattr(h, "category", "") == "hopelessness" for h in (sup.crisis_hits or [])
    )
    items = suggestions_for(sup.final_status, hopelessness=has_hopelessness)
    if not items:
        return

    label = mode_label(sup.final_status)
    pill_kind = _SUGGESTION_MODE_KIND.get(label, "info")
    pill = status_pill(label, pill_kind)

    cards = "".join(
        f"""
<div style="
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 12px;
    padding: 12px 14px;
    display: flex;
    gap: 10px;
    align-items: flex-start;">
  <div style="font-size: 1.35rem; line-height: 1.2;">{html.escape(s.icon)}</div>
  <div style="flex: 1;">
    <div style="font-weight: 700; color: #e2e8f0; font-size: 0.92rem; margin-bottom: 2px;">
      {html.escape(s.title)}
    </div>
    <div style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.45;">
      {html.escape(s.detail)}
    </div>
  </div>
</div>"""
        for s in items
    )

    body = f"""
<div style="
    background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(139,92,246,0.06));
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 16px;
    padding: 16px 18px;
    margin: 0.6rem 0 0.4rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div style="font-weight:700;color:#e2e8f0;letter-spacing:-0.01em;">
      🌱 Personalized suggestions
    </div>
    <div>{pill}</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;">
    {cards}
  </div>
  <div style="margin-top:10px;font-size:0.74rem;color:#94a3b8;">
    Based on the final mode <em>{html.escape(sup.final_status)}</em>. Generic
    self-care prompts only — not medical advice.
  </div>
</div>
"""
    _html(body)


# ---------------------------------------------------------------------
# Main results
# ---------------------------------------------------------------------
def render_analysis_results(
    result: AnalysisResult,
    *,
    elapsed_ms: float | None = None,
) -> None:
    """Full results area after a successful `system.analyze(...)`."""
    dip = result.dip
    ocr = result.ocr
    nlp = result.nlp
    cnn = result.cnn
    df = result.deepface
    img_fuse = result.image_fusion
    mm = result.multimodal
    sup = result.supervisor
    comp = result.comparison

    # ---------- crisis banner (renders FIRST when SafetyAgent fires) ----
    if sup.flagged_crisis:
        detected = [
            (_CRISIS_CATEGORY_LABELS.get(h.category, h.category), h.phrase)
            for h in sup.crisis_hits
        ]
        region = st.session_state.get("mh_region_value", "auto")
        hotlines = get_hotlines_for_region(region)
        _html(crisis_banner(detected, hotlines))

    # ---------- decision snapshot ----------
    status_cls = _status_class(sup.final_status)
    badges = [
        status_pill(f"Text: {nlp.short_label} · {nlp.confidence * 100:.0f}%", _status_kind(nlp.short_label) if nlp.short_label != "Negative" else "danger"),
        status_pill(f"Face: {img_fuse.final_emotion.title()} · {img_fuse.final_confidence * 100:.0f}%", _band_kind(img_fuse.confidence_band)),
        status_pill(f"Agreement: {'yes' if img_fuse.agreement else 'no'}", "success" if img_fuse.agreement else "warn"),
        status_pill(f"Mode: {mm.decision}", _status_kind(mm.decision)),
    ]
    if elapsed_ms is not None:
        badges.append(status_pill(f"⚡ {elapsed_ms / 1000:.2f}s", "info"))

    decision_html = f"""
<div class="mh-decision">
  <div class="mh-decision-grid">
    <div style="display:flex;justify-content:center;">
      {confidence_ring(img_fuse.final_confidence * 100, label='Face conf.', color='#818cf8', size=148)}
    </div>
    <div class="mh-decision-headline">
      <p class="mh-decision-status {status_cls}">{html.escape(sup.final_status)}</p>
      <p class="mh-decision-explanation">{html.escape(sup.final_explanation)}</p>
      <div class="mh-decision-badges">{''.join(badges)}</div>
    </div>
  </div>
</div>
"""
    _html(decision_html)

    # ---------- DIP + OCR preprocessing info ----------
    dip_band_color = "#22c55e" if dip.confidence_band == "high" else "#eab308" if dip.confidence_band == "medium" else "#ef4444"
    ocr_confidence_str = f"{ocr.confidence:.2f}" if ocr.confidence > 0 else "N/A"
    dip_ocr_html = f"""
<div class="mh-mini-grid">
  <div class="mh-mini-card" style="border-left:4px solid {dip_band_color};">
    <div class="label">🎨 DIP Processing</div>
    <div class="value" style="font-size:0.9rem;">{html.escape(dip.confidence_band.capitalize())} quality</div>
    <div class="note">Face: {'✓ Found' if dip.face_detected else '✗ Not found'} · Quality: {dip.quality_score:.2f}/1.0</div>
    <div class="note" style="font-size:0.8rem;margin-top:4px;">{html.escape(dip.enhancement_applied)}</div>
  </div>
  <div class="mh-mini-card" style="border-left:4px solid #06b6d4;">
    <div class="label">📝 OCR (Text Extraction)</div>
    <div class="value" style="font-size:0.9rem;">{ocr.num_detections} region{'s' if ocr.num_detections != 1 else ''}</div>
    <div class="note">Confidence: {ocr_confidence_str} · {html.escape(ocr.note[:60])}</div>
    {f'<div class="note" style="font-size:0.8rem;margin-top:4px;color:#475569;">Text: {html.escape(ocr.extracted_text[:100])}{"…" if len(ocr.extracted_text) > 100 else ""}</div>' if ocr.extracted_text else '<div class="note" style="font-size:0.8rem;margin-top:4px;color:#999;">No text extracted</div>'}
  </div>
</div>
"""
    _html(dip_ocr_html)

    # ---------- mini grid: text · face · fusion ----------
    df_emo = df.emotion.title() if df.emotion else "N/A"
    df_conf_pct = (df.confidence or 0) * 100
    mini_html = f"""
<div class="mh-mini-grid">
  <div class="mh-mini-card text">
    <div class="label">Text · NLP</div>
    <div class="value">{html.escape(nlp.short_label)}</div>
    <div class="note">P(Positive) {nlp.proba_positive:.2f} · P(Negative) {nlp.proba_negative:.2f}</div>
    <div class="meter"><span style="width:{nlp.confidence * 100:.0f}%"></span></div>
  </div>
  <div class="mh-mini-card face">
    <div class="label">Face · CNN</div>
    <div class="value">{html.escape(cnn.emotion.title())}</div>
    <div class="note">DeepFace ({html.escape(df.backend)}): {html.escape(df_emo)} · {df_conf_pct:.0f}%</div>
    <div class="meter"><span style="width:{cnn.confidence * 100:.0f}%"></span></div>
  </div>
  <div class="mh-mini-card fusion">
    <div class="label">Fusion · resolved</div>
    <div class="value">{html.escape(img_fuse.final_emotion.title())}</div>
    <div class="note">Source: <strong>{html.escape(img_fuse.chosen_source)}</strong> · band {html.escape(img_fuse.confidence_band)}</div>
    <div class="meter"><span style="width:{img_fuse.final_confidence * 100:.0f}%"></span></div>
  </div>
</div>
"""
    _html(mini_html)

    st.caption(
        "📄 **Structured report** — open the **Report** tab to **print** (browser dialog) "
        "or download `.txt` / `.md`."
    )

    # ---------- supervisor warnings (only if any) ----------
    if sup.warnings:
        with st.expander(f"⚠ Supervisor warnings ({len(sup.warnings)})", expanded=False):
            for w in sup.warnings:
                st.write(f"- {w}")

    # ---------- mode-aware wellbeing suggestions ----------
    _render_suggestions(sup)

    # ---------- tabs ----------
    tab_overview, tab_agents, tab_xai, tab_report = st.tabs(
        ["📊 Overview", "🤖 Agent details", "🔍 Explainability (XAI)", "📄 Report"]
    )

    # ===== Tab 1 — Overview ==========================================
    with tab_overview:
        st.markdown("##### Image emotion distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("CV Agent · CNN (your trained model)")
            cnn_df = (
                pd.DataFrame(
                    sorted(cnn.probabilities.items(), key=lambda kv: -kv[1]),
                    columns=["emotion", "probability"],
                )
                .set_index("emotion")
            )
            st.bar_chart(cnn_df, height=240, color="#818cf8")
        with c2:
            st.caption(f"DeepFace · {df.backend} (pretrained benchmark)")
            if df.probabilities:
                df_probs = (
                    pd.DataFrame(
                        sorted(df.probabilities.items(), key=lambda kv: -kv[1]),
                        columns=["emotion", "probability"],
                    )
                    .set_index("emotion")
                )
                st.bar_chart(df_probs, height=240, color="#34d399")
            else:
                st.info(f"DeepFace returned no scores ({df.note or 'no backend'}).")

        _html(
            comparison_panel(
                cnn.emotion,
                cnn.confidence,
                df.emotion,
                df.confidence,
                comp.chosen_source,
                comp.agreement,
                comp.note,
            )
        )

        st.markdown("")
        st.markdown("##### Multimodal decision")
        mm_kind = _status_kind(mm.decision)
        decision_pills = (
            status_pill(f"Text: {mm.text_signal}", "success" if mm.text_signal == "positive" else "danger")
            + status_pill(f"Face: {mm.face_signal}", "success" if mm.face_signal == "positive" else "warn" if mm.face_signal == "neutral" else "danger")
            + status_pill(f"→ {mm.decision}", mm_kind)
        )
        _html(
            f"""
<div class="mh-detail-card" style="margin-top:0.4rem;">
  <div class="mh-detail-head">
    <div class="mh-detail-title">
      <span class="ix">∑</span><span class="name">Resolved decision</span>
    </div>
    <div>{decision_pills}</div>
  </div>
  <div class="mh-detail-desc" style="margin-bottom:0.2rem;">{html.escape(mm.explanation)}</div>
</div>
"""
        )

    # ===== Tab 2 — Agent details =====================================
    with tab_agents:
        cleaned_preview = nlp.cleaned_text if len(nlp.cleaned_text) <= 220 else nlp.cleaned_text[:217] + "…"

        # 0 — DIP Processor
        dip_quality_issues = ", ".join(dip.quality_issues) if dip.quality_issues else "none"
        _html(
            detail_card(
                "0",
                "DIP Agent — Digital Image Processing",
                "Face detection, enhancement, and quality assessment. Uses OpenCV Cascade + MediaPipe fallback for face detection; CLAHE for contrast enhancement; denoising for noise reduction.",
                {
                    "Face detected": "yes" if dip.face_detected else "no",
                    "Quality score": f"{dip.quality_score:.2f}/1.0",
                    "Quality band": dip.confidence_band,
                    "Quality issues": dip_quality_issues,
                    "Enhancement": dip.enhancement_applied,
                },
                pill_text=dip.confidence_band,
                pill_kind=_band_kind(dip.confidence_band),
            )
        )

        # 0.5 — OCR Agent
        ocr_preview = ocr.extracted_text[:100] + "…" if len(ocr.extracted_text) > 100 else ocr.extracted_text
        _html(
            detail_card(
                "0.5",
                "OCR Agent — Optical Character Recognition",
                "Extracts text from image using EasyOCR. Detected text is merged with user-provided text before NLP analysis.",
                {
                    "Regions detected": str(ocr.num_detections),
                    "Avg confidence": f"{ocr.confidence:.2f}" if ocr.confidence > 0 else "N/A",
                    "Extracted text": f'"{ocr_preview}"' if ocr_preview else "(none)",
                    "Status": ocr.note,
                },
                pill_text="OCR" if ocr.num_detections > 0 else "no text",
                pill_kind="success" if ocr.num_detections > 0 else "info",
            )
        )

        # 1 — NLP
        _html(
            detail_card(
                "1",
                "NLP Agent — language sentiment",
                "TF-IDF (word 1-2 + char 3-5) → Logistic Regression. Predicts Positive / Negative with calibrated probabilities; LIME reuses these probabilities for word-level explanations.",
                {
                    "Prediction": nlp.short_label,
                    "Confidence": f"{nlp.confidence * 100:.1f}%",
                    "P(Positive)": f"{nlp.proba_positive:.3f}",
                    "P(Negative)": f"{nlp.proba_negative:.3f}",
                    "Cleaned text": f'"{cleaned_preview}"' if cleaned_preview else "(empty after cleaning)",
                },
                pill_text=nlp.short_label,
                pill_kind="success" if nlp.short_label == "Positive" else "danger",
            )
        )

        # 2 — HPT
        _html(
            detail_card(
                "2",
                "HPT Agent — hyperparameter tuning",
                "Runs <em>at training time</em> (not per click). Uses GridSearchCV across <code>C</code>, <code>max_iter</code>, and <code>solver</code> for the LR head, then keeps the best estimator. Enable with <code>train_and_save_nlp(run_grid_search=True)</code>.",
                {"Stage": "training", "Search": "GridSearchCV", "Tuned params": "C, max_iter, solver", "Status": "active when training"},
                pill_text="train-time",
                pill_kind="info",
            )
        )

        # 3 — CV / CNN
        top3 = sorted(cnn.probabilities.items(), key=lambda kv: -kv[1])[:3]
        top3_str = ", ".join(f"{n} {p * 100:.0f}%" for n, p in top3)
        _html(
            detail_card(
                "3",
                "CV Agent — CNN emotions",
                "Compact PyTorch CNN (3 conv blocks) on 48×48 RGB images (FER-style). Returns the top emotion and the full probability distribution shown in the Overview tab.",
                {
                    "Top-1": cnn.emotion.title(),
                    "Confidence": f"{cnn.confidence * 100:.1f}%",
                    "Top-3": top3_str,
                    "Classes": str(len(cnn.probabilities)),
                },
                pill_text=cnn.emotion.title(),
                pill_kind=_band_kind("high" if cnn.confidence >= 0.65 else "medium" if cnn.confidence >= 0.45 else "low"),
            )
        )

        # 4 — DeepFace
        df_pill = df.backend if df.backend != "none" else "offline"
        df_kind = "success" if df.backend in ("deepface", "hf-vit") else "warn"
        _html(
            detail_card(
                "4",
                "DeepFace Agent — pretrained benchmark",
                "Pretrained reference model. Uses real <code>deepface</code> when TensorFlow is available; otherwise falls back to a HuggingFace ViT (<code>dima806/facial_emotions_image_detection</code>).",
                {
                    "Backend": df.backend,
                    "Top-1": df.emotion or "N/A",
                    "Confidence": f"{df.confidence * 100:.1f}%" if df.confidence is not None else "N/A",
                    "Note": df.note or "—",
                },
                pill_text=df_pill,
                pill_kind=df_kind,
            )
        )

        # 5 — Comparison
        _html(
            detail_card(
                "5",
                "Comparison Agent — CNN vs DeepFace",
                "Validates the two image-side sources. On <strong>agreement</strong> the shared label is kept with the higher confidence; on <strong>disagreement</strong> DeepFace is preferred as the external benchmark (project policy).",
                {
                    "Final emotion": comp.final_emotion.title(),
                    "Agreement": "yes" if comp.agreement else "no",
                    "Chosen source": comp.chosen_source,
                    "Band": comp.confidence_band,
                    "Final score": f"{comp.final_confidence:.3f}",
                },
                pill_text=comp.confidence_band,
                pill_kind=_band_kind(comp.confidence_band),
            )
        )

        # 6 — Fusion
        _html(
            detail_card(
                "6",
                "Fusion Agent — text + face",
                "Multimodal rule-based fusion that combines NLP sentiment with the resolved face emotion. Outputs one of <em>High Risk · Stable · Conflict · Mixed/monitor</em>.",
                {
                    "Text signal": mm.text_signal,
                    "Face signal": mm.face_signal,
                    "Decision": mm.decision,
                    "Image final": img_fuse.final_emotion.title(),
                    "Chosen source": img_fuse.chosen_source,
                },
                pill_text=mm.decision,
                pill_kind=_status_kind(mm.decision),
            )
        )

        # 7 — Safety (deterministic crisis lexicon)
        n_crisis = sum(
            1 for h in sup.crisis_hits if h.category != "hopelessness"
        )
        n_hope = sum(1 for h in sup.crisis_hits if h.category == "hopelessness")
        if sup.flagged_crisis:
            safety_pill_text, safety_pill_kind = "override fired", "danger"
        elif n_hope:
            safety_pill_text, safety_pill_kind = "hopelessness flagged", "warn"
        else:
            safety_pill_text, safety_pill_kind = "no hits", "success"
        sample_hits = ", ".join(repr(h.phrase) for h in sup.crisis_hits[:4]) or "—"
        _html(
            detail_card(
                "7",
                "Safety Agent — crisis lexicon",
                "Deterministic regex-based safety net for crisis language (suicide, self-harm, planned attempt, hopelessness). Tolerates common misspellings and repeated letters; on a high-severity hit it <strong>overrides</strong> the model output to <em>Crisis — seek support</em>.",
                {
                    "Crisis hits": str(n_crisis),
                    "Hopelessness hits": str(n_hope),
                    "Override fired": "yes" if sup.flagged_crisis else "no",
                    "Sample phrases": sample_hits,
                },
                pill_text=safety_pill_text,
                pill_kind=safety_pill_kind,
            )
        )

        # 8 — Supervisor
        _html(
            detail_card(
                "8",
                "Supervisor Agent — final decision",
                f"Reviews every upstream signal, attaches warnings on low confidence (&lt;{LOW_CONFIDENCE_THRESHOLD:.2f}), short text, conflict, etc., softens <em>High Risk</em> to <em>Likely High Risk</em> when any signal is weak, and respects the SafetyAgent override.",
                {
                    "Final status": sup.final_status,
                    "Warnings": str(len(sup.warnings)),
                    "Low-conf flag": "yes" if sup.flagged_low_confidence else "no",
                    "Conflict flag": "yes" if sup.flagged_conflict else "no",
                    "Crisis flag": "yes" if sup.flagged_crisis else "no",
                },
                pill_text=sup.final_status,
                pill_kind=_status_kind(sup.final_status),
            )
        )

        # 9 — XAI
        _html(
            detail_card(
                "9",
                "XAI Agent — explainability",
                "Uses LIME to surface the most influential words in your text (color-coded in the next tab) and produces a structured CV-vs-DeepFace summary for the image side.",
                {
                    "LIME words": str(len(result.xai.text_words)),
                    "Image explainer": "structured summary",
                },
                pill_text="LIME · text + image",
                pill_kind="info",
            )
        )

    # ===== Tab 3 — XAI ===============================================
    with tab_xai:
        st.markdown("##### Most influential words (LIME)")
        _html(lime_legend())
        if result.xai.text_words:
            chips_html = '<div class="mh-lime-row">' + "".join(
                lime_chip(w.word, w.weight) for w in result.xai.text_words
            ) + "</div>"
            _html(chips_html)

            st.markdown("##### Highlighted text")
            weight_map = {w.word.lower(): w.weight for w in result.xai.text_words}
            _html(lime_highlighted_text(nlp.cleaned_text, weight_map))
        else:
            st.info("No LIME words available — text was empty after cleaning, or LIME failed silently.")

        st.markdown("##### Image-side explanation")
        _html(
            f"""
<div class="mh-detail-card" style="margin-top:0.4rem;">
  <div class="mh-detail-desc" style="margin:0;">{html.escape(result.xai.image_explanation)}</div>
</div>
"""
        )

    # ===== Tab 4 — Report ============================================
    with tab_report:
        st.markdown("##### Held-out evaluation (recomputed without retraining)")
        nlp_eval = _load_nlp_eval()
        cnn_eval = _load_cnn_eval()
        if nlp_eval or cnn_eval:
            ev_cols = st.columns(2)
            with ev_cols[0]:
                st.caption("NLP — TF-IDF + LR")
                if nlp_eval:
                    st.write(
                        {
                            "accuracy": round(float(nlp_eval.get("accuracy", 0)), 4),
                            "macro_f1": round(float(nlp_eval.get("macro_f1", 0)), 4),
                            "weighted_f1": round(float(nlp_eval.get("weighted_f1", 0)), 4),
                            "ROC-AUC": round(float(nlp_eval.get("roc_auc", 0)), 4),
                            "PR-AUC": round(float(nlp_eval.get("pr_auc", 0)), 4),
                            "n_test": int(nlp_eval.get("n_test", 0)),
                            "threshold": float(nlp_eval.get("threshold", 0.5)),
                        }
                    )
                    if Path(NLP_CONFUSION_PNG).is_file():
                        st.image(str(NLP_CONFUSION_PNG), caption="NLP confusion matrix")
                else:
                    st.caption("Run `python -m scripts.eval_nlp` to compute.")
            with ev_cols[1]:
                st.caption("CNN — facial emotion (FER held-out)")
                if cnn_eval:
                    st.write(
                        {
                            "accuracy": round(float(cnn_eval.get("accuracy", 0)), 4),
                            "macro_f1": round(float(cnn_eval.get("macro_f1", 0)), 4),
                            "weighted_f1": round(float(cnn_eval.get("weighted_f1", 0)), 4),
                            "n_test": int(cnn_eval.get("n_test", 0)),
                            "tta": bool(cnn_eval.get("tta", False)),
                        }
                    )
                    if Path(CNN_CONFUSION_PNG).is_file():
                        st.image(str(CNN_CONFUSION_PNG), caption="CNN confusion matrix")
                else:
                    st.caption("Run `python -m scripts.eval_cnn --tta` to compute.")
            st.divider()

        st.markdown("##### Structured report (print & export)")
        st.caption(
            "Use **Print report** to open your browser’s print dialog (save as PDF or paper). "
            "Downloads use the same structured content as the text view below."
        )
        try:
            components.html(
                build_printable_report_html(result),
                height=820,
                scrolling=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error("Could not render the printable report.")
            st.caption(str(exc))

        report_txt = format_report(result)
        try:
            report_md = analysis_to_markdown(result)
        except Exception:  # noqa: BLE001
            report_md = "# Report\n\n" + report_txt

        st.divider()
        cdl1, cdl2, cdl3 = st.columns(3)
        with cdl1:
            st.download_button(
                "⬇ Download (.txt)",
                data=report_txt,
                file_name="multimodal_analysis_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with cdl2:
            st.download_button(
                "⬇ Download (.md)",
                data=report_md,
                file_name="multimodal_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with cdl3:
            st.caption(f"~{len(report_txt):,} characters · plain text")

        rep_tab1, rep_tab2 = st.tabs(["📝 Summary (markdown)", "📋 Full plain text"])
        with rep_tab1:
            try:
                st.markdown(report_md)
            except Exception:  # noqa: BLE001
                st.caption("Markdown render failed — see Full plain text tab.")
        with rep_tab2:
            st.code(report_txt, language="text")


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
def render_disclaimer() -> None:
    st.divider()
    body = (
        '<div class="mh-footer">'
        'This is a <strong>multimodal AI demo system</strong>. Results are experimental and '
        'intended for <strong>educational analysis only</strong>. '
        '<span style="opacity:0.92;">Not for diagnosis, treatment, or emergency decisions.</span>'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)
