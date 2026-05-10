"""System orchestrator wiring all agents together (used by app + notebooks)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape as he

from PIL import Image

from agents import (
    CVAgent,
    DeepFaceAgent,
    DIImageProcessor,
    DIProcessingResult,
    FusionAgent,
    ImageFusion,
    MultimodalFusion,
    NLPAgent,
    NLPResult,
    OCRAgent,
    OCRResult,
    SupervisorAgent,
    SupervisorReport,
    XAIAgent,
    suggestions_for,
)
from agents.cnn_agent import CNNResult
from agents.deepface_agent import DeepFaceResult
from agents.comparison_agent import ComparisonResult
from agents.suggestions_agent import mode_label
from agents.xai_agent import XAIReport


@dataclass
class AnalysisResult:
    dip: DIProcessingResult
    ocr: OCRResult
    nlp: NLPResult
    cnn: CNNResult
    deepface: DeepFaceResult
    comparison: ComparisonResult
    image_fusion: ImageFusion
    multimodal: MultimodalFusion
    supervisor: SupervisorReport
    xai: XAIReport


class System:
    """Loads every agent once and exposes a single `analyze()` entry point."""

    def __init__(self) -> None:
        self.dip = DIImageProcessor()
        self.ocr = OCRAgent()
        self.nlp = NLPAgent()
        self.cnn = CVAgent()
        self.deepface = DeepFaceAgent()
        self.fusion = FusionAgent()
        self.supervisor = SupervisorAgent()
        self.xai = XAIAgent(self.nlp)

    def analyze(self, text: str, image: Image.Image) -> AnalysisResult:
        # 1. DIP Preprocessing + OCR (single-threaded, must be done first)
        dip_res = self.dip.preprocess(image)
        processed_image = dip_res.processed_image

        ocr_res = self.ocr.extract(image)
        merged_text = self.ocr.merge_text(text, ocr_res.extracted_text)

        # 2. Run NLP + CNN + DeepFace in parallel on preprocessed inputs
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_nlp = pool.submit(self.nlp.predict, merged_text)
            fut_cnn = pool.submit(self.cnn.predict, processed_image)
            fut_df = pool.submit(self.deepface.predict, processed_image)
            nlp_res = fut_nlp.result()
            cnn_res = fut_cnn.result()
            df_res = fut_df.result()

        # 3. Rest of pipeline (unchanged)
        img_fuse, comparison = self.fusion.fuse_image(cnn_res, df_res)
        mm = self.fusion.fuse_multimodal(nlp_res, img_fuse)
        # Pass the *raw* merged text so the SafetyAgent can scan punctuation tricks
        sup = self.supervisor.review(nlp_res, cnn_res, df_res, img_fuse, mm, raw_text=merged_text)
        try:
            xai = self.xai.explain(nlp_res, cnn_res, df_res, img_fuse)
        except Exception:  # noqa: BLE001 — LIME must not break the pipeline
            try:
                img_ex = self.xai.explain_image(cnn_res, df_res, img_fuse)
            except Exception:  # noqa: BLE001
                img_ex = "Summary unavailable."
            xai = XAIReport(text_words=[], image_explanation=img_ex)

        return AnalysisResult(
            dip=dip_res,
            ocr=ocr_res,
            nlp=nlp_res,
            cnn=cnn_res,
            deepface=df_res,
            comparison=comparison,
            image_fusion=img_fuse,
            multimodal=mm,
            supervisor=sup,
            xai=xai,
        )



def _report_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_report(result: AnalysisResult) -> str:
    """Structured plain-text report (downloads + parity with HTML print view)."""
    dip = result.dip
    ocr = result.ocr
    nlp = result.nlp
    cnn = result.cnn
    df = result.deepface
    comp = result.comparison
    img_fuse = result.image_fusion
    mm = result.multimodal
    sup = result.supervisor
    xai = result.xai

    df_emo = df.emotion.title() if df.emotion else "N/A"
    df_conf = f"{df.confidence * 100:.0f}%" if df.confidence is not None else "N/A"
    bar = "=" * 76

    lines: list[str] = [
        bar,
        " MULTIMODAL ANALYSIS REPORT",
        bar,
        f"Generated: {_report_timestamp()}",
        "",
        "─" * 76,
        " 1. FINAL STATUS",
        "─" * 76,
        f"  Decision     : {sup.final_status}",
        f"  Explanation  : {sup.final_explanation}",
    ]
    if sup.flagged_crisis:
        lines.extend(
            [
                "",
                "  ⚠ CRISIS SAFETY NET — override active. Seek professional or crisis support.",
            ]
        )

    lines.extend(
        [
            "",
            "─" * 76,
            " 2. DIP PREPROCESSING (Digital Image Processing)",
            "─" * 76,
            f"  Face detected: {dip.face_detected}",
            f"  Quality score: {dip.quality_score:.2f}/1.0 ({dip.confidence_band})",
            f"  Enhancement  : {dip.enhancement_applied}",
            f"  Status       : {dip.note}",
        ]
    )

    lines.extend(
        [
            "",
            "─" * 76,
            " 3. OCR (Optical Character Recognition)",
            "─" * 76,
            f"  Text found   : {ocr.num_detections} region(s)",
            f"  Confidence   : {ocr.confidence:.2f}" if ocr.confidence > 0 else "  Confidence   : N/A",
            f"  Status       : {ocr.note}",
        ]
    )
    if ocr.extracted_text:
        preview = ocr.extracted_text[:200] + "…" if len(ocr.extracted_text) > 200 else ocr.extracted_text
        lines.append(f"  Extracted    : {preview}")

    lines.extend(
        [
            "",
            "─" * 76,
            " 4. TEXT SIGNAL (NLP)",
            "─" * 76,
            f"  Label        : {nlp.short_label}",
            f"  Confidence   : {nlp.confidence * 100:.0f}%",
            f"  P(Positive)  : {nlp.proba_positive:.4f}",
            f"  P(Negative)  : {nlp.proba_negative:.4f}",
        ]
    )
    preview = nlp.cleaned_text
    if len(preview) > 400:
        preview = preview[:397] + "…"
    lines.append(f"  Cleaned text : {preview or '(empty after cleaning)'}")

    lines.extend(
        [
            "",
            "─" * 76,
            " 5. IMAGE SIGNAL (CV + DeepFace + Comparison)",
            "─" * 76,
            f"  CNN top-1    : {cnn.emotion.title()} ({cnn.confidence * 100:.0f}%)",
            f"  DeepFace     : {df_emo} ({df_conf})  [backend: {df.backend}]",
            f"  Comparison   : {comp.final_emotion.title()}",
            f"    · band     : {comp.confidence_band}",
            f"    · agreement: {comp.agreement}",
            f"    · source   : {comp.chosen_source}",
            f"  Resolved face: {img_fuse.final_emotion.title()} "
            f"({img_fuse.final_confidence * 100:.0f}%, band={img_fuse.confidence_band}, "
            f"source={img_fuse.chosen_source})",
            "",
            "─" * 76,
            " 6. MULTIMODAL FUSION",
            "─" * 76,
            f"  Text signal  : {mm.text_signal}",
            f"  Face signal  : {mm.face_signal}",
            f"  Fusion mode  : {mm.decision}",
            f"  Rationale    : {mm.explanation}",
        ]
    )

    if sup.crisis_hits:
        lines.extend(["", "─" * 76, " 7. SAFETY (lexicon hits)", "─" * 76])
        for h in sup.crisis_hits:
            lines.append(f"  · [{h.category}] {h.phrase}")

    lines.extend(
        [
            "",
            "─" * 76,
            " 8. EXPLAINABILITY (XAI)",
            "─" * 76,
            f"  Image summary: {xai.image_explanation}",
        ]
    )
    if xai.text_words:
        lines.append("  LIME (words) :")
        for w in xai.text_words[:15]:
            lines.append(f"    · {w.word:20s}  weight={w.weight:+.3f}  ({w.strength})")
    else:
        lines.append("  LIME (words) : (none — empty text or LIME skipped)")

    if sup.warnings:
        lines.extend(["", "─" * 76, " 9. WARNINGS", "─" * 76])
        for w in sup.warnings:
            lines.append(f"  · {w}")

    has_hopelessness = any(h.category == "hopelessness" for h in sup.crisis_hits)
    suggestions = suggestions_for(sup.final_status, hopelessness=has_hopelessness)
    if suggestions:
        lines.extend(
            [
                "",
                "─" * 76,
                f" 10. SUGGESTIONS ({mode_label(sup.final_status)} mode)",
                "─" * 76,
            ]
        )
        for s in suggestions:
            lines.append(f"  · {s.title}")
            lines.append(f"      {s.detail}")

    lines.extend(
        [
            "",
            "─" * 76,
            " 11. AGENT PIPELINE (inference order)",
            "─" * 76,
            "  0. DIP Processor (face detection + enhancement)",
            "  1. OCR Agent (text extraction)",
            "  2. NLP Agent",
            "  3. CV Agent (CNN)",
            "  4. DeepFace Agent",
            "  5. Comparison Agent",
            "  6. Fusion Agent",
            "  7. Safety Agent (lexicon)",
            "  8. Supervisor Agent",
            "  9. Suggestions Agent",
            "  10. XAI Agent",
            "  (HPT Agent runs at training time only.)",
        ]
    )

    lines.extend(
        [
            "",
            bar,
            " End of report — experimental; educational use only.",
            bar,
        ]
    )
    return "\n".join(lines)


def build_printable_report_html(result: AnalysisResult) -> str:
    """Self-contained HTML fragment for Streamlit ``components.html`` + browser Print."""
    dip = result.dip
    ocr = result.ocr
    nlp = result.nlp
    cnn = result.cnn
    df = result.deepface
    comp = result.comparison
    img_fuse = result.image_fusion
    mm = result.multimodal
    sup = result.supervisor
    xai = result.xai

    df_emo = df.emotion.title() if df.emotion else "N/A"
    df_conf = f"{df.confidence * 100:.0f}%" if df.confidence is not None else "N/A"
    ocr_conf = f"{ocr.confidence:.2f}" if ocr.confidence > 0 else "N/A"
    ts = _report_timestamp()
    preview = nlp.cleaned_text or ""
    if len(preview) > 600:
        preview = preview[:597] + "…"

    crisis_block = ""
    if sup.flagged_crisis:
        phrases = ", ".join(he(h.phrase) for h in sup.crisis_hits[:8])
        crisis_block = f"""
<div class="crisis">
  <strong>Crisis safety net triggered.</strong> Phrases detected: {phrases}.
  This overrides probabilistic model outputs. Please reach out to qualified support.
</div>"""

    lime_rows = ""
    if xai.text_words:
        for w in xai.text_words[:20]:
            lime_rows += (
                f"<tr><td>{he(w.word)}</td><td>{w.weight:+.4f}</td><td>{he(w.strength)}</td></tr>"
            )
    else:
        lime_rows = "<tr><td colspan='3'><em>No LIME tokens</em></td></tr>"

    warn_list = ""
    if sup.warnings:
        for w in sup.warnings:
            warn_list += f"<li>{he(w)}</li>"
    else:
        warn_list = "<li><em>None</em></li>"

    safety_rows = ""
    if sup.crisis_hits:
        for h in sup.crisis_hits:
            safety_rows += (
                f"<tr><td>{he(h.category)}</td><td>{he(h.phrase)}</td></tr>"
            )
    else:
        safety_rows = "<tr><td colspan='2'><em>No lexicon hits</em></td></tr>"

    has_hopelessness = any(h.category == "hopelessness" for h in sup.crisis_hits)
    suggestion_items = suggestions_for(sup.final_status, hopelessness=has_hopelessness)
    sugg_label = mode_label(sup.final_status)
    suggestion_rows = "".join(
        f"<tr><td style='width:6%;font-size:1.1rem;text-align:center;'>{he(s.icon)}</td>"
        f"<td><strong>{he(s.title)}</strong><br/><span style='color:#475569;'>{he(s.detail)}</span></td></tr>"
        for s in suggestion_items
    )

    return f"""
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<style>
  :root {{
    --ink: #0f172a; --muted: #475569; --line: #e2e8f0; --accent: #4f46e5;
    --crisis-bg: #fef2f2; --crisis-bd: #fecaca;
  }}
  body {{
    font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px; line-height: 1.5; color: var(--ink); margin: 0; padding: 12px 14px;
    background: #fff;
  }}
  h1 {{ font-size: 1.35rem; margin: 0 0 4px 0; letter-spacing: -0.02em; }}
  .meta {{ color: var(--muted); font-size: 12px; margin-bottom: 14px; }}
  h2 {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted); margin: 18px 0 8px 0; border-bottom: 1px solid var(--line);
    padding-bottom: 4px;
  }}
  table {{ width: 100%; border-collapse: collapse; margin: 6px 0 4px 0; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  .kv td:first-child {{ width: 38%; color: var(--muted); font-weight: 600; }}
  .crisis {{
    background: var(--crisis-bg); border: 1px solid var(--crisis-bd);
    border-radius: 8px; padding: 10px 12px; margin: 10px 0; font-size: 12px;
  }}
  .footer {{
    margin-top: 16px; padding-top: 10px; border-top: 1px solid var(--line);
    font-size: 11px; color: var(--muted);
  }}
  .no-print {{
    margin-bottom: 12px;
  }}
  button.print-btn {{
    background: linear-gradient(135deg,#6366f1,#8b5cf6); color: #fff; border: none;
    padding: 8px 16px; border-radius: 8px; font-weight: 600; cursor: pointer;
    font-size: 13px;
  }}
  button.print-btn:hover {{ filter: brightness(1.06); }}
  @media print {{
    .no-print {{ display: none !important; }}
    body {{ padding: 0; }}
    button {{ display: none !important; }}
  }}
</style></head><body>
<div class="no-print">
  <button type="button" class="print-btn" onclick="window.print()">Print report</button>
</div>
<h1>Multimodal analysis report</h1>
<div class="meta">Generated: {he(ts)} · Multimodal AI demo · educational analysis only</div>
{crisis_block}
<h2>1. Final status</h2>
<table class="kv"><tbody>
<tr><td>Decision</td><td><strong>{he(sup.final_status)}</strong></td></tr>
<tr><td>Explanation</td><td>{he(sup.final_explanation)}</td></tr>
</tbody></table>
<h2>2. DIP (Digital Image Processing)</h2>
<table class="kv"><tbody>
<tr><td>Face detected</td><td>{dip.face_detected}</td></tr>
<tr><td>Quality score</td><td>{dip.quality_score:.2f}/1.0 ({he(dip.confidence_band)})</td></tr>
<tr><td>Enhancement</td><td>{he(dip.enhancement_applied)}</td></tr>
<tr><td>Status</td><td>{he(dip.note)}</td></tr>
</tbody></table>
<h2>3. OCR (Optical Character Recognition)</h2>
<table class="kv"><tbody>
<tr><td>Text regions found</td><td>{ocr.num_detections}</td></tr>
<tr><td>Confidence</td><td>{ocr_conf}</td></tr>
<tr><td>Status</td><td>{he(ocr.note)}</td></tr>
""" + (
        f"<tr><td>Extracted text</td><td>{he(ocr.extracted_text[:300] + '…' if len(ocr.extracted_text) > 300 else ocr.extracted_text)}</td></tr>"
        if ocr.extracted_text
        else "<tr><td>Extracted text</td><td><em>None</em></td></tr>"
    ) + f"""
</tbody></table>
<h2>4. Text (NLP)</h2>
<table class="kv"><tbody>
<tr><td>Label</td><td>{he(nlp.short_label)}</td></tr>
<tr><td>Confidence</td><td>{nlp.confidence * 100:.0f}%</td></tr>
<tr><td>P(Positive) / P(Negative)</td><td>{nlp.proba_positive:.4f} / {nlp.proba_negative:.4f}</td></tr>
<tr><td>Cleaned text</td><td>{he(preview) if preview else "(empty)"}</td></tr>
</tbody></table>
<h2>5. Image (CV · DeepFace · comparison)</h2>
<table class="kv"><tbody>
<tr><td>CNN</td><td>{he(cnn.emotion.title())} ({cnn.confidence * 100:.0f}%)</td></tr>
<tr><td>DeepFace ({he(df.backend)})</td><td>{he(df_emo)} ({he(df_conf)})</td></tr>
<tr><td>Comparison</td><td>{he(comp.final_emotion.title())} · band {he(comp.confidence_band)} ·
 agree={comp.agreement} · source={he(comp.chosen_source)}</td></tr>
<tr><td>Resolved face</td><td>{he(img_fuse.final_emotion.title())}
 ({img_fuse.final_confidence * 100:.0f}%) · {he(img_fuse.confidence_band)} · {he(img_fuse.chosen_source)}</td></tr>
</tbody></table>
<h2>6. Multimodal fusion</h2>
<table class="kv"><tbody>
<tr><td>Text / face signals</td><td>{he(mm.text_signal)} / {he(mm.face_signal)}</td></tr>
<tr><td>Mode</td><td><strong>{he(mm.decision)}</strong></td></tr>
<tr><td>Rationale</td><td>{he(mm.explanation)}</td></tr>
</tbody></table>
<h2>7. Safety lexicon</h2>
<table><thead><tr><th>Category</th><th>Phrase</th></tr></thead><tbody>{safety_rows}</tbody></table>
<h2>8. Warnings</h2>
<ul style="margin:6px 0 0 18px;padding:0;">{warn_list}</ul>
<h2>9. Suggestions ({he(sugg_label)} mode)</h2>
<table><tbody>{suggestion_rows}</tbody></table>
<h2>10. Explainability (XAI)</h2>
<p style="margin:6px 0 8px 0;">{he(xai.image_explanation)}</p>
<table><thead><tr><th>Word</th><th>Weight</th><th>Strength</th></tr></thead><tbody>{lime_rows}</tbody></table>
<div class="footer">
  This is a multimodal AI demo system. Results are experimental and intended for educational analysis only.
  Not for diagnosis, treatment, or emergency decisions.
</div>
</body></html>
"""


def analysis_to_markdown(result: AnalysisResult) -> str:
    """Same information as the Streamlit overview — use in Jupyter for parity with the app.

    Note: this is also the body of the downloadable .md report. Warnings are
    intentionally NOT duplicated here because the UI shows them in a dedicated
    expander above — see ``streamlit_ui.render_analysis_results``.
    """
    nlp = result.nlp
    cnn = result.cnn
    df = result.deepface
    comp = result.comparison
    img_fuse = result.image_fusion
    mm = result.multimodal
    sup = result.supervisor
    xai = result.xai
    df_conf = (
        f"{df.confidence * 100:.0f}%"
        if df.confidence is not None
        else "N/A"
    )
    df_note = (df.note or "").replace("|", "/")

    # Safety summary line (used both in the bullet list and the Final-status callout)
    n_crisis = sum(1 for h in sup.crisis_hits if h.category != "hopelessness")
    n_hope = sum(1 for h in sup.crisis_hits if h.category == "hopelessness")
    if sup.flagged_crisis:
        safety_line = f"**override fired** ({n_crisis} crisis hit{'s' if n_crisis != 1 else ''})"
    elif n_hope:
        safety_line = f"hopelessness flagged ({n_hope}) — no override"
    else:
        safety_line = "no hits"

    crisis_callout = ""
    if sup.flagged_crisis:
        phrases = ", ".join(f"`{h.phrase}`" for h in sup.crisis_hits[:5])
        crisis_callout = (
            "\n> 🚨 **Crisis safety net triggered.** Detected: "
            f"{phrases}. The SafetyAgent has overridden the probabilistic models — "
            "please consult a mental-health professional or hotline.\n"
        )

    has_hopelessness = any(h.category == "hopelessness" for h in sup.crisis_hits)
    suggestion_items = suggestions_for(sup.final_status, hopelessness=has_hopelessness)
    sugg_label = mode_label(sup.final_status)
    suggestion_block = "\n".join(
        f"- {s.icon} **{s.title}** — {s.detail}" for s in suggestion_items
    )

    md = f"""### Final status
**{sup.final_status}** — {sup.final_explanation}{crisis_callout}

### Agent pipeline (same order as `System.analyze`)

1. **NLP Agent** — {nlp.short_label} ({nlp.confidence * 100:.0f}%)
2. **CV Agent (CNN)** — {cnn.emotion} ({cnn.confidence * 100:.0f}%)
3. **DeepFace Agent** (`{df.backend}`) — {df.emotion or "N/A"} ({df_conf}){f" — {df_note}" if df_note else ""}
4. **Comparison Agent** — `{comp.final_emotion}` ({comp.confidence_band}), agreement={comp.agreement}, source=*{comp.chosen_source}*
5. **Fusion Agent** — multimodal: text + face → **{mm.decision}** — {mm.explanation}
6. **Safety Agent** — {safety_line}
7. **Supervisor Agent** — {sup.final_status}
8. **Suggestions Agent** — {sugg_label} mode ({len(suggestion_items)} tips)
9. **XAI Agent** — LIME (text) + summary (image) below

*(HPT Agent runs only during training — see `agents/hpt_agent.py` / `training_utils.py`.)*

### Multimodal detail
- Text signal: `{mm.text_signal}` · Face signal: `{mm.face_signal}` · Image final: `{img_fuse.final_emotion}` ({img_fuse.confidence_band})

### Suggestions ({sugg_label} mode)
{suggestion_block}

### Image explanation
{xai.image_explanation}
"""
    return md
