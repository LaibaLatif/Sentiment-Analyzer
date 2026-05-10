"""Supervisor Agent: final decision, conflict / low-confidence / crisis handling.

The supervisor consumes every upstream signal and produces:
    * a single final status string,
    * a one-line explanation,
    * a list of human-readable warnings,
    * boolean flags (low-confidence, conflict, crisis).

Crisis override
---------------
For mental-health crisis language (suicidal ideation, self-harm, planned
attempt, …) we cannot trust the probabilistic NLP head — a misspelled
5-word sentence can score 94% Positive. The SafetyAgent runs a deterministic
keyword lexicon on both the raw and the cleaned text and, on any high-severity
hit, **forces** the final status to ``CRISIS_LABEL`` regardless of the rest of
the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import (
    CRISIS_LABEL,
    LOW_CONFIDENCE_THRESHOLD,
    NLP_MIN_CHARS_FOR_RELIABLE,
    NLP_MIN_WORDS_FOR_RELIABLE,
)

from .cnn_agent import CNNResult
from .deepface_agent import DeepFaceResult
from .fusion_agent import ImageFusion, MultimodalFusion
from .nlp_agent import NLPResult
from .safety_lexicon import (
    CAT_HOPELESS,
    CAT_SELF_HARM,
    CAT_SEVERE,
    CAT_SUICIDE,
    CrisisHit,
    SafetyAgent,
    has_crisis,
)


_CATEGORY_LABELS: dict[str, str] = {
    CAT_SUICIDE: "suicidal language",
    CAT_SEVERE: "imminent-action wording",
    CAT_SELF_HARM: "self-harm",
    CAT_HOPELESS: "hopelessness",
}


@dataclass
class SupervisorReport:
    final_status: str
    final_explanation: str
    warnings: list[str] = field(default_factory=list)
    flagged_low_confidence: bool = False
    flagged_conflict: bool = False
    flagged_crisis: bool = False
    crisis_hits: list[CrisisHit] = field(default_factory=list)


class SupervisorAgent:
    """Consumes upstream agent outputs and produces the final decision."""

    def __init__(self, low_conf_threshold: float = LOW_CONFIDENCE_THRESHOLD) -> None:
        self.low_conf_threshold = low_conf_threshold
        self._safety = SafetyAgent()

    def review(
        self,
        nlp: NLPResult,
        cnn: CNNResult,
        deepface: DeepFaceResult,
        image_fusion: ImageFusion,
        multimodal: MultimodalFusion,
        raw_text: str = "",
    ) -> SupervisorReport:
        warnings: list[str] = []

        # --- Crisis safety net (deterministic; runs first) ----------------
        crisis_hits = self._safety.scan(raw_text or "", nlp.cleaned_text or "")
        is_crisis = has_crisis(crisis_hits)

        # --- Standard reliability warnings --------------------------------
        ct = (nlp.cleaned_text or "").strip()
        n_words = len(ct.split()) if ct else 0
        if ct and (
            len(ct) < NLP_MIN_CHARS_FOR_RELIABLE
            or n_words < NLP_MIN_WORDS_FOR_RELIABLE
        ):
            warnings.append(
                "Short or informal text — the NLP model was trained on longer sentences; "
                "typos and slang can yield misleading high confidence. Prefer a clear, "
                "full sentence when possible."
            )

        if nlp.confidence < self.low_conf_threshold:
            warnings.append(
                f"Low text confidence ({nlp.confidence:.2f}; threshold {self.low_conf_threshold:.2f})."
            )

        # Face side: single consolidated warning — CNN / DeepFace / resolved often repeat
        # the same underlying issue (weak lighting, angle, blur).
        t = self.low_conf_threshold
        cnn_c, fus_c = cnn.confidence, image_fusion.final_confidence
        df_c = deepface.confidence

        sub: list[str] = []
        if cnn_c < t:
            sub.append(f"CNN {cnn_c:.2f}")
        if df_c is not None and df_c < t:
            sub.append(f"DeepFace {df_c:.2f}")
        if fus_c < t:
            dup = (
                (cnn_c < t and abs(fus_c - cnn_c) < 0.01)
                or (df_c is not None and df_c < t and abs(fus_c - df_c) < 0.01)
            )
            if not dup:
                sub.append(f"resolved {fus_c:.2f}")

        if sub:
            warnings.append(
                "Facial emotion confidence is weak "
                f"(threshold {t:.2f}): {', '.join(sub)}. "
                "Poor lighting, angle, blur, or a small face in the frame often causes this."
            )

        flagged_conflict = multimodal.decision == "Conflict"
        flagged_low = bool(warnings)

        # --- Crisis OVERRIDE ---------------------------------------------
        # A high-severity safety hit replaces the multimodal decision and
        # blocks the soft "Likely" downgrade — Crisis stays Crisis.
        if is_crisis:
            cats_present = sorted(
                {h.category for h in crisis_hits if h.category != CAT_HOPELESS}
            )
            cats_str = ", ".join(_CATEGORY_LABELS.get(c, c) for c in cats_present)
            phrases = ", ".join(repr(h.phrase) for h in crisis_hits[:5])
            warnings.insert(
                0,
                f"Crisis safety net triggered ({cats_str}). Detected: {phrases}. "
                "This overrides the probabilistic NLP/CNN models — please consult a "
                "mental-health professional or hotline.",
            )
            return SupervisorReport(
                final_status=CRISIS_LABEL,
                final_explanation=(
                    "Your text contains explicit crisis language. The system has overridden "
                    "the model output and recommends contacting a mental-health professional "
                    "or hotline immediately."
                ),
                warnings=warnings,
                flagged_low_confidence=flagged_low,
                flagged_conflict=flagged_conflict,
                flagged_crisis=True,
                crisis_hits=crisis_hits,
            )

        # --- Hopelessness-only signal: nudge but do not override ---------
        if any(h.category == CAT_HOPELESS for h in crisis_hits):
            phrases = ", ".join(repr(h.phrase) for h in crisis_hits[:3])
            warnings.append(
                f"Hopelessness language detected ({phrases}). Not a crisis override, but worth attention."
            )

        # --- Standard final status ---------------------------------------
        final_status = multimodal.decision
        if final_status == "High Risk" and flagged_low:
            final_status = "Likely High Risk"

        return SupervisorReport(
            final_status=final_status,
            final_explanation=multimodal.explanation,
            warnings=warnings,
            flagged_low_confidence=flagged_low,
            flagged_conflict=flagged_conflict,
            flagged_crisis=False,
            crisis_hits=crisis_hits,
        )
