"""Fusion Agent: image-level fusion (CNN + DeepFace) and multimodal fusion (text + image)."""
from __future__ import annotations

from dataclasses import dataclass

from .cnn_agent import CNNResult
from .comparison_agent import ComparisonAgent, ComparisonResult
from .deepface_agent import DeepFaceResult
from .nlp_agent import NLPResult


@dataclass
class ImageFusion:
    final_emotion: str
    final_confidence: float
    chosen_source: str  # "cnn" | "deepface" | "agreement"
    agreement: bool
    note: str
    confidence_band: str = "medium"  # "high" | "medium" | "low" (from Comparison Agent)


@dataclass
class MultimodalFusion:
    decision: str  # "High Risk" | "Stable" | "Conflict" | "Mixed / monitor"
    explanation: str
    text_signal: str  # "negative" | "positive"
    face_signal: str  # "negative" | "positive" | "neutral"


NEG_FACE = {"sad", "angry", "fear", "disgust"}
POS_FACE = {"happy", "surprise"}
NEU_FACE = {"neutral"}


def _norm(name: str | None) -> str:
    return (name or "").strip().lower()


class FusionAgent:
    """Combines model outputs at two levels.

    Image path delegates to the **Comparison Agent** (CNN vs DeepFace), then
    multimodal rules combine NLP + resolved face emotion.
    """

    def __init__(self) -> None:
        self._comparison = ComparisonAgent()

    # ---- Image-level: Comparison Agent → ImageFusion -------------------
    def fuse_image(
        self, cnn: CNNResult, df: DeepFaceResult
    ) -> tuple[ImageFusion, ComparisonResult]:
        cr = self._comparison.compare(cnn, df)
        fused = ImageFusion(
            final_emotion=cr.final_emotion,
            final_confidence=cr.final_confidence,
            chosen_source=cr.chosen_source,
            agreement=cr.agreement,
            note=cr.note,
            confidence_band=cr.confidence_band,
        )
        return fused, cr

    # ---- Multimodal fusion (Step 10) -----------------------------------
    def fuse_multimodal(
        self, nlp: NLPResult, image_fusion: ImageFusion
    ) -> MultimodalFusion:
        text_signal = "negative" if nlp.short_label == "Negative" else "positive"
        fe = _norm(image_fusion.final_emotion)

        if fe in NEG_FACE:
            face_signal = "negative"
        elif fe in POS_FACE:
            face_signal = "positive"
        elif fe in NEU_FACE:
            face_signal = "neutral"
        else:
            face_signal = "unknown"

        if text_signal == "negative" and face_signal == "negative":
            return MultimodalFusion(
                "High Risk",
                "Negative text and distressing facial emotion align.",
                text_signal,
                face_signal,
            )
        if text_signal == "positive" and face_signal == "positive":
            return MultimodalFusion(
                "Stable",
                "Positive text and a positive facial emotion align.",
                text_signal,
                face_signal,
            )
        if text_signal == "positive" and face_signal == "neutral":
            return MultimodalFusion(
                "Stable",
                "Text appears stable; face is neutral.",
                text_signal,
                face_signal,
            )
        if text_signal == "negative" and face_signal == "neutral":
            return MultimodalFusion(
                "Mixed / monitor",
                "Text suggests concern but face is neutral — monitor.",
                text_signal,
                face_signal,
            )
        return MultimodalFusion(
            "Conflict",
            "Text and face signals disagree.",
            text_signal,
            face_signal,
        )
