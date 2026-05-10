"""Comparison Agent: validates CNN vs DeepFace and picks the image-side emotion.

Policy (submission spec):
    • Agreement  → high confidence, keep shared label.
    • Disagreement → prefer DeepFace as the external benchmark (when available).
"""
from __future__ import annotations

from dataclasses import dataclass

from .cnn_agent import CNNResult
from .deepface_agent import DeepFaceResult


def _norm(name: str | None) -> str:
    return (name or "").strip().lower()


def _confidence_band(p: float) -> str:
    if p >= 0.65:
        return "high"
    if p >= 0.45:
        return "medium"
    return "low"


@dataclass
class ComparisonResult:
    final_emotion: str
    final_confidence: float
    agreement: bool
    confidence_band: str  # "high" | "medium" | "low"
    chosen_source: str  # "agreement" | "deepface" | "cnn"
    note: str

    def to_dict(self) -> dict:
        return {
            "final_emotion": self.final_emotion,
            "confidence": self.confidence_band,
            "agreement": self.agreement,
            "final_score": round(self.final_confidence, 4),
            "chosen_source": self.chosen_source,
            "note": self.note,
        }


class ComparisonAgent:
    """Evaluates two face-emotion sources and applies explicit decision rules."""

    def compare(self, cnn: CNNResult, df: DeepFaceResult) -> ComparisonResult:
        cnn_emo = _norm(cnn.emotion)
        df_emo = _norm(df.emotion)

        if not df_emo:
            band = _confidence_band(cnn.confidence)
            return ComparisonResult(
                final_emotion=cnn_emo,
                final_confidence=cnn.confidence,
                agreement=False,
                confidence_band=band,
                chosen_source="cnn",
                note="DeepFace unavailable — using CNN only.",
            )

        if cnn_emo == df_emo:
            conf = max(cnn.confidence, df.confidence or 0.0)
            return ComparisonResult(
                final_emotion=cnn_emo,
                final_confidence=conf,
                agreement=True,
                confidence_band=_confidence_band(conf),
                chosen_source="agreement",
                note="CNN and DeepFace agree — high reliability.",
            )

        # Disagreement: prefer DeepFace (benchmark / pretrained specialist)
        df_c = float(df.confidence or 0.0)
        return ComparisonResult(
            final_emotion=df_emo,
            final_confidence=df_c,
            agreement=False,
            confidence_band=_confidence_band(df_c),
            chosen_source="deepface",
            note="Models disagree — preferring DeepFace (comparison policy).",
        )
