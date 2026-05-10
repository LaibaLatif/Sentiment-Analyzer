"""XAI Agent: LIME word importances for text + structured face explanation."""
from __future__ import annotations

from dataclasses import dataclass

from .cnn_agent import CNNResult
from .deepface_agent import DeepFaceResult
from .fusion_agent import ImageFusion
from .nlp_agent import NLPAgent, NLPResult


@dataclass
class WordImportance:
    word: str
    weight: float
    strength: str  # "strong" | "medium" | "low"


@dataclass
class XAIReport:
    text_words: list[WordImportance]
    image_explanation: str


def _strength(weight: float) -> str:
    a = abs(weight)
    if a >= 0.15:
        return "strong"
    if a >= 0.08:
        return "medium"
    return "low"


class XAIAgent:
    """Uses LIME for text. For images, summarises CNN vs DeepFace agreement."""

    def __init__(self, nlp_agent: NLPAgent, num_features: int = 12) -> None:
        self.nlp_agent = nlp_agent
        self.num_features = num_features
        self._lime_explainer = None

    def _get_explainer(self):
        """Lazy init — LIME setup is slow; reuse across Analyze clicks."""
        if self._lime_explainer is None:
            from lime.lime_text import LimeTextExplainer

            self._lime_explainer = LimeTextExplainer(
                class_names=["Positive", "Negative"]
            )
        return self._lime_explainer

    # ---- Text -----------------------------------------------------------
    def explain_text(self, nlp_result: NLPResult) -> list[WordImportance]:
        if not nlp_result.cleaned_text:
            return []
        try:
            pred_idx = 1 if nlp_result.short_label == "Negative" else 0
            exp = self._get_explainer().explain_instance(
                nlp_result.cleaned_text,
                self.nlp_agent.predict_proba_raw,
                num_features=self.num_features,
                top_labels=1,
            )
            weights = exp.as_list(label=pred_idx)
            return [
                WordImportance(w, float(v), _strength(float(v))) for w, v in weights
            ]
        except Exception:  # noqa: BLE001
            return []

    # ---- Image ----------------------------------------------------------
    def explain_image(
        self, cnn: CNNResult, deepface: DeepFaceResult, fusion: ImageFusion
    ) -> str:
        df_emo = deepface.emotion or "n/a"
        df_conf = (
            f"{deepface.confidence:.2f}"
            if deepface.confidence is not None
            else "n/a"
        )
        agree = "agree" if fusion.agreement else "differ"
        return (
            f"CNN={cnn.emotion} ({cnn.confidence:.2f}), "
            f"DeepFace={df_emo} ({df_conf}). "
            f"Models {agree}; final emotion: {fusion.final_emotion} "
            f"({fusion.final_confidence:.2f})."
        )

    # ---- One-shot -------------------------------------------------------
    def explain(
        self,
        nlp_result: NLPResult,
        cnn: CNNResult,
        deepface: DeepFaceResult,
        fusion: ImageFusion,
    ) -> XAIReport:
        return XAIReport(
            text_words=self.explain_text(nlp_result),
            image_explanation=self.explain_image(cnn, deepface, fusion),
        )
