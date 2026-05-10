"""Agent-based architecture for the multimodal mental-health system.

Agents:
    DIImageProcessor   - digital image processing (face detection, enhancement)
    OCRAgent           - optical character recognition (text extraction)
    NLPAgent           - text -> sentiment + confidence
    HPTAgent           - training-time grid search for logistic regression
    CVAgent            - face image -> emotion (trained CNN; coursework name)
    CNNAgent           - alias base class for the CNN weights
    DeepFaceAgent      - pretrained face emotion (DeepFace or HF fallback)
    ComparisonAgent    - CNN vs DeepFace validation / benchmark selection
    FusionAgent        - multimodal fusion (text + resolved face emotion)
    SafetyAgent        - deterministic crisis-keyword override (lexicon)
    SupervisorAgent    - final decision, conflicts, low-confidence handling,
                         consumes SafetyAgent for crisis override
    SuggestionsAgent   - mode-aware wellbeing tips (deterministic, offline)
    XAIAgent           - LIME + image-side explanation
"""

from .cnn_agent import CNNAgent
from .comparison_agent import ComparisonAgent, ComparisonResult
from .cv_agent import CVAgent
from .deepface_agent import DeepFaceAgent
from .dip_processor import DIImageProcessor, DIProcessingResult
from .fusion_agent import FusionAgent, ImageFusion, MultimodalFusion
from .hpt_agent import HPTAgent
from .nlp_agent import NLPAgent, NLPResult
from .ocr_agent import OCRAgent, OCRResult
from .safety_lexicon import (
    CrisisHit,
    SafetyAgent,
    find_crisis_signals,
    get_hotlines_for_region,
)
from .suggestions_agent import Suggestion, mode_label, suggestions_for
from .supervisor_agent import SupervisorAgent, SupervisorReport
from .xai_agent import XAIAgent

__all__ = [
    "CNNAgent",
    "ComparisonAgent",
    "ComparisonResult",
    "CrisisHit",
    "CVAgent",
    "DeepFaceAgent",
    "DIImageProcessor",
    "DIProcessingResult",
    "FusionAgent",
    "HPTAgent",
    "ImageFusion",
    "MultimodalFusion",
    "NLPAgent",
    "NLPResult",
    "OCRAgent",
    "OCRResult",
    "SafetyAgent",
    "Suggestion",
    "SupervisorAgent",
    "SupervisorReport",
    "XAIAgent",
    "find_crisis_signals",
    "get_hotlines_for_region",
    "mode_label",
    "suggestions_for",
]
