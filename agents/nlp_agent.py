"""NLP Agent: TF-IDF + Logistic Regression sentiment predictor.

Decision rule
-------------
By default the predicted label is the argmax of ``predict_proba``. If
``models/nlp_threshold.json`` exists (produced by
``scripts/calibrate_nlp_threshold.py``), the agent uses ``P(negative) >= thr``
instead — calibrated on the held-out 20% split so we don't have to retrain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from config import NLP_MODEL_PATH, NLP_THRESHOLD_JSON, NLP_VECTORIZER_PATH
from text_clean import clean_text


@dataclass
class NLPResult:
    label: str  # "Positive (Normal)" or "Negative (Depression)"
    short_label: str  # "Positive" or "Negative"
    confidence: float
    proba_positive: float
    proba_negative: float
    cleaned_text: str


def _load_threshold() -> float:
    """Read the calibrated threshold; fall back to argmax (0.5) if absent."""
    p = Path(NLP_THRESHOLD_JSON)
    if not p.is_file():
        return 0.5
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        thr = float(data.get("threshold", 0.5))
        if 0.05 <= thr <= 0.95:
            return thr
    except Exception:  # noqa: BLE001
        pass
    return 0.5


class NLPAgent:
    """Loads the trained vectorizer + logistic-regression classifier."""

    def __init__(self, threshold: float | None = None) -> None:
        self.vectorizer = joblib.load(NLP_VECTORIZER_PATH)
        self.model = joblib.load(NLP_MODEL_PATH)
        classes = list(self.model.classes_)
        self.idx_pos = classes.index(0) if 0 in classes else 0
        self.idx_neg = classes.index(1) if 1 in classes else 1
        self.threshold = float(threshold) if threshold is not None else _load_threshold()

    def predict_proba_raw(self, raw_texts: list[str]) -> np.ndarray:
        """Used by LIME: returns columns [P(positive), P(negative)]."""
        cleaned = [clean_text(t) for t in raw_texts]
        X = self.vectorizer.transform(cleaned)
        probs = self.model.predict_proba(X)
        return np.stack([probs[:, self.idx_pos], probs[:, self.idx_neg]], axis=1)

    def predict(self, text: str) -> NLPResult:
        cleaned = clean_text(text)
        if not cleaned:
            return NLPResult(
                label="Positive (Normal)",
                short_label="Positive",
                confidence=0.0,
                proba_positive=1.0,
                proba_negative=0.0,
                cleaned_text="",
            )
        # One transform + one predict (LIME still uses `predict_proba_raw`).
        X = self.vectorizer.transform([cleaned])
        row = self.model.predict_proba(X)[0]
        p_pos, p_neg = float(row[self.idx_pos]), float(row[self.idx_neg])
        # Calibrated threshold on P(negative); fall back to argmax when thr=0.5.
        if p_neg >= self.threshold:
            return NLPResult(
                label="Negative (Depression)",
                short_label="Negative",
                confidence=p_neg,
                proba_positive=p_pos,
                proba_negative=p_neg,
                cleaned_text=cleaned,
            )
        return NLPResult(
            label="Positive (Normal)",
            short_label="Positive",
            confidence=p_pos,
            proba_positive=p_pos,
            proba_negative=p_neg,
            cleaned_text=cleaned,
        )
