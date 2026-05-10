"""DeepFace Agent: pretrained, no-training emotion detector.

Tries the real `deepface` library first (preferred when TensorFlow is available).
On Python versions where TensorFlow has no wheel (e.g. 3.13/3.14), falls back to
a HuggingFace ViT-based facial emotion classifier so the project still works.

Both backends expose the same interface and emotion vocabulary so downstream
fusion/supervisor code does not change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class DeepFaceResult:
    emotion: str | None
    confidence: float | None
    probabilities: dict[str, float]
    backend: str  # "deepface" | "hf-vit" | "none"
    note: str  # error / status message (empty on success)


class DeepFaceAgent:
    def __init__(self) -> None:
        self.backend = "none"
        self._deepface = None
        self._hf_pipe = None
        self._init_error = ""

        try:
            from deepface import DeepFace  # noqa: F401
            self._deepface = DeepFace
            self.backend = "deepface"
            return
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"deepface unavailable ({exc.__class__.__name__})"

        try:
            from transformers import pipeline

            self._hf_pipe = pipeline(
                task="image-classification",
                model="dima806/facial_emotions_image_detection",
                top_k=None,
            )
            self.backend = "hf-vit"
        except Exception as exc:  # noqa: BLE001
            self._init_error += f"; hf fallback failed ({exc})"
            self.backend = "none"

    # ---------- backend implementations ----------
    def _predict_deepface(self, image: Image.Image) -> DeepFaceResult:
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        try:
            out = self._deepface.analyze(  # type: ignore[union-attr]
                bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:  # noqa: BLE001
            return DeepFaceResult(None, None, {}, "deepface", f"DeepFace error: {exc}")
        if isinstance(out, list):
            out = out[0]
        emo = {k.lower(): float(v) for k, v in (out.get("emotion") or {}).items()}
        # DeepFace returns percentages; normalise to 0-1
        total = sum(emo.values()) or 1.0
        probs = {k: v / total for k, v in emo.items()}
        if not probs:
            return DeepFaceResult(None, None, {}, "deepface", "no face emotion returned")
        dom = max(probs, key=probs.get)
        return DeepFaceResult(dom, float(probs[dom]), probs, "deepface", "")

    def _predict_hf(self, image: Image.Image) -> DeepFaceResult:
        try:
            out = self._hf_pipe(image.convert("RGB"))  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            return DeepFaceResult(None, None, {}, "hf-vit", f"HF model error: {exc}")
        probs = {item["label"].lower(): float(item["score"]) for item in out}
        if not probs:
            return DeepFaceResult(None, None, {}, "hf-vit", "no scores returned")
        dom = max(probs, key=probs.get)
        return DeepFaceResult(dom, float(probs[dom]), probs, "hf-vit", "")

    # ---------- public API ----------
    def predict(self, image: Image.Image) -> DeepFaceResult:
        if self.backend == "deepface":
            return self._predict_deepface(image)
        if self.backend == "hf-vit":
            return self._predict_hf(image)
        return DeepFaceResult(None, None, {}, "none", self._init_error or "no backend")
