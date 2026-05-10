"""OCR Agent: Text extraction from images using EasyOCR.

Handles multilingual OCR, confidence filtering, and quality assessment.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


@dataclass
class OCRResult:
    """Encapsulates OCR output and metadata."""

    extracted_text: str
    confidence: float  # Average confidence of detected text
    num_detections: int
    languages: list[str]
    note: str


class OCRAgent:
    """Extracts text from images using EasyOCR.

    Falls back gracefully if EasyOCR unavailable.
    """

    def __init__(self) -> None:
        """Initialize OCR engine."""
        self.reader = None
        self.available = False
        self.backend = "none"
        self._init_error = ""

        try:
            import easyocr

            # Initialize for common languages (English + common multilingual)
            self.reader = easyocr.Reader(
                ["en"],  # Extend to more languages if needed
                gpu=False,  # CPU-only to avoid memory issues
            )
            self.available = True
            self.backend = "easyocr"
        except ImportError as e:
            self._init_error = f"EasyOCR not installed: {e}"
            self.backend = "none"
        except Exception as e:
            self._init_error = f"EasyOCR init failed: {e}"
            self.backend = "none"

    def extract(self, image: Image.Image, confidence_threshold: float = 0.3) -> OCRResult:
        """Extract text from image.

        Args:
            image: PIL Image
            confidence_threshold: Only include detections above this confidence

        Returns:
            OCRResult with extracted text and metadata
        """
        if not self.available or self.reader is None:
            return OCRResult(
                extracted_text="",
                confidence=0.0,
                num_detections=0,
                languages=["unknown"],
                note=f"OCR unavailable ({self.backend}): {self._init_error}",
            )

        try:
            # Convert PIL to numpy array
            img_array = np.asarray(image)

            # Run OCR
            results = self.reader.readtext(img_array, detail=1)

            if not results:
                return OCRResult(
                    extracted_text="",
                    confidence=0.0,
                    num_detections=0,
                    languages=["en"],
                    note="No text detected in image.",
                )

            # Filter by confidence and extract text
            detections = [
                (item[1], item[2]) for item in results if item[2] >= confidence_threshold
            ]

            if not detections:
                return OCRResult(
                    extracted_text="",
                    confidence=0.0,
                    num_detections=len(results),
                    languages=["en"],
                    note=f"Text found but confidence too low (threshold: {confidence_threshold}).",
                )

            texts = [det[0] for det in detections]
            confidences = [det[1] for det in detections]
            extracted = " ".join(texts)
            avg_confidence = float(np.mean(confidences))

            note = f"✓ Extracted {len(detections)} text region(s), avg confidence: {avg_confidence:.2f}"

            return OCRResult(
                extracted_text=extracted,
                confidence=avg_confidence,
                num_detections=len(detections),
                languages=["en"],
                note=note,
            )

        except Exception as e:
            return OCRResult(
                extracted_text="",
                confidence=0.0,
                num_detections=0,
                languages=["en"],
                note=f"OCR extraction failed: {e}",
            )

    def merge_text(self, user_text: str, ocr_text: str) -> str:
        """Merge user-provided text with OCR-extracted text.

        Args:
            user_text: Text provided by user
            ocr_text: Text extracted from image via OCR

        Returns:
            Merged text (user text first, then OCR)
        """
        parts = []
        if user_text and user_text.strip():
            parts.append(user_text.strip())
        if ocr_text and ocr_text.strip():
            parts.append(f"[From image]: {ocr_text.strip()}")
        return " ".join(parts) if parts else ""
