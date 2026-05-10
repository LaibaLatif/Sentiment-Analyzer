"""Digital Image Processing (DIP) Preprocessing Agent.

Handles image enhancement, face detection, and quality assessment.
Techniques:
    • Contrast Limited Adaptive Histogram Equalization (CLAHE)
    • Non-Local Means Denoising
    • Face detection (OpenCV Cascade + MediaPipe fallback)
    • Quality assessment (blur detection, brightness check)
    • Automatic color space conversions
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Suppress warnings from optional imports
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class DIProcessingResult:
    """Encapsulates preprocessing output and metadata."""

    original_image: Image.Image
    processed_image: Image.Image
    face_detected: bool
    face_region: tuple[int, int, int, int] | None  # (x, y, w, h) or None
    quality_score: float  # 0.0 to 1.0
    quality_issues: list[str]  # ["blurry", "too_dark", "too_bright"] etc
    enhancement_applied: str  # e.g., "CLAHE + Denoise"
    confidence_band: str  # "high" | "medium" | "low"
    note: str  # Status/warning message


class DIImageProcessor:
    """Preprocesses images using DIP enhancement and face detection."""

    def __init__(self) -> None:
        """Initialize cascade classifiers and preprocessing parameters."""
        self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        try:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            self.has_cascade = not self.face_cascade.empty()
        except Exception:
            self.has_cascade = False

        # Try to load mediapipe for better face detection (optional)
        self._mediapipe_detector = None
        try:
            import mediapipe as mp  # noqa: F401

            self._mediapipe_available = True
        except ImportError:
            self._mediapipe_available = False

        # DIP parameters
        self.clahe_clip_limit = 2.0
        self.clahe_tile_size = 8
        self.denoise_h = 10
        self.blur_threshold = 100  # Laplacian variance threshold
        self.min_brightness = 30
        self.max_brightness = 220

    def preprocess(self, image: Image.Image) -> DIProcessingResult:
        """Main preprocessing pipeline.

        Args:
            image: PIL Image (any format/size)

        Returns:
            DIProcessingResult with processed image and metadata
        """
        original = image.copy()

        # Convert PIL to OpenCV (BGR)
        img_cv = self._pil_to_cv(image)
        original_cv = img_cv.copy()

        # 1. Assess quality
        quality_score, quality_issues = self._assess_quality(img_cv)

        # 2. Detect face
        face_region = self._detect_face(img_cv)

        # 3. Extract / crop face if detected
        if face_region is not None:
            x, y, w, h = face_region
            # Add small padding
            pad = max(5, w // 10)
            y1 = max(0, y - pad)
            x1 = max(0, x - pad)
            y2 = min(img_cv.shape[0], y + h + pad)
            x2 = min(img_cv.shape[1], x + w + pad)
            face_crop = img_cv[y1:y2, x1:x2].copy()
        else:
            face_crop = img_cv

        # 4. Apply enhancements to detected/full region
        enhanced = self._enhance_image(face_crop)
        enhancement_applied = self._get_enhancement_label(quality_issues)

        # 5. Determine confidence band
        confidence_band = self._get_confidence_band(quality_score, face_region is not None)

        # Convert back to PIL
        processed_pil = self._cv_to_pil(enhanced)

        note = self._build_note(face_region, quality_issues, quality_score)

        return DIProcessingResult(
            original_image=original,
            processed_image=processed_pil,
            face_detected=face_region is not None,
            face_region=face_region,
            quality_score=quality_score,
            quality_issues=quality_issues,
            enhancement_applied=enhancement_applied,
            confidence_band=confidence_band,
            note=note,
        )

    def _pil_to_cv(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV (BGR)."""
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _cv_to_pil(self, img_cv: np.ndarray) -> Image.Image:
        """Convert OpenCV (BGR) to PIL Image."""
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _assess_quality(self, img_cv: np.ndarray) -> tuple[float, list[str]]:
        """Assess image quality: blur, brightness, contrast.

        Returns:
            (quality_score: 0-1, issues: list of problem strings)
        """
        issues = []
        score = 1.0

        # 1. Blur detection (Laplacian variance)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.blur_threshold:
            issues.append("blurry")
            score -= 0.3

        # 2. Brightness assessment
        brightness = np.mean(gray)
        if brightness < self.min_brightness:
            issues.append("too_dark")
            score -= 0.2
        elif brightness > self.max_brightness:
            issues.append("too_bright")
            score -= 0.15

        # 3. Contrast check
        contrast = np.std(gray)
        if contrast < 20:
            issues.append("low_contrast")
            score -= 0.15

        return max(0.0, score), issues

    def _detect_face(self, img_cv: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect faces using cascades + mediapipe fallback.

        Returns:
            (x, y, w, h) of largest face, or None if no face found
        """
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Try cascade first (fast)
        if self.has_cascade:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                # Return largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                return tuple(largest)

        # Try mediapipe if cascade didn't find face
        if self._mediapipe_available:
            try:
                import mediapipe as mp

                face_detection = mp.solutions.face_detection.FaceDetection()
                h, w = img_cv.shape[:2]
                rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb)

                if results.detections:
                    # Get largest detection
                    largest_det = max(
                        results.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height,
                    )
                    bbox = largest_det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    fw = int(bbox.width * w)
                    fh = int(bbox.height * h)
                    return (x, y, fw, fh)
            except Exception:
                pass

        return None

    def _enhance_image(self, img_cv: np.ndarray) -> np.ndarray:
        """Apply DIP enhancement: CLAHE + denoising + histogram equalization.

        Args:
            img_cv: BGR image (OpenCV format)

        Returns:
            Enhanced BGR image
        """
        # Convert to LAB for better contrast enhancement
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
        )
        l_clahe = clahe.apply(l)

        # Merge back
        lab_enhanced = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Apply non-local means denoising
        try:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, h=self.denoise_h, templateWindowSize=7, searchWindowSize=21
            )
        except Exception:
            pass  # Fallback: continue without denoising

        return enhanced

    def _get_enhancement_label(self, quality_issues: list[str]) -> str:
        """Generate descriptive label of applied enhancements."""
        base = "CLAHE + Denoise"
        if not quality_issues:
            return base
        return f"{base} (fixed: {', '.join(quality_issues)})"

    def _get_confidence_band(self, quality_score: float, face_found: bool) -> str:
        """Map quality score + face detection to confidence band."""
        if not face_found:
            return "low"
        if quality_score >= 0.7:
            return "high"
        if quality_score >= 0.4:
            return "medium"
        return "low"

    def _build_note(
        self, face_region: tuple | None, quality_issues: list[str], quality_score: float
    ) -> str:
        """Build user-friendly status message."""
        parts = []
        if face_region is None:
            parts.append("⚠ No face detected; using full image.")
        else:
            parts.append("✓ Face detected and extracted.")

        if quality_issues:
            parts.append(f"Issues found: {', '.join(quality_issues)}.")
        else:
            parts.append("✓ Image quality good.")

        parts.append(f"Quality score: {quality_score:.2f}/1.0")
        return " ".join(parts)
