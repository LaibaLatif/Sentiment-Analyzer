"""CNN Agent: our trained PyTorch emotion classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config import CNN_CLASS_NAMES, CNN_CLASS_NAMES_JSON, CNN_MODEL_PATH, IMG_SIZE


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


@dataclass
class CNNResult:
    emotion: str
    confidence: float
    probabilities: dict[str, float]


def _load_class_names() -> list[str]:
    p = Path(CNN_CLASS_NAMES_JSON)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return list(CNN_CLASS_NAMES)


class CNNAgent:
    """Loads the trained CNN. Supports test-time augmentation (TTA).

    TTA averages the softmax over the original image and its horizontal mirror.
    It costs one extra forward pass and is on by default (small accuracy bump,
    no retraining). Disable by passing ``tta=False`` for benchmarks.
    """

    def __init__(self, device: str | None = None, *, tta: bool = True) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tta = bool(tta)
        self.class_names = _load_class_names()
        try:
            payload = torch.load(
                CNN_MODEL_PATH, map_location=self.device, weights_only=True
            )
        except TypeError:
            payload = torch.load(CNN_MODEL_PATH, map_location=self.device)
        if isinstance(payload, dict) and "state_dict" in payload:
            num_classes = int(payload.get("num_classes", len(self.class_names)))
            state = payload["state_dict"]
        else:
            state = payload
            num_classes = len(self.class_names)
        self.model = SmallCNN(num_classes).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> CNNResult:
        x = self._preprocess(image)
        with torch.inference_mode():
            logits = self.model(x)
            if self.tta:
                # Horizontal-flip TTA — averages two forward passes for a small
                # accuracy bump on near-symmetric face crops.
                logits = (logits + self.model(torch.flip(x, dims=[3]))) / 2
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return CNNResult(
            emotion=self.class_names[idx],
            confidence=float(probs[idx]),
            probabilities={n: float(p) for n, p in zip(self.class_names, probs)},
        )
