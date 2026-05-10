"""CV Agent — computer-vision path for facial emotion (trained CNN).

Coursework wording: “independently analyzes facial data using a trained CNN”.
Implementation inherits `CNNAgent` unchanged.
"""
from __future__ import annotations

from .cnn_agent import CNNAgent, CNNResult, SmallCNN

__all__ = ["CVAgent", "CNNResult", "SmallCNN"]


class CVAgent(CNNAgent):
    """Face emotion prediction using our trained convolutional network."""

    pass
