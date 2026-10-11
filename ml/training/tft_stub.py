"""
Temporal Fusion Transformer (TFT) advanced forecasting stub.

Production path: train on multi-station spatiotemporal panels with pytorch-forecasting.
This module documents the interface and provides a horizon-averaging placeholder that
the API can call once real weights are registered in MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TFTConfig:
    max_encoder_length: int = 48
    max_prediction_length: int = 24
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1


def predict_horizons(history_pm25: List[float], horizons: List[int]) -> dict:
    """Placeholder multi-horizon forecast from recent history persistence + drift."""
    if not history_pm25:
        base = 80.0
    else:
        base = float(sum(history_pm25[-6:]) / min(6, len(history_pm25)))
    out = {}
    for h in horizons:
        mean = base + 0.3 * h
        out[h] = {"mean": mean, "p10": max(1.0, mean * 0.75), "p90": mean * 1.25, "model": "tft_stub"}
    return out


if __name__ == "__main__":
    print(predict_horizons([70, 75, 90, 88, 95], [1, 6, 12, 24]))
