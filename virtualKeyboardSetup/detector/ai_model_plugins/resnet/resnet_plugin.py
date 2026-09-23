"""
plugins/resnet/resnet_plugin.py

ResNet1D Touch Detection Model: ResNet1D_All_Combined (all_joints_coords_vel_speed).
Achieved 90.50% Test Accuracy, 90.17% Touch Recall.

Architecture:
  - TouchResNet1D: residual 1D CNN blocks with skip connections over 45 features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.interfaces.touch_model import ITouchModel, register_model
from core.pipeline.feature_extractor import (
    compute_window_velocities,
    extract_variant_tensor,
    scale_variant_tensor,
    unroll_per_finger_window,
)

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
VARIANT_NAME = "all_joints_coords_vel_speed"


class _ResBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class _TouchResNet1D(nn.Module):
    def __init__(self, input_features: int = 45, hidden_dim: int = 48, dropout: float = 0.25):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.b1   = _ResBlock1D(hidden_dim, dropout)
        self.b2   = _ResBlock1D(hidden_dim, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        z = self.in_conv(x.permute(0, 2, 1))
        z = self.b1(z)
        z = self.b2(z)
        return self.head(self.pool(z)).squeeze(-1)



@register_model(
    name="ResNet1D All-Combined (90.5% Acc)",
    description="1D Residual CNN with skip connections trained on all joints coords, velocities and speeds.",
    weights_file="ResNet1D_All_Combined_cfg01.pth",
)
class ResNet1DPlugin(ITouchModel):
    """Production plugin for ResNet1D model."""

    def __init__(self) -> None:
        self._model: _TouchResNet1D | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, weights_path: str) -> None:
        self._model = _TouchResNet1D(input_features=45, hidden_dim=48, dropout=0.25)
        state = torch.load(weights_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, window_5_frames: list[dict[str, float]]) -> dict[str, float]:
        if self._model is None or len(window_5_frames) < 5:
            return {f: 0.0 for f in FINGERS}

        v_steps = compute_window_velocities(window_5_frames)
        finger_rows = unroll_per_finger_window(window_5_frames, v_steps)
        X_raw = extract_variant_tensor(finger_rows, VARIANT_NAME)
        X_scaled = scale_variant_tensor(X_raw, VARIANT_NAME)

        tensor = torch.from_numpy(X_scaled).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        return dict(zip(FINGERS, [float(p) for p in probs]))
