"""
plugins/bilstm_all_combined/bilstm_plugin.py

Top Benchmark BiLSTM Model: BiLSTM_All_Combined (all_joints_coords_vel_speed).
Achieved 91.59% Test Accuracy, 92.14% Touch Recall, 0.9164 F1-Score.

Architecture:
  - BiLSTM: input_features=45, hidden_units=48, num_layers=2, bidirectional=True
  - Feature Variant: all_joints_coords_vel_speed (4 steps x 45 features per finger)
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


class _BiLSTM(nn.Module):
    """Bidirectional LSTM matching trained BiLSTM weights."""

    def __init__(self, input_features: int = 45, hidden_units: int = 48, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@register_model(
    name="BiLSTM All-Combined (91.6% Acc)",
    description="Bidirectional LSTM trained on all hand joints coords, velocities and speeds (45 features, 4 steps).",
    weights_file="BiLSTM_All_Combined_cfg01.pth",
)
class BiLSTMAllCombinedPlugin(ITouchModel):
    """Production plugin for BiLSTM_All_Combined model."""

    def __init__(self) -> None:
        self._model: _BiLSTM | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, weights_path: str) -> None:
        self._model = _BiLSTM(input_features=45, hidden_units=48, num_layers=2)
        state = torch.load(weights_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, window_5_frames: list[dict[str, float]]) -> dict[str, float]:
        if self._model is None or len(window_5_frames) < 5:
            return {f: 0.0 for f in FINGERS}

        # 1. Compute 4 velocity steps
        v_steps = compute_window_velocities(window_5_frames)

        # 2. Unroll 5 per-finger rows
        finger_rows = unroll_per_finger_window(window_5_frames, v_steps)

        # 3. Extract all_joints_coords_vel tensor: shape (5, 4, 36)
        X_raw = extract_variant_tensor(finger_rows, VARIANT_NAME)

        # 4. Standard scale
        X_scaled = scale_variant_tensor(X_raw, VARIANT_NAME)

        # 5. Forward inference
        tensor = torch.from_numpy(X_scaled).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        return dict(zip(FINGERS, [float(p) for p in probs]))
