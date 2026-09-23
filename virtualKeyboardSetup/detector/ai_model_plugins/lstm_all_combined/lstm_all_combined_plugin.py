"""
plugins/lstm_all_combined/lstm_all_combined_plugin.py

Champion Touch Detection Model: LSTM_All_Combined (all_joints_coords_vel_speed).
Achieved top benchmark 94.34% Test Accuracy, 94.97% Touch Recall, 0.9437 F1-Score.

Architecture:
  - SequenceLSTM: input_features=45, hidden_units=48, num_layers=2, dropout=0.25
  - Classifier Head: Linear(48, 24) -> ReLU -> Dropout(0.25) -> Linear(24, 1)
  - Feature Variant: all_joints_coords_vel_speed (4 steps x 45 features per finger)
  - Pre-scaled via dataset StandardScaler (scalers.npz: all_joints_coords_vel_speed_mean/scale)
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


class _SequenceLSTM(nn.Module):
    """PyTorch LSTM Touch Detection Network matching trained weights."""

    def __init__(
        self,
        input_features: int = 45,
        hidden_units: int = 48,
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@register_model(
    name="LSTM All-Combined (94.3% Acc)",
    description="Champion 2-layer LSTM trained on all hand joints coords, velocities and speeds (45 features, 4 steps).",
    weights_file="LSTM_All_Combined_cfg01.pth",
)
class LSTMAllCombinedPlugin(ITouchModel):
    """Production plugin for LSTM_All_Combined model."""

    def __init__(self) -> None:
        self._model: _SequenceLSTM | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, weights_path: str) -> None:
        self._model = _SequenceLSTM(input_features=45, hidden_units=48, num_layers=2, dropout=0.25)
        state = torch.load(weights_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, window_5_frames: list[dict[str, float]]) -> dict[str, float]:
        if self._model is None or len(window_5_frames) < 5:
            return {f: 0.0 for f in FINGERS}

        # 1. Compute 4 velocity steps across 5 frames (Step 8)
        v_steps = compute_window_velocities(window_5_frames)

        # 2. Unroll 5 per-finger rows (Step 11)
        finger_rows = unroll_per_finger_window(window_5_frames, v_steps)

        # 3. Extract all_joints_coords_vel tensor: shape (5, 4, 36)
        X_raw = extract_variant_tensor(finger_rows, VARIANT_NAME)

        # 4. Standard scale using training distribution parameters
        X_scaled = scale_variant_tensor(X_raw, VARIANT_NAME)

        # 5. Forward inference
        tensor = torch.from_numpy(X_scaled).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        return dict(zip(FINGERS, [float(p) for p in probs]))
