"""
plugins/lstm_v1/lstm_v1_plugin.py

Production inference plugin for the LSTM touch-detection model trained in
mediapipeDetector/.

Architecture summary
────────────────────
Model  : 2-layer LSTM (input_size=8, hidden_size=64, num_layers=2)
         Classifier: Linear(64, 32) -> ReLU -> Dropout -> Linear(32, 1)
Input  : (batch=5, seq_len=5, feature_dim=8)
         5 fingers × 5-frame sliding window × 8 features per finger
Output : (batch=5,), binary touch probability per finger
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.interfaces.touch_model import ITouchModel, register_model
from core.pipeline.feature_extractor import scale_variant_tensor

# ── Model architecture (matches best_finger_touch_lstm.pth exactly) ───────────

class _FingerTouchLSTM(nn.Module):
    """PyTorch LSTM touch classification network matching trained weights."""

    def __init__(
        self,
        input_features: int = 8,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_units, out_features=32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_features)
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]  # (batch_size, hidden_units)
        logits = self.classifier(last_timestep)  # (batch_size, 1)
        return logits.squeeze(-1)


# ── Finger joint mappings ──────────────────────────────────────────────────────

_FINGER_JOINT_MAP: dict[str, dict[str, str]] = {
    "Thumb":  {"pip": "thumb_mcp", "dip": "thumb_ip",  "tip": "thumb_tip"},
    "Index":  {"pip": "index_pip", "dip": "index_dip", "tip": "index_tip"},
    "Middle": {"pip": "middle_pip", "dip": "middle_dip", "tip": "middle_tip"},
    "Ring":   {"pip": "ring_pip",   "dip": "ring_dip",   "tip": "ring_tip"},
    "Pinky":  {"pip": "pinky_pip",  "dip": "pinky_dip",  "tip": "pinky_tip"},
}

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FEATURE_DIM = 8


# ── Plugin class ───────────────────────────────────────────────────────────────

@register_model(
    name="LSTM v1 (Coords+Vel)",
    description="2-layer LSTM touch detection model (8 features per finger, 5-frame temporal window).",
    weights_file="best_finger_touch_lstm.pth",
)
class LSTMv1Plugin(ITouchModel):
    """Inference adapter for the PyTorch LSTM touch detection model."""

    def __init__(self) -> None:
        self._model: _FingerTouchLSTM | None = None
        self._device = torch.device("cpu")

    # ── ITouchModel interface ──────────────────────────────────────────────────

    def load(self, weights_path: str) -> None:
        self._model = _FingerTouchLSTM(input_features=FEATURE_DIM, hidden_units=64, num_layers=2)
        state = torch.load(weights_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, window_5_frames: list[dict[str, float]]) -> dict[str, float]:
        """
        Parameters
        ----------
        window_5_frames : 5 norm-dicts from HandScaleNormalizer.build_norm_dict().

        Returns
        -------
        {"Thumb": p, "Index": p, "Middle": p, "Ring": p, "Pinky": p}
        """
        if self._model is None:
            return {f: 0.0 for f in FINGERS}

        # Build (5_fingers, 5_frames, 8) tensor
        batch = np.stack(
            [self._extract_per_finger_sequence(window_5_frames, f) for f in FINGERS],
            axis=0,
        )
        batch = scale_variant_tensor(batch, "coords_2d")

        tensor = torch.from_numpy(batch.astype(np.float32)).to(self._device)

        with torch.inference_mode():
            logits = self._model(tensor)          # (5,)
            probs  = torch.sigmoid(logits).cpu().numpy().tolist()

        return dict(zip(FINGERS, probs))

    # ── Private feature extraction ─────────────────────────────────────────────

    def _extract_per_finger_sequence(
        self,
        window_5_frames: list[dict[str, float]],
        finger: str,
    ) -> np.ndarray:
        """
        Extracts 8 features for the specified finger across 5 frames:
        [wrist_x, wrist_y, pip_x, pip_y, dip_x, dip_y, tip_x, tip_y]
        """
        j_map = _FINGER_JOINT_MAP[finger]
        pip_name = j_map["pip"]
        dip_name = j_map["dip"]
        tip_name = j_map["tip"]

        rows = []
        for f_data in window_5_frames:
            wx = f_data.get("wrist_x", 0.0)
            wy = f_data.get("wrist_y", 0.0)
            px = f_data.get(f"{pip_name}_x", 0.0)
            py = f_data.get(f"{pip_name}_y", 0.0)
            dx = f_data.get(f"{dip_name}_x", 0.0)
            dy = f_data.get(f"{dip_name}_y", 0.0)
            tx = f_data.get(f"{tip_name}_x", 0.0)
            ty = f_data.get(f"{tip_name}_y", 0.0)
            rows.append([wx, wy, px, py, dx, dy, tx, ty])

        return np.array(rows, dtype=np.float32)  # shape: (5, 8)

