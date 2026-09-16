"""
plugins/lstm_v1/lstm_v1_plugin.py

Example inference plugin for the LSTM touch-detection model trained in
mediapipeDetector/deepLearningModels/.

This plugin expects the best_finger_touch_lstm.pth weights from the training
pipeline to be placed alongside this file.

Architecture summary
────────────────────
Input  : (batch=5, seq_len=5, feature_dim=84)
          5 fingers × 5-frame window × 84 features per finger
Output : (batch=5,) — one sigmoid probability per finger

Feature extraction (matches training pipeline stage5 + stage6):
  Per frame: 21-landmark normalised coords (63 values)
  Velocities: computed across 4 frame pairs (delta_x, delta_y, delta_z + speed_2d, speed_3d)
  Final features for LSTM: coords + velocities = 63 + (21×5) = 168 → then filtered to 84
  NOTE: Update FEATURE_DIM and _extract_features() to match your exact trained variant.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from core.interfaces.touch_model import ITouchModel, register_model

# ── Model architecture (must match training-time definition exactly) ───────────

class _LSTMTouchNet(nn.Module):
    """Compact single-layer LSTM for 5-finger binary touch classification."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Use last time step's output
        logits = self.fc(out[:, -1, :])   # (batch, 1)
        return logits.squeeze(-1)          # (batch,)


# ── Landmark metadata ──────────────────────────────────────────────────────────

_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip",  "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp",  "ring_pip",  "ring_dip",  "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

_FINGER_JOINT_INDICES: dict[str, list[int]] = {
    "Thumb":  [0, 1, 2, 3, 4],    # wrist + 4 thumb joints
    "Index":  [0, 5, 6, 7, 8],
    "Middle": [0, 9, 10, 11, 12],
    "Ring":   [0, 13, 14, 15, 16],
    "Pinky":  [0, 17, 18, 19, 20],
}

# Feature dimension per finger per frame
# coords: 5 joints × 3 (x,y,z) = 15 per frame
# velocities: 4 steps × 5 joints × 2 (vx, vy) = 40 per window → averaged to per-frame = N/A
# Adjust FEATURE_DIM to match your training variant exactly.
FEATURE_DIM = 84    # ← UPDATE THIS to match your trained model's input_size

# ── Plugin class ───────────────────────────────────────────────────────────────

@register_model(
    name="LSTM v1 — Coords+Vel",
    description="Single-layer LSTM trained on combined coords+velocity features. 5-frame window, 84-dim input.",
    weights_file="best_finger_touch_lstm.pth",
)
class LSTMv1Plugin(ITouchModel):
    """
    Inference adapter for the LSTM touch detection model.

    Adapt _extract_features() to match the exact feature engineering used during
    training (check your training notebook / run_all.py for the active variant).
    """

    def __init__(self) -> None:
        self._model: _LSTMTouchNet | None = None
        self._device = torch.device("cpu")

    # ── ITouchModel interface ──────────────────────────────────────────────────

    def load(self, weights_path: str) -> None:
        self._model = _LSTMTouchNet(input_size=FEATURE_DIM)
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
            return {f: 0.0 for f in _FINGER_JOINT_INDICES}

        fingers = list(_FINGER_JOINT_INDICES.keys())
        # Build (5_fingers, 5_frames, feature_dim) tensor
        batch = np.stack(
            [self._extract_per_finger_sequence(window_5_frames, f) for f in fingers],
            axis=0,
        )   # shape (5, 5, feature_dim)

        tensor = torch.from_numpy(batch.astype(np.float32)).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)          # (5,)
            probs  = torch.sigmoid(logits).cpu().numpy().tolist()

        return dict(zip(fingers, probs))

    # ── Private feature extraction ─────────────────────────────────────────────

    def _extract_per_finger_sequence(
        self,
        window_5_frames: list[dict[str, float]],
        finger: str,
    ) -> np.ndarray:
        """
        Build a (5, FEATURE_DIM) array for one finger across 5 frames.
        Modify this method to match your training feature engineering exactly.

        Default here: coords (x,y,z) of the finger's 5 joints + wrist = 15 values,
        padded with zeros to FEATURE_DIM.
        """
        joint_indices = _FINGER_JOINT_INDICES[finger]   # 5 joint indices
        rows = []
        for frame_dict in window_5_frames:
            coords = []
            for j_idx in joint_indices:
                name = _LANDMARK_NAMES[j_idx]
                coords.extend([
                    frame_dict.get(f"{name}_x", 0.0),
                    frame_dict.get(f"{name}_y", 0.0),
                    frame_dict.get(f"{name}_z", 0.0),
                ])
            # Pad or trim to FEATURE_DIM
            vec = coords[:FEATURE_DIM]
            vec.extend([0.0] * max(0, FEATURE_DIM - len(vec)))
            rows.append(vec)
        return np.array(rows, dtype=np.float32)   # (5, FEATURE_DIM)
