# Plugins Directory

Place your trained touch-detection model files here.

## Structure

Each model lives in its own sub-directory:

```
plugins/
└── my_model/
    ├── my_model_plugin.py     ← Plugin adapter file
    └── my_weights.pth         ← Trained PyTorch weights
```

## How to create a plugin

1. Create a sub-directory under `plugins/`.
2. Copy your trained `.pth` weights file into it.
3. Create a Python adapter file with the `@register_model` decorator:

```python
from core.interfaces.touch_model import register_model, ITouchModel
import torch
import numpy as np

@register_model(
    name="My Model — Short Name",
    description="Brief description of architecture and training data.",
    weights_file="my_weights.pth",      # filename relative to this plugin dir
)
class MyTouchModel(ITouchModel):

    def load(self, weights_path: str) -> None:
        # Define your model class (must match training-time architecture)
        self.model = MyLSTMNet(...)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, window_5_frames: list[dict[str, float]]) -> dict[str, float]:
        # window_5_frames: list of 5 dicts, each with 63 keys
        #   "<landmark_name>_x/y/z" for all 21 landmarks
        # Your model decides which features to extract.

        # Example: extract fingertip + wrist coords for all 5 frames
        features = ...   # shape (5, feature_dim)

        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.model(tensor)
            probs  = torch.sigmoid(logits).squeeze().tolist()

        fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        return dict(zip(fingers, probs))
```

4. Restart the detector app — the plugin is automatically discovered and shown in the model dropdown.

## Notes

- The detector calls `predict()` on every 5-frame window (~every 250 ms at 12 FPS, shift=2).
- `window_5_frames[0]` is the oldest frame; `window_5_frames[4]` is the newest.
- All landmark values are **scale-normalised** (divided by L_hand) and **wrist-centred** (wrist = origin).
- Return values must be in `[0.0, 1.0]`. Values >= `TOUCH_THRESHOLD` (default 0.5) trigger touch resolution.
