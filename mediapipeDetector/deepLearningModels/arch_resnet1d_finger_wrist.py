"""
arch_resnet1d_finger_wrist.py
=============================
Architecture variant: ResNet1D_Finger_Wrist
Data Feature Variant: finger_wrist
(Uses 2D Coords + 2D Velocities of Wrist + finger joints: Wrist, base MCP, PIP, DIP, TIP: 4x20 tensor)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    TouchResNet1D, run_model_benchmark
)

ARCH_NAME = "ResNet1D_Finger_Wrist"
VARIANT   = "finger_wrist"
CONFIGS   = [{'id': 1, 'hidden_dim': 32, 'dropout': 0.2, 'lr': 0.001, 'bs': 32}]

def create_model(feature_dim, cfg):
    return TouchResNet1D(input_features=feature_dim, hidden_dim=cfg.get("hidden_dim", 32), dropout=cfg.get("dropout", 0.2))

def main():
    run_model_benchmark(
        arch_name=ARCH_NAME,
        variant_name=VARIANT,
        create_model_fn=create_model,
        configs=CONFIGS,
        default_epochs=70
    )

if __name__ == "__main__":
    main()
