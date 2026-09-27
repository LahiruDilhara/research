"""
arch_tcn_finger_wrist.py
========================
Architecture variant: TCN_Finger_Wrist
Data Feature Variant: finger_wrist
(Uses 2D Coords + 2D Velocities of Wrist + finger joints: Wrist, base MCP, PIP, DIP, TIP: 4x20 tensor)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    TouchTCN, run_model_benchmark
)

ARCH_NAME = "TCN_Finger_Wrist"
VARIANT   = "finger_wrist"
CONFIGS   = [{'id': 1, 'tcn_channels': 32, 'num_levels': 2, 'dropout': 0.20, 'lr': 0.0015, 'bs': 32}]

def create_model(feature_dim, cfg):
    return TouchTCN(input_features=feature_dim, tcn_channels=cfg.get("tcn_channels", 32), num_levels=cfg.get("num_levels", 2), dropout=cfg.get("dropout", 0.20))

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
