"""
arch_bilstm_finger_only.py
==========================
Architecture variant: BiLSTM_Finger_Only
Data Feature Variant: finger_only
(Uses 2D Coords + 2D Velocities of isolated finger joints: base MCP, PIP, DIP, TIP: 4x16 tensor)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    BiLSTM, run_model_benchmark
)

ARCH_NAME = "BiLSTM_Finger_Only"
VARIANT   = "finger_only"
CONFIGS   = [{'id': 1, 'hidden': 32, 'layers': 2, 'dropout': 0.25, 'lr': 0.001, 'bs': 32}]

def create_model(feature_dim, cfg):
    return BiLSTM(input_features=feature_dim, hidden_units=cfg.get("hidden", 32), num_layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.25))

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
