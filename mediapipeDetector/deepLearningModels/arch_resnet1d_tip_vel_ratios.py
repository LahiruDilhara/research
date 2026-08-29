"""
arch_resnet1d_tip_vel_ratios.py
===============================
Architecture variant: ResNet1D_Tip_Vel_Ratios
Data Feature Variant: fingertip_velocity_ratios
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    TouchResNet1D, run_model_benchmark
)

ARCH_NAME = "ResNet1D_Tip_Vel_Ratios"
VARIANT   = "fingertip_velocity_ratios"
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
