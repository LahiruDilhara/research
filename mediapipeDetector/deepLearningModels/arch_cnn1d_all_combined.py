"""
arch_cnn1d_all_combined.py
==========================
Architecture variant: CNN1D_All_Combined
Data Feature Variant: all_joints_coords_vel
(Uses 2D Coords + 2D Velocities of ALL 9 Joints: 4x36 tensor)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    TouchCNN1D, run_model_benchmark
)

ARCH_NAME = "CNN1D_All_Combined"
VARIANT   = "all_joints_coords_vel"
CONFIGS   = [{'id': 1, 'conv_ch': 32, 'fc_hid': 32, 'dropout': 0.2, 'lr': 0.001, 'bs': 32}]

def create_model(feature_dim, cfg):
    return TouchCNN1D(input_features=feature_dim, conv_channels=cfg.get("conv_ch", 32), fc_hidden=cfg.get("fc_hid", 32), dropout=cfg.get("dropout", 0.2))

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
