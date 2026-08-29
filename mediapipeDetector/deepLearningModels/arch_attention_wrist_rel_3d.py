"""
arch_attention_wrist_rel_3d.py
==============================
Architecture variant: Attention_Wrist_Rel_3D
Data Feature Variant: wrist_relative_3d
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    TouchAttentionNet, run_model_benchmark
)

ARCH_NAME = "Attention_Wrist_Rel_3D"
VARIANT   = "wrist_relative_3d"
CONFIGS   = [{'id': 1, 'embed_dim': 32, 'num_heads': 4, 'dropout': 0.2, 'lr': 0.001, 'bs': 32}]

def create_model(feature_dim, cfg):
    return TouchAttentionNet(input_features=feature_dim, embed_dim=cfg.get("embed_dim", 32), num_heads=cfg.get("num_heads", 4), dropout=cfg.get("dropout", 0.2))

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
