"""
arch_lstm_coords.py
===================
Architecture variant: LSTM_Coords
Data Feature Variant: coords_2d
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_arch import (
    SequenceLSTM, run_model_benchmark
)

ARCH_NAME = "LSTM_Coords"
VARIANT   = "coords_2d"
CONFIGS   = [{'id': 1, 'hidden': 32, 'layers': 2, 'dropout': 0.2, 'lr': 0.001, 'bs': 32}]

def create_model(feature_dim, cfg):
    return SequenceLSTM(input_features=feature_dim, hidden_units=cfg.get("hidden", 32), num_layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.2))

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
