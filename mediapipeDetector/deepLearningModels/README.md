# Deep Learning Benchmark Suite & CLI Flags Reference

This directory contains the deep learning benchmarking framework for per-finger touch detection. It evaluates **10 model architectures and data representation views** (LSTM variants, BiLSTM, 1D CNN, 1D ResNet, Self-Attention Transformer, and Temporal Convolutional Networks).

---

## 🚀 Quick Start

To run all 10 benchmark models sequentially with default parameters and live epoch progress logging:

```bash
python3 deepLearningModels/run_all.py
```

---

## 🎛️ Command-Line (CLI) Flags

`run_all.py` and all individual architecture scripts (`arch_*.py`) support command-line parameter overrides. When passed to `run_all.py`, the flags apply to **all internal models**. If omitted, each model uses its tuned default values.

| Flag           | Short Flag |  Type   | Default | Description                                                              |
| :------------- | :--------: | :-----: | :-----: | :----------------------------------------------------------------------- |
| `--epochs`     |    `-e`    |  `int`  |  `70`   | Number of training epochs per configuration                              |
| `--dropout`    |    `-d`    | `float` | `0.20`  | Dropout probability applied across layers                                |
| `--lr`         |            | `float` | `0.001` | Adam optimizer learning rate                                             |
| `--batch-size` |   `-bs`    |  `int`  |  `32`   | Training and evaluation batch size                                       |
| `--hidden`     |            |  `int`  |  `32`   | Hidden dimension / channels (LSTM units, CNN channels, ResNet dim, etc.) |

---

## 💡 Usage Examples

### 1. Override Epochs and Dropout for All Models

Run all benchmark models for 50 epochs with a 0.30 dropout rate:

```bash
python3 deepLearningModels/run_all.py --epochs 50 --dropout 0.30
```

### 2. Override Learning Rate and Batch Size

Train all models with a smaller learning rate (`0.0005`) and larger batch size (`64`):

```bash
python3 deepLearningModels/run_all.py --lr 0.0005 -bs 64
```

### 3. Override All Parameters Simultaneously

Customize epochs, dropout, learning rate, batch size, and hidden dimension across all models:

```bash
python3 deepLearningModels/run_all.py -e 60 -d 0.25 --lr 0.001 -bs 32 --hidden 64
```

### 4. Run an Individual Model Script Directly

Any architecture script can be executed independently with custom parameter overrides:

```bash
python3 deepLearningModels/arch_lstm_velocities.py --epochs 50 --dropout 0.25 -bs 64
python3 deepLearningModels/arch_resnet1d.py --epochs 60 --hidden 64
```

---

## 🏗️ Model Architectures & Data Views

The suite benchmark 10 model variants:

|   #    | Architecture Script           | Data View / Representation                                                                                                       | Input Shape  |
| :----: | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------- | :----------: |
| **1**  | `arch_lstm_velocities.py`     | **LSTM (Velocities Only)**: 4 transition steps $\times$ 8 2D velocity features (`wrist`, `pip`, `dip`, `tip` $vx, vy$)           | `(N, 4, 8)`  |
| **2**  | `arch_lstm_coords.py`         | **LSTM (Coordinates Only)**: 5 frame timesteps $\times$ 8 2D coordinate features (`wrist`, `pip`, `dip`, `tip` $x, y$)           | `(N, 5, 8)`  |
| **3**  | `arch_lstm_combined.py`       | **LSTM (Combined)**: 4 transition steps $\times$ 16 features (8 2D coords + 8 2D vels)                                           | `(N, 4, 16)` |
| **4**  | `arch_lstm_vel_speed.py`      | **LSTM (Velocities + 2D Speed)**: 4 transition steps $\times$ 12 features (8 2D vels + 4 2D speed magnitudes $\sqrt{vx^2+vy^2}$) | `(N, 4, 12)` |
| **5**  | `arch_lstm_all_joints_vel.py` | **LSTM (All 9 Joints Velocities)**: 4 transition steps $\times$ 18 2D velocity features across all hand joints                   | `(N, 4, 18)` |
| **6**  | `arch_bilstm.py`              | **Bidirectional LSTM**: BiLSTM processing combined features forward & backward                                                   | `(N, 4, 16)` |
| **7**  | `arch_cnn1d.py`               | **1D CNN**: 2-layer 1D Convolutional network with global average pooling                                                         | `(N, 4, 16)` |
| **8**  | `arch_resnet1d.py`            | **1D ResNet**: 1D Residual network with skip connections                                                                         | `(N, 4, 16)` |
| **9**  | `arch_attention.py`           | **Self-Attention Transformer**: Transformer encoder with Multi-Head Attention                                                    | `(N, 4, 16)` |
| **10** | `arch_tcn.py`                 | **Temporal Convolutional Network**: TCN with exponentially growing dilated convolutions                                          | `(N, 4, 16)` |

---

## 📊 Benchmark Output & Reports

- **Console Stream**: Unbuffered real-time loss and accuracy printed for every single training epoch (`Epoch: 01`, `Epoch: 02`, ...).
- **Log Files**: Full stdout logs for each model are saved under `deepLearningModels/results/<arch_name>.log`.
- **CSV Results**: Summary metric CSVs are generated in `deepLearningModels/results/`.
- **Ranked Summary**: `compare_results.py` automatically generates a ranked summary table (`deepLearningModels/results/summary_all.csv`) sorting models by test accuracy, touch F1-score, precision, recall, and training time.
