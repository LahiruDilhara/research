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

| Flag           | Short Flag            |  Type   | Default    | Description                                                              |
| :------------- | :-------------------- | :-----: | :--------- | :----------------------------------------------------------------------- |
| `--epochs`     | `-e`                  | `int`   | `70`       | Number of training epochs per configuration                              |
| `--dropout`    | `-d`                  | `float` | `0.20`     | Dropout probability applied across layers                                |
| `--lr`         |                       | `float` | `0.001`    | Adam optimizer learning rate                                             |
| `--batch-size` | `-bs`                 | `int`   | `32`       | Training and evaluation batch size                                       |
| `--hidden`     |                       | `int`   | `32`       | Hidden dimension / channels (LSTM units, CNN channels, ResNet dim, etc.) |
| `--plot`       | `-p`, `--plot-curves` | `flag`  | `Disabled` | Generates Matplotlib publication-quality Loss & Acc plot PNGs for each model and a master comparison grid |

---

## 💡 Usage Examples

### 1. Generate Matplotlib Curve Plots for All Models
Run all benchmark models and generate Jupyter-notebook style Matplotlib Loss & Accuracy curve PNGs (`results/plots/<arch_name>.png` and `results/plots/all_models_curves_grid.png`):

```bash
python3 deepLearningModels/run_all.py --plot
```

### 2. Combine Parameter Overrides with Matplotlib Plotting
Run for 50 epochs with 0.25 dropout and generate plots:

```bash
python3 deepLearningModels/run_all.py --epochs 50 --dropout 0.25 --plot
```

### 3. Generate Plot for a Single Model Script
```bash
python3 deepLearningModels/arch_lstm_velocities.py --epochs 50 --plot
```

### 4. Override All Parameters Simultaneously

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

- **Real-Time Epoch Progress**: Unbuffered live loss and accuracy printed for every single epoch (`Epoch: 01`, `Epoch: 02`, ...).
- **Terminal Loss & Accuracy Curves**: Renders ASCII line graphs using `asciichartpy` immediately after training each model:
  - **Loss Curve**: 🔴 Red (Train Loss) vs. 🔵 Cyan (Test Loss)
  - **Accuracy Curve**: 🟢 Green (Train Acc %) vs. 🟡 Yellow (Test Acc %)
- **Confusion Matrix & Classification Metrics**: Right after model evaluation, a formatted 2x2 binary confusion matrix (True Positives, True Negatives, False Positives, False Negatives) along with Accuracy, Precision, Recall, Specificity, and F1-score is printed to stdout.
- **Jupyter Notebook-Style Graphical Confusion Matrix Heatmaps**: When running with `--plot` / `-p`, publication-quality Matplotlib Confusion Matrix heatmaps are rendered:
  - Per-model heatmap PNGs: `results/plots/<arch_name>_confusion_matrix.png`
  - Master multi-model confusion matrix grid PNG: `results/plots/all_models_confusion_matrices_grid.png`
- **Overfitting & Underfitting Analytics**: Right after training each model, real-time analytics are computed and printed:
  - **Diagnosis Status**: Automatically classifies training as `OVERFITTING (Severe/Moderate/Mild)`, `UNDERFITTING (High)`, or `GOOD FIT (OPTIMAL)`.
  - **Overfit Onset Epoch**: Exact epoch where test loss minimum was reached before rising or where generalization gap started widening.
  - **Accuracy Gap Scale (`max_gap_pct`)**: Maximum percentage difference between Train Acc and Test Acc (`+X.XX%`).
  - **Actionable Recommendation**: Suggests early stopping epoch, dropout increase, or capacity adjustments.
- **Single-Line Master Experiment Log**: Appends **exactly one single line** to `deepLearningModels/results/experiment_history.log` on every benchmark execution detailing:
  - **Timestamp**: Execution date and time.
  - **All Model Performance**: Best accuracy, F1-score, precision, recall, and training duration for **every** model.
  - **All Model Hyperparameters**: Epochs, dropout, learning rate, batch size, and hidden units per model.
  - **Dataset Statistics**: Training samples count (`train_len`), testing samples count (`test_len`), and touch target class distribution.
  - **Pipeline Parameters (`process.sh`)**: Preserves all data processing parameters and execution flags line-by-line (e.g. 1Euro filter `min`/`beta`/`d`, window split, dataset cleaning flags, train/test split percentages, seed, `no-video-leak`).
