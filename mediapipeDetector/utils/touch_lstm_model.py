# %% [markdown]
# # Per-Finger Touch Detection Model (LSTM in PyTorch)
# 
# This notebook builds and trains a **PyTorch LSTM model** to detect touch events for a target finger using **8 input velocities**:
# - 2 Wrist velocities: `(vx_wrist, vy_wrist)`
# - 6 Finger joint velocities: `(vx_j1, vy_j1, vx_j2, vy_j2, vx_j3, vy_j3)`
# 
# Sequence length = 5 (sliding window of 5 consecutive 30fps frames).
# Output = Binary classification (0 = No Touch, 1 = Touch).

# %% [markdown]
# ## 1. Imports and Hyperparameters Setup

# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Hyperparameters
NUM_SAMPLES = 1000
SEQ_LEN = 5        # 5 consecutive frames
FEATURE_DIM = 8    # 2 wrist vels + 6 finger joint vels
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Creating Synthetic Touch Dataset for Demonstration

# %%
# Function to generate synthetic velocity sequence data
def generate_touch_dataset(n_samples=1000, seq_len=5, feature_dim=8):
    X = np.zeros((n_samples, seq_len, feature_dim), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    for i in range(n_samples):
        is_touch = np.random.choice([0, 1])
        y[i] = is_touch
        
        if is_touch == 1:
            # Touch gesture: sharp decelerations / velocity spikes across the 5-frame window
            base_vel = np.random.uniform(-0.5, 0.5, size=(seq_len, feature_dim))
            # Inject impact deceleration spike in middle frames (frame index 2 and 3)
            base_vel[2:4, 2:] += np.random.uniform(-2.5, -1.0, size=(2, 6))
            X[i] = base_vel
        else:
            # Idle / Normal movement: low smooth velocities
            X[i] = np.random.uniform(-0.3, 0.3, size=(seq_len, feature_dim))
            
    return X, y

# Generate synthetic dataset
X_data, y_data = generate_touch_dataset(n_samples=NUM_SAMPLES, seq_len=SEQ_LEN, feature_dim=FEATURE_DIM)

# Convert to PyTorch Tensors
X_tensor = torch.from_numpy(X_data).type(torch.float32)
y_tensor = torch.from_numpy(y_data).type(torch.float32)

# Train/Test Split (80% Train, 20% Test)
split_idx = int(0.8 * NUM_SAMPLES)
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

# %% [markdown]
# ## 3. PyTorch Dataset and DataLoader

# %%
class VelocitySequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = VelocitySequenceDataset(X_train, y_train)
test_dataset = VelocitySequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
# ## 4. Building the LSTM Touch Detection Model

# %%
class FingerTouchLSTM(nn.Module):
    def __init__(self, input_features: int = 8, hidden_units: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # LSTM Layer processing (batch_size, seq_len, input_features)
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully connected layer stack for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_units, out_features=16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=16, out_features=1)
        )
        
    def forward(self, x):
        # Forward pass through LSTM: lstm_out shape -> (batch_size, seq_len, hidden_units)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Select the output from the final frame timestep (t = seq_len - 1)
        last_timestep = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_units)
        
        # Pass raw unnormalized logits to classifier head
        logits = self.classifier(last_timestep)
        return logits

# Instantiate model and send to target device
model_0 = FingerTouchLSTM(input_features=FEATURE_DIM, hidden_units=32, num_layers=2).to(device)
print(model_0)

# %% [markdown]
# ## 5. Loss Function, Optimizer, and Accuracy Function

# %%
# Loss Function & Optimizer
loss_fn = nn.BCEWithLogitsLoss() # Combines Sigmoid layer and Binary Cross Entropy Loss in one single class for numerical stability
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)

# Accuracy Calculation Function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# %% [markdown]
# ## 6. Initial Un-trained Model Evaluation (Checking Logits & Probabilities)

# %%
model_0.eval()
with torch.inference_mode():
    sample_X = X_train[:5].to(device)
    sample_y = y_train[:5].to(device)
    
    # 1. Raw Logits
    y_logits = model_0(sample_X)
    
    # 2. Convert Logits to Prediction Probabilities using Sigmoid
    y_pred_probs = torch.sigmoid(y_logits)
    
    # 3. Convert Prediction Probabilities to Prediction Labels (0 or 1 thresholded at 0.5)
    y_pred_labels = torch.round(y_pred_probs)

print("Sample Raw Logits:\n", y_logits.squeeze())
print("Sample Prediction Probabilities:\n", y_pred_probs.squeeze())
print("Sample Predicted Labels:\n", y_pred_labels.squeeze())
print("True Target Labels:\n", sample_y.squeeze())

# %% [markdown]
# ## 7. Training and Testing Loop

# %%
# Set training loop
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(1, EPOCHS + 1):
    # --- Training Phase ---
    model_0.train()
    train_loss, train_acc = 0.0, 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 1. Forward Pass
        y_logits = model_0(X_batch)
        
        # 2. Calculate Loss and Accuracy
        loss = loss_fn(y_logits, y_batch)
        y_preds = torch.round(torch.sigmoid(y_logits))
        acc = accuracy_fn(y_batch, y_preds)
        
        # 3. Optimizer Zero Grad
        optimizer.zero_grad()
        
        # 4. Backward Pass (Backpropagation)
        loss.backward()
        
        # 5. Optimizer Step (Weight update)
        optimizer.step()
        
        train_loss += loss.item() * len(X_batch)
        train_acc += (acc / 100.0) * len(X_batch)
        
    train_loss /= len(train_dataset)
    train_acc = (train_acc / len(train_dataset)) * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # --- Testing Phase ---
    model_0.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            test_logits = model_0(X_batch)
            t_loss = loss_fn(test_logits, y_batch)
            test_preds = torch.round(torch.sigmoid(test_logits))
            t_acc = accuracy_fn(y_batch, test_preds)
            
            test_loss += t_loss.item() * len(X_batch)
            test_acc += (t_acc / 100.0) * len(X_batch)
            
        test_loss /= len(test_dataset)
        test_acc = (test_acc / len(test_dataset)) * 100
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch: {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

# %% [markdown]
# ## 8. Plotting Training and Testing Curves

# %%
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="blue")
plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss", color="red", linestyle="--")
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy", color="blue")
plt.plot(range(1, EPOCHS + 1), test_accuracies, label="Test Accuracy", color="red", linestyle="--")
plt.title("Accuracy Curves")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Single Sequence Inference Helper Function

# %%
def predict_touch(model, velocity_window_5x8):
    """
    Predicts touch event for a single 5x8 velocity window sequence.
    
    Args:
        velocity_window_5x8 (list or np.ndarray): Shape (5, 8) containing:
            [vx_wrist, vy_wrist, vx_j1, vy_j1, vx_j2, vy_j2, vx_j3, vy_j3] across 5 frames.
            
    Returns:
        tuple: (prob_score (float), is_touch (bool))
    """
    model.eval()
    tensor_in = torch.tensor(velocity_window_5x8, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        logits = model(tensor_in)
        prob = torch.sigmoid(logits).item()
        
    is_touch = prob >= 0.5
    return prob, is_touch

# Test with a single sample from test set
sample_seq = X_test[0].numpy()
prob, is_touch = predict_touch(model_0, sample_seq)
print(f"Predicted Touch Probability: {prob:.4f} -> Is Touch: {is_touch}")

# %% [markdown]
# ## 10. Saving Model Weights

# %%
MODEL_SAVE_PATH = "finger_touch_lstm.pth"
torch.save(model_0.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved successfully to: {MODEL_SAVE_PATH}")
