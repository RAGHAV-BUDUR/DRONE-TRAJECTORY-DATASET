"""
Quickstart: Loading preprocessed UAV trajectory data for Model Training
"""
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Load preprocessed tensors
data = np.load("processed_data/sequences_dataset.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

print("Loaded Sequence Shapes:")
print(f"  Train: X={X_train.shape}, y={y_train.shape}")
print(f"  Val:   X={X_val.shape},   y={y_val.shape}")
print(f"  Test:  X={X_test.shape},  y={y_test.shape}")

# 2. Convert to PyTorch DataLoaders
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 3. Sample batch inspection
sample_X, sample_y = next(iter(train_loader))
print(f"\nBatch sample -> X shape: {sample_X.shape}, y shape: {sample_y.shape}")

# 4. Load scaler parameters to de-normalize predictions at test time
with open("processed_data/scaler_params.json") as f:
    scaler = json.load(f)

print("\nScaler feature columns:", scaler["feature_cols"])
print("Target columns:", scaler["target_cols"])
