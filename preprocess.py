"""
UAV Flight Trajectories Dataset - Preprocessing Pipeline
=======================================================
This script performs complete, leakage-free preprocessing on the 5,093 raw UAV
trajectory files from RIOTU-Lab.

Steps performed:
1. Trajectory Categorization (2D, 3D, Standard).
2. Kinematic Feature Engineering:
   - Relative position from trajectory start: (rx, ry, rz)
   - Displacements: (dx, dy, dz)
   - Velocities: (vx, vy, vz) and speed magnitude
   - Accelerations: (ax, ay, az) and acceleration magnitude
   - Heading / yaw angle (rad)
   - Elapsed time: tau = t - t0
3. Leakage-Free Stratified Data Split (70% Train, 15% Val, 15% Test at trajectory level).
4. Feature Normalization (StandardScaler fitted strictly on training data).
5. Sliding Window Generation for AI/DL Sequence Models (X: past history, y: future trajectory).
6. Export to fast-loading consolidated Parquet and NumPy formats.
"""

import os
import glob
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def get_trajectory_type(filename: str) -> str:
    name = Path(filename).name
    if "trajectory2D" in name:
        return "2D"
    elif "trajectory3D" in name:
        return "3D"
    else:
        return "standard"


def extract_trajectory_id(filename: str) -> int:
    name = Path(filename).stem
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def process_single_trajectory(filepath: str, traj_idx: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    traj_type = get_trajectory_type(filepath)
    file_id = extract_trajectory_id(filepath)
    
    # Sort by timestamp to ensure strict chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    t = df["timestamp"].values
    x = df["tx"].values
    y = df["ty"].values
    z = df["tz"].values
    
    # Elapsed time from trajectory start
    t0 = t[0]
    elapsed_time = t - t0
    
    # Relative coordinates from trajectory origin (0, 0, 0)
    rx = x - x[0]
    ry = y - y[0]
    rz = z - z[0]
    
    # Displacements between consecutive steps (delta t = 0.1s)
    # prepend 0 for the initial step
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])
    dt = np.diff(t, prepend=t[0] - 0.1)
    # Guard against zero dt
    dt = np.where(dt <= 0, 0.1, dt)
    
    # Velocities (m/s)
    vx = dx / dt
    vy = dy / dt
    vz = dz / dt
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Accelerations (m/s^2)
    ax = np.diff(vx, prepend=vx[0]) / dt
    ay = np.diff(vy, prepend=vy[0]) / dt
    az = np.diff(vz, prepend=vz[0]) / dt
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    
    # Heading angle in radians: atan2(vy, vx)
    heading = np.arctan2(vy, vx)
    
    processed_df = pd.DataFrame({
        "traj_id": traj_idx,
        "file_name": Path(filepath).name,
        "traj_type": traj_type,
        "step": np.arange(len(df)),
        "timestamp": t,
        "elapsed_time": elapsed_time,
        "tx": x,
        "ty": y,
        "tz": z,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "speed": speed,
        "ax": ax,
        "ay": ay,
        "az": az,
        "acc_mag": acc_mag,
        "heading": heading
    })
    
    return processed_df


def create_sliding_windows(
    traj_dfs: list,
    feature_cols: list,
    target_cols: list,
    obs_len: int = 20,
    pred_len: int = 10,
    step_stride: int = 2
):
    """
    Creates (X, y) sliding window sequences:
    X: shape (N, obs_len, num_features)
    y: shape (N, pred_len, num_targets)
    """
    X_list, y_list = [], []
    
    for df in traj_dfs:
        features = df[feature_cols].values
        targets = df[target_cols].values
        seq_len = len(df)
        total_window = obs_len + pred_len
        
        if seq_len < total_window:
            continue
            
        for i in range(0, seq_len - total_window + 1, step_stride):
            X_list.append(features[i : i + obs_len])
            y_list.append(targets[i + obs_len : i + total_window])
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Preprocess UAV Trajectory Dataset")
    parser.add_argument("--data_dir", type=str, default="repo_gh", help="Path to raw CSV files")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory")
    parser.add_argument("--obs_len", type=int, default=20, help="Observation window length (default: 20 steps / 2.0s)")
    parser.add_argument("--pred_len", type=int, default=10, help="Prediction horizon length (default: 10 steps / 1.0s)")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UAV Trajectory Dataset Preprocessing Pipeline")
    print("=" * 60)
    
    # 1. Discover all files
    raw_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    print(f"Found {len(raw_files)} raw CSV trajectories in '{args.data_dir}'.")
    if not raw_files:
        print("No CSV files found. Please check data_dir.")
        return

    # 2. Process all files into DataFrames
    print("\n[Step 1/5] Computing kinematic features (velocities, accelerations, displacements)...")
    all_traj_dfs = []
    metadata = []
    
    for idx, f in enumerate(raw_files):
        df = process_single_trajectory(f, traj_idx=idx)
        all_traj_dfs.append(df)
        metadata.append({
            "traj_id": idx,
            "file_name": Path(f).name,
            "traj_type": df["traj_type"].iloc[0],
            "num_steps": len(df)
        })
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(raw_files):
            print(f"  Processed {idx + 1}/{len(raw_files)} files...")
            
    meta_df = pd.DataFrame(metadata)
    print(f"\nDistribution of Trajectory Types:")
    print(meta_df["traj_type"].value_counts().to_string())

    # 3. Stratified Data Split (70% Train, 15% Val, 15% Test)
    print("\n[Step 2/5] Creating stratified trajectory-level split (70% / 15% / 15%)...")
    meta_df["split"] = "train"
    
    for t_type in meta_df["traj_type"].unique():
        type_indices = meta_df[meta_df["traj_type"] == t_type].index.values
        np.random.shuffle(type_indices)
        
        n_total = len(type_indices)
        n_train = int(n_total * args.train_ratio)
        n_val = int(n_total * args.val_ratio)
        
        train_idx = type_indices[:n_train]
        val_idx = type_indices[n_train : n_train + n_val]
        test_idx = type_indices[n_train + n_val :]
        
        meta_df.loc[train_idx, "split"] = "train"
        meta_df.loc[val_idx, "split"] = "val"
        meta_df.loc[test_idx, "split"] = "test"
        
    print("Split summary:")
    print(pd.crosstab(meta_df["traj_type"], meta_df["split"]))

    # Map split back to each trajectory dataframe
    split_map = dict(zip(meta_df["traj_id"], meta_df["split"]))
    for df in all_traj_dfs:
        df["split"] = split_map[df["traj_id"].iloc[0]]

    # 4. Feature Normalization (Strictly fitted on Train set)
    print("\n[Step 3/5] Normalizing features (StandardScaler fitted exclusively on Training set)...")
    feature_cols = [
        "rx", "ry", "rz",       # Origin-relative positions
        "dx", "dy", "dz",       # Displacements
        "vx", "vy", "vz",       # Velocities
        "speed",                # Scalar speed
        "ax", "ay", "az",       # Accelerations
        "acc_mag",              # Scalar acceleration
        "heading"               # Heading angle
    ]
    target_cols = ["dx", "dy", "dz"] # Future step-by-step displacements (or "rx", "ry", "rz")
    
    train_dfs = [df for df in all_traj_dfs if df["split"].iloc[0] == "train"]
    val_dfs   = [df for df in all_traj_dfs if df["split"].iloc[0] == "val"]
    test_dfs  = [df for df in all_traj_dfs if df["split"].iloc[0] == "test"]
    
    # Compute mean and std exclusively from training data
    train_concat = pd.concat([df[feature_cols] for df in train_dfs], ignore_index=True)
    scaler_means = train_concat.mean().to_dict()
    scaler_stds = train_concat.std().replace(0, 1.0).to_dict()
    
    # Save scaler parameters to JSON for inference / test de-normalization
    scaler_info = {
        "mean": scaler_means,
        "std": scaler_stds,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "obs_len": args.obs_len,
        "pred_len": args.pred_len
    }
    with open(out_dir / "scaler_params.json", "w") as f:
        json.dump(scaler_info, f, indent=4)
    print(f"  Scaler parameters saved to '{out_dir / 'scaler_params.json'}'.")

    # Apply scaling: create normalized columns with prefix 'norm_'
    for df in all_traj_dfs:
        for col in feature_cols:
            df[f"norm_{col}"] = (df[col] - scaler_means[col]) / scaler_stds[col]

    # 5. Export Consolidated Datasets
    print("\n[Step 4/5] Exporting consolidated dataset files...")
    full_df = pd.concat(all_traj_dfs, ignore_index=True)
    
    csv_gz_path = out_dir / "all_trajectories_preprocessed.csv.gz"
    full_df.to_csv(csv_gz_path, index=False, compression="gzip")
    print(f"  Full consolidated dataset: {csv_gz_path} ({csv_gz_path.stat().st_size / (1024*1024):.1f} MB)")
    
    meta_path = out_dir / "trajectories_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"  Dataset metadata index: {meta_path}")

    # 6. Generate ML/DL Sliding Window Sequences
    print(f"\n[Step 5/5] Generating sliding window sequences (Obs: {args.obs_len} steps, Pred: {args.pred_len} steps)...")
    norm_feature_cols = [f"norm_{col}" for col in feature_cols]
    norm_target_cols = [f"norm_{col}" for col in target_cols]
    
    X_train, y_train = create_sliding_windows(train_dfs, norm_feature_cols, norm_target_cols, args.obs_len, args.pred_len)
    X_val,   y_val   = create_sliding_windows(val_dfs,   norm_feature_cols, norm_target_cols, args.obs_len, args.pred_len)
    X_test,  y_test  = create_sliding_windows(test_dfs,  norm_feature_cols, norm_target_cols, args.obs_len, args.pred_len)
    
    print(f"  Train sequences -> X: {X_train.shape}, y: {y_train.shape}")
    print(f"  Val sequences   -> X: {X_val.shape}, y: {y_val.shape}")
    print(f"  Test sequences  -> X: {X_test.shape}, y: {y_test.shape}")
    
    # Save NumPy compressed tensors
    npz_path = out_dir / "sequences_dataset.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test
    )
    print(f"  Compressed sequence tensors saved to: {npz_path} ({npz_path.stat().st_size / (1024*1024):.1f} MB)")
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
