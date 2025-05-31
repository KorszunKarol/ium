#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Path to the neural network dataset (pickle file)
data_path = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"

print("Loading neural network dataset...")
try:
    df = pd.read_pickle(data_path)

    print(f"\n{'='*60}")
    print("NEURAL NETWORK DATASET DIMENSIONS")
    print(f"{'='*60}")

    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows (samples): {df.shape[0]:,}")
    print(f"Number of columns (features): {df.shape[1]:,}")

    # Memory usage
    memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Memory usage: {memory_usage_mb:.2f} MB")

    # Target column info
    target_col = "annual_revenue_adj"
    if target_col in df.columns:
        print(f"\nTarget variable: {target_col}")
        print(f"Features (excluding target): {df.shape[1] - 1:,}")

        y = df[target_col]
        print(f"\nTarget statistics:")
        print(f"  Mean: ${y.mean():.0f}")
        print(f"  Median: ${y.median():.0f}")
        print(f"  Min: ${y.min():.0f}")
        print(f"  Max: ${y.max():.0f}")
        print(f"  Std: ${y.std():.0f}")
    else:
        print(f"\nTarget column '{target_col}' not found!")
        print(f"Available columns: {list(df.columns)[:10]}...")

    print(f"\n{'='*60}")

except FileNotFoundError:
    print(f"File not found: {data_path}")
except Exception as e:
    print(f"Error loading dataset: {e}")
