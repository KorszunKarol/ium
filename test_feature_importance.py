#!/usr/bin/env python3
"""
Test Feature Importance Analysis for Neural Network
==================================================

Quick test to verify the feature importance methods work correctly
before running the full training pipeline.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from models.neural_net import NeuralNetworkModel

def test_feature_importance():
    """Test the feature importance analysis methods"""
    print("Testing Feature Importance Analysis...")

    # Check if dataset exists
    data_path = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please ensure the modeling dataset has been created.")
        return

    # Create model instance
    nn_model = NeuralNetworkModel(log_dir="logs/feature_importance_test")

    try:
        print("\n1. Loading and preprocessing data...")
        nn_model.load_and_preprocess_data(data_path)

        print("\n2. Building simple model for testing...")
        # Use smaller architecture for faster testing
        nn_model.build_model(hidden_layers=[64, 32], dropout_rate=0.1)
        nn_model.compile_model(optimizer="adam", learning_rate=0.001)

        print("\n3. Training minimal model (5 epochs for testing)...")
        nn_model.setup_tensorboard_callbacks()
        nn_model.train_model(epochs=5, batch_size=32)

        print("\n4. Testing feature importance analysis...")
        # Test with fewer features and smaller top_k for faster execution
        importance_results = nn_model.analyze_feature_importance(
            method='all',  # Test all methods
            top_k=10,      # Show top 10 features
            save_plots=True
        )

        print("\n=== FEATURE IMPORTANCE TEST RESULTS ===")
        for method, results_df in importance_results.items():
            print(f"\n{method.upper()} Method:")
            if 'importance_mean' in results_df.columns:
                top_feature = results_df.iloc[0]
                print(f"  Top feature: {top_feature['feature']} (importance: {top_feature['importance_mean']:.4f})")
            else:
                top_feature = results_df.iloc[0]
                print(f"  Top feature: {top_feature['feature']} (importance: {top_feature['importance']:.4f})")

        print(f"\n✅ Feature importance analysis test completed successfully!")
        print(f"Check the log directory: {nn_model.log_dir}")

        return True

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_importance()
    if success:
        print("\n🎉 All tests passed! The feature importance analysis is ready to use.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
