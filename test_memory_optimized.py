#!/usr/bin/env python3
"""
Memory-optimized test script for neural network with custom visualizations
"""

import sys
import os
sys.path.append('/home/karolito/IUM')

import pandas as pd
import numpy as np
from models.neural_net_optimized import NeuralNetworkModel

def create_small_test_data():
    """Create a small test dataset that mimics your real data structure"""
    np.random.seed(42)
    n_samples = 1000  # Small dataset for testing

    # Create synthetic data similar to your Airbnb dataset
    data = {
        'annual_revenue_adj': np.random.lognormal(8, 1, n_samples),  # Target variable
        'neighbourhood_cleansed': np.random.choice([
            'Westminster', 'Camden', 'Kensington and Chelsea', 'Islington',
            'Hackney', 'Tower Hamlets', 'Southwark', 'Lambeth'
        ], n_samples),
        'property_type': np.random.choice([
            'Entire home/apt', 'Private room', 'Hotel room', 'Shared room'
        ], n_samples),
        'accommodates': np.random.randint(1, 10, n_samples),
        'bathrooms': np.random.uniform(0.5, 5, n_samples),
        'bedrooms': np.random.randint(0, 6, n_samples),
        'beds': np.random.randint(1, 8, n_samples),
        'price': np.random.lognormal(4, 0.5, n_samples),
        'minimum_nights': np.random.randint(1, 30, n_samples),
        'maximum_nights': np.random.randint(30, 365, n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'latitude': np.random.uniform(51.3, 51.7, n_samples),
        'longitude': np.random.uniform(-0.5, 0.2, n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Hotel room'], n_samples),
        'instant_bookable': np.random.choice(['t', 'f'], n_samples),
    }

    # Add some numerical features
    for i in range(5):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)

    return pd.DataFrame(data)

def test_memory_optimized_neural_network():
    """Test the memory-optimized neural network"""
    print("Testing Memory-Optimized Neural Network")
    print("=" * 50)

    try:
        # Create test dataset
        print("Creating small test dataset...")
        test_df = create_small_test_data()
        test_path = "/tmp/test_dataset.pkl"
        test_df.to_pickle(test_path)
        print(f"Test dataset created with shape: {test_df.shape}")

        # Initialize model
        print("\nInitializing memory-optimized neural network...")
        nn_model = NeuralNetworkModel(log_dir="logs/test_neural_net")

        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        nn_model.load_and_preprocess_data(test_path, sample_size=800)  # Even smaller for testing

        # Build model
        print("\nBuilding model...")
        nn_model.build_model(hidden_layers=[32, 16], dropout_rate=0.3)  # Very small model

        # Compile model
        print("\nCompiling model...")
        nn_model.compile_model(optimizer="adam", learning_rate=0.01)

        # Setup callbacks
        print("\nSetting up callbacks...")
        nn_model.setup_tensorboard_callbacks(enable_advanced_plots=False)  # No advanced plots for testing

        # Train model
        print("\nTraining model (short training for testing)...")
        nn_model.train_model(epochs=5, batch_size=32)  # Very short training

        # Evaluate model
        print("\nEvaluating model...")
        results = nn_model.evaluate_model()

        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"MAE: £{results['mae']:.2f}")
        print(f"RMSE: £{results['rmse']:.2f}")
        print(f"R² Score: {results['r2_score']:.4f}")

        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)

        print("\n✅ Memory-optimized neural network test completed successfully!")
        print("\nNow you can run the optimized version on your full dataset:")
        print("python models/neural_net_optimized.py")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_optimized_neural_network()
    sys.exit(0 if success else 1)
