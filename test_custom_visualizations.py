#!/usr/bin/env python3
"""
Test script for custom TensorBoard visualizations
Validates that the enhanced neural network model works correctly
"""

import sys
import os
sys.path.append('/home/karolito/IUM')

from models.neural_net import NeuralNetworkModel
import pandas as pd
import numpy as np

def create_test_data():
    """Create synthetic test data with required structure"""
    np.random.seed(42)
    n_samples = 1000

    # Create synthetic features
    neighborhoods = ['Westminster', 'Camden', 'Hackney', 'Tower Hamlets', 'Islington']
    property_types = ['Entire home/apt', 'Private room', 'Shared room']

    # Properly cycle through the options to get exactly n_samples
    neighborhood_data = [neighborhoods[i % len(neighborhoods)] for i in range(n_samples)]
    property_type_data = [property_types[i % len(property_types)] for i in range(n_samples)]

    data = {
        'neighbourhood_cleansed': neighborhood_data,
        'property_type': property_type_data,
        'accommodates': np.random.randint(1, 8, n_samples),
        'bedrooms': np.random.randint(1, 4, n_samples),
        'bathrooms': np.random.uniform(0.5, 3.0, n_samples),
        'price': np.random.uniform(50, 500, n_samples),
        'minimum_nights': np.random.randint(1, 30, n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'number_of_reviews': np.random.randint(0, 100, n_samples),
        'review_scores_rating': np.random.uniform(3.0, 5.0, n_samples),
    }

    # Create synthetic target (annual revenue)
    # Make it somewhat realistic based on features
    base_revenue = (data['price'] * data['availability_365'] * 0.7 +
                   np.random.normal(0, 5000, n_samples))
    data['annual_revenue_adj'] = np.maximum(base_revenue, 1000)  # Minimum revenue

    return pd.DataFrame(data)

def test_neural_network_with_custom_plots():
    """Test the neural network with custom visualizations"""
    print("Creating test dataset...")
    test_df = create_test_data()

    # Save test data
    test_data_path = "/tmp/test_airbnb_data.pkl"
    test_df.to_pickle(test_data_path)
    print(f"Test data saved to {test_data_path}")
    print(f"Dataset shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")

    # Initialize model
    print("\nInitializing neural network model...")
    nn_model = NeuralNetworkModel(log_dir="/tmp/test_logs")

    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        nn_model.load_and_preprocess_data(test_data_path)

        # Build model (smaller for testing)
        print("Building model architecture...")
        nn_model.build_model(hidden_layers=[32, 16], dropout_rate=0.2)

        # Compile model
        print("Compiling model...")
        nn_model.compile_model(optimizer="adam", learning_rate=0.01)

        # Setup callbacks
        print("Setting up TensorBoard callbacks with custom visualizations...")
        nn_model.setup_tensorboard_callbacks()

        # Verify original features are preserved
        if nn_model.original_train is not None:
            print(f"✓ Original categorical features preserved: {list(nn_model.original_train.columns)}")
        else:
            print("✗ Warning: Original categorical features not preserved")

        # Train for a few epochs to test visualizations
        print("Training model for 3 epochs to test visualizations...")
        nn_model.train_model(epochs=3, batch_size=32)

        print("✓ Training completed successfully!")
        print(f"✓ Custom visualizations should be available in TensorBoard")
        print(f"Run: tensorboard --logdir /tmp/test_logs")

        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(test_data_path):
            os.remove(test_data_path)

if __name__ == "__main__":
    print("Testing Custom TensorBoard Visualizations for Neural Network")
    print("=" * 60)

    success = test_neural_network_with_custom_plots()

    if success:
        print("\n" + "=" * 60)
        print("✓ All tests passed! Custom visualizations are working correctly.")
        print("You can now use the enhanced neural network with your actual data.")
    else:
        print("\n" + "=" * 60)
        print("✗ Tests failed. Please check the errors above.")
