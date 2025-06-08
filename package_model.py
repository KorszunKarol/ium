"""
Model Packaging Script for Deployment

This script takes a training log directory, loads the best model and its
corresponding artifacts, and saves them in a clean, deployable format.

Usage:
    python package_model.py path/to/your/training/log/directory

Example:
    python package_model.py logs/neural_net/neural_net_20250530-053253

This will create a `models_deploy` directory (or overwrite the existing one)
with the following files, ready for the API:
- neural_net_model.keras
- neural_net_scaler.pkl
- neural_net_feature_names.pkl
- neural_net_metadata.pkl
"""
import os
import sys
from models.neural_net import NeuralNetworkModel, NeuralNetConfig


def main(model_log_dir):
    """
    Packages a trained neural network model and its artifacts for deployment.

    This script performs the following steps:
    1.  Initializes a NeuralNetConfig to get data paths and feature lists.
    2.  Runs the data preprocessing step to generate the correct scaler and feature names.
    3.  Loads the specific, pre-trained model from the specified log directory.
    4.  Saves the loaded model, the scaler, and feature names into the final
        deployment directory, ready for the API to use.

    Args:
        model_log_dir (str): The path to the training log directory containing
                             the 'best_model.keras' file.
    """
    print("--- Starting Model Packaging Process ---")

    # --- 1. Configuration ---
    # Use the default config to ensure feature lists and preprocessing match
    config = NeuralNetConfig()

    # The final destination for the API to load models from
    deployment_dir = "models_deploy"

    # Path to the specific model file you want to deploy
    model_path = os.path.join(model_log_dir, "best_model.keras")

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)

    print(f"Target model: {model_path}")
    print(f"Deployment destination: {deployment_dir}")
    print(f"Using {len(config.selected_features)} features.")

    # --- 2. Recreate Scaler and Feature Names ---
    # We instantiate the model class but don't train it. We only need to
    # run the preprocessing to get the scaler and feature names.
    nn_model = NeuralNetworkModel(log_dir="temp_packaging_logs")
    nn_model.config = config

    print("\n--- Running data preprocessing to generate artifacts ---")
    nn_model.load_and_preprocess_data(data_path=config.data_path,
                                      feature_subset=config.selected_features,
                                      cap_outliers=config.cap_outliers,
                                      cap_percentile=config.cap_percentile)
    print("✅ Scaler and feature names have been recreated.")

    # --- 3. Load the specific pre-trained model ---
    print(f"\n--- Loading the pre-trained model from {model_path} ---")
    nn_model.load_saved_model(model_path=model_path)
    print("✅ Specific model loaded successfully.")

    # --- 4. Save all artifacts for deployment ---
    print(f"\n--- Saving all artifacts to {deployment_dir} ---")
    nn_model.save_for_deployment(output_dir=deployment_dir)
    print("\n--- Model Packaging Complete ---")
    print(
        f"✅ All artifacts for model '{model_log_dir}' are now in '{deployment_dir}'."
    )
    print("You can now restart the API.")


if __name__ == "__main__":
    # The user-specified directory containing the desired model.
    # This is the directory you mentioned: 20_features_20250606-213537
    target_log_dir = "logs/neural_net/20_features_20250606-213537"
    main(target_log_dir)
