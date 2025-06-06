import os
import pickle
import numpy as np
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define custom functions locally to avoid import issues
@tf.function
def mae_original_scale_loss(y_true, y_pred):
    """Calculates Mean Absolute Error on the original dollar scale."""
    y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 5000000.0)
    loss = tf.reduce_mean(tf.abs(y_true - y_pred_clipped))
    return loss

@tf.function
def male_original_scale_metric(y_true, y_pred):
    """Keras metric wrapper for Mean Absolute Log Error (MALE) on original scale."""
    y_true_safe = tf.maximum(y_true, 0.0)
    y_pred_safe = tf.maximum(y_pred, 0.0)
    log_true = tf.math.log1p(y_true_safe)
    log_pred = tf.math.log1p(y_pred_safe)
    male = tf.reduce_mean(tf.abs(log_true - log_pred))
    return male

@tf.function
def wape_original_scale_metric(y_true, y_pred):
    """Keras metric wrapper for Weighted Absolute Percentage Error (WAPE) on original scale."""
    total_actual = tf.reduce_sum(tf.abs(y_true))
    safe_total_actual = tf.maximum(total_actual, tf.keras.backend.epsilon())
    total_error = tf.reduce_sum(tf.abs(y_true - y_pred))
    wape = (total_error / safe_total_actual) * 100.0
    return wape

@tf.function
def asymmetric_weighted_male_loss(w_u=1.5, w_o=1.0):
    """Creates a weighted loss function that penalizes underestimation more than overestimation."""
    def loss(y_true, y_pred):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)
        y_true_safe = tf.maximum(y_true_float, 0.0)
        y_pred_safe = tf.maximum(y_pred_float, 0.0)
        log_true = tf.math.log1p(y_true_safe)
        log_pred = tf.math.log1p(y_pred_safe)
        abs_log_diff = tf.abs(log_true - log_pred)
        is_under = tf.cast(log_pred < log_true, tf.float32)
        weights = is_under * w_u + (1.0 - is_under) * w_o
        weighted_abs_diff = abs_log_diff * weights
        return tf.reduce_mean(weighted_abs_diff)
    return loss

def check_model():
    """
    Loads the deployed neural network model and its artifacts to verify its
    integrity and input requirements.
    """
    logging.info("Starting Neural Network Model Integrity Check...")

    models_dir = "models_deploy"
    model_path = os.path.join(models_dir, "neural_net_model.keras")
    scaler_path = os.path.join(models_dir, "neural_net_scaler.pkl")
    features_path = os.path.join(models_dir, "neural_net_feature_names.pkl")

    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            logging.error(f"Required file not found: {path}")
            logging.error("Please ensure you have run a training and deployment script to place the necessary files in `models_deploy/`.")
            return

    try:
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        logging.info(f"Successfully loaded feature names from: {features_path}")
        logging.info(f"Number of features from file: {len(feature_names)}")
        logging.info(f"Feature names: {feature_names}")
    except Exception as e:
        logging.error(f"Failed to load feature names from {features_path}: {e}")
        return

    num_features = len(feature_names)

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info(f"Successfully loaded scaler from: {scaler_path}")
        if hasattr(scaler, 'n_features_in_'):
             logging.info(f"Scaler was fitted on {scaler.n_features_in_} features.")
             if scaler.n_features_in_ != num_features:
                 logging.warning(f"Mismatch! Scaler expects {scaler.n_features_in_} features, but feature list has {num_features}.")
        else:
            logging.warning("Could not determine number of features the scaler was fitted on.")

    except Exception as e:
        logging.error(f"Failed to load scaler from {scaler_path}: {e}")
        return

    try:
        custom_objects = {
            "mae_original_scale_loss": mae_original_scale_loss,
            "asymmetric_weighted_male_loss": asymmetric_weighted_male_loss,
            "male_original_scale_metric": male_original_scale_metric,
            "wape_original_scale_metric": wape_original_scale_metric
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logging.info(f"Successfully loaded Keras model from: {model_path}")
        model.summary()
        model_input_shape = model.input_shape
        logging.info(f"Model input shape: {model_input_shape}")
        if model_input_shape[1] != num_features:
            logging.warning(f"Mismatch! Model expects input of shape {model_input_shape}, but feature list has {num_features} features.")

    except Exception as e:
        logging.error(f"Failed to load Keras model from {model_path}: {e}")
        return

    logging.info("="*50)
    logging.info("PERFORMING TEST PREDICTION")
    logging.info("="*50)
    try:
        dummy_input = np.zeros((1, num_features))
        logging.info(f"Created dummy input with shape: {dummy_input.shape}")

        scaled_input = scaler.transform(dummy_input)
        logging.info(f"Scaled dummy input with shape: {scaled_input.shape}")

        prediction = model.predict(scaled_input, verbose=0)
        logging.info(f"Prediction successful!")
        logging.info(f"Predicted value for dummy input: {prediction[0][0]}")

    except Exception as e:
        logging.error(f"Test prediction FAILED: {e}")
        logging.error("There is a problem with the model or its artifacts. The API will not work correctly.")
        return

    logging.info("\n" + "="*50)
    logging.info("MODEL INTEGRITY CHECK COMPLETE")
    logging.info("="*50)
    logging.info("Conclusion: The model, scaler, and feature list seem compatible.")
    logging.info("The primary issue in the API is likely the PREPROCESSING step before prediction, which fails to create the correct input format.")


if __name__ == "__main__":
    check_model()