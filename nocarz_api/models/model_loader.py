"""
Model loading and prediction utilities

This module handles loading and inference for both the neural network
and baseline models.

Author: Deployment Team
Created: 2025-05-31
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, Optional
import uuid
import sys
from datetime import datetime


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

        # Log transformation for MALE calculation
        y_true_safe = tf.maximum(y_true_float, 0.0)
        y_pred_safe = tf.maximum(y_pred_float, 0.0)

        log_true = tf.math.log1p(y_true_safe)
        log_pred = tf.math.log1p(y_pred_safe)

        # Calculate absolute log differences
        abs_log_diff = tf.abs(log_true - log_pred)

        # Create weights based on under/over estimation
        is_under = tf.cast(log_pred < log_true, tf.float32)
        weights = is_under * w_u + (1.0 - is_under) * w_o

        # Apply weights to absolute differences
        weighted_abs_diff = abs_log_diff * weights

        # Mean across the batch
        return tf.reduce_mean(weighted_abs_diff)

    return loss


logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and inference for ML models"""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.neural_net_model = None
        self.neural_net_scaler = None
        self.neural_net_features = None
        self.neural_net_metadata = None
        self.baseline_model = None
        self.numerical_features = [
            'latitude', 'longitude', 'distance_to_center', 'accommodates',
            'bedrooms', 'bathrooms', 'price_log', 'amenities_count',
            'review_scores_rating', 'number_of_reviews', 'host_is_superhost',
            'host_days_active', 'instant_bookable', 'neighbourhood_price_rank',
            'property_type_frequency', 'host_response_rate',
            'review_scores_location', 'minimum_nights',
            'calculated_host_listings_count', 'name_positive_sentiment'
        ]
        self.categorical_features = [
            'property_type', 'neighbourhood_cleansed', 'room_type'
        ]

    def load_models(self):
        """Load all models and preprocessing artifacts"""
        logger.info(f"Loading models from: {self.models_dir}")

        try:

            self._load_neural_net()
            logger.info("Neural network model loaded successfully")

            self._load_baseline()
            logger.info("Baseline model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    def _load_neural_net(self):
        """Load neural network model and preprocessing components"""

        model_path = os.path.join(self.models_dir, "neural_net_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Neural net model not found: {model_path}")

        # Define custom_objects dictionary for Keras model loading
        custom_objects = {
            "mae_original_scale_loss": mae_original_scale_loss,
            "asymmetric_weighted_male_loss": asymmetric_weighted_male_loss,
            "male_original_scale_metric": male_original_scale_metric,
            "wape_original_scale_metric": wape_original_scale_metric
        }

        # Pass custom_objects when loading the model
        self.neural_net_model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects)
        logger.info(f"Loaded neural net model from: {model_path}")

        scaler_path = os.path.join(self.models_dir, "neural_net_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Neural net scaler not found: {scaler_path}")

        with open(scaler_path, 'rb') as f:
            self.neural_net_scaler = pickle.load(f)
        logger.info(f"Loaded neural net scaler from: {scaler_path}")

        features_path = os.path.join(self.models_dir,
                                     "neural_net_feature_names.pkl")
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Neural net features not found: {features_path}")

        with open(features_path, 'rb') as f:
            self.neural_net_features = pickle.load(f)
        logger.info(
            f"Loaded {len(self.neural_net_features)} neural net features (post-encoding)"
        )

        metadata_path = os.path.join(self.models_dir,
                                     "neural_net_metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Neural net metadata not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            self.neural_net_metadata = pickle.load(f)
        logger.info("Loaded neural net metadata")

    def _load_baseline(self):
        """Load baseline model"""
        model_path = os.path.join(self.models_dir, "baseline_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Baseline model not found: {model_path}")

        import joblib
        self.baseline_model = joblib.load(model_path)
        logger.info(f"Loaded baseline model from: {model_path}")

    def health_check(self) -> Dict[str, bool]:
        """Perform health check on loaded models"""
        return {
            "models_loaded": True,
            "neural_net_available": self.neural_net_model is not None,
            "baseline_available": self.baseline_model is not None
        }

    def predict_neural_net(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using neural network model

        Args:
            input_df: DataFrame containing preprocessed features aligned with model's expected features
        """
        if self.neural_net_model is None:
            raise RuntimeError("Neural network model not loaded")

        try:
            # The input_df is now already preprocessed by to_dataframe method in schemas.py
            # We only need to scale it and make predictions

            features_scaled = self.neural_net_scaler.transform(input_df)

            prediction = self.neural_net_model.predict(features_scaled,
                                                       verbose=0)
            predicted_revenue = float(prediction[0][0])

            predicted_revenue = max(0, predicted_revenue)

            prediction_id = f"nn_{uuid.uuid4().hex[:8]}"

            logger.info(
                f"Neural net prediction: ${predicted_revenue:.2f} (ID: {prediction_id})"
            )

            return {
                "predicted_revenue": predicted_revenue,
                "prediction_id": prediction_id,
                "confidence_interval": None
            }

        except Exception as e:
            logger.error(f"Neural net prediction failed: {str(e)}")
            raise

    def predict_baseline(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using baseline model

        Args:
            input_df: DataFrame containing preprocessed features
        """
        if self.baseline_model is None:
            raise RuntimeError("Baseline model not loaded")

        try:
            # Extract neighborhood from the input DataFrame
            if 'neighbourhood_cleansed' in input_df.columns:
                # Get the first row's neighborhood value (there should be only one row)
                neighborhood = input_df['neighbourhood_cleansed'].iloc[
                    0] if 'neighbourhood_cleansed_' not in input_df.columns else 'Unknown'
            else:
                # Try to find one-hot encoded neighborhood columns
                neighborhood = 'Unknown'
                for col in input_df.columns:
                    if col.startswith('neighbourhood_cleansed_'
                                      ) and input_df[col].iloc[0] == 1:
                        neighborhood = col.replace('neighbourhood_cleansed_',
                                                   '')
                        break

            if hasattr(self.baseline_model, 'predict'):
                prediction = self.baseline_model.predict([neighborhood])
                predicted_revenue = float(prediction[0])
            else:
                predicted_revenue = self.baseline_model.get(
                    neighborhood, 15000.0)

            predicted_revenue = max(0, predicted_revenue)

            prediction_id = f"bl_{uuid.uuid4().hex[:8]}"

            logger.info(
                f"Baseline prediction: ${predicted_revenue:.2f} (ID: {prediction_id})"
            )

            return {
                "predicted_revenue": predicted_revenue,
                "prediction_id": prediction_id,
                "confidence_interval": None
            }

        except Exception as e:
            logger.error(f"Baseline prediction failed: {str(e)}")
            raise

    def _prepare_neural_net_input(self, input_data: Dict[str,
                                                         Any]) -> pd.DataFrame:
        """
        Prepares the input data for the neural network model, ensuring it
        matches the feature set used during training.
        """
        # Convert the input dictionary to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Get the list of all feature names from the trained model
        final_feature_names = self.neural_net_features

        # One-hot encode categorical features present in the input
        # Note: This uses metadata loaded from training to handle categories correctly
        if 'categorical_encoder' in self.neural_net_metadata:
            cat_encoder = self.neural_net_metadata['categorical_encoder']
            try:
                # Select only the categorical columns the encoder was trained on
                input_cat_features = [
                    col for col in cat_encoder.feature_names_in_
                    if col in input_df.columns
                ]

                if input_cat_features:
                    encoded_cat_df = pd.DataFrame(
                        cat_encoder.transform(
                            input_df[input_cat_features]).toarray(),
                        columns=cat_encoder.get_feature_names_out(
                            input_cat_features),
                        index=input_df.index)
                    # Drop original categorical columns and join encoded ones
                    input_df = input_df.drop(
                        columns=input_cat_features).join(encoded_cat_df)

            except Exception as e:
                logger.error(f"Error during one-hot encoding: {e}")
                # Create an empty DataFrame with expected columns if encoding fails
                # This prevents crashes but prediction will likely be poor
                encoded_columns = cat_encoder.get_feature_names_out()
                empty_encoded_df = pd.DataFrame(0,
                                                index=input_df.index,
                                                columns=encoded_columns)
                input_df = input_df.join(empty_encoded_df)

        # Reindex the DataFrame to match the exact feature set of the trained model
        # - Aligns column order
        # - Adds missing columns with a value of 0
        # - Removes columns from input that were not in the training data
        final_df = input_df.reindex(columns=final_feature_names, fill_value=0)

        logger.debug(
            f"Prepared DataFrame with {len(final_df.columns)} features.")

        return final_df

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        info = {
            "neural_net": {
                "loaded":
                self.neural_net_model is not None,
                "features_count":
                len(self.neural_net_features)
                if self.neural_net_features else 0,
                "model_type":
                "Keras Sequential"
            },
            "baseline": {
                "loaded": self.baseline_model is not None,
                "model_type": "Neighborhood Median"
            }
        }

        if self.neural_net_metadata:
            info["neural_net"]["metadata"] = {
                "input_dim":
                self.neural_net_metadata.get("input_dim"),
                "cap_value":
                self.neural_net_metadata.get("cap_value"),
                "training_samples":
                self.neural_net_metadata.get("training_samples")
            }

        return info

    def get_neural_net_features(self) -> list:
        """
        Returns the list of feature names that the neural network model was trained on.
        This is used for proper data preprocessing in the API.

        Returns:
            list: List of feature names expected by the neural network model
        """
        if self.neural_net_features is None:
            logger.warning("Neural network features not loaded")
            return []
        return self.neural_net_features

    def get_baseline_features(self) -> list:
        """
        Returns the list of feature names needed for the baseline model.
        For the neighborhood median model, this is simpler as it mainly needs the neighborhood.

        Returns:
            list: List of feature names used by the baseline model
        """
        # For the baseline model, we primarily need the neighborhood feature
        # Adding other potentially useful features for future model improvements
        return [
            'neighbourhood_cleansed', 'property_type', 'room_type',
            'accommodates'
        ]
