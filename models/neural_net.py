import os
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from models.custom_callbacks import asymmetric_weighted_male_loss
import matplotlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import pickle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from models.custom_callbacks import CustomMetricsCallback

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - feature analysis will use alternative methods")

print(f"Matplotlib backend set to: {matplotlib.get_backend()}")

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")


def mae_original_scale_loss(y_true, y_pred):
    """Calculates Mean Absolute Error on the original dollar scale.
  Note: Uses a hardcoded clipping value for y_pred.
  """
    y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 5000000.0)
    loss = tf.reduce_mean(tf.abs(y_true - y_pred_clipped))
    return loss


def male_original_scale_metric(y_true, y_pred):
    """Keras metric wrapper for Mean Absolute Log Error (MALE) on original scale."""
    y_true_safe = tf.maximum(y_true, 0.0)
    y_pred_safe = tf.maximum(y_pred, 0.0)

    log_true = tf.math.log1p(y_true_safe)
    log_pred = tf.math.log1p(y_pred_safe)
    male = tf.reduce_mean(tf.abs(log_true - log_pred))
    return male


def wape_original_scale_metric(y_true, y_pred):
    """Keras metric wrapper for Weighted Absolute Percentage Error (WAPE) on original scale."""
    total_actual = tf.reduce_sum(tf.abs(y_true))
    safe_total_actual = tf.maximum(total_actual, tf.keras.backend.epsilon())
    total_error = tf.reduce_sum(tf.abs(y_true - y_pred))
    wape = (total_error / safe_total_actual) * 100.0
    return wape


def create_combined_male_mae_loss(w_male: float, w_orig_mae: float,
                                  max_clip_value: float):
    """
  Creates a combined loss function of MALE and MAE on the original scale.
  Args:
      w_male: Weight for the MALE component.
      w_orig_mae: Weight for the MAE (original scale) component.
      max_clip_value: Maximum value to clip predictions to.
  Returns:
      A Keras-compatible loss function.
  """

    @tf.function
    def combined_loss(y_true, y_pred):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)

        y_pred_clipped = tf.clip_by_value(y_pred_float, 0.0, max_clip_value)

        y_true_safe_log = tf.maximum(y_true_float, 0.0)
        y_pred_safe_log = tf.maximum(y_pred_clipped, 0.0)

        log_true = tf.math.log1p(y_true_safe_log)
        log_pred = tf.math.log1p(y_pred_safe_log)
        male_val = tf.reduce_mean(tf.abs(log_true - log_pred))

        mae_val = tf.reduce_mean(tf.abs(y_true_float - y_pred_clipped))

        return w_male * male_val + w_orig_mae * mae_val

    combined_loss.__name__ = f'combined_male_{w_male}_mae_{w_orig_mae}'
    return combined_loss


class NeuralNetworkModel:

    def __init__(self, log_dir="logs"):
        self.model = None
        self.scaler = StandardScaler()
        self.log_dir = log_dir
        self.history = None
        self.is_regression = True

        self.max_revenue_value = 1000000.0
        self.callbacks_list = []
        self.evaluation_results = None

    def load_saved_model(self, model_path: str):
        """Loads a pre-trained Keras model from the given path."""
        print(f"Loading pre-trained model from: {model_path}")
        try:

            custom_objects = {
                "mae_original_scale_loss": mae_original_scale_loss,
                "asymmetric_weighted_male_loss": asymmetric_weighted_male_loss,
                "male_original_scale_metric": male_original_scale_metric,
                "wape_original_scale_metric": wape_original_scale_metric
            }

            self.model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects)
            print("Pre-trained model loaded successfully.")
            self.model.summary()
        except Exception as e:
            print(f"Error loading saved model: {e}")
            print(
                "Ensure the model path is correct and all custom objects (losses/metrics) used during training are provided."
            )
            raise

    def load_and_preprocess_data(self,
                                 data_path,
                                 feature_subset: list[str] = None,
                                 cap_outliers: bool = True,
                                 cap_percentile: float = 0.99):
        """Load and preprocess data for neural network regression (no log transformation)
      Args:
          data_path: Path to the dataset
          feature_subset: List of features to use (if None, uses all)
          cap_outliers: Whether to cap extreme outliers in target variable
          cap_percentile: Percentile threshold for capping (e.g., 0.99 = cap at 99th percentile)
      """

        if data_path.endswith(".pkl"):
            df = pd.read_pickle(data_path)
        else:
            df = pd.read_csv(data_path)

        print(f"Dataset shape: {df.shape}")

        target_col = "annual_revenue_adj"
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}"
            )

        X = df.drop(target_col, axis=1)
        y_original = df[target_col].values.astype(np.float32)

        print(f"Target variable: {target_col}")
        print(
            f"Original target statistics: mean=${y_original.mean():.0f}, std=${y_original.std():.0f}, min=${y_original.min():.0f}, max=${y_original.max():.0f}"
        )

        if cap_outliers:
            cap_value = np.percentile(y_original, cap_percentile * 100)
            outliers_count = np.sum(y_original > cap_value)
            outliers_percentage = (outliers_count / len(y_original)) * 100

            print(f"\nOutlier Analysis:")
            print(f"  Cap percentile: {cap_percentile*100:.1f}%")
            print(f"  Cap value: ${cap_value:.0f}")
            print(
                f"  Outliers found: {outliers_count} ({outliers_percentage:.2f}% of data)"
            )
            print(f"  Max outlier value: ${y_original.max():.0f}")

            if outliers_count > 0:
                y_original = np.clip(y_original, a_min=None, a_max=cap_value)
                print(f"  Applied capping at ${cap_value:.0f}")

                self.max_revenue_value = float(cap_value)
                print(
                    f"  Updated max_revenue_value to: ${self.max_revenue_value:.0f}"
                )
            else:
                print("  No outliers to cap")
        else:
            print("\nSkipping outlier capping")

        print(
            f"Final target statistics: mean=${y_original.mean():.0f}, std=${y_original.std():.0f}, min=${y_original.min():.0f}, max=${y_original.max():.0f}"
        )

        print(
            "Working directly with original scale revenue values (no log transformation)"
        )

        id_cols = ["listing_id", "id", "host_id"]
        X = X.drop(columns=[col for col in id_cols if col in X.columns],
                   errors="ignore")

        categorical_columns = X.select_dtypes(
            include=["object", "category"]).columns
        numerical_columns = X.select_dtypes(
            exclude=["object", "category"]).columns

        print(
            f"Categorical columns ({len(categorical_columns)}): {list(categorical_columns)[:5]}{'...' if len(categorical_columns) > 5 else ''}"
        )
        print(
            f"Numerical columns ({len(numerical_columns)}): {list(numerical_columns)[:5]}{'...' if len(numerical_columns) > 5 else ''}"
        )

        if len(categorical_columns) > 0:
            X_cat_copy = X[categorical_columns].copy()
            for col in categorical_columns:
                X_cat_copy[col] = X_cat_copy[col].astype(str)

            X_categorical = pd.get_dummies(X_cat_copy,
                                           drop_first=True,
                                           dtype=np.float32)
            X_numerical = X[numerical_columns].astype(np.float32)
            X_processed = pd.concat([X_numerical, X_categorical], axis=1)
        else:
            X_processed = X[numerical_columns].astype(np.float32)

        if feature_subset is not None and isinstance(X_processed,
                                                     pd.DataFrame):
            available_features = [
                f for f in feature_subset if f in X_processed.columns
            ]
            missing_features = [
                f for f in feature_subset if f not in X_processed.columns
            ]

            if missing_features:
                print(
                    f"Warning: {len(missing_features)} features not found in dataset: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
                )

            print(
                f"Selecting {len(available_features)} features from {len(feature_subset)} requested features"
            )
            print(f"Available features: {available_features}")

            X_processed = X_processed[available_features]
            print(
                f"Dataset shape after feature selection: {X_processed.shape}")
        else:
            print(
                f"Using all {len(X_processed.columns)} features - no feature subset specified"
            )

        y_processed = y_original

        self.original_features = (X[[
            "neighbourhood_cleansed", "property_type"
        ]] if all(
            col in X.columns
            for col in ["neighbourhood_cleansed", "property_type"]) else None)

        X_train_val, X_test_processed, y_train_val, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42)

        X_train_processed, X_val_processed, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42)

        self.y_train_original_scale = y_train
        self.y_val_original_scale = y_val
        self.y_test_original_scale = y_test

        if self.original_features is not None:
            original_train_val, self.original_test = train_test_split(
                self.original_features, test_size=0.2, random_state=42)
            self.original_train, self.original_features_val = train_test_split(
                original_train_val, test_size=0.2, random_state=42)
        else:
            self.original_train = None
            self.original_test = None
            self.original_features_val = None

        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        X_val_scaled = self.scaler.transform(X_val_processed)
        X_test_scaled = self.scaler.transform(X_test_processed)

        self.X_train = X_train_scaled.astype(np.float32)
        self.X_val = X_val_scaled.astype(np.float32)
        self.X_test = X_test_scaled.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.y_val = y_val.astype(np.float32)
        self.y_test = y_test.astype(np.float32)

        self.feature_names = list(X_processed.columns)
        self.input_dim = X_train_scaled.shape[1]
        self.is_regression = True

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print("Task type: Regression (predicting original scale target)")

        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def build_model(self,
                    hidden_layers=[128, 64, 32],
                    dropout_rate=0.3,
                    activation="relu"):
        """Build neural network architecture for regression with regularization"""
        self.model = keras.Sequential()

        self.model.add(
            layers.Dense(
                hidden_layers[0],
                activation=activation,
                input_shape=(self.input_dim, ),
                kernel_regularizer=keras.regularizers.l2(0.001),
            ))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(dropout_rate))

        for units in hidden_layers[1:]:
            self.model.add(
                layers.Dense(
                    units,
                    activation=activation,
                    kernel_regularizer=keras.regularizers.l2(0.001),
                ))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dropout(dropout_rate))

        self.model.add(
            layers.Dense(1,
                         activation=None,
                         kernel_regularizer=keras.regularizers.l2(0.001)))

        print("Model architecture for regression:")
        self.model.summary()

        return self.model

    def compile_model(self,
                      optimizer="adam",
                      learning_rate=0.001,
                      use_asymmetric_loss: bool = False,
                      asymmetric_weights: tuple = (1.5, 1.0),
                      use_combined_loss: bool = False,
                      combined_loss_weights: tuple = (1.0, 0.0001)):
        """Compile the model with selected loss function for original scale.
      Args:
          optimizer (str or keras.optimizers.Optimizer): Optimizer to use.
          learning_rate (float): Learning rate for the optimizer.
          use_asymmetric_loss (bool): Whether to use asymmetric_weighted_male_loss.
          asymmetric_weights (tuple): Weights (under, over) for asymmetric loss.
          use_combined_loss (bool): Whether to use the combined MALE + MAE loss.
          combined_loss_weights (tuple): Weights (w_male, w_orig_mae) for combined loss.
      """
        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer

        loss_fn = None
        if use_combined_loss:
            if use_asymmetric_loss:
                print(
                    "Warning: Both use_combined_loss and use_asymmetric_loss are True. Prioritizing combined_loss."
                )
            loss_fn = create_combined_male_mae_loss(
                w_male=combined_loss_weights[0],
                w_orig_mae=combined_loss_weights[1],
                max_clip_value=self.max_revenue_value)
            print(
                f"Using Combined MALE + MAE loss with weights: MALE={combined_loss_weights[0]}, MAE={combined_loss_weights[1]}, clip_max={self.max_revenue_value}"
            )
        elif use_asymmetric_loss:
            loss_fn = asymmetric_weighted_male_loss(w_u=asymmetric_weights[0],
                                                    w_o=asymmetric_weights[1])
            print(
                f"Using Asymmetric Weighted MALE loss with weights: under={asymmetric_weights[0]}, over={asymmetric_weights[1]}"
            )
        else:
            loss_fn = mae_original_scale_loss
            print(
                f"Using custom Mean Absolute Error loss (mae_original_scale_loss). Note: This uses a hardcoded clip_max of 5,000,000."
            )

        metrics = [
            "mean_absolute_error",
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsolutePercentageError(name="mape"),
            male_original_scale_metric, wape_original_scale_metric
        ]

        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=metrics)

    def setup_tensorboard_callbacks(
            self,
            enable_histograms=False,
            asymmetric_weights=(1.5, 1.0),
            early_stopping_patience_override: Optional[int] = None):
        """Setup comprehensive TensorBoard logging
      Args:
          enable_histograms (bool): Enable weight/bias histogram logging (resource intensive)
          asymmetric_weights (tuple): Weights for asymmetric loss monitoring
          early_stopping_patience_override (Optional[int]): If provided, overrides the default patience for EarlyStopping.
      """

        checkpoint_filepath = os.path.join(self.log_dir, "best_model.keras")

        os.makedirs(self.log_dir, exist_ok=True)

        tensorboard_callback = callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1 if enable_histograms else 0,
            write_graph=False,
            write_images=True,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
        )

        patience = early_stopping_patience_override if early_stopping_patience_override is not None else 40

        early_stopping = callbacks.EarlyStopping(monitor="val_loss",
                                                 patience=patience,
                                                 restore_best_weights=True,
                                                 verbose=1)

        checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                factor=0.5,
                                                patience=5,
                                                min_lr=1e-10,
                                                verbose=1)

        custom_metrics_callback = CustomMetricsCallback(
            log_dir=self.log_dir,
            X_val=self.X_val,
            y_val_log=self.y_val,
            y_val_original_scale=self.y_val_original_scale,
            original_features_val=self.original_features_val,
            plape_thresholds_pct=[10, 20, 30],
            max_log_value=None,
            max_revenue_value=self.max_revenue_value,
            asymmetric_weights=asymmetric_weights)

        self.callbacks_list = [
            tensorboard_callback, checkpoint, reduce_lr,
            custom_metrics_callback, early_stopping
        ]

    def train_model(self, epochs=100, batch_size=32):
        """Train the neural network model"""
        print("Starting training...")

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=self.callbacks_list,
            verbose=1,
        )

        print("Training completed!")
        return self.history

    def evaluate_model(self, model=None, model_name="Neural Network"):
        """
      Comprehensive model evaluation with robust metrics (updated for original scale)
      Args:
          model: Model to evaluate (uses self.model if None)
          model_name (str): Name for logging
      Returns:
          dict: Comprehensive evaluation metrics
      """
        if model is None:
            model = self.model

        if model is None:
            raise ValueError(
                "No model available for evaluation. Train a model first.")

        print(f"\n{'=' * 80}")
        print(f"{model_name.upper()} - COMPREHENSIVE MODEL EVALUATION")
        print(f"{'=' * 80}")

        train_loss_metrics = model.evaluate(self.X_train,
                                            self.y_train,
                                            verbose=0)
        val_loss_metrics = model.evaluate(self.X_val, self.y_val, verbose=0)
        test_loss_metrics = model.evaluate(self.X_test, self.y_test, verbose=0)

        train_loss = train_loss_metrics[0]
        val_loss = val_loss_metrics[0]
        test_loss = test_loss_metrics[0]

        print(f"\nLoss Metrics (Original Scale):")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")

        y_train_pred = model.predict(self.X_train, verbose=0).flatten()
        y_val_pred = model.predict(self.X_val, verbose=0).flatten()
        y_test_pred = model.predict(self.X_test, verbose=0).flatten()

        if self.max_revenue_value is not None:
            y_train_pred = np.clip(y_train_pred,
                                   a_min=0,
                                   a_max=self.max_revenue_value)
            y_val_pred = np.clip(y_val_pred,
                                 a_min=0,
                                 a_max=self.max_revenue_value)
            y_test_pred = np.clip(y_test_pred,
                                  a_min=0,
                                  a_max=self.max_revenue_value)

        def calculate_metrics(y_true, y_pred, set_name):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            non_zero_mask = y_true != 0
            mape = np.nan
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(
                    np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) /
                           y_true[non_zero_mask])) * 100

            median_ae = np.median(np.abs(y_true - y_pred))
            max_error = np.max(np.abs(y_true - y_pred))

            within_10pct = np.mean(
                np.abs(y_true - y_pred) / np.maximum(y_true, 1) <= 0.1) * 100
            within_25pct = np.mean(
                np.abs(y_true - y_pred) / np.maximum(y_true, 1) <= 0.25) * 100
            within_50pct = np.mean(
                np.abs(y_true - y_pred) / np.maximum(y_true, 1) <= 0.5) * 100

            residuals = y_true - y_pred
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)

            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'median_ae': median_ae,
                'max_error': max_error,
                'within_10pct': within_10pct,
                'within_25pct': within_25pct,
                'within_50pct': within_50pct,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'predictions': y_pred
            }

        train_metrics = calculate_metrics(self.y_train_original_scale,
                                          y_train_pred, "Train")
        val_metrics = calculate_metrics(self.y_val_original_scale, y_val_pred,
                                        "Val")
        test_metrics = calculate_metrics(self.y_test_original_scale,
                                         y_test_pred, "Test")

        print(f"\n{'=' * 60}")
        print("REGRESSION METRICS (Original Scale)")
        print(f"{'=' * 60}")

        metrics_names = [
            'MAE', 'RMSE', 'R²', 'MAPE (%)', 'Median AE', 'Max Error'
        ]
        print(f"{'Metric':<12} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 60)

        print(
            f"{'MAE':<12} ${train_metrics['mae']:<11.0f} ${val_metrics['mae']:<11.0f} ${test_metrics['mae']:<11.0f}"
        )
        print(
            f"{'RMSE':<12} ${train_metrics['rmse']:<11.0f} ${val_metrics['rmse']:<11.0f} ${test_metrics['rmse']:<11.0f}"
        )
        print(
            f"{'R²':<12} {train_metrics['r2']:<12.4f} {val_metrics['r2']:<12.4f} {test_metrics['r2']:<12.4f}"
        )
        print(
            f"{'MAPE (%)':<12} {train_metrics['mape']:<12.2f} {val_metrics['mape']:<12.2f} {test_metrics['mape']:<12.2f}"
        )
        print(
            f"{'Median AE':<12} ${train_metrics['median_ae']:<11.0f} ${val_metrics['median_ae']:<11.0f} ${test_metrics['median_ae']:<11.0f}"
        )
        print(
            f"{'Max Error':<12} ${train_metrics['max_error']:<11.0f} ${val_metrics['max_error']:<11.0f} ${test_metrics['max_error']:<11.0f}"
        )

        print(f"\n{'=' * 60}")
        print("PREDICTION ACCURACY ANALYSIS")
        print(f"{'=' * 60}")

        print(f"{'Accuracy':<15} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 60)
        print(
            f"{'Within 10%':<15} {train_metrics['within_10pct']:<12.1f} {val_metrics['within_10pct']:<12.1f} {test_metrics['within_10pct']:<12.1f}"
        )
        print(
            f"{'Within 25%':<15} {train_metrics['within_25pct']:<12.1f} {val_metrics['within_25pct']:<12.1f} {test_metrics['within_25pct']:<12.1f}"
        )
        print(
            f"{'Within 50%':<15} {train_metrics['within_50pct']:<12.1f} {val_metrics['within_50pct']:<12.1f} {test_metrics['within_50pct']:<12.1f}"
        )

        print(f"\n{'=' * 60}")
        print("RESIDUAL ANALYSIS")
        print(f"{'=' * 60}")

        print(f"{'Statistic':<15} {'Train':<12} {'Val':<12} {'Test':<12}")
        print("-" * 60)
        print(
            f"{'Mean':<15} ${train_metrics['residual_mean']:<11.0f} ${val_metrics['residual_mean']:<11.0f} ${test_metrics['residual_mean']:<11.0f}"
        )
        print(
            f"{'Std Dev':<15} ${train_metrics['residual_std']:<11.0f} ${val_metrics['residual_std']:<11.0f} ${test_metrics['residual_std']:<11.0f}"
        )

        print(f"\n{'=' * 60}")
        print("GENERALIZATION ASSESSMENT")
        print(f"{'=' * 60}")

        mae_train_test_ratio = test_metrics['mae'] / train_metrics['mae']
        rmse_train_test_ratio = test_metrics['rmse'] / train_metrics['rmse']
        r2_degradation = train_metrics['r2'] - test_metrics['r2']

        print(
            f"MAE Train/Test Ratio: {mae_train_test_ratio:.3f} ({'Good' if mae_train_test_ratio < 1.2 else 'Concerning' if mae_train_test_ratio < 1.5 else 'Poor'})"
        )
        print(
            f"RMSE Train/Test Ratio: {rmse_train_test_ratio:.3f} ({'Good' if rmse_train_test_ratio < 1.2 else 'Concerning' if rmse_train_test_ratio < 1.5 else 'Poor'})"
        )
        print(
            f"R² Degradation: {r2_degradation:.4f} ({'Good' if r2_degradation < 0.05 else 'Concerning' if r2_degradation < 0.1 else 'Poor'})"
        )

        if hasattr(self, 'history') and self.history is not None:
            print(f"\n{'=' * 60}")
            print("TRAINING CONVERGENCE ANALYSIS")
            print(f"{'=' * 60}")

            final_train_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            best_val_loss = min(self.history.history['val_loss'])
            best_epoch = np.argmin(self.history.history['val_loss']) + 1
            total_epochs = len(self.history.history['loss'])

            print(f"Total Epochs Trained: {total_epochs}")
            print(
                f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})"
            )
            print(f"Final Training Loss: {final_train_loss:.4f}")
            print(f"Final Validation Loss: {final_val_loss:.4f}")
            print(f"Early Stopping: {'Yes' if total_epochs < 200 else 'No'}")

        self.evaluation_results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'loss_metrics': {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss
            },
            'generalization': {
                'mae_ratio': mae_train_test_ratio,
                'rmse_ratio': rmse_train_test_ratio,
                'r2_degradation': r2_degradation
            }
        }

        self._plot_comprehensive_predictions_analysis(
            self.y_test_original_scale, test_metrics['predictions'],
            f"{model_name}: Test Set Analysis")

        self._plot_residuals_analysis(
            self.y_test_original_scale - test_metrics['predictions'],
            f"{model_name}: Residuals Analysis")

        self._plot_prediction_accuracy_distribution(
            self.y_test_original_scale, test_metrics['predictions'],
            f"{model_name}: Prediction Accuracy Distribution")

        return self.evaluation_results

    def _plot_comprehensive_predictions_analysis(self, y_true, y_pred, title):
        """Enhanced predictions vs actual plot with multiple views"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val, max_val = min(y_true.min(),
                               y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val],
                        'r--',
                        lw=2,
                        label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Annual Revenue ($)')
        axes[0, 0].set_ylabel('Predicted Annual Revenue ($)')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(np.log1p(y_true), np.log1p(y_pred), alpha=0.5, s=20)
        log_min, log_max = min(np.log1p(y_true).min(),
                               np.log1p(y_pred).min()), max(
                                   np.log1p(y_true).max(),
                                   np.log1p(y_pred).max())
        axes[0, 1].plot([log_min, log_max], [log_min, log_max], 'r--', lw=2)
        axes[0, 1].set_xlabel('Log(1 + Actual)')
        axes[0, 1].set_ylabel('Log(1 + Predicted)')
        axes[0, 1].set_title('Log Scale View')
        axes[0, 1].grid(True, alpha=0.3)

        relative_error = (y_pred - y_true) / np.maximum(y_true, 1) * 100
        axes[1, 0].scatter(y_true, relative_error, alpha=0.5, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].axhline(y=25,
                           color='orange',
                           linestyle=':',
                           lw=1,
                           label='±25%')
        axes[1, 0].axhline(y=-25, color='orange', linestyle=':', lw=1)
        axes[1, 0].set_xlabel('Actual Annual Revenue ($)')
        axes[1, 0].set_ylabel('Relative Error (%)')
        axes[1, 0].set_title('Relative Error vs Actual')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        abs_error = np.abs(y_pred - y_true)
        axes[1, 1].scatter(y_true, abs_error, alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Actual Annual Revenue ($)')
        axes[1, 1].set_ylabel('Absolute Error ($)')
        axes[1, 1].set_title('Absolute Error vs Actual')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir,
                                 "comprehensive_predictions_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive predictions analysis saved to: {plot_path}")
        plt.close()

    def _plot_residuals_analysis(self, residuals, title):
        """Enhanced residuals analysis with multiple diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].hist(residuals,
                        bins=50,
                        alpha=0.7,
                        edgecolor='black',
                        density=True)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].axvline(x=np.mean(residuals),
                           color='orange',
                           linewidth=2,
                           label=f'Mean: ${np.mean(residuals):.0f}')
        axes[0, 0].set_xlabel('Residuals ($)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Residuals Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Check)')
        axes[0, 1].grid(True, alpha=0.3)

        fitted_values = self.y_test_original_scale - residuals
        axes[1, 0].scatter(fitted_values, residuals, alpha=0.5, s=20)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Fitted Values ($)')
        axes[1, 0].set_ylabel('Residuals ($)')
        axes[1, 0].set_title('Residuals vs Fitted')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].boxplot(residuals, vert=True)
        axes[1, 1].set_ylabel('Residuals ($)')
        axes[1, 1].set_title('Residuals Box Plot')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, "residuals_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Residuals analysis saved to: {plot_path}")
        plt.close()

    def _plot_prediction_accuracy_distribution(self, y_true, y_pred, title):
        """Plot distribution of prediction accuracy"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        ape = np.abs(y_pred - y_true) / np.maximum(y_true, 1) * 100
        axes[0].hist(ape, bins=50, alpha=0.7, edgecolor='black', density=True)
        axes[0].axvline(x=10, color='green', linestyle='--', label='10%')
        axes[0].axvline(x=25, color='orange', linestyle='--', label='25%')
        axes[0].axvline(x=50, color='red', linestyle='--', label='50%')
        axes[0].set_xlabel('Absolute Percentage Error (%)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Prediction Accuracy Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        ape_sorted = np.sort(ape)
        cumulative_pct = np.arange(1,
                                   len(ape_sorted) + 1) / len(ape_sorted) * 100
        axes[1].plot(ape_sorted, cumulative_pct, linewidth=2)
        axes[1].axvline(x=10, color='green', linestyle='--', label='10%')
        axes[1].axvline(x=25, color='orange', linestyle='--', label='25%')
        axes[1].axvline(x=50, color='red', linestyle='--', label='50%')
        axes[1].set_xlabel('Absolute Percentage Error (%)')
        axes[1].set_ylabel('Cumulative Percentage of Predictions')
        axes[1].set_title('Cumulative Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir,
                                 "prediction_accuracy_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Prediction accuracy distribution saved to: {plot_path}")
        plt.close()

    def visualize_selected_features_distributions(self):
        """
      Visualize distributions of the selected features used for training
      """
        print("Visualizing distributions of selected features...")

        if self.X_processed_for_viz is not None:
            all_features = self.X_processed_for_viz.columns
            num_total_features = len(all_features)
            max_features_per_plot = 10
            num_plots = (num_total_features + max_features_per_plot -
                         1) // max_features_per_plot

            for plot_idx in range(num_plots):
                start_idx = plot_idx * max_features_per_plot
                end_idx = min(start_idx + max_features_per_plot,
                              num_total_features)
                current_features = all_features[start_idx:end_idx]
                num_current_features = len(current_features)

                if num_current_features == 0:
                    continue

                fig, axes = plt.subplots(nrows=num_current_features,
                                         ncols=1,
                                         figsize=(10,
                                                  5 * num_current_features))

                if num_current_features == 1:
                    axes = [axes]

                for ax, column in zip(axes, current_features):
                    ax.hist(self.X_processed_for_viz[column],
                            bins=30,
                            alpha=0.7,
                            color='blue')
                    ax.set_title(f'Distribution of {column}')
                    ax.set_xlabel(column)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_filename = f"selected_features_distributions_part_{plot_idx + 1}.png"
                plot_path = os.path.join(self.log_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(
                    f"Selected features distributions (part {plot_idx + 1}) saved to: {plot_path}"
                )
                plt.close(fig)
        else:
            print("No processed features available for visualization")

    def plot_training_history(self):
        """Plots the training and validation loss and primary metrics."""
        if self.history is None or not hasattr(self.history, 'history'):
            print(
                "No training history found or history object is not as expected. Train the model first."
            )
            return

        history_df = pd.DataFrame(self.history.history)

        if history_df.empty:
            print("Training history is empty. Cannot plot.")
            return

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        if 'loss' in history_df.columns:
            plt.plot(history_df['loss'], label='Training Loss')
        if 'val_loss' in history_df.columns:
            plt.plot(history_df['val_loss'], label='Validation Loss')

        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        if 'loss' in history_df.columns or 'val_loss' in history_df.columns:
            plt.legend()
        plt.grid(True, alpha=0.5)

        key_metrics_to_plot = [
            'mean_absolute_error', 'male_original_scale_metric'
        ]
        num_metrics = len(key_metrics_to_plot)

        primary_metric_plot_idx = 0

        metric_to_plot_on_subplot2 = None
        for metric_name in key_metrics_to_plot:
            if metric_name in history_df.columns and f'val_{metric_name}' in history_df.columns:
                metric_to_plot_on_subplot2 = metric_name
                break

        if metric_to_plot_on_subplot2:
            plt.subplot(2, 1, 2)
            metric_display_name = metric_to_plot_on_subplot2.replace(
                "_", " ").title()
            plt.plot(history_df[metric_to_plot_on_subplot2],
                     label=f'Training {metric_display_name}')
            plt.plot(history_df[f'val_{metric_to_plot_on_subplot2}'],
                     label=f'Validation {metric_display_name}')
            plt.title(f'Model {metric_display_name} Over Epochs')
            plt.ylabel(metric_display_name)
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.5)
        else:
            print(
                f"None of the primary metrics ({', '.join(key_metrics_to_plot)}) found in history for the second subplot."
            )

        plt.tight_layout()

        os.makedirs(self.log_dir, exist_ok=True)
        plot_path = os.path.join(self.log_dir, "training_history.png")
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {plot_path}")
        except Exception as e:
            print(f"Error saving training history plot: {e}")
        plt.close()

    def save_for_deployment(self, output_dir: str):
        """
        Saves all necessary artifacts for API deployment.

        Args:
            output_dir (str): The directory to save the artifacts to.
        """
        print(f"\n{'=' * 80}")
        print(f"SAVING ARTIFACTS FOR DEPLOYMENT to: {output_dir}")
        print(f"{'=' * 80}")

        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "neural_net_model.keras")
        self.model.save(model_path)
        print(f"✅ Model saved to: {model_path}")

        scaler_path = os.path.join(output_dir, "neural_net_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved to: {scaler_path}")

        features_path = os.path.join(output_dir,
                                     "neural_net_feature_names.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(
            f"✅ {len(self.feature_names)} feature names saved to: {features_path}"
        )

        metadata = {
            "training_time":
            datetime.now().isoformat(),
            "input_dim":
            self.input_dim,
            "cap_value":
            self.max_revenue_value,
            "training_samples":
            len(self.X_train),
            "original_features_selected":
            self.config.selected_features if hasattr(self, 'config') else "N/A"
        }
        metadata_path = os.path.join(output_dir, "neural_net_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadata saved to: {metadata_path}")
        print("\nDeployment artifacts ready.")


@dataclass
class NeuralNetConfig:
    """Configuration class for Neural Network model training and evaluation."""

    data_path: str = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"
    selected_features: List[str] = None
    cap_outliers: bool = True
    cap_percentile: float = 0.97

    hidden_layers: List[int] = None
    dropout_rate: float = 0.1
    activation: str = "relu"

    optimizer: str = "adam"
    learning_rate: float = 0.001
    epochs: int = 200
    batch_size: int = 64

    loss_choice: str = "mse"
    combined_loss_weights: Tuple[float, float] = (1.0, 0.00001)
    asymmetric_weights: Tuple[float, float] = (1.5, 1.0)

    enable_histograms: bool = False
    log_dir_base: str = "logs/neural_net"
    experiment_name: str = "50_features_expanded"

    def __post_init__(self):
        """Set default values that depend on other attributes."""
        if self.selected_features is None:
            self.selected_features = [
                # --- Core 20 Features ---
                'latitude',
                'longitude',
                'distance_to_center',
                'accommodates',
                'bedrooms',
                'bathrooms',
                'price_log',
                'amenities_count',
                'review_scores_rating',
                'number_of_reviews',
                'host_is_superhost',
                'instant_bookable',
                'neighbourhood_price_rank',
                'property_type_frequency',
                'host_response_rate',
                'review_scores_location',
                'minimum_nights',
                'calculated_host_listings_count',
                'name_positive_sentiment',

                # --- Adding 30 More for a Richer Model ---

                # More specific review scores
                'review_scores_cleanliness',
                'review_scores_checkin',
                'review_scores_communication',
                'review_scores_value',
                'review_scores_accuracy',

                # Deeper host analysis
                'host_acceptance_rate',
                'host_identity_verified',
                'host_total_listings_count',
                'host_has_profile_pic',

                # Core property characteristics
                'beds',
                'property_size',  # Engineered feature

                # Engineered features for more context
                'beds_per_person',
                'bedrooms_per_person',
                'total_review_months',
                'review_intensity',  # Engineered feature
                'price_competitiveness',  # Engineered feature

                # Text-based features
                'name_length',
                'name_word_count',
                'description_length',
                'description_word_count',

                # Key amenities (as binary flags)
                'has_wifi',
                'has_kitchen',
                'has_air_conditioning',
                'has_parking',
                'has_elevator',
                'has_gym',

                # Amenity counts by type
                'luxury_amenities_count',
                'basic_amenities_count',

                # Host listing distribution
                'calculated_host_listings_count_entire_homes',
                'calculated_host_listings_count_private_rooms',
            ]

        if self.hidden_layers is None:
            self.hidden_layers = [512, 256, 128, 64]


if __name__ == "__main__":

    config = NeuralNetConfig()
    config

    print("--- Using Configuration for Training ---")

    print(pd.Series(asdict(config)).to_string())
    print("-" * 40)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join(config.log_dir_base,
                               f"{config.experiment_name}_{timestamp}")
    print(
        f"--> Logs and artifacts for this run will be saved in: {run_log_dir}")

    nn_model = NeuralNetworkModel(log_dir=run_log_dir)

    nn_model.config = config

    nn_model.load_and_preprocess_data(data_path=config.data_path,
                                      feature_subset=config.selected_features,
                                      cap_outliers=config.cap_outliers,
                                      cap_percentile=config.cap_percentile)

    nn_model.build_model(hidden_layers=config.hidden_layers,
                         dropout_rate=config.dropout_rate,
                         activation=config.activation)

    use_combined = config.loss_choice == 'combined'
    use_asymm = config.loss_choice == 'asymmetric'

    nn_model.compile_model(optimizer=config.optimizer,
                           learning_rate=config.learning_rate,
                           use_asymmetric_loss=use_asymm,
                           asymmetric_weights=config.asymmetric_weights,
                           use_combined_loss=use_combined,
                           combined_loss_weights=config.combined_loss_weights)

    nn_model.setup_tensorboard_callbacks(
        enable_histograms=config.enable_histograms)

    nn_model.train_model(epochs=config.epochs, batch_size=config.batch_size)

    nn_model.plot_training_history()

    print("\n--- Evaluating Final Model on Test Set ---")
    evaluation_results = nn_model.evaluate_model(
        model_name="Final Neural Network (50+ features)")

    nn_model.visualize_selected_features_distributions()

    deployment_dir = "models/deployment_artifacts/neural_net_50_features"
    nn_model.save_for_deployment(output_dir=deployment_dir)

    print(
        "\n--- Training, Evaluation, and Deployment Artifact Generation Complete ---"
    )
    print(
        f"✅ Model and artifacts for the 50+ feature model saved in: {deployment_dir}"
    )
    print(
        f"To see detailed logs, run: tensorboard --logdir {config.log_dir_base}"
    )
