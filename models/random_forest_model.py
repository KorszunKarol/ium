import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


class RandomForestModel:

    def __init__(self, log_dir="logs/random_forest"):
        """
      Initialize Random Forest Model for regression
      Args:
          log_dir (str): Directory to save logs, models, and plots
      """
        self.model = None
        self.best_model = None
        self.log_dir = log_dir
        self.feature_names = None
        self.is_regression = True

        self.log_prediction_cap_for_expm1 = 80.0
        self.max_revenue_value = 1000000

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_train_original_scale = None
        self.y_val_original_scale = None
        self.y_test_original_scale = None

        self.cv_results = None
        self.evaluation_results = None
        self.feature_importance_results = None

        os.makedirs(log_dir, exist_ok=True)

    def load_and_preprocess_data(self, data_path):
        """
      Load and preprocess data for Random Forest regression
      This method mirrors the Neural Network preprocessing but without scaling
      numerical features, as Random Forest handles mixed scales well.
      Args:
          data_path (str): Path to the dataset pickle file
      Returns:
          tuple: Processed training, validation, and test sets
      """
        print("Loading Random Forest dataset...")

        if data_path.endswith(".pkl"):
            df = pd.read_pickle(data_path)
        else:
            df = pd.read_csv(data_path)

        print(f"Dataset shape: {df.shape}")

        target_col = "annual_revenue_adj"
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in dataset. "
                f"Available columns: {list(df.columns)}")

        X = df.drop(target_col, axis=1)
        y_original = df[target_col]

        print(f"Target variable: {target_col}")
        print(f"Original Target statistics:")
        print(f"  Mean: ${y_original.mean():.0f}")
        print(f"  Median: ${y_original.median():.0f}")
        print(f"  Min: ${y_original.min():.0f}")
        print(f"  Max: ${y_original.max():.0f}")
        print(f"  Std: ${y_original.std():.0f}")

        y_log_transformed = np.log1p(y_original.values.astype(np.float32))
        print(f"Log-Transformed Target statistics:")
        print(f"  Mean: {y_log_transformed.mean():.2f}")
        print(f"  Std: {y_log_transformed.std():.2f}")
        print(f"  Min: {y_log_transformed.min():.2f}")
        print(f"  Max: {y_log_transformed.max():.2f}")

        id_cols = ["listing_id", "id", "host_id"]
        X = X.drop(columns=[col for col in id_cols if col in X.columns],
                   errors="ignore")

        categorical_columns = X.select_dtypes(
            include=["object", "category"]).columns
        numerical_columns = X.select_dtypes(
            exclude=["object", "category"]).columns

        print(
            f"Categorical columns ({len(categorical_columns)}): "
            f"{list(categorical_columns)[:5]}{'...' if len(categorical_columns) > 5 else ''}"
        )
        print(
            f"Numerical columns ({len(numerical_columns)}): "
            f"{list(numerical_columns)[:5]}{'...' if len(numerical_columns) > 5 else ''}"
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


        nn_selected_features = [
            'latitude', 'longitude', 'distance_to_center', 'accommodates',
            'bedrooms', 'bathrooms', 'price_log', 'amenities_count',
            'review_scores_rating', 'number_of_reviews', 'host_is_superhost',
            'host_days_active', 'instant_bookable', 'neighbourhood_price_rank',
            'property_type_frequency', 'host_response_rate',
            'review_scores_location', 'minimum_nights',
            'calculated_host_listings_count', 'name_positive_sentiment'
        ]







        common_features = [
            feature for feature in nn_selected_features
            if feature in X_processed.columns
        ]
        missing_features = [
            feature for feature in nn_selected_features
            if feature not in X_processed.columns
        ]

        if missing_features:
            print(
                f"Warning: The following features from the NN's selected list were not found "
                f"in the Random Forest's processed columns and will be excluded: {missing_features}"
            )





        if not common_features:
            raise ValueError(
                "No common features found between NN's selected list and RF's processed columns. "
                "Please check feature names and preprocessing steps.")

        X_processed = X_processed[common_features]

        self.feature_names = list(X_processed.columns)
        print(
            f"Total features after preprocessing and subset selection: {len(self.feature_names)}"
        )

        X_train_val, X_test_processed, y_train_val_log, y_test_log = train_test_split(
            X_processed,
            y_log_transformed,
            test_size=0.2,
            random_state=RANDOM_STATE)

        X_train_processed, X_val_processed, y_train_log, y_val_log = train_test_split(
            X_train_val,
            y_train_val_log,
            test_size=0.2,
            random_state=RANDOM_STATE)

        y_train_val_original, y_test_original_scale = train_test_split(
            y_original.values.astype(np.float32),
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        y_train_original_scale, y_val_original_scale = train_test_split(
            y_train_val_original, test_size=0.2, random_state=RANDOM_STATE)

        self.X_train = X_train_processed.values.astype(np.float32)
        self.X_val = X_val_processed.values.astype(np.float32)
        self.X_test = X_test_processed.values.astype(np.float32)
        self.y_train = y_train_log.astype(np.float32)
        self.y_val = y_val_log.astype(np.float32)
        self.y_test = y_test_log.astype(np.float32)
        self.y_train_original_scale = y_train_original_scale
        self.y_val_original_scale = y_val_original_scale
        self.y_test_original_scale = y_test_original_scale

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print("Task type: Regression (predicting log-transformed target)")
        print(
            "Note: Features are NOT scaled (Random Forest handles mixed scales)"
        )

        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def train_baseline_model(self,
                             n_estimators=100,
                             max_depth=None,
                             min_samples_split=2,
                             min_samples_leaf=1):
        """
      Train a baseline Random Forest model with default/simple parameters
      Args:
          n_estimators (int): Number of trees
          max_depth (int): Maximum depth of trees
          min_samples_split (int): Minimum samples required to split
          min_samples_leaf (int): Minimum samples required at leaf
      Returns:
          RandomForestRegressor: Trained baseline model
      """
        print("Training baseline Random Forest model...")

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )

        self.model.fit(self.X_train, self.y_train)

        print("Baseline model training completed!")
        print(f"Model parameters: {self.model.get_params()}")

        return self.model

    def hyperparameter_tuning(self,
                              search_type="randomized",
                              cv_folds=3,
                              n_iter=50):
        """
      Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
      Args:
          search_type (str): "grid" or "randomized"
          cv_folds (int): Number of cross-validation folds
          n_iter (int): Number of iterations for randomized search
      Returns:
          Best estimator from the search
      """
        print(f"Starting {search_type} hyperparameter tuning...")

        param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

        rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

        if search_type.lower() == "grid":
            search = GridSearchCV(
                rf,
                param_grid,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                rf,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1,
            )

        X_train_val = np.vstack([self.X_train, self.X_val])
        y_train_val = np.hstack([self.y_train, self.y_val])

        search.fit(X_train_val, y_train_val)

        self.cv_results = search
        self.best_model = search.best_estimator_

        print("Hyperparameter tuning completed!")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score (neg_MSE): {search.best_score_:.4f}")

        return self.best_model

    def evaluate_model(self, model=None, model_name="Random Forest"):
        """
      Comprehensive model evaluation on test set
      Args:
          model: Model to evaluate (uses best_model if None)
          model_name (str): Name for logging
      Returns:
          dict: Evaluation metrics
      """
        if model is None:
            model = self.best_model if self.best_model is not None else self.model

        if model is None:
            raise ValueError(
                "No model available for evaluation. Train a model first.")

        print(f"\n{'=' * 60}")
        print(f"{model_name.upper()} MODEL EVALUATION - REGRESSION")
        print(f"{'=' * 60}")

        y_pred_log = model.predict(self.X_test)

        if self.log_prediction_cap_for_expm1 is not None:
            y_pred_log_orig = y_pred_log.copy()
            y_pred_log = np.clip(y_pred_log,
                                 a_min=None,
                                 a_max=self.log_prediction_cap_for_expm1)
            clipped_count = np.sum(
                y_pred_log_orig > self.log_prediction_cap_for_expm1)
            if clipped_count > 0:
                print(
                    f"Clipped {clipped_count} predictions "
                    f"({clipped_count / len(y_pred_log) * 100:.2f}%) to prevent overflow"
                )
                print(
                    f"Max predicted log value before clipping: {np.max(y_pred_log_orig):.4f}"
                )
                print(
                    f"Max predicted log value after clipping: {np.max(y_pred_log):.4f}"
                )

        y_pred_original_scale = np.expm1(y_pred_log)

        if self.max_revenue_value is not None:
            y_pred_original_scale_orig = y_pred_original_scale.copy()
            y_pred_original_scale = np.clip(y_pred_original_scale,
                                            a_min=0,
                                            a_max=self.max_revenue_value)
            orig_scale_clipped = np.sum(
                y_pred_original_scale_orig > self.max_revenue_value)
            if orig_scale_clipped > 0:
                print(
                    f"Additionally clipped {orig_scale_clipped} original-scale predictions "
                    f"to ${self.max_revenue_value}")

        mae = mean_absolute_error(self.y_test_original_scale,
                                  y_pred_original_scale)
        rmse = np.sqrt(
            mean_squared_error(self.y_test_original_scale,
                               y_pred_original_scale))
        r2 = r2_score(self.y_test_original_scale, y_pred_original_scale)

        non_zero_mask = self.y_test_original_scale != 0
        mape = np.nan
        if np.sum(non_zero_mask) > 0:
            mape = (np.mean(
                np.abs((self.y_test_original_scale[non_zero_mask] -
                        y_pred_original_scale[non_zero_mask]) /
                       self.y_test_original_scale[non_zero_mask])) * 100)

        print(f"\nRegression Metrics (Original Scale):")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        self.evaluation_results = {
            "mae_original_scale": mae,
            "rmse_original_scale": rmse,
            "r2_score_original_scale": r2,
            "mape_original_scale": mape,
            "predictions_log": y_pred_log,
            "predictions_original": y_pred_original_scale,
        }

        self._plot_predictions_vs_actual(
            self.y_test_original_scale,
            y_pred_original_scale,
            f"{model_name}: Predictions vs Actual (Original Scale)",
        )

        residuals = self.y_test_original_scale - y_pred_original_scale
        self._plot_residuals_histogram(
            residuals,
            f"{model_name}: Residuals Distribution (Original Scale)")

        return self.evaluation_results

    def analyze_feature_importance(self,
                                   model=None,
                                   top_k=20,
                                   include_permutation=True):
        """
      Comprehensive feature importance analysis
      Args:
          model: Model to analyze (uses best_model if None)
          top_k (int): Number of top features to display
          include_permutation (bool): Whether to include permutation importance
      Returns:
          dict: Feature importance results
      """
        if model is None:
            model = self.best_model if self.best_model is not None else self.model

        if model is None:
            raise ValueError(
                "No model available for analysis. Train a model first.")

        print(f"\n{'=' * 60}")
        print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
        print(f"{'=' * 60}")

        results = {}

        print(
            "\n1. Built-in Feature Importance (Mean Decrease in Impurity)...")

        importance_mdi = pd.DataFrame({
            "feature": self.feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        print(
            f"   Top {min(top_k, len(importance_mdi))} features by MDI importance:"
        )
        for idx, row in importance_mdi.head(top_k).iterrows():
            print(f"     {row['feature']}: {row['importance']:.6f}")

        results["mdi"] = importance_mdi

        self._plot_feature_importance(
            importance_mdi.head(top_k),
            "Random Forest Feature Importance (Mean Decrease in Impurity)",
            "mdi_importance.png",
        )

        if include_permutation:
            print("\n2. Permutation Importance...")

            n_samples = min(1000, len(self.X_test))
            X_test_subset = self.X_test[:n_samples]
            y_test_subset = self.y_test_original_scale[:n_samples]

            def rf_predict_original_scale(X):
                pred_log = model.predict(X)
                if self.log_prediction_cap_for_expm1 is not None:
                    pred_log = np.clip(pred_log,
                                       a_min=None,
                                       a_max=self.log_prediction_cap_for_expm1)
                pred_original = np.expm1(pred_log)
                if self.max_revenue_value is not None:
                    pred_original = np.clip(pred_original,
                                            a_min=0,
                                            a_max=self.max_revenue_value)
                return pred_original

            class ModelWrapper:

                def __init__(self, predict_func):
                    self.predict = predict_func

                def fit(self, X, y):
                    return self

            model_wrapper = ModelWrapper(rf_predict_original_scale)

            perm_importance = permutation_importance(
                model_wrapper,
                X_test_subset,
                y_test_subset,
                n_repeats=10,
                random_state=RANDOM_STATE,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )

            importance_perm = pd.DataFrame({
                "feature":
                self.feature_names,
                "importance_mean":
                perm_importance.importances_mean,
                "importance_std":
                perm_importance.importances_std,
            }).sort_values("importance_mean", ascending=False)

            print(
                f"   Top {min(top_k, len(importance_perm))} features by permutation importance:"
            )
            for idx, row in importance_perm.head(top_k).iterrows():
                print(f"     {row['feature']}: {row['importance_mean']:.4f} "
                      f"(±{row['importance_std']:.4f})")

            results["permutation"] = importance_perm

            self._plot_feature_importance(
                importance_perm.head(top_k),
                "Random Forest Permutation Importance",
                "permutation_importance.png",
                error_bars=True,
            )

        self.feature_importance_results = results
        return results

    def _plot_feature_importance(self,
                                 importance_df,
                                 title,
                                 filename,
                                 error_bars=False):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))

        if error_bars and "importance_std" in importance_df.columns:
            plt.barh(
                range(len(importance_df)),
                importance_df["importance_mean"].values,
                xerr=importance_df["importance_std"].values,
                alpha=0.7,
            )
            x_col = "importance_mean"
        else:
            plt.barh(range(len(importance_df)),
                     importance_df["importance"].values,
                     alpha=0.7)
            x_col = "importance"

        plt.yticks(range(len(importance_df)), importance_df["feature"].values)
        plt.xlabel("Importance")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_path = os.path.join(self.log_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"   Feature importance plot saved to: {plot_path}")
        plt.close()

    def _plot_predictions_vs_actual(self, y_true, y_pred, title):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 8))

        plt.scatter(y_true, y_pred, alpha=0.5, s=20)

        min_val, max_val = (
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        )
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        plt.text(
            0.05,
            0.95,
            f"MAE: ${mae:.0f}\nR²: {r2:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.xlabel("Actual Annual Revenue ($)")
        plt.ylabel("Predicted Annual Revenue ($)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.gca().ticklabel_format(style="plain", axis="both")

        plot_path = os.path.join(self.log_dir, "predictions_vs_actual.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Predictions vs Actual plot saved to: {plot_path}")
        plt.close()

    def _plot_residuals_histogram(self, residuals, title):
        """Plot histogram of residuals"""
        plt.figure(figsize=(10, 6))

        plt.hist(residuals,
                 bins=50,
                 alpha=0.7,
                 edgecolor="black",
                 density=True)

        plt.axvline(x=0,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Zero Residual")

        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        plt.axvline(
            x=mean_res,
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Mean: ${mean_res:.0f}",
        )

        plt.text(
            0.05,
            0.95,
            f"Mean: ${mean_res:.0f}\nStd: ${std_res:.0f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.xlabel("Residuals ($)")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(self.log_dir, "residuals_histogram.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Residuals histogram saved to: {plot_path}")
        plt.close()

    def save_model(self, model=None, filename=None):
        """
      Save the trained model
      Args:
          model: Model to save (uses best_model if None)
          filename (str): Custom filename (auto-generated if None)
      Returns:
          str: Path to saved model
      """
        if model is None:
            model = self.best_model if self.best_model is not None else self.model

        if model is None:
            raise ValueError(
                "No model available to save. Train a model first.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"random_forest_model_{timestamp}.joblib"

        model_path = os.path.join(self.log_dir, filename)
        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")
        return model_path

    def load_model(self, model_path):
        """
      Load a saved model
      Args:
          model_path (str): Path to saved model
      Returns:
          Loaded model
      """
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print("Model loaded successfully!")
        return self.model

    def generate_comparison_report(self, nn_results=None):
        """
      Generate a comparison report between Random Forest and Neural Network
      Args:
          nn_results (dict): Neural Network evaluation results for comparison
      """
        print(f"\n{'=' * 80}")
        print("MODEL COMPARISON REPORT: RANDOM FOREST vs NEURAL NETWORK")
        print(f"{'=' * 80}")

        if self.evaluation_results is None:
            print("Random Forest results not available. Run evaluation first.")
            return

        rf_results = self.evaluation_results

        print(f"\nRandom Forest Results:")
        print(f"  MAE: ${rf_results['mae_original_scale']:.2f}")
        print(f"  RMSE: ${rf_results['rmse_original_scale']:.2f}")
        print(f"  R² Score: {rf_results['r2_score_original_scale']:.4f}")
        print(f"  MAPE: {rf_results['mape_original_scale']:.2f}%")

        if nn_results is not None:
            print(f"\nNeural Network Results (for comparison):")
            print(f"  MAE: ${nn_results.get('mae_original_scale', 'N/A')}")
            print(f"  RMSE: ${nn_results.get('rmse_original_scale', 'N/A')}")
            print(
                f"  R² Score: {nn_results.get('r2_score_original_scale', 'N/A')}"
            )
            print(f"  MAPE: {nn_results.get('mape_original_scale', 'N/A')}%")

            if all(key in nn_results for key in [
                    "mae_original_scale",
                    "rmse_original_scale",
                    "r2_score_original_scale",
            ]):
                mae_improvement = ((nn_results["mae_original_scale"] -
                                    rf_results["mae_original_scale"]) /
                                   nn_results["mae_original_scale"]) * 100
                rmse_improvement = ((nn_results["rmse_original_scale"] -
                                     rf_results["rmse_original_scale"]) /
                                    nn_results["rmse_original_scale"]) * 100
                r2_improvement = (rf_results["r2_score_original_scale"] -
                                  nn_results["r2_score_original_scale"])

                print(f"\nImprovement (Random Forest vs Neural Network):")
                print(
                    f"  MAE: {mae_improvement:+.2f}% ({'better' if mae_improvement > 0 else 'worse'})"
                )
                print(
                    f"  RMSE: {rmse_improvement:+.2f}% ({'better' if rmse_improvement > 0 else 'worse'})"
                )
                print(
                    f"  R² Score: {r2_improvement:+.4f} ({'better' if r2_improvement > 0 else 'worse'})"
                )

        print(f"\nModel Characteristics:")
        print(f"Random Forest:")
        print(f"  - No feature scaling required")
        print(f"  - Built-in feature importance")
        print(f"  - Faster training and prediction")
        print(f"  - Better interpretability")
        print(f"  - Handles mixed data types well")

        print(f"\nNeural Network:")
        print(f"  - Requires feature scaling")
        print(f"  - Complex feature interactions")
        print(f"  - Slower training, requires tuning")
        print(f"  - Less interpretable")
        print(f"  - Better for complex non-linear patterns")


def main():
    """
  Main function to demonstrate Random Forest model usage
  """

    rf_model = RandomForestModel()

    data_path = (
        "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"
    )

    try:
        rf_model.load_and_preprocess_data(data_path)

        rf_model.train_baseline_model(n_estimators=100)

        print("\n" + "=" * 60)
        print("BASELINE MODEL EVALUATION")
        print("=" * 60)
        rf_model.evaluate_model(model_name="Baseline Random Forest")

        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        rf_model.hyperparameter_tuning(search_type="randomized", n_iter=30)

        print("\n" + "=" * 60)
        print("TUNED MODEL EVALUATION")
        print("=" * 60)
        rf_model.evaluate_model(model_name="Tuned Random Forest")

        rf_model.analyze_feature_importance(top_k=20, include_permutation=True)

        rf_model.save_model()

        rf_model.generate_comparison_report()

        print(f"\n{'=' * 60}")
        print("RANDOM FOREST PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {rf_model.log_dir}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"Error in Random Forest pipeline: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
