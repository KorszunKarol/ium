import os
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


import xgboost as xgb
import lightgbm as lgb


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


@dataclass
class GradientBoostConfig:
    """Configuration class for Gradient Boosting model training and evaluation."""


    data_path: str = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"
    selected_features: List[str] = None
    cap_outliers: bool = True
    cap_percentile: float = 0.99
    use_scaled_features: bool = True


    model_type: str = "xgboost"


    xgb_n_estimators: int = 200
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: int = 3
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0


    lgb_n_estimators: int = 200
    lgb_max_depth: int = 7
    lgb_learning_rate: float = 0.05
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_min_child_samples: int = 20
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0


    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    verbose: int = 0


    tune_hyperparameters: bool = False
    tuning_method: str = "randomized"
    cv_folds: int = 5
    n_iter: int = 50


    log_dir: str = "logs/gradient_boost"

    def __post_init__(self):
        """Set default values that depend on other attributes."""
        if self.selected_features is None:
            self.selected_features = [
                'latitude', 'longitude', 'distance_to_center', 'accommodates',
                'bedrooms', 'bathrooms', 'price_log', 'amenities_count',
                'review_scores_rating', 'number_of_reviews',
                'host_is_superhost', 'host_days_active', 'instant_bookable',
                'neighbourhood_price_rank', 'property_type_frequency',
                'host_response_rate', 'review_scores_location',
                'minimum_nights', 'calculated_host_listings_count',
                'name_positive_sentiment'
            ]


class GradientBoostModel:

    def __init__(self, config: GradientBoostConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if config.use_scaled_features else None
        self.feature_names = None
        self.max_revenue_value = None
        self.evaluation_results = None


        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_train_original_scale = None
        self.y_val_original_scale = None
        self.y_test_original_scale = None


        os.makedirs(config.log_dir, exist_ok=True)

    def load_and_preprocess_data(self):
        """Load and preprocess data using the same pipeline as neural network."""
        print("Loading and preprocessing data for Gradient Boosting...")

        if self.config.data_path.endswith(".pkl"):
            df = pd.read_pickle(self.config.data_path)
        else:
            df = pd.read_csv(self.config.data_path)

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


        if self.config.cap_outliers:
            cap_value = np.percentile(y_original,
                                      self.config.cap_percentile * 100)
            outliers_count = np.sum(y_original > cap_value)
            outliers_percentage = (outliers_count / len(y_original)) * 100

            print(f"\nOutlier Analysis:")
            print(f"  Cap percentile: {self.config.cap_percentile*100:.1f}%")
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


        if self.config.selected_features is not None:
            available_features = [
                f for f in self.config.selected_features
                if f in X_processed.columns
            ]
            missing_features = [
                f for f in self.config.selected_features
                if f not in X_processed.columns
            ]

            if missing_features:
                print(f"Warning: Missing features: {missing_features}")

            X_processed = X_processed[available_features]
            print(f"Using {len(available_features)} selected features")


        X_train_val, X_test_processed, y_train_val, y_test = train_test_split(
            X_processed, y_original, test_size=0.2, random_state=42)

        X_train_processed, X_val_processed, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42)


        self.y_train_original_scale = y_train
        self.y_val_original_scale = y_val
        self.y_test_original_scale = y_test


        if self.config.use_scaled_features:
            print("Applying StandardScaler to features (for NN compatibility)")
            X_train_scaled = self.scaler.fit_transform(X_train_processed)
            X_val_scaled = self.scaler.transform(X_val_processed)
            X_test_scaled = self.scaler.transform(X_test_processed)

            self.X_train = X_train_scaled.astype(np.float32)
            self.X_val = X_val_scaled.astype(np.float32)
            self.X_test = X_test_scaled.astype(np.float32)
        else:
            print(
                "Using unscaled features (tree models typically don't need scaling)"
            )
            self.X_train = X_train_processed.values.astype(np.float32)
            self.X_val = X_val_processed.values.astype(np.float32)
            self.X_test = X_test_processed.values.astype(np.float32)

        self.y_train = y_train.astype(np.float32)
        self.y_val = y_val.astype(np.float32)
        self.y_test = y_test.astype(np.float32)

        self.feature_names = list(X_processed.columns)

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print("Data preprocessing completed!")

        return (self.X_train, self.X_val, self.X_test, self.y_train,
                self.y_val, self.y_test)

    def build_model(self):
        """Build the gradient boosting model based on configuration."""
        print(f"Building {self.config.model_type.upper()} model...")

        if self.config.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                min_child_weight=self.config.xgb_min_child_weight,
                reg_alpha=self.config.xgb_reg_alpha,
                reg_lambda=self.config.xgb_reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbosity=self.config.verbose)
        elif self.config.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.lgb_n_estimators,
                max_depth=self.config.lgb_max_depth,
                learning_rate=self.config.lgb_learning_rate,
                subsample=self.config.lgb_subsample,
                colsample_bytree=self.config.lgb_colsample_bytree,
                min_child_samples=self.config.lgb_min_child_samples,
                reg_alpha=self.config.lgb_reg_alpha,
                reg_lambda=self.config.lgb_reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbosity=self.config.verbose)
        else:
            raise ValueError(
                f"Unsupported model type: {self.config.model_type}")

        print(f"{self.config.model_type.upper()} model created successfully!")
        return self.model

    def train_model(self):
        """Train the gradient boosting model."""
        print(f"Training {self.config.model_type.upper()} model...")

        if self.config.tune_hyperparameters:
            print(
                f"Performing hyperparameter tuning using {self.config.tuning_method} search..."
            )

            param_dist = {}
            fit_params = {}

            if self.config.model_type == "xgboost":
                param_dist = {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [3, 5, 7, 9, 11],
                    'learning_rate': [0.01, 0.02, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5, 7],
                    'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
                    'reg_lambda': [0.01, 0.1, 1, 5, 10]
                }


                if self.config.early_stopping_rounds and self.config.early_stopping_rounds > 0:
                    fit_params = {
                        'early_stopping_rounds':
                        self.config.early_stopping_rounds,
                        'eval_set': [(self.X_val, self.y_val)],
                        'verbose':
                        False
                    }
                base_model = xgb.XGBRegressor(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    verbosity=0
                )

            elif self.config.model_type == "lightgbm":
                param_dist = {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [3, 5, 7, 9, 11, -1],
                    'learning_rate': [0.01, 0.02, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'min_child_samples': [10, 20, 30, 50],
                    'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
                    'reg_lambda': [0.01, 0.1, 1, 5, 10]
                }
                if self.config.early_stopping_rounds and self.config.early_stopping_rounds > 0:
                    fit_params = {
                        'callbacks': [
                            lgb.early_stopping(stopping_rounds=self.config.
                                               early_stopping_rounds,
                                               verbose=-1)
                        ],
                        'eval_set': [(self.X_val, self.y_val)]
                    }
                base_model = lgb.LGBMRegressor(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    verbosity=-1
                )
            else:
                raise ValueError(
                    f"Unsupported model type for tuning: {self.config.model_type}"
                )

            search_cv = None
            if self.config.tuning_method == "randomized":
                search_cv = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_dist,
                    n_iter=self.config.n_iter,
                    cv=self.config.cv_folds,
                    scoring='neg_mean_absolute_error',
                    n_jobs=self.config.n_jobs,
                    random_state=self.config.random_state,
                    verbose=1
                )
            elif self.config.tuning_method == "grid":
                search_cv = GridSearchCV(
                    estimator=base_model,
                    param_grid=
                    param_dist,
                    cv=self.config.cv_folds,
                    scoring='neg_mean_absolute_error',
                    n_jobs=self.config.n_jobs,
                    verbose=1
                )
            else:
                raise ValueError(
                    f"Unsupported tuning_method: {self.config.tuning_method}")

            print(
                f"Starting {self.config.tuning_method} search with {self.config.n_iter if self.config.tuning_method == 'randomized' else 'all'} iterations and {self.config.cv_folds} CV folds."
            )
            if fit_params:
                print(
                    f"Using early stopping with {self.config.early_stopping_rounds} rounds based on X_val, y_val."
                )

            search_cv.fit(self.X_train, self.y_train, **fit_params)

            self.model = search_cv.best_estimator_
            print(
                f"\nBest parameters found by {self.config.tuning_method} search:"
            )
            for param, value in search_cv.best_params_.items():
                print(f"  {param}: {value}")
            print(f"Best CV score (Negative MAE): {search_cv.best_score_:.4f}")

            for param, value in search_cv.best_params_.items():
                if self.config.model_type == "xgboost" and hasattr(
                        self.config, f"xgb_{param}"):
                    setattr(self.config, f"xgb_{param}", value)
                elif self.config.model_type == "lightgbm" and hasattr(
                        self.config, f"lgb_{param}"):
                    setattr(self.config, f"lgb_{param}", value)

        else:
            if self.model is None:
                self.build_model()

            if self.config.model_type == "xgboost":
                self.model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=
                    False



                )
            elif self.config.model_type == "lightgbm":
                self.model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    callbacks=[
                        lgb.early_stopping(
                            self.config.early_stopping_rounds,
                            verbose=False if self.config.verbose < 1 else True)
                    ])

        print(f"{self.config.model_type.upper()} training completed!")


        self._print_feature_importance()

        return self.model

    def _print_feature_importance(self, top_n=10):
        """Print top feature importances."""
        if self.model is None:
            return

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"\nTop {top_n} Feature Importances:")
            print("-" * 40)
            for i, (_, row) in enumerate(
                    feature_importance_df.head(top_n).iterrows()):
                print(
                    f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")

    def evaluate_model(self, model_name=None):
        """Comprehensive model evaluation matching neural network format."""
        if model_name is None:
            model_name = f"{self.config.model_type.upper()} Gradient Boost"

        if self.model is None:
            raise ValueError(
                "No model available for evaluation. Train a model first.")

        print(f"\n{'=' * 80}")
        print(f"{model_name.upper()} - COMPREHENSIVE MODEL EVALUATION")
        print(f"{'=' * 80}")


        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)
        y_test_pred = self.model.predict(self.X_test)


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

        self.evaluation_results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'generalization': {
                'mae_ratio': mae_train_test_ratio,
                'rmse_ratio': rmse_train_test_ratio,
                'r2_degradation': r2_degradation
            }
        }


        self._plot_predictions_analysis(self.y_test_original_scale,
                                        test_metrics['predictions'],
                                        model_name)
        self._plot_feature_importance()

        return self.evaluation_results

    def _plot_predictions_analysis(self, y_true, y_pred, title):
        """Plot predictions vs actual analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))


        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val, max_val = min(y_true.min(),
                               y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val],
                     'r--',
                     lw=2,
                     label='Perfect Prediction')
        axes[0].set_xlabel('Actual Annual Revenue ($)')
        axes[0].set_ylabel('Predicted Annual Revenue ($)')
        axes[0].set_title(f'{title}: Predictions vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)


        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Annual Revenue ($)')
        axes[1].set_ylabel('Residuals ($)')
        axes[1].set_title(f'{title}: Residuals vs Predicted')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(
            self.config.log_dir,
            f"{self.config.model_type}_predictions_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Predictions analysis saved to: {plot_path}")
        plt.close()

    def _plot_feature_importance(self, top_n=20):
        """Plot feature importance."""
        if self.model is None or not hasattr(self.model,
                                             'feature_importances_'):
            return

        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(
            f'Top {top_n} Feature Importances - {self.config.model_type.upper()}'
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_path = os.path.join(
            self.config.log_dir,
            f"{self.config.model_type}_feature_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {plot_path}")
        plt.close()


if __name__ == "__main__":
    print("Starting Gradient Boosting Model Training and Evaluation...")


    models_to_run = [
        (
            "xgboost",
            GradientBoostConfig(
                model_type="xgboost",
                cap_percentile=0.99,
                use_scaled_features=True,
                tune_hyperparameters=False,
                log_dir="logs/gradient_boost/xgboost")),
        (
            "lightgbm",
            GradientBoostConfig(
                model_type="lightgbm",
                cap_percentile=0.99,
                use_scaled_features=True,
                tune_hyperparameters=False,
                log_dir="logs/gradient_boost/lightgbm"))
    ]

    results_summary = {}

    for model_name, config in models_to_run:
        print(f"\n{'='*80}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*80}")

        try:

            gb_model = GradientBoostModel(config)


            gb_model.load_and_preprocess_data()


            gb_model.build_model()


            gb_model.train_model()


            evaluation_results = gb_model.evaluate_model()


            results_summary[model_name] = {
                'test_mae':
                evaluation_results['test_metrics']['mae'],
                'test_rmse':
                evaluation_results['test_metrics']['rmse'],
                'test_r2':
                evaluation_results['test_metrics']['r2'],
                'test_mape':
                evaluation_results['test_metrics']['mape'],
                'mae_ratio':
                evaluation_results['generalization']['mae_ratio'],
                'r2_degradation':
                evaluation_results['generalization']['r2_degradation']
            }

            print(f"\n{model_name.upper()} training and evaluation completed!")
            print(f"Results saved to: {config.log_dir}")

        except Exception as e:
            print(f"Error in {model_name} pipeline: {e}")
            import traceback
            traceback.print_exc()


    if results_summary:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(
            f"{'Model':<12} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10} {'Test MAPE':<12} {'MAE Ratio':<10} {'R² Degrad':<10}"
        )
        print("-" * 88)

        for model_name, results in results_summary.items():
            print(
                f"{model_name.upper():<12} ${results['test_mae']:<11.0f} ${results['test_rmse']:<11.0f} {results['test_r2']:<10.4f} {results['test_mape']:<12.2f} {results['mae_ratio']:<10.3f} {results['r2_degradation']:<10.4f}"
            )


        best_model = min(results_summary.items(),
                         key=lambda x: x[1]['test_mae'])
        print(
            f"\nBest model by MAE: {best_model[0].upper()} (MAE: ${best_model[1]['test_mae']:.0f})"
        )

    print("\nGradient Boosting model comparison completed!")
