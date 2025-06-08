import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

DEFAULT_DATA_PATH = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"
DEFAULT_MODEL_SAVE_DIR = "logs/baseline"
TARGET_COL = 'annual_revenue_adj'
NEIGHBORHOOD_COL = 'neighbourhood_cleansed'


class NeighborhoodMedianBaseline:
    """
  Baseline model that predicts annual revenue based on neighborhood median values.
  Falls back to global median for unseen neighborhoods.
  """

    def __init__(self, model_save_dir: str = DEFAULT_MODEL_SAVE_DIR):
        self.model_save_dir = model_save_dir
        self.model_artifact = None
        self.is_trained = False

        os.makedirs(self.model_save_dir, exist_ok=True)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
      Load and clean data for baseline model training/evaluation.
      Args:
          file_path: Path to the dataset file
      Returns:
          Cleaned DataFrame with necessary columns
      """
        print(f"Loading data from: {file_path}")

        if file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            df = pd.read_csv(file_path)

        print(f"Original dataset shape: {df.shape}")

        required_cols = [TARGET_COL, NEIGHBORHOOD_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_clean = df[required_cols].copy()

        print("\nHandling missing values...")
        print(f"Missing values before cleaning:")
        print(f"  {TARGET_COL}: {df_clean[TARGET_COL].isna().sum()}")
        print(
            f"  {NEIGHBORHOOD_COL}: {df_clean[NEIGHBORHOOD_COL].isna().sum()}")

        df_clean = df_clean.dropna(subset=[TARGET_COL])

        df_clean[NEIGHBORHOOD_COL] = df_clean[NEIGHBORHOOD_COL].fillna(
            "Unknown")

        print(f"Final dataset shape after cleaning: {df_clean.shape}")
        print(f"Unique neighborhoods: {df_clean[NEIGHBORHOOD_COL].nunique()}")

        return df_clean

    def train(self, df_train: pd.DataFrame) -> Dict:
        """
      Train the baseline model by calculating neighborhood medians.
      Args:
          df_train: Training DataFrame
      Returns:
          Dictionary containing model artifacts
      """
        print(f"Training baseline model on {len(df_train)} samples...")

        global_median = df_train[TARGET_COL].median()
        print(f"Global median revenue: ${global_median:,.2f}")

        neighborhood_stats = df_train.groupby(
            NEIGHBORHOOD_COL)[TARGET_COL].agg(
                ['median', 'mean', 'count', 'std']).reset_index()

        neighborhood_stats.columns = [
            NEIGHBORHOOD_COL, 'median_revenue', 'mean_revenue', 'count',
            'std_revenue'
        ]

        neighborhood_medians = neighborhood_stats.set_index(
            NEIGHBORHOOD_COL)['median_revenue'].to_dict()

        print(
            f"Calculated medians for {len(neighborhood_medians)} neighborhoods"
        )
        print(f"Neighborhoods with most listings (top 5):")
        top_neighborhoods = neighborhood_stats.nlargest(5, 'count')
        for _, row in top_neighborhoods.iterrows():
            print(
                f"  {row[NEIGHBORHOOD_COL]}: {row['count']} listings, median=${row['median_revenue']:,.2f}"
            )

        self.model_artifact = {
            'neighborhood_medians': neighborhood_medians,
            'global_median': global_median,
            'neighborhood_stats': neighborhood_stats,
            'training_size': len(df_train)
        }

        self.is_trained = True
        return self.model_artifact

    def save_model(self, filename: str = "neighborhood_median_model.joblib"):
        """
      Save the trained model artifacts.
      Args:
          filename: Name of the file to save the model
      """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_path = os.path.join(self.model_save_dir, filename)
        joblib.dump(self.model_artifact, model_path)
        print(f"Model saved to: {model_path}")
        return model_path

    def load_model(self, filename: str = "neighborhood_median_model.joblib"):
        """
      Load a previously saved model.
      Args:
          filename: Name of the saved model file
      """
        model_path = os.path.join(self.model_save_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_artifact = joblib.load(model_path)
        self.is_trained = True
        print(f"Model loaded from: {model_path}")
        return self.model_artifact

    def predict(self, df_predict: pd.DataFrame) -> pd.Series:
        """
      Make predictions using the trained baseline model.
      Args:
          df_predict: DataFrame with neighborhood information
      Returns:
          Series of predictions
      """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained or loaded before making predictions")

        predictions = []
        neighborhoods_not_found = set()

        for neighborhood in df_predict[NEIGHBORHOOD_COL]:
            if neighborhood in self.model_artifact['neighborhood_medians']:
                pred = self.model_artifact['neighborhood_medians'][
                    neighborhood]
            else:
                pred = self.model_artifact['global_median']
                neighborhoods_not_found.add(neighborhood)
            predictions.append(pred)

        if neighborhoods_not_found:
            print(
                f"Warning: {len(neighborhoods_not_found)} neighborhoods not seen during training. Using global median for these."
            )
            print(
                f"Unseen neighborhoods: {list(neighborhoods_not_found)[:5]}{'...' if len(neighborhoods_not_found) > 5 else ''}"
            )

        return pd.Series(predictions, index=df_predict.index)

    def evaluate(self, y_true: pd.Series,
                 y_pred: pd.Series) -> Dict[str, float]:
        """
      Evaluate predictions using comprehensive regression metrics.
      Args:
          y_true: True target values
          y_pred: Predicted values
      Returns:
          Dictionary of evaluation metrics
      """

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        non_zero_mask = y_true != 0
        mape = np.nan
        if non_zero_mask.sum() > 0:
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
            'residual_std': residual_std
        }

    def print_evaluation_summary(self,
                                 metrics: Dict[str, float],
                                 dataset_name: str = "Test"):
        """
      Print a comprehensive evaluation summary.
      Args:
          metrics: Dictionary of evaluation metrics
          dataset_name: Name of the dataset being evaluated
      """
        print(f"\n{'='*60}")
        print(f"BASELINE MODEL EVALUATION - {dataset_name.upper()} SET")
        print(f"{'='*60}")

        print(f"\nRegression Metrics:")
        print(f"  MAE:           ${metrics['mae']:,.2f}")
        print(f"  RMSE:          ${metrics['rmse']:,.2f}")
        print(f"  R²:            {metrics['r2']:.4f}")
        print(f"  MAPE:          {metrics['mape']:.2f}%")
        print(f"  Median AE:     ${metrics['median_ae']:,.2f}")
        print(f"  Max Error:     ${metrics['max_error']:,.2f}")

        print(f"\nPrediction Accuracy:")
        print(f"  Within 10%:    {metrics['within_10pct']:.1f}%")
        print(f"  Within 25%:    {metrics['within_25pct']:.1f}%")
        print(f"  Within 50%:    {metrics['within_50pct']:.1f}%")

        print(f"\nResidual Analysis:")
        print(f"  Mean:          ${metrics['residual_mean']:,.2f}")
        print(f"  Std Dev:       ${metrics['residual_std']:,.2f}")


def main():
    """Main execution function for baseline model training and evaluation."""
    print("=" * 80)
    print("NEIGHBORHOOD MEDIAN BASELINE MODEL")
    print("=" * 80)

    baseline = NeighborhoodMedianBaseline()

    try:
        df = baseline.load_data(DEFAULT_DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"\nSplitting data for evaluation...")
    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=42,
                                         stratify=None)
    print(f"Training set: {len(df_train)} samples")
    print(f"Test set: {len(df_test)} samples")

    try:
        model_artifact = baseline.train(df_train)
        print(f"\nTraining completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    try:
        model_path = baseline.save_model()
        print(f"Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
        return

    try:
        baseline_loaded = NeighborhoodMedianBaseline()
        baseline_loaded.load_model()
        print(f"Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        print(f"\nMaking predictions on test set...")
        test_predictions = baseline_loaded.predict(df_test)
        test_metrics = baseline_loaded.evaluate(df_test[TARGET_COL],
                                                test_predictions)
        baseline_loaded.print_evaluation_summary(test_metrics, "Test")

        print(f"\nMaking predictions on training set (sanity check)...")
        train_predictions = baseline_loaded.predict(df_train)
        train_metrics = baseline_loaded.evaluate(df_train[TARGET_COL],
                                                 train_predictions)
        baseline_loaded.print_evaluation_summary(train_metrics, "Training")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    print(f"\n{'='*80}")
    print("BASELINE MODEL PROCESSING COMPLETED!")
    print(f"{'='*80}")
    print(f"Model artifacts saved in: {baseline.model_save_dir}")
    print(
        f"This baseline can now be used for comparison with more complex models."
    )


if __name__ == "__main__":
    main()
