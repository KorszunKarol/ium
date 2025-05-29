#!/usr/bin/env python3
"""
Baseline Models Comparison for Final Modeling Dataset
====================================================

This script runs comprehensive baseline model comparison on the final modeling dataset
created by create_modeling_dataset.py, which includes feature-engineered data and
reliable revenue targets.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Model imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def load_modeling_dataset(dataset_path):
    """
    Load the final modeling dataset created by create_modeling_dataset.py

    Args:
        dataset_path (str): Path to the modeling dataset pickle file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading modeling dataset from: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Modeling dataset not found at {dataset_path}")

    df = pd.read_pickle(dataset_path)
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def prepare_features_and_target(df, target_col='annual_revenue_adj'):
    """
    Prepare features and target for modeling with comprehensive data leakage prevention.

    This function implements extensive feature exclusion to prevent data leakage
    that was causing unrealistic performance in tree-based models.

    Args:
        df (pd.DataFrame): The modeling dataset
        target_col (str): Name of the target column

    Returns:
        tuple: (X, y, groups) where X is features, y is target, groups is host_id for GroupKFold
    """
    print("\n=== Preparing Features and Target ===")

    # Comprehensive exclusion list to prevent data leakage
    exclude_cols = [
        # Target variable
        target_col,

        # Identifiers that could leak information
        'listing_id', 'id', 'scrape_id', 'host_id',

        # Revenue calculation components (direct leakage)
        'total_bookings', 'observation_days', 'annualization_factor',

        # Review-related features (highly correlated with revenue)
        'number_of_reviews', 'reviews_per_month', 'first_review', 'last_review',

        # Availability features (outcome of bookings, not predictor)
        'availability_30', 'availability_60', 'availability_90', 'availability_365',
        'minimum_nights', 'maximum_nights',

        # Calendar-based availability features
        'calendar_last_scraped', 'calendar_updated',

        # High-cardinality text features that trees can memorize
        'name', 'summary', 'space', 'description', 'experiences_offered',
        'neighborhood_overview', 'notes', 'transit', 'access', 'interaction',
        'house_rules', 'host_name', 'host_about',

        # URL and image features
        'listing_url', 'scrape_id', 'thumbnail_url', 'medium_url', 'picture_url',
        'xl_picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url',

        # Exact location coordinates (too specific)
        'latitude', 'longitude',

        # Features that are post-booking outcomes
        'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',

        # Other potential leakers
        'license', 'jurisdiction_names', 'requires_license', 'is_business_travel_ready',
    ]

    # Extract groups for grouped cross-validation (before removing host_id)
    if 'host_id' not in df.columns:
        raise ValueError("host_id column not found in dataset - needed for grouped CV")
    groups = df['host_id'].values

    # Remove excluded columns
    available_exclude = [col for col in exclude_cols if col in df.columns]
    print(f"Excluding {len(available_exclude)} columns: {available_exclude[:10]}{'...' if len(available_exclude) > 10 else ''}")

    # Prepare features and target
    X = df.drop(columns=available_exclude)
    y = df[target_col]

    # State validation assertions
    assert target_col not in X.columns, f"Target column {target_col} found in features!"
    assert 'host_id' not in X.columns, "host_id found in features - should be excluded!"
    assert len(X) == len(y), "Features and target have different lengths!"
    assert len(groups) == len(X), "Groups and features have different lengths!"

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target statistics: mean=${y.mean():.0f}, std=${y.std():.0f}, min=${y.min():.0f}, max=${y.max():.0f}")
    print(f"Groups (hosts): {len(np.unique(groups))} unique hosts")

    return X, y, groups


def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for the features.

    Returns:
        sklearn.compose.ColumnTransformer: Preprocessing pipeline
    """
    # We'll handle this based on the actual data types in the prepared features
    # For now, create a simple pipeline that can handle mixed data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return numeric_transformer, categorical_transformer


def get_baseline_models():
    """
    Define the baseline models to compare (reduced set to prevent overfitting).

    Returns:
        dict: Dictionary of model names and instances
    """
    models = {
        # Baseline/Dummy models
        'Dummy (Mean)': DummyRegressor(strategy='mean'),
        'Dummy (Median)': DummyRegressor(strategy='median'),

        # Linear models (robust to leakage)
        'Ridge': Ridge(alpha=1.0, random_state=42),

        # Tree-based models (reduced complexity)
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
    }

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, groups_train, cv_folds=3):
    """
    Evaluate a single model with both train/test split and grouped cross-validation.

    Args:
        model: sklearn model instance
        X_train, X_test, y_train, y_test: Train/test splits
        groups_train: Group labels for grouped cross-validation
        cv_folds: Number of CV folds

    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    results = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_r2': r2_score(y_test, y_test_pred),
    }

    # Grouped cross-validation to check for overfitting
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=GroupKFold(n_splits=cv_folds),
            groups=groups_train,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        results['cv_mae_mean'] = -cv_scores.mean()
        results['cv_mae_std'] = cv_scores.std()

        # Calculate overfitting indicators
        results['overfitting_train_test'] = results['train_mae'] / results['test_mae']
        results['overfitting_cv_test'] = results['cv_mae_mean'] / results['test_mae']

    except Exception as e:
        print(f"Warning: Cross-validation failed: {e}")
        results['cv_mae_mean'] = np.nan
        results['cv_mae_std'] = np.nan
        results['overfitting_train_test'] = np.nan
        results['overfitting_cv_test'] = np.nan

    return results


def run_baseline_comparison(X, y, groups, test_size=0.2, random_state=42):
    """
    Run the baseline model comparison with grouped train/test split.

    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels for splitting
        test_size: Proportion of data for testing
        random_state: Random seed

    Returns:
        dict: Results for all models
    """
    print("\n=== Running Baseline Model Comparison ===")

    # Use GroupShuffleSplit to ensure no host appears in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]

    print(f"Train set: {len(X_train)} samples, {len(np.unique(groups_train))} unique hosts")
    print(f"Test set: {len(X_test)} samples, {len(np.unique(groups[test_idx]))} unique hosts")

    # Verify no host overlap between train and test
    train_hosts = set(groups_train)
    test_hosts = set(groups[test_idx])
    overlap = train_hosts.intersection(test_hosts)
    assert len(overlap) == 0, f"Host overlap between train/test: {len(overlap)} hosts"
    print("✓ Verified no host overlap between train and test sets")

    # Create preprocessing pipeline based on actual data
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Simple preprocessing for mixed data
    if len(categorical_features) > 0:
        # For now, drop categorical features to avoid complexity
        print("Dropping categorical features for baseline comparison")
        X_train = X_train[numeric_features]
        X_test = X_test[numeric_features]

    # Handle any remaining missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames for consistency
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print(f"Final preprocessed shape: {X_train.shape}")

    # Get models
    models = get_baseline_models()
    results = {}

    print(f"\nEvaluating {len(models)} models...")

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        try:
            model_results = evaluate_model(
                model, X_train, X_test, y_train, y_test, groups_train
            )
            results[name] = model_results

            # Print key metrics
            print(f"  Test MAE: £{model_results['test_mae']:,.0f}")
            print(f"  Test R²: {model_results['test_r2']:.3f}")
            print(f"  CV MAE: £{model_results['cv_mae_mean']:,.0f} ± £{model_results['cv_mae_std']:,.0f}")

            # Check for potential overfitting
            if model_results['test_mae'] < 2000:  # Suspiciously low error
                print(f"  ⚠️  WARNING: Suspiciously low test MAE - possible data leakage!")

        except Exception as e:
            print(f"  ERROR: Failed to evaluate {name}: {e}")
            results[name] = {'error': str(e)}

    return results


def save_results(results, output_dir="results"):
    """
    Save the comparison results to CSV and JSON files.

    Args:
        results (dict): Model evaluation results
        output_dir (str): Output directory
    """
    print(f"\n=== Saving Results to {output_dir} ===")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert results to DataFrame
    summary_data = []
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            row = {'Model': model_name}
            row.update(metrics)
            summary_data.append(row)

    if summary_data:
        df_results = pd.DataFrame(summary_data)

        # Sort by test MAE
        df_results = df_results.sort_values('test_mae')

        # Save to CSV
        summary_path = os.path.join(output_dir, "baseline_models_comparison.csv")
        df_results.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")

        # Save detailed results to JSON
        json_path = os.path.join(output_dir, "baseline_models_detailed.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: {json_path}")

        # Print summary table
        print("\n=== BASELINE MODEL COMPARISON RESULTS ===")
        pd.set_option('display.precision', 3)
        pd.set_option('display.width', None)
        print(df_results[['Model', 'test_mae', 'test_r2', 'cv_mae_mean', 'overfitting_train_test']].to_string(index=False))

        # Analysis
        print(f"\n=== ANALYSIS ===")
        best_model = df_results.iloc[0]['Model']
        best_mae = df_results.iloc[0]['test_mae']
        print(f"Best model: {best_model} (Test MAE: £{best_mae:,.0f})")

        # Check for data leakage indicators
        suspicious_models = df_results[df_results['test_mae'] < 2000]
        if len(suspicious_models) > 0:
            print(f"\n⚠️  WARNING: {len(suspicious_models)} models show suspiciously low errors:")
            for _, row in suspicious_models.iterrows():
                print(f"  - {row['Model']}: £{row['test_mae']:,.0f}")
            print("This may indicate remaining data leakage!")
        else:
            print("✓ No suspicious low errors detected - data leakage mitigation appears successful")

    else:
        print("No valid results to save.")


def main():
    """Main execution function."""
    print("="*60)
    print("BASELINE MODELS COMPARISON")
    print("="*60)

    # Configuration
    dataset_path = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset.pkl"
    output_dir = "/home/karolito/IUM/results/baseline_comparison"

    try:
        # Load data
        print("Step 1: Loading modeling dataset...")
        df = load_modeling_dataset(dataset_path)

        # Prepare features and target
        print("Step 2: Preparing features and target...")
        X, y, groups = prepare_features_and_target(df)

        # Run comparison
        print("Step 3: Running baseline model comparison...")
        results = run_baseline_comparison(X, y, groups)

        # Save results
        print("Step 4: Saving results...")
        save_results(results, output_dir)

        print("\n✓ Baseline comparison completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during baseline comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
