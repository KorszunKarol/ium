"""
Nocarz Etap 2: Quick Baseline Model
==================================

This script implements a quick baseline model for the Nocarz Airbnb revenue prediction project.
The goal is to establish an initial performance benchmark using minimal feature engineering.

Based on the "Etap 2 Data Analysis Plan" strategy document.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Load the processed Etap 2 datasets."""
    print("Loading Etap 2 datasets...")

    listings_df = pd.read_pickle("data/processed/etap2/listings_e2_df.pkl")
    calendar_df = pd.read_pickle("data/processed/etap2/calendar_e2_df.pkl")

    print(f"Loaded listings: {listings_df.shape}")
    print(f"Loaded calendar: {calendar_df.shape}")

    return listings_df, calendar_df


def select_core_features(listings_df):
    """Select a minimal set of core features for the baseline model."""
    print("\nSelecting core features for baseline model...")

    core_features = [
        "id",
        "accommodates",
        "room_type",
        "neighbourhood_cleansed",
        "review_scores_rating",
        "price_cleaned",
        "bedrooms",
        "number_of_reviews",
    ]

    available_features = [col for col in core_features if col in listings_df.columns]
    missing_features = [col for col in core_features if col not in listings_df.columns]

    if missing_features:
        print(f"Missing features: {missing_features}")

    baseline_df = listings_df[available_features].copy()
    print(
        f"Selected {len(available_features) - 1} features: {[f for f in available_features if f != 'id']}"
    )

    return baseline_df


def create_simple_target(listings_df, calendar_df):
    """Create a simple target variable using basic calendar aggregation."""
    print("\nCreating simple target variable...")

    price_filtered = calendar_df[
        (calendar_df["price_cleaned"] > 0) & (calendar_df["price_cleaned"] < 2000)
    ].copy()

    print(f"Filtered calendar from {len(calendar_df)} to {len(price_filtered)} records")
    print(f"Removed {len(calendar_df) - len(price_filtered)} anomalous price records")

    target_df = (
        price_filtered.groupby("listing_id")
        .agg({"price_cleaned": ["mean", "count"]})
        .round(2)
    )

    target_df.columns = ["mean_price_obs", "obs_count"]
    target_df = target_df.reset_index()

    target_df = target_df[target_df["obs_count"] >= 5].copy()

    print(f"Created target for {len(target_df)} listings (min 5 observations each)")
    print(
        f"Target stats: Mean=${target_df['mean_price_obs'].mean():.2f}, "
        f"Std=${target_df['mean_price_obs'].std():.2f}"
    )

    return target_df


def basic_preprocessing(df, target_col="mean_price_obs"):
    """Apply basic preprocessing for the baseline model."""
    print("\nApplying basic preprocessing...")

    feature_cols = [
        col for col in df.columns if col not in ["id", "listing_id", target_col]
    ]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    if "neighbourhood_cleansed" in categorical_features:
        neighbourhood_counts = X["neighbourhood_cleansed"].value_counts()
        X["neighbourhood_frequency"] = X["neighbourhood_cleansed"].map(
            neighbourhood_counts
        )

        top_neighbourhoods = neighbourhood_counts.head(10).index
        X["neighbourhood_top10"] = X["neighbourhood_cleansed"].apply(
            lambda x: x if x in top_neighbourhoods else "Other"
        )

        categorical_features.remove("neighbourhood_cleansed")
        categorical_features.append("neighbourhood_top10")
        numerical_features.append("neighbourhood_frequency")
        X = X.drop("neighbourhood_cleansed", axis=1)

        print("Transformed neighbourhood_cleansed to frequency + top10 encoding")

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Created preprocessing pipeline")

    return X, y, preprocessor


def train_baseline_models(X, y, preprocessor):
    """Train multiple baseline models and compare performance."""
    print("\nTraining baseline models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Preprocessed features shape: {X_train_processed.shape}")

    models = {
        "Dummy (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train_processed, y_train)

        y_pred_train = model.predict(X_train_processed)
        y_pred_test = model.predict(X_test_processed)

        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        results[name] = {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "model": model,
        }

        print(
            f"  Train MAE: ${train_mae:.2f}, RMSE: ${train_rmse:.2f}, R²: {train_r2:.3f}"
        )
        print(
            f"  Test MAE:  ${test_mae:.2f}, RMSE: ${test_rmse:.2f}, R²: {test_r2:.3f}"
        )

    return results, X_test, y_test, X_test_processed, preprocessor


def analyze_feature_importance(results, preprocessor, feature_names):
    """Analyze feature importance from the best performing model."""
    print("\nFeature Importance Analysis...")

    best_model_name = max(results.keys(), key=lambda k: results[k]["test_r2"])
    best_model = results[best_model_name]["model"]

    print(f"Best model: {best_model_name}")

    if hasattr(best_model, "feature_importances_"):
        try:
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names_out = preprocessor.get_feature_names_out()
            else:
                feature_names_out = [
                    f"feature_{i}" for i in range(len(best_model.feature_importances_))
                ]

            importance_df = pd.DataFrame(
                {
                    "feature": feature_names_out,
                    "importance": best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            print("\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

        except Exception as e:
            print(f"Could not extract feature names: {e}")

    return best_model_name


def save_baseline_results(results):
    """Save baseline results to a summary file."""
    print("\nSaving baseline results...")

    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append(
            {
                "model": model_name,
                "test_mae": metrics["test_mae"],
                "test_rmse": metrics["test_rmse"],
                "test_r2": metrics["test_r2"],
                "train_mae": metrics["train_mae"],
                "train_rmse": metrics["train_rmse"],
                "train_r2": metrics["train_r2"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("test_r2", ascending=False)

    summary_df.to_csv("docs/etap2_baseline_results.csv", index=False)
    print("Saved results to docs/etap2_baseline_results.csv")

    return summary_df


def main():
    """Main execution function."""
    print("Nocarz Etap 2: Quick Baseline Model")
    print("=" * 50)

    try:
        listings_df, calendar_df = load_data()

        baseline_df = select_core_features(listings_df)

        target_df = create_simple_target(listings_df, calendar_df)

        modeling_df = baseline_df.merge(
            target_df, left_on="id", right_on="listing_id", how="inner"
        )
        print(f"\nFinal modeling dataset: {modeling_df.shape}")

        X, y, preprocessor = basic_preprocessing(modeling_df)

        results, X_test, y_test, X_test_processed, preprocessor = train_baseline_models(
            X, y, preprocessor
        )

        best_model = analyze_feature_importance(
            results, preprocessor, X.columns.tolist()
        )

        summary_df = save_baseline_results(results)

        print("\n" + "=" * 50)
        print("BASELINE MODEL SUMMARY")
        print("=" * 50)
        print(summary_df.to_string(index=False, float_format="%.3f"))

        print(f"\nBest performing model: {best_model}")
        print(f"Target variable: Mean observed price per listing (£)")
        print(f"Sample size: {len(modeling_df)} listings")
        print(f"Features used: {len(X.columns)}")

        print("\nBaseline model creation completed!")
        print(
            "This establishes the performance floor to beat with advanced feature engineering."
        )

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
