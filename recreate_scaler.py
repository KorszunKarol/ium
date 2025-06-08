

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def main():
    print("="*60)
    print("RECREATING FITTED STANDARDSCALER FOR DEPLOYMENT")
    print("="*60)


    data_path = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"
    output_dir = "/home/karolito/IUM/nocarz_api/models_deploy"


    selected_features = [
        'latitude', 'longitude', 'distance_to_center', 'accommodates',
        'bedrooms', 'bathrooms', 'price_log', 'amenities_count',
        'review_scores_rating', 'number_of_reviews', 'host_is_superhost',
        'host_days_active', 'instant_bookable', 'neighbourhood_price_rank',
        'property_type_frequency', 'host_response_rate', 'review_scores_location',
        'minimum_nights', 'calculated_host_listings_count', 'name_positive_sentiment'
    ]


    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_pickle(data_path)
    print(f"Dataset shape: {df.shape}")


    target_col = 'annual_revenue_adj'
    print(f"Target column: {target_col}")


    y_original = df[target_col].copy()
    cap_percentile = 0.97
    cap_value = y_original.quantile(cap_percentile)
    outliers_count = (y_original > cap_value).sum()

    if outliers_count > 0:
        y_original = np.clip(y_original, a_min=None, a_max=cap_value)
        print(f"Applied outlier capping at {cap_percentile*100}th percentile: ${cap_value:.0f}")
        print(f"Capped {outliers_count} outliers")


    X = df.drop(columns=[target_col])


    id_cols = ["listing_id", "id", "host_id"]
    X = X.drop(columns=[col for col in id_cols if col in X.columns], errors="ignore")


    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    numerical_columns = X.select_dtypes(exclude=["object", "category"]).columns

    print(f"Categorical columns: {len(categorical_columns)}")
    print(f"Numerical columns: {len(numerical_columns)}")


    if len(categorical_columns) > 0:
        X_cat_copy = X[categorical_columns].copy()
        for col in categorical_columns:
            X_cat_copy[col] = X_cat_copy[col].astype(str)

        X_categorical = pd.get_dummies(X_cat_copy, drop_first=True, dtype=np.float32)
        X_numerical = X[numerical_columns].astype(np.float32)
        X_processed = pd.concat([X_numerical, X_categorical], axis=1)
    else:
        X_processed = X[numerical_columns].astype(np.float32)

    print(f"Processed features shape: {X_processed.shape}")
    print(f"Available feature columns: {len(X_processed.columns)}")


    missing_features = [f for f in selected_features if f not in X_processed.columns]
    available_features = [f for f in selected_features if f in X_processed.columns]

    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")

    print(f"Using {len(available_features)} out of {len(selected_features)} selected features")


    X_selected = X_processed[available_features]


    print("\nApplying train/validation/test split...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_selected, y_original, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")


    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"Scaler fitted on {X_train.shape[0]} training samples")
    print(f"Feature means (first 5): {scaler.mean_[:5]}")
    print(f"Feature scales (first 5): {scaler.scale_[:5]}")


    scaler_path = os.path.join(output_dir, "neural_net_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved fitted scaler to: {scaler_path}")


    feature_names_path = os.path.join(output_dir, "neural_net_feature_names.pkl")
    with open(feature_names_path, 'wb') as f:
        pickle.dump(available_features, f)
    print(f"Saved feature names to: {feature_names_path}")


    metadata = {
        'selected_features': selected_features,
        'available_features': available_features,
        'missing_features': missing_features,
        'input_dim': len(available_features),
        'cap_percentile': cap_percentile,
        'cap_value': float(cap_value),
        'training_samples': X_train.shape[0],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }

    metadata_path = os.path.join(output_dir, "neural_net_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to: {metadata_path}")


    print("\nTesting scaler transformation...")
    X_val_scaled = scaler.transform(X_val)
    print(f"Validation set transformed shape: {X_val_scaled.shape}")
    print(f"Transformed validation mean (should be ~0): {X_val_scaled.mean():.6f}")
    print(f"Transformed validation std (should be ~1): {X_val_scaled.std():.6f}")

    print("\n" + "="*60)
    print("SCALER RECREATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Files saved to: {output_dir}")
    print("- neural_net_scaler.pkl: Fitted StandardScaler")
    print("- neural_net_feature_names.pkl: Feature names list")
    print("- neural_net_metadata.pkl: Training metadata")

if __name__ == "__main__":
    main()
