#!/usr/bin/env python3
"""
Modeling Dataset Creation Script
Combines feature-engineered listings data with seasonally-adjusted target variables
to create the final modeling dataset.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


def load_data(processed_data_base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load feature-engineered listings and reliable target variables.

    Args:
        processed_data_base_dir: The base directory for processed data 
                                 (e.g., 'data/processed/etap2/').
    
    Returns:
        A tuple containing two DataFrames:
        - listings_feature_engineered_df
        - reliable_targets_df
    """
    feature_engineered_path = os.path.join(
        processed_data_base_dir, 
        "feature_engineered", 
        "listings_feature_engineered.pkl"
    )
    targets_path = os.path.join(
        processed_data_base_dir,
        "target_variables",
        "reliable_targets.pkl"
    )

    print(f"Loading feature-engineered listings from: {feature_engineered_path}")
    listings_feature_engineered_df = pd.read_pickle(feature_engineered_path)
    
    print(f"Loading reliable targets from: {targets_path}")
    reliable_targets_df = pd.read_pickle(targets_path)
    
    return listings_feature_engineered_df, reliable_targets_df


def create_modeling_dataset(
    listings_feature_engineered_df: pd.DataFrame, 
    reliable_targets_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create the final modeling dataset by joining feature-engineered listings
    with reliable target variables.

    Args:
        listings_feature_engineered_df: DataFrame with feature-engineered listings.
        reliable_targets_df: DataFrame with reliable target variables.

    Returns:
        The final modeling DataFrame.
    """
    print("Joining feature-engineered listings with target variables...")
    
    # Perform an inner join to keep only listings present in both DataFrames
    # 'id' from feature_engineered listings and 'listing_id' from targets
    modeling_dataset = pd.merge(
        listings_feature_engineered_df,
        reliable_targets_df,
        left_on="id",  # Assuming 'id' is the primary key in feature_engineered_df
        right_on="listing_id", # Assuming 'listing_id' is the key in targets_df
        how="inner"
    )

    # If 'id' and 'listing_id' are identical and one is redundant, drop one.
    # For example, if 'listing_id' becomes redundant after merge:
    if 'listing_id' in modeling_dataset.columns and 'id' in modeling_dataset.columns and 'listing_id' != 'id':
         if pd.Series(modeling_dataset['id'] == modeling_dataset['listing_id']).all():
            modeling_dataset = modeling_dataset.drop(columns=['listing_id'])
    elif 'listing_id' in modeling_dataset.columns and 'id' not in modeling_dataset.columns:
        modeling_dataset = modeling_dataset.rename(columns={'listing_id': 'id'})


    print(f"Initial feature-engineered listings: {len(listings_feature_engineered_df)}")
    print(f"Reliable targets available for: {len(reliable_targets_df)}")
    print(f"Merged modeling dataset size: {len(modeling_dataset)}")

    if len(modeling_dataset) == 0:
        raise ValueError(
            "Merging resulted in an empty DataFrame. "
            "Check keys and data: 'id' in feature engineered data, "
            "'listing_id' in target data."
        )
        
    # Feature preparation is now mostly done in the feature engineering pipeline.
    # We might select a subset of columns or drop specific ones if needed.
    # For now, keeping all merged columns.

    return modeling_dataset


def analyze_dataset_quality(modeling_dataset: pd.DataFrame) -> dict:
    """Analyze the quality and completeness of the modeling dataset."""
    total_rows = len(modeling_dataset)
    print(f"Analyzing dataset with {total_rows} rows.")

    # Key feature completeness - these should ideally be high after feature engineering
    # This list can be expanded based on the output of feature_engineering_pipeline.py
    key_features_for_check = [
        "price", # This might be 'price_cleaned' or just 'price' after engineering
        "room_type_Entire home/apt", # Example of an encoded feature
        "accommodates",
        "bedrooms",
        "beds",
        "latitude",
        "longitude",
        "review_scores_rating", # This might be an average or imputed version
        "number_of_reviews",
        "amenities_count",
        "distance_to_center",
        "host_is_superhost" # Example boolean/binary
    ]
    
    # Add a check for price column name
    price_col_to_check = 'price'
    if 'price_cleaned' in modeling_dataset.columns:
        price_col_to_check = 'price_cleaned'
    elif 'price' not in modeling_dataset.columns: # if neither exists, it's an issue
        print(f"Warning: Neither 'price' nor 'price_cleaned' found for quality check.")
        # key_features_for_check.remove("price") # remove if not present
    
    if price_col_to_check not in key_features_for_check and price_col_to_check in modeling_dataset.columns:
         key_features_for_check = [f if f != 'price' else price_col_to_check for f in key_features_for_check]


    feature_completeness = {}
    for feature in key_features_for_check:
        if feature in modeling_dataset.columns:
            missing_count = modeling_dataset[feature].isna().sum()
            completeness_pct = (1 - missing_count / total_rows) * 100
            feature_completeness[feature] = {
                "missing_count": int(missing_count),
                "completeness_percentage": float(completeness_pct),
            }
        else:
            feature_completeness[feature] = {
                "missing_count": total_rows, # Mark as fully missing if column doesn't exist
                "completeness_percentage": 0.0,
            }


    # Target variable statistics - updated to 'annual_revenue_adj'
    target_col = "annual_revenue_adj"
    if target_col not in modeling_dataset.columns:
        raise KeyError(f"Target column '{target_col}' not found in the modeling dataset.")
        
    target_stats = {
        target_col: {
            "count": int(modeling_dataset[target_col].count()),
            "mean": float(modeling_dataset[target_col].mean()),
            "median": float(modeling_dataset[target_col].median()),
            "std": float(modeling_dataset[target_col].std()),
            "min": float(modeling_dataset[target_col].min()),
            "max": float(modeling_dataset[target_col].max()),
            "q25": float(modeling_dataset[target_col].quantile(0.25)),
            "q75": float(modeling_dataset[target_col].quantile(0.75)),
        }
    }

    quality_report = {
        "dataset_size": total_rows,
        "feature_count": modeling_dataset.shape[1],
        "feature_completeness": feature_completeness,
        "target_statistics": target_stats,
        "confidence_score_distribution": modeling_dataset["confidence_score"].describe().to_dict()
        if "confidence_score" in modeling_dataset.columns
        else "Not available",
        "geographic_coverage": {
            "unique_neighbourhoods": (
                int(modeling_dataset["neighbourhood_cleansed"].nunique())
                if "neighbourhood_cleansed" in modeling_dataset.columns # This might be encoded
                else "Encoded or Not available"
            ),
            "coordinate_completeness_count": (
                int(
                    (
                        modeling_dataset["latitude"].notna()
                        & modeling_dataset["longitude"].notna()
                    ).sum()
                )
                if "latitude" in modeling_dataset.columns and "longitude" in modeling_dataset.columns
                else 0
            ),
        },
    }
    return quality_report


def save_modeling_dataset(modeling_dataset: pd.DataFrame, quality_report: dict, output_dir: str) -> dict :
    """Save the modeling dataset and quality report."""
    os.makedirs(output_dir, exist_ok=True)

    dataset_path_csv = os.path.join(output_dir, "modeling_dataset.csv")
    dataset_path_pkl = os.path.join(output_dir, "modeling_dataset.pkl")
    quality_path = os.path.join(output_dir, "dataset_quality_report.json")

    print(f"Saving modeling dataset (CSV) to: {dataset_path_csv}")
    modeling_dataset.to_csv(dataset_path_csv, index=False)
    print(f"Saving modeling dataset (PKL) to: {dataset_path_pkl}")
    modeling_dataset.to_pickle(dataset_path_pkl)
    
    print(f"Saving dataset quality report to: {quality_path}")
    with open(quality_path, "w") as f:
        json.dump(quality_report, f, indent=2, default=str) # Added default=str for numpy types

    return {
        "dataset_csv": dataset_path_csv,
        "dataset_pkl": dataset_path_pkl,
        "quality_report": quality_path,
    }


def main():
    """Main function to create the modeling dataset."""
    processed_data_base_dir = "data/processed/etap2/" # Base directory
    output_dir = os.path.join(processed_data_base_dir, "modeling/")

    print("Starting modeling dataset creation process...")
    
    listings_feature_engineered_df, reliable_targets_df = load_data(processed_data_base_dir)

    print("\nCreating modeling dataset...")
    modeling_dataset = create_modeling_dataset(
        listings_feature_engineered_df, reliable_targets_df
    )

    print("\nAnalyzing dataset quality...")
    quality_report = analyze_dataset_quality(modeling_dataset)

    print("\nSaving modeling dataset and quality report...")
    file_paths = save_modeling_dataset(modeling_dataset, quality_report, output_dir)

    print("\n--- Modeling Dataset Creation Complete! ---")
    print(f"Dataset size: {quality_report['dataset_size']:,} listings")
    print(f"Features: {quality_report['feature_count']} columns")
    
    target_col_name = "annual_revenue_adj" # Updated target name
    if quality_report['target_statistics'].get(target_col_name):
        print(
            f"Target ({target_col_name}) range: "
            f"${quality_report['target_statistics'][target_col_name]['min']:,.0f} - "
            f"${quality_report['target_statistics'][target_col_name]['max']:,.0f}"
        )
        print(
            f"Target ({target_col_name}) median: "
            f"${quality_report['target_statistics'][target_col_name]['median']:,.0f}"
        )
    else:
        print(f"Target statistics for {target_col_name} not available in report.")

    print("\nFiles saved:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")

    print(f"\nKey feature completeness snapshot (selected features):")
    if quality_report.get("feature_completeness"):
        for feature, stats in quality_report["feature_completeness"].items():
            if feature in key_features_for_check: # Only print for those we listed
                 print(f"  {feature}: {stats['completeness_percentage']:.1f}% missing: {stats['missing_count']}")
    else:
        print("  Feature completeness data not available.")
    print("--------------------------------------------")

if __name__ == "__main__":
    main()

# Example key_features_for_check list, to be kept in sync with analyze_dataset_quality
key_features_for_check = [
    "price", "room_type_Entire home/apt", "accommodates", "bedrooms", "beds",
    "latitude", "longitude", "review_scores_rating", "number_of_reviews",
    "amenities_count", "distance_to_center", "host_is_superhost"
]
