#!/usr/bin/env python3
"""
Modeling Dataset Creation Script
Combines listings data with calculated revenue metrics to create final modeling dataset.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


def load_data(processed_data_dir):
    """Load all required datasets."""
    listings_df = pd.read_pickle(
        os.path.join(processed_data_dir, "feature_engineered", "listings_feature_engineered.pkl")
    )
    target_variables_df = pd.read_pickle(
        os.path.join(processed_data_dir, "target_variables", "reliable_targets.pkl")
    )
    return listings_df, target_variables_df


def prepare_modeling_features(listings_df):
    """
    Prepare and clean features for modeling.

    Since listings_feature_engineered.pkl is already processed by FeatureEngineeringPipeline,
    minimal additional processing is needed.
    """
    modeling_features = listings_df.copy()

    # Ensure ID column is properly named for merging
    if 'id' in modeling_features.columns and 'listing_id' not in modeling_features.columns:
        modeling_features['listing_id'] = modeling_features['id']

    return modeling_features


def create_modeling_dataset(listings_df, target_variables_df):
    """Create the final modeling dataset by joining listings with revenue metrics."""
    # Prepare features
    modeling_features = prepare_modeling_features(listings_df)

    # Determine the correct ID column for merging
    if 'listing_id' in modeling_features.columns:
        left_on = 'listing_id'
    else:
        left_on = 'id'

    # Join with target variables - ONLY include target and metadata, NOT target components
    modeling_dataset = modeling_features.merge(
        target_variables_df[[
            'listing_id',
            'annual_revenue_adj',
            'confidence_score',
            'confidence_level'
        ]],
        left_on=left_on,
        right_on='listing_id',
        how='inner'
    )

    # Remove duplicate ID columns if created
    if 'listing_id_x' in modeling_dataset.columns:
        modeling_dataset = modeling_dataset.drop(['listing_id_y'], axis=1)
        modeling_dataset = modeling_dataset.rename(columns={'listing_id_x': 'listing_id'})

    # CRITICAL: Remove data leakage sources
    leaky_features = [
        # Target components - these can perfectly reconstruct the target!
        'occupancy_rate_adj', 'adr_adj',

        # Calendar-derived features that might use same data as target calculation
        'price_cleaned_mean', 'price_cleaned_median', 'price_cleaned_std',
        'available_mean', 'available_sum', 'weekend_availability_rate',

        # Availability features derived from calendar data
        'availability_rate', 'availability_30', 'availability_60',
        'availability_90', 'availability_365',

        # Target-related metadata that shouldn't be used for prediction
        'observation_days', 'total_bookings', 'total_observations',
        'annualization_factor',

        # Review temporal features that might leak future information
        'reviews_per_month', 'number_of_reviews_ltm', 'number_of_reviews_l30d',

        # Scaled versions of the above
        'occupancy_rate_adj_scaled', 'adr_adj_scaled',
        'price_cleaned_mean_scaled', 'price_cleaned_median_scaled', 'price_cleaned_std_scaled',
        'available_mean_scaled', 'available_sum_scaled', 'weekend_availability_rate_scaled',
        'availability_rate_scaled', 'availability_30_scaled', 'availability_60_scaled',
        'availability_90_scaled', 'availability_365_scaled', 'reviews_per_month_scaled',
        'number_of_reviews_ltm_scaled', 'number_of_reviews_l30d_scaled'
    ]

    # Remove leaky features if they exist
    features_to_remove = [col for col in leaky_features if col in modeling_dataset.columns]
    if features_to_remove:
        print(f"🚨 REMOVING {len(features_to_remove)} potentially leaky features:")
        for feature in features_to_remove:
            print(f"   - {feature}")
        modeling_dataset = modeling_dataset.drop(columns=features_to_remove)

    print(f"✅ Clean dataset created with {modeling_dataset.shape[1]} features (down from {modeling_features.shape[1]})")

    return modeling_dataset


def analyze_dataset_quality(modeling_dataset):
    """Analyze the quality and completeness of the modeling dataset."""
    total_rows = len(modeling_dataset)

    # Key feature completeness
    key_features = [
        "neighbourhood_cleansed",
        "property_type",
        "room_type",
        "accommodates",
        "bedrooms",
        "beds",
        "latitude",
        "longitude",
        "review_scores_rating",
        "number_of_reviews",
    ]

    feature_completeness = {}
    for feature in key_features:
        if feature in modeling_dataset.columns:
            missing_count = modeling_dataset[feature].isna().sum()
            completeness_pct = (1 - missing_count / total_rows) * 100
            feature_completeness[feature] = {
                "missing_count": int(missing_count),
                "completeness_percentage": float(completeness_pct),
            }

    # Target variable statistics
    target_stats = {
        'annual_revenue_adj': {
            'count': int(modeling_dataset['annual_revenue_adj'].count()),
            'mean': float(modeling_dataset['annual_revenue_adj'].mean()),
            'median': float(modeling_dataset['annual_revenue_adj'].median()),
            'std': float(modeling_dataset['annual_revenue_adj'].std()),
            'min': float(modeling_dataset['annual_revenue_adj'].min()),
            'max': float(modeling_dataset['annual_revenue_adj'].max()),
            'q25': float(modeling_dataset['annual_revenue_adj'].quantile(0.25)),
            'q75': float(modeling_dataset['annual_revenue_adj'].quantile(0.75))
        }
    }

    quality_report = {
        'dataset_size': total_rows,
        'feature_completeness': feature_completeness,
        'target_statistics': target_stats,
        'confidence_distribution': modeling_dataset['confidence_level'].value_counts().to_dict(),
        'geographic_coverage': {
            'unique_neighbourhoods': (
                int(modeling_dataset['neighbourhood_cleansed'].nunique())
                if 'neighbourhood_cleansed' in modeling_dataset.columns
                else 0
            ),
            'coordinate_completeness': (
                int(
                    (modeling_dataset['latitude'].notna() & modeling_dataset['longitude'].notna()).sum()
                )
                if 'latitude' in modeling_dataset.columns and 'longitude' in modeling_dataset.columns
                else 0
            )
        }
    }

    return quality_report


def save_modeling_dataset(modeling_dataset, quality_report, output_dir):
    """Save the modeling dataset and quality report."""
    os.makedirs(output_dir, exist_ok=True)

    # Save dataset in multiple formats
    dataset_path_csv = os.path.join(output_dir, "modeling_dataset.csv")
    dataset_path_pkl = os.path.join(output_dir, "modeling_dataset.pkl")
    quality_path = os.path.join(output_dir, "dataset_quality_report.json")

    modeling_dataset.to_csv(dataset_path_csv, index=False)
    modeling_dataset.to_pickle(dataset_path_pkl)

    with open(quality_path, "w") as f:
        json.dump(quality_report, f, indent=2)

    return {
        "dataset_csv": dataset_path_csv,
        "dataset_pkl": dataset_path_pkl,
        "quality_report": quality_path,
    }


def main():
    """Main function to create the modeling dataset."""
    processed_data_dir = 'data/processed/etap2/'
    output_dir = 'data/processed/etap2/modeling/'

    print('Loading datasets...')
    listings_df, target_variables_df = load_data(processed_data_dir)

    print('Creating modeling dataset...')
    modeling_dataset = create_modeling_dataset(listings_df, target_variables_df)

    print('Analyzing dataset quality...')
    quality_report = analyze_dataset_quality(modeling_dataset)

    print('Saving modeling dataset...')
    file_paths = save_modeling_dataset(modeling_dataset, quality_report, output_dir)

    print('\nModeling dataset creation complete!')
    print(f"Dataset size: {quality_report['dataset_size']:,} listings")
    print(f"Features: {modeling_dataset.shape[1]} columns")
    print(f"Annual Revenue range: £{quality_report['target_statistics']['annual_revenue_adj']['min']:,.0f} - £{quality_report['target_statistics']['annual_revenue_adj']['max']:,.0f}")
    print(f"Annual Revenue median: £{quality_report['target_statistics']['annual_revenue_adj']['median']:,.0f}")

    print('\nConfidence Level Distribution:')
    for level, count in quality_report['confidence_distribution'].items():
        print(f"  {level}: {count:,} listings ({count/quality_report['dataset_size']*100:.1f}%)")

    print('\nFiles saved:')
    for name, path in file_paths.items():
        print(f"  {name}: {path}")

    print('\nKey feature completeness:')
    for feature, stats in quality_report['feature_completeness'].items():
        print(f"  {feature}: {stats['completeness_percentage']:.1f}%")


if __name__ == '__main__':
    main()
