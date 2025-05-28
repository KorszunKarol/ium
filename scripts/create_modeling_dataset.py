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
    listings_df = pd.read_pickle(os.path.join(processed_data_dir, 'listings_e2_df.pkl'))
    reliable_listings = pd.read_pickle(os.path.join(processed_data_dir, 'analysis', 'reliable_listings.pkl'))
    return listings_df, reliable_listings


def prepare_modeling_features(listings_df):
    """Prepare and clean features for modeling."""
    modeling_features = listings_df.copy()

    # Clean numeric columns
    numeric_columns = ['accommodates', 'bedrooms', 'beds', 'minimum_nights',
                      'maximum_nights', 'availability_365', 'number_of_reviews',
                      'review_scores_rating', 'review_scores_accuracy',
                      'review_scores_cleanliness', 'review_scores_checkin',
                      'review_scores_communication', 'review_scores_location',
                      'review_scores_value', 'calculated_host_listings_count',
                      'reviews_per_month']

    for col in numeric_columns:
        if col in modeling_features.columns:
            modeling_features[col] = pd.to_numeric(modeling_features[col], errors='coerce')

    # Clean boolean columns
    boolean_columns = ['host_is_superhost', 'host_has_profile_pic',
                      'host_identity_verified', 'instant_bookable']

    for col in boolean_columns:
        if col in modeling_features.columns:
            modeling_features[col] = modeling_features[col].map({'t': True, 'f': False})

    # Clean price column from listings
    if 'price' in modeling_features.columns and modeling_features['price'].dtype == 'object':
        modeling_features['price_listings'] = modeling_features['price'].str.replace(r'[\$,]', '', regex=True)
        modeling_features['price_listings'] = pd.to_numeric(modeling_features['price_listings'], errors='coerce')

    return modeling_features


def create_modeling_dataset(listings_df, reliable_listings):
    """Create the final modeling dataset by joining listings with revenue metrics."""
    # Prepare features
    modeling_features = prepare_modeling_features(listings_df)

    # Join with revenue metrics
    modeling_dataset = modeling_features.merge(
        reliable_listings[['daily_revenue', 'annual_revenue_estimate', 'occupancy_rate',
                          'avg_price', 'confidence_score', 'observation_period_days']],
        left_on='id',
        right_index=True,
        how='inner'
    )

    return modeling_dataset


def analyze_dataset_quality(modeling_dataset):
    """Analyze the quality and completeness of the modeling dataset."""
    total_rows = len(modeling_dataset)

    # Key feature completeness
    key_features = ['neighbourhood_cleansed', 'property_type', 'room_type',
                   'accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
                   'review_scores_rating', 'number_of_reviews']

    feature_completeness = {}
    for feature in key_features:
        if feature in modeling_dataset.columns:
            missing_count = modeling_dataset[feature].isna().sum()
            completeness_pct = (1 - missing_count / total_rows) * 100
            feature_completeness[feature] = {
                'missing_count': int(missing_count),
                'completeness_percentage': float(completeness_pct)
            }

    # Target variable statistics
    target_stats = {
        'annual_revenue_estimate': {
            'count': int(modeling_dataset['annual_revenue_estimate'].count()),
            'mean': float(modeling_dataset['annual_revenue_estimate'].mean()),
            'median': float(modeling_dataset['annual_revenue_estimate'].median()),
            'std': float(modeling_dataset['annual_revenue_estimate'].std()),
            'min': float(modeling_dataset['annual_revenue_estimate'].min()),
            'max': float(modeling_dataset['annual_revenue_estimate'].max()),
            'q25': float(modeling_dataset['annual_revenue_estimate'].quantile(0.25)),
            'q75': float(modeling_dataset['annual_revenue_estimate'].quantile(0.75))
        }
    }

    quality_report = {
        'dataset_size': total_rows,
        'feature_completeness': feature_completeness,
        'target_statistics': target_stats,
        'geographic_coverage': {
            'unique_neighbourhoods': int(modeling_dataset['neighbourhood_cleansed'].nunique()) if 'neighbourhood_cleansed' in modeling_dataset.columns else 0,
            'coordinate_completeness': int((modeling_dataset['latitude'].notna() & modeling_dataset['longitude'].notna()).sum()) if 'latitude' in modeling_dataset.columns else 0
        }
    }

    return quality_report


def save_modeling_dataset(modeling_dataset, quality_report, output_dir):
    """Save the modeling dataset and quality report."""
    os.makedirs(output_dir, exist_ok=True)

    # Save dataset in multiple formats
    dataset_path_csv = os.path.join(output_dir, 'modeling_dataset.csv')
    dataset_path_pkl = os.path.join(output_dir, 'modeling_dataset.pkl')
    quality_path = os.path.join(output_dir, 'dataset_quality_report.json')

    modeling_dataset.to_csv(dataset_path_csv, index=False)
    modeling_dataset.to_pickle(dataset_path_pkl)

    with open(quality_path, 'w') as f:
        json.dump(quality_report, f, indent=2)

    return {
        'dataset_csv': dataset_path_csv,
        'dataset_pkl': dataset_path_pkl,
        'quality_report': quality_path
    }


def main():
    """Main function to create the modeling dataset."""
    processed_data_dir = 'data/processed/etap2/'
    output_dir = 'data/processed/etap2/modeling/'

    print("Loading datasets...")
    listings_df, reliable_listings = load_data(processed_data_dir)

    print("Creating modeling dataset...")
    modeling_dataset = create_modeling_dataset(listings_df, reliable_listings)

    print("Analyzing dataset quality...")
    quality_report = analyze_dataset_quality(modeling_dataset)

    print("Saving modeling dataset...")
    file_paths = save_modeling_dataset(modeling_dataset, quality_report, output_dir)

    print("\nModeling dataset creation complete!")
    print(f"Dataset size: {quality_report['dataset_size']:,} listings")
    print(f"Features: {modeling_dataset.shape[1]} columns")
    print(f"Target range: £{quality_report['target_statistics']['annual_revenue_estimate']['min']:,.0f} - £{quality_report['target_statistics']['annual_revenue_estimate']['max']:,.0f}")
    print(f"Target median: £{quality_report['target_statistics']['annual_revenue_estimate']['median']:,.0f}")

    print("\nFiles saved:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")

    print(f"\nKey feature completeness:")
    for feature, stats in quality_report['feature_completeness'].items():
        print(f"  {feature}: {stats['completeness_percentage']:.1f}%")


if __name__ == "__main__":
    main()
