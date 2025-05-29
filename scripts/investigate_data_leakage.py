#!/usr/bin/env python3
"""
Data Leakage Investigation Script
Identifies and analyzes sources of data leakage in the modeling dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_analyze_leakage():
    """Load dataset and analyze potential data leakage sources."""

    # Load the modeling dataset
    df = pd.read_pickle('data/processed/etap2/modeling/modeling_dataset.pkl')

    print("="*60)
    print("DATA LEAKAGE INVESTIGATION REPORT")
    print("="*60)

    # 1. CRITICAL LEAKAGE: Target components in features
    print("\n🚨 CRITICAL LEAKAGE DETECTED:")
    print("The following target components are present as features:")

    target_components = ['occupancy_rate_adj', 'adr_adj']
    target = df['annual_revenue_adj']

    for component in target_components:
        if component in df.columns:
            corr = df[component].corr(target)
            print(f"  • {component}: correlation = {corr:.4f}")

    # Test if target can be reconstructed
    print("\n🔍 RECONSTRUCTION TEST:")
    if all(comp in df.columns for comp in target_components):
        reconstructed = df['occupancy_rate_adj'] * df['adr_adj'] * 365
        reconstruction_corr = reconstructed.corr(target)
        print(f"Correlation between 'occupancy_rate_adj * adr_adj * 365' and target: {reconstruction_corr:.6f}")

        # Show sample comparison
        print("\nSample verification (first 5 rows):")
        sample_comparison = pd.DataFrame({
            'actual_target': target.head(),
            'reconstructed': reconstructed.head(),
            'difference': (target.head() - reconstructed.head()).abs(),
            'pct_diff': ((target.head() - reconstructed.head()).abs() / target.head() * 100)
        })
        print(sample_comparison)

    # 2. SUSPICIOUS PRICE FEATURES
    print("\n"+"="*60)
    print("🔍 SUSPICIOUS PRICE-RELATED FEATURES:")

    price_features = [
        'price_cleaned_mean', 'price_cleaned_median', 'price_cleaned_std',
        'available_mean', 'available_sum', 'weekend_availability_rate'
    ]

    print("\nCorrelations with target:")
    high_corr_features = []
    for feature in price_features:
        if feature in df.columns:
            corr = df[feature].corr(target)
            print(f"  • {feature:25s}: {corr:.4f}")
            if abs(corr) > 0.7:  # High correlation threshold
                high_corr_features.append((feature, corr))

    # 3. OTHER LEAKY FEATURES
    print("\n"+"="*60)
    print("🔍 OTHER POTENTIALLY LEAKY FEATURES:")

    # Features that might be derived from the same calendar data used for targets
    other_suspicious = [
        'availability_rate', 'availability_30', 'availability_60',
        'availability_90', 'availability_365'
    ]

    for feature in other_suspicious:
        if feature in df.columns:
            corr = df[feature].corr(target)
            print(f"  • {feature:25s}: {corr:.4f}")
            if abs(corr) > 0.3:
                high_corr_features.append((feature, corr))

    # 4. SUMMARY AND RECOMMENDATIONS
    print("\n"+"="*60)
    print("📋 SUMMARY AND RECOMMENDATIONS:")
    print("="*60)

    print("\n🚨 IMMEDIATE ACTIONS REQUIRED:")
    print("1. REMOVE TARGET COMPONENTS from features:")
    print("   - occupancy_rate_adj")
    print("   - adr_adj")
    print("   - Any scaled versions of these")

    print("\n2. INVESTIGATE CALENDAR-DERIVED FEATURES:")
    print("   These might be calculated from the same data used for targets:")
    for feature in ['price_cleaned_mean', 'price_cleaned_median', 'available_mean']:
        if feature in df.columns:
            print(f"   - {feature}")

    print("\n3. SAFE FEATURES TO KEEP:")
    safe_features = [
        'neighbourhood_cleansed', 'property_type', 'room_type', 'accommodates',
        'bedrooms', 'beds', 'latitude', 'longitude', 'host_is_superhost',
        'number_of_reviews', 'review_scores_rating', 'amenities_count',
        'price'  # Original listing price should be OK
    ]
    print("   These are likely safe as they're from listing data, not calendar data:")
    for feature in safe_features:
        if feature in df.columns:
            print(f"   ✓ {feature}")

    return df, high_corr_features


def create_clean_feature_list():
    """Create a list of definitely safe features for modeling."""

    # Features that are definitely NOT derived from calendar data
    safe_features = [
        # Basic listing attributes
        'neighbourhood_cleansed', 'property_type', 'room_type',
        'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'latitude', 'longitude',

        # Host characteristics
        'host_is_superhost', 'host_response_rate', 'host_acceptance_rate',
        'host_listings_count', 'host_total_listings_count',
        'host_identity_verified',

        # Reviews (historical, not forward-looking)
        'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location', 'review_scores_value',

        # Amenities and property features
        'amenities_count', 'has_wifi', 'has_kitchen', 'has_air_conditioning',
        'has_pool', 'has_parking', 'has_tv', 'luxury_amenities_count',

        # Basic listing settings
        'minimum_nights', 'maximum_nights', 'instant_bookable',

        # Text features
        'name_length', 'description_length',

        # Geographic features
        'distance_to_center',

        # Original price (should be listing price, not calculated from calendar)
        'price'
    ]

    # Add one-hot encoded versions
    categorical_prefixes = [
        'neighbourhood_cleansed_grouped_',
        'property_type_grouped_',
        'room_type_'
    ]

    return safe_features, categorical_prefixes


if __name__ == '__main__':
    df, high_corr_features = load_and_analyze_leakage()
    safe_features, categorical_prefixes = create_clean_feature_list()

    print("\n"+"="*60)
    print("🛠️  NEXT STEPS:")
    print("="*60)
    print("1. Create a clean modeling dataset with only safe features")
    print("2. Re-run baseline models to get realistic performance estimates")
    print("3. Implement proper cross-validation with spatial splitting")
    print("4. Add features incrementally and monitor for leakage")
