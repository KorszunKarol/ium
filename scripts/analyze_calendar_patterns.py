#!/usr/bin/env python3
"""
Calendar Data Analysis Script
Analyzes availability patterns and calculates occupancy metrics for Airbnb listings.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


def load_calendar_data(processed_data_dir):
    """Load calendar data from pickle file."""
    calendar_path = os.path.join(processed_data_dir, 'calendar_e2_df.pkl')
    return pd.read_pickle(calendar_path)


def calculate_listing_statistics(calendar_df):
    """Calculate comprehensive statistics for each listing."""
    listing_stats = calendar_df.groupby('listing_id').agg({
        'date': ['count', 'min', 'max'],
        'available': ['sum', 'mean'],
        'price_cleaned': ['mean', 'median', 'count']
    }).round(2)

    listing_stats.columns = ['total_days', 'first_date', 'last_date',
                            'available_days', 'availability_rate',
                            'avg_price', 'median_price', 'price_records']

    listing_stats['observation_period_days'] = (
        listing_stats['last_date'] - listing_stats['first_date']
    ).dt.days + 1
    listing_stats['booked_days'] = listing_stats['total_days'] - listing_stats['available_days']
    listing_stats['occupancy_rate'] = listing_stats['booked_days'] / listing_stats['total_days']

    return listing_stats


def generate_summary_statistics(listing_stats):
    """Generate summary statistics for the analysis."""
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return float(obj) if pd.notnull(obj) else None

    obs_stats = listing_stats['observation_period_days'].describe()
    occ_stats = listing_stats['occupancy_rate'].describe()
    price_stats = listing_stats['avg_price'].describe()

    summary = {
        'total_listings': int(len(listing_stats)),
        'observation_period_stats': {k: convert_to_json_serializable(v) for k, v in obs_stats.to_dict().items()},
        'occupancy_rate_stats': {k: convert_to_json_serializable(v) for k, v in occ_stats.to_dict().items()},
        'price_stats': {k: convert_to_json_serializable(v) for k, v in price_stats.to_dict().items()},
        'data_quality_metrics': {
            'listings_with_price_data': int(listing_stats['avg_price'].notna().sum()),
            'median_observation_period': float(listing_stats['observation_period_days'].median()),
            'mean_occupancy_rate': float(listing_stats['occupancy_rate'].mean()),
            'listings_with_min_30_days': int((listing_stats['observation_period_days'] >= 30).sum()),
            'listings_with_min_20_price_records': int((listing_stats['price_records'] >= 20).sum())
        }
    }
    return summary


def filter_reliable_listings(listing_stats, min_days=30, min_price_records=20):
    """Filter for listings with sufficient data quality."""
    reliable_listings = listing_stats[
        (listing_stats['observation_period_days'] >= min_days) &
        (listing_stats['price_records'] >= min_price_records) &
        (listing_stats['avg_price'].notna())
    ].copy()

    reliable_listings['daily_revenue'] = (
        reliable_listings['avg_price'] * reliable_listings['occupancy_rate']
    )
    reliable_listings['annual_revenue_estimate'] = reliable_listings['daily_revenue'] * 365

    reliable_listings['confidence_score'] = np.minimum(
        reliable_listings['observation_period_days'] / 365,
        1.0
    ) * np.minimum(
        reliable_listings['price_records'] / 100,
        1.0
    )

    return reliable_listings


def save_results(listing_stats, reliable_listings, summary, output_dir):
    """Save analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full listing statistics
    listing_stats.to_csv(os.path.join(output_dir, 'listing_statistics.csv'))
    listing_stats.to_pickle(os.path.join(output_dir, 'listing_statistics.pkl'))

    # Save reliable listings for modeling
    reliable_listings.to_csv(os.path.join(output_dir, 'reliable_listings.csv'))
    reliable_listings.to_pickle(os.path.join(output_dir, 'reliable_listings.pkl'))

    # Save summary statistics
    with open(os.path.join(output_dir, 'calendar_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return {
        'listing_statistics': os.path.join(output_dir, 'listing_statistics.csv'),
        'reliable_listings': os.path.join(output_dir, 'reliable_listings.csv'),
        'summary': os.path.join(output_dir, 'calendar_analysis_summary.json')
    }


def main():
    """Main analysis function."""
    processed_data_dir = 'data/processed/etap2/'
    output_dir = 'data/processed/etap2/analysis/'

    print("Loading calendar data...")
    calendar_df = load_calendar_data(processed_data_dir)

    print("Calculating listing statistics...")
    listing_stats = calculate_listing_statistics(calendar_df)

    print("Filtering reliable listings...")
    reliable_listings = filter_reliable_listings(listing_stats)

    print("Generating summary statistics...")
    summary = generate_summary_statistics(listing_stats)
    summary['reliable_listings_count'] = int(len(reliable_listings))
    summary['reliable_listings_percentage'] = float(len(reliable_listings) / len(listing_stats) * 100)

    print("Saving results...")
    file_paths = save_results(listing_stats, reliable_listings, summary, output_dir)

    print("\nAnalysis complete. Results saved to:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")

    print(f"\nSummary:")
    print(f"  Total listings: {summary['total_listings']:,}")
    print(f"  Reliable listings: {int(summary['reliable_listings_count']):,} ({summary['reliable_listings_percentage']:.1f}%)")
    print(f"  Median observation period: {summary['data_quality_metrics']['median_observation_period']:.0f} days")
    print(f"  Mean occupancy rate: {summary['data_quality_metrics']['mean_occupancy_rate']:.3f}")


if __name__ == "__main__":
    main()
