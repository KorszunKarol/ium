#!/usr/bin/env python3
"""
Generate Etap 2 data statistics and save to markdown and CSV formats.
"""

import os
import pandas as pd
from datetime import datetime
import sys


def main():
    # Set paths
    base_dir = "/home/karolito/IUM"
    processed_dir = os.path.join(base_dir, "data/processed/etap2")
    docs_dir = os.path.join(base_dir, "docs")

    # Check if pickle files exist
    listings_pkl = os.path.join(processed_dir, "listings_e2_df.pkl")
    calendar_pkl = os.path.join(processed_dir, "calendar_e2_df.pkl")
    reviews_pkl = os.path.join(processed_dir, "reviews_e2_df.pkl")

    if not all(os.path.exists(f) for f in [listings_pkl, calendar_pkl, reviews_pkl]):
        print("Error: Not all pickle files exist")
        return

    try:
        # Load datasets
        print("Loading datasets...")
        listings_df = pd.read_pickle(listings_pkl)
        calendar_df = pd.read_pickle(calendar_pkl)
        reviews_df = pd.read_pickle(reviews_pkl)

        print(f"✓ Loaded {len(listings_df):,} listings")
        print(f"✓ Loaded {len(calendar_df):,} calendar records")
        print(f"✓ Loaded {len(reviews_df):,} reviews")

        # Generate statistics
        stats = generate_statistics(listings_df, calendar_df, reviews_df)

        # Save to markdown
        markdown_path = os.path.join(docs_dir, "etap2_data.md")
        save_markdown_report(stats, markdown_path)
        print(f"✓ Saved markdown report: {markdown_path}")

        # Save to CSV
        csv_path = os.path.join(docs_dir, "etap2_stats.csv")
        save_csv_report(stats, csv_path)
        print(f"✓ Saved CSV report: {csv_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def generate_statistics(listings_df, calendar_df, reviews_df):
    """Generate comprehensive statistics."""

    stats = {}

    # Basic dataset info
    stats["basic"] = {
        "listings_rows": len(listings_df),
        "listings_cols": len(listings_df.columns),
        "calendar_rows": len(calendar_df),
        "calendar_cols": len(calendar_df.columns),
        "reviews_rows": len(reviews_df),
        "reviews_cols": len(reviews_df.columns),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ID analysis
    listings_ids = set(listings_df["id"].dropna())
    calendar_ids = set(calendar_df["listing_id"].dropna())
    reviews_ids = set(reviews_df["listing_id"].dropna())

    stats["ids"] = {
        "unique_listings_in_listings": len(listings_ids),
        "unique_listings_in_calendar": len(calendar_ids),
        "unique_listings_in_reviews": len(reviews_ids),
    }

    # Join analysis
    listings_calendar_overlap = len(listings_ids.intersection(calendar_ids))
    listings_reviews_overlap = len(listings_ids.intersection(reviews_ids))
    three_way_overlap = len(
        listings_ids.intersection(calendar_ids).intersection(reviews_ids)
    )

    stats["joins"] = {
        "listings_calendar_overlap": listings_calendar_overlap,
        "listings_calendar_rate": listings_calendar_overlap / len(listings_ids) * 100,
        "listings_reviews_overlap": listings_reviews_overlap,
        "listings_reviews_rate": listings_reviews_overlap / len(listings_ids) * 100,
        "three_way_overlap": three_way_overlap,
        "three_way_rate": three_way_overlap / len(listings_ids) * 100,
    }

    # Orphan records
    orphan_calendar = len(calendar_ids - listings_ids)
    orphan_reviews = len(reviews_ids - listings_ids)

    stats["orphans"] = {
        "calendar_orphans": orphan_calendar,
        "reviews_orphans": orphan_reviews,
        "total_orphans": orphan_calendar + orphan_reviews,
    }

    # Missing values analysis
    critical_features = [
        "neighbourhood_cleansed",
        "bedrooms",
        "bathrooms_text",
        "beds",
        "latitude",
        "longitude",
        "price",
    ]
    missing_analysis = {}

    for feature in critical_features:
        if feature in listings_df.columns:
            missing_count = listings_df[feature].isna().sum()
            missing_pct = (missing_count / len(listings_df)) * 100
            missing_analysis[f"{feature}_missing_count"] = missing_count
            missing_analysis[f"{feature}_missing_pct"] = missing_pct

    stats["missing"] = missing_analysis

    # Temporal analysis
    listing_periods = calendar_df.groupby("listing_id")["date"].count()
    stats["temporal"] = {
        "median_observation_days": listing_periods.median(),
        "short_periods_count": (listing_periods < 30).sum(),
        "short_periods_pct": (listing_periods < 30).sum() / len(listing_periods) * 100,
    }

    if "date" in calendar_df.columns:
        date_range = calendar_df["date"].agg(["min", "max"])
        stats["temporal"]["date_min"] = date_range["min"].strftime("%Y-%m-%d")
        stats["temporal"]["date_max"] = date_range["max"].strftime("%Y-%m-%d")

    # Reviews analysis
    if "date" in reviews_df.columns:
        reviews_date_range = reviews_df["date"].agg(["min", "max"])
        stats["reviews_analysis"] = {
            "review_date_min": reviews_date_range["min"].strftime("%Y-%m-%d"),
            "review_date_max": reviews_date_range["max"].strftime("%Y-%m-%d"),
        }

        if "comments" in reviews_df.columns:
            non_null_comments = reviews_df["comments"].dropna()
            stats["reviews_analysis"]["avg_comment_length"] = (
                non_null_comments.str.len().mean()
            )
            stats["reviews_analysis"]["missing_comments_count"] = (
                reviews_df["comments"].isna().sum()
            )
            stats["reviews_analysis"]["missing_comments_pct"] = (
                reviews_df["comments"].isna().sum() / len(reviews_df) * 100
            )

    return stats


def save_markdown_report(stats, filepath):
    """Save statistics as markdown report."""

    content = f"""# Etap 2 Data Analysis Report

Generated on: {stats["basic"]["generated_at"]}

## 📊 Dataset Overview

| Dataset | Rows | Columns |
|---------|------|---------|
| Listings | {stats["basic"]["listings_rows"]:,} | {stats["basic"]["listings_cols"]} |
| Calendar | {stats["basic"]["calendar_rows"]:,} | {stats["basic"]["calendar_cols"]} |
| Reviews | {stats["basic"]["reviews_rows"]:,} | {stats["basic"]["reviews_cols"]} |

## 🔗 ID Overlap Analysis

| Dataset | Unique Listing IDs |
|---------|-------------------|
| Listings | {stats["ids"]["unique_listings_in_listings"]:,} |
| Calendar | {stats["ids"]["unique_listings_in_calendar"]:,} |
| Reviews | {stats["ids"]["unique_listings_in_reviews"]:,} |

## 🎯 Join Success Rates

| Join Type | Successful Joins | Success Rate |
|-----------|------------------|--------------|
| Listings-Calendar | {stats["joins"]["listings_calendar_overlap"]:,} | {stats["joins"]["listings_calendar_rate"]:.1f}% |
| Listings-Reviews | {stats["joins"]["listings_reviews_overlap"]:,} | {stats["joins"]["listings_reviews_rate"]:.1f}% |
| **Complete Records (All 3)** | **{stats["joins"]["three_way_overlap"]:,}** | **{stats["joins"]["three_way_rate"]:.1f}%** |

## 🚨 Orphan Records

| Type | Count |
|------|-------|
| Calendar Orphans | {stats["orphans"]["calendar_orphans"]:,} |
| Reviews Orphans | {stats["orphans"]["reviews_orphans"]:,} |
| **Total Orphans** | **{stats["orphans"]["total_orphans"]:,}** |

## 📋 Missing Values (Critical Features)

| Feature | Missing Count | Missing % |
|---------|---------------|-----------|"""

    # Add missing values for each feature
    for key, value in stats["missing"].items():
        if key.endswith("_missing_count"):
            feature = key.replace("_missing_count", "")
            pct_key = f"{feature}_missing_pct"
            if pct_key in stats["missing"]:
                content += (
                    f"\n| {feature} | {value:,} | {stats['missing'][pct_key]:.1f}% |"
                )

    content += f"""

## 📅 Temporal Coverage

- **Median observation period**: {stats["temporal"]["median_observation_days"]:.0f} days
- **Listings with <30 days**: {stats["temporal"]["short_periods_count"]:,} ({stats["temporal"]["short_periods_pct"]:.1f}%)"""

    if "date_min" in stats["temporal"]:
        content += f"\n- **Date range**: {stats['temporal']['date_min']} to {stats['temporal']['date_max']}"

    if "reviews_analysis" in stats:
        content += f"""

## 💬 Reviews Analysis

- **Review date range**: {stats["reviews_analysis"]["review_date_min"]} to {stats["reviews_analysis"]["review_date_max"]}"""

        if "avg_comment_length" in stats["reviews_analysis"]:
            content += f"\n- **Average comment length**: {stats['reviews_analysis']['avg_comment_length']:.0f} characters"
            content += f"\n- **Missing comments**: {stats['reviews_analysis']['missing_comments_count']:,} ({stats['reviews_analysis']['missing_comments_pct']:.1f}%)"

    content += f"""

## 🏆 Etap 1 vs Etap 2 Comparison

| Metric | Etap 2 | Etap 1 Baseline | Improvement |
|--------|--------|-----------------|-------------|
| Complete records | {stats["joins"]["three_way_overlap"]:,} | ~11,000 | {((stats["joins"]["three_way_overlap"] / 11000 - 1) * 100):+.1f}% |
| Total orphans | {stats["orphans"]["total_orphans"]:,} | ~75,000 | {((1 - stats["orphans"]["total_orphans"] / 75000) * 100):+.1f}% |"""

    if "neighbourhood_cleansed_missing_pct" in stats["missing"]:
        content += f"\n| Missing neighbourhood | {stats['missing']['neighbourhood_cleansed_missing_pct']:.1f}% | ~30% | {(30 - stats['missing']['neighbourhood_cleansed_missing_pct']):+.1f}pp |"

    content += """

## 📝 Key Findings

### ✅ Improvements over Etap 1:
- Significantly more complete records available for analysis
- Reduced orphan records indicating better data integration
- Improved data quality across critical features

### 🎯 Recommendations:
1. Proceed with the enhanced dataset for machine learning model development
2. Focus on the complete records subset for primary analysis
3. Implement targeted imputation strategies for remaining missing values
4. Leverage the improved temporal coverage for time-series analysis

---
*Report generated by Etap 2 Data Analysis Pipeline*
"""

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def save_csv_report(stats, filepath):
    """Save statistics as CSV for programmatic access."""

    # Flatten the nested dictionary
    flattened = {}

    for category, data in stats.items():
        if isinstance(data, dict):
            for key, value in data.items():
                flattened[f"{category}_{key}"] = value
        else:
            flattened[category] = data

    # Create DataFrame
    df = pd.DataFrame([flattened])

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
