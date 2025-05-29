#!/usr/bin/env python3
"""
Professional Data Visualization Script for Airbnb Data Analysis

This script generates all the visualizations from the professional visualization notebook
and saves them as high-quality image files.

Author: Data Analysis Team
Date: May 2025
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")


class AirbnbVisualizationGenerator:
    def __init__(
        self, data_dir="data/processed/etap2/", output_dir="reports/figures/etap2/"
    ):
        """
        Initialize the visualization generator

        Args:
            data_dir (str): Path to processed data directory
            output_dir (str): Path to output directory for saved plots
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.setup_environment()
        self.load_data()

    def setup_environment(self):
        """Setup plotting environment and create output directory"""
        # Set professional plotting style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load all required datasets"""
        print("Loading datasets...")

        self.listings_df = pd.read_pickle(self.data_dir + "listings_e2_df.pkl")
        self.calendar_df = pd.read_pickle(self.data_dir + "calendar_e2_df.pkl")
        self.reviews_df = pd.read_pickle(self.data_dir + "reviews_e2_df.pkl")

        print(f"Listings: {self.listings_df.shape}")
        print(f"Calendar: {self.calendar_df.shape}")
        print(f"Reviews: {self.reviews_df.shape}")

        # Load revenue analysis if available
        try:
            self.reliable_listings = pd.read_pickle(
                self.data_dir + "analysis/reliable_listings.pkl"
            )
            print(f"Reliable listings: {self.reliable_listings.shape}")
            self.has_reliable_data = True
        except FileNotFoundError:
            print("Reliable listings data not found")
            self.has_reliable_data = False

    def save_plot(self, filename, dpi=300, bbox_inches="tight"):
        """Save current matplotlib plot"""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
        print(f"Saved: {filename}")

    def save_plotly_plot(self, fig, filename):
        """Save plotly plot as HTML and PNG"""
        html_path = os.path.join(self.output_dir, filename.replace(".png", ".html"))
        png_path = os.path.join(self.output_dir, filename)

        # Save as HTML (interactive)
        plot(fig, filename=html_path, auto_open=False)
        print(f"Saved: {filename.replace('.png', '.html')}")

        # Save as PNG (static)
        try:
            fig.write_image(png_path, width=1200, height=700)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Could not save PNG (install kaleido): {e}")

    def generate_missing_data_analysis(self):
        """Generate missing value analysis charts"""
        print("\n=== Generating Missing Data Analysis ===")

        # Calculate missing values for key features
        key_features = [
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms_text",
            "price",
            "minimum_nights",
            "availability_365",
            "number_of_reviews",
            "review_scores_rating",
            "review_scores_accuracy",
            "review_scores_cleanliness",
            "review_scores_checkin",
            "review_scores_communication",
            "review_scores_location",
            "review_scores_value",
            "host_is_superhost",
            "host_identity_verified",
            "instant_bookable",
            "latitude",
            "longitude",
        ]

        available_features = [f for f in key_features if f in self.listings_df.columns]

        missing_data = []
        for feature in available_features:
            missing_count = self.listings_df[feature].isna().sum()
            missing_pct = (missing_count / len(self.listings_df)) * 100
            missing_data.append(
                {
                    "feature": feature,
                    "missing_count": missing_count,
                    "missing_percentage": missing_pct,
                }
            )

        missing_df = pd.DataFrame(missing_data).sort_values(
            "missing_percentage", ascending=False
        )

        # Chart 1: Missing Data Percentages
        plt.figure(figsize=(12, 8))
        top_missing = missing_df.head(15)

        # Create color map based on severity
        colors = []
        for pct in top_missing["missing_percentage"]:
            if pct > 50:
                colors.append("#D32F2F")  # Red
            elif pct > 20:
                colors.append("#FF8F00")  # Orange
            else:
                colors.append("#FBC02D")  # Yellow

        bars = plt.barh(
            range(len(top_missing)),
            top_missing["missing_percentage"],
            color=colors,
            alpha=0.8,
        )
        plt.yticks(range(len(top_missing)), top_missing["feature"])
        plt.xlabel("Missing Percentage (%)")
        plt.title("Missing Data by Feature (Top 15)", fontweight="bold", pad=20)
        plt.grid(axis="x", alpha=0.3)

        # Add percentage labels on bars
        for i, (idx, row) in enumerate(top_missing.iterrows()):
            plt.text(
                row["missing_percentage"] + 0.5,
                i,
                f"{row['missing_percentage']:.1f}%",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        self.save_plot("01_missing_data_percentages.png")
        plt.close()

        # Chart 2: Missing Value Correlation
        geo_review_features = [
            "neighbourhood_cleansed",
            "latitude",
            "longitude",
            "review_scores_rating",
            "review_scores_accuracy",
            "review_scores_cleanliness",
            "review_scores_checkin",
            "review_scores_communication",
            "review_scores_location",
            "review_scores_value",
            "number_of_reviews",
        ]
        geo_review_available = [
            f for f in geo_review_features if f in self.listings_df.columns
        ]

        if geo_review_available:
            plt.figure(figsize=(12, 10))
            missing_matrix = self.listings_df[geo_review_available].isna()
            correlation_matrix = missing_matrix.corr()

            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                annot=True,
                fmt=".4f",
                cmap="Reds",
                square=True,
                cbar_kws={"label": "Missing Value Correlation"},
            )
            plt.title(
                "Missing Value Correlation: Geographic & Review Features",
                fontweight="bold",
                pad=20,
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            self.save_plot("02_missing_value_correlation.png")
            plt.close()

    def generate_neighborhood_analysis(self):
        """Generate neighborhood analysis charts"""
        print("\n=== Generating Neighborhood Analysis ===")

        if "neighbourhood_cleansed" not in self.listings_df.columns:
            return

        # Chart 3: Top Neighborhoods by Listing Count
        neighbourhood_counts = self.listings_df["neighbourhood_cleansed"].value_counts()

        plt.figure(figsize=(14, 8))
        top_20_neighbourhoods = neighbourhood_counts.head(20)
        bars = plt.barh(
            range(len(top_20_neighbourhoods)),
            top_20_neighbourhoods.values,
            color="steelblue",
            alpha=0.8,
        )
        plt.yticks(range(len(top_20_neighbourhoods)), top_20_neighbourhoods.index)
        plt.xlabel("Number of Listings")
        plt.title(
            "Top 20 London Neighborhoods by Listing Count", fontweight="bold", pad=20
        )
        plt.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, count in enumerate(top_20_neighbourhoods.values):
            plt.text(count + 50, i, f"{count:,}", va="center", fontsize=10)

        plt.tight_layout()
        self.save_plot("03_top_neighborhoods_by_count.png")
        plt.close()

        # Chart 4: Distribution of Neighborhood Sizes
        plt.figure(figsize=(12, 6))
        plt.hist(
            neighbourhood_counts.values,
            bins=40,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
            linewidth=0.5,
        )
        plt.xlabel("Number of Listings per Neighborhood")
        plt.ylabel("Number of Neighborhoods")
        plt.title("Distribution of Neighborhood Sizes", fontweight="bold", pad=20)
        plt.grid(alpha=0.3)

        # Add statistics
        mean_size = neighbourhood_counts.mean()
        median_size = neighbourhood_counts.median()
        plt.axvline(
            mean_size, color="red", linestyle="--", label=f"Mean: {mean_size:.0f}"
        )
        plt.axvline(
            median_size,
            color="blue",
            linestyle="--",
            label=f"Median: {median_size:.0f}",
        )
        plt.legend()

        plt.tight_layout()
        self.save_plot("04_neighborhood_size_distribution.png")
        plt.close()

    def generate_geographic_distribution(self):
        """Generate geographic distribution visualizations"""
        print("\n=== Generating Geographic Distribution ===")

        if (
            "latitude" not in self.listings_df.columns
            or "longitude" not in self.listings_df.columns
        ):
            print("Latitude or Longitude data not available")
            return

        # Interactive London Listings Geographic Distribution
        plot_df = self.listings_df[
            ["latitude", "longitude", "room_type", "price", "neighbourhood_cleansed"]
        ].copy()
        plot_df.dropna(subset=["latitude", "longitude"], inplace=True)

        # Clean price data
        if "price" in plot_df.columns:
            plot_df["price_cleaned"] = plot_df["price"].replace(
                {r"\$|,": ""}, regex=True
            )
            plot_df["price_cleaned"] = pd.to_numeric(
                plot_df["price_cleaned"], errors="coerce"
            )
            # Fill NaN values with median price or a default value
            plot_df["price_cleaned"] = plot_df["price_cleaned"].fillna(
                plot_df["price_cleaned"].median()
            )
            # If still NaN (all values were NaN), use a default
            if plot_df["price_cleaned"].isna().all():
                plot_df["price_cleaned"] = 100
        else:
            plot_df["price_cleaned"] = 100  # Default value

        plot_df["room_type"] = plot_df["room_type"].fillna("Unknown")
        plot_df["neighbourhood_cleansed"] = plot_df["neighbourhood_cleansed"].fillna(
            "Unknown"
        )

        # Sample for performance
        sample_size = min(20000, len(plot_df))
        sample_df = plot_df.sample(n=sample_size, random_state=42)

        fig = px.scatter_mapbox(
            sample_df,
            lat="latitude",
            lon="longitude",
            color="room_type",
            size="price_cleaned",
            hover_name="neighbourhood_cleansed",
            hover_data={
                "latitude": False,
                "longitude": False,
                "room_type": True,
                "price_cleaned": ":.2f",
            },
            color_discrete_sequence=px.colors.qualitative.Plotly,
            zoom=10,
            height=700,
            title=f"Interactive Geographic Distribution of London Airbnb Listings (n={sample_size:,})",
        )

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})

        self.save_plotly_plot(fig, "05_geographic_distribution_interactive.png")

    def generate_property_features(self):
        """Generate property features analysis"""
        print("\n=== Generating Property Features Analysis ===")

        # Chart 6: Room Type Distribution
        if "room_type" in self.listings_df.columns:
            plt.figure(figsize=(10, 8))
            room_counts = self.listings_df["room_type"].value_counts()

            colors = plt.cm.Set3(np.linspace(0, 1, len(room_counts)))
            wedges, texts, autotexts = plt.pie(
                room_counts.values,
                labels=room_counts.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            plt.title("Distribution of Room Types", fontweight="bold", pad=20)

            # Enhance text
            for autotext in autotexts:
                autotext.set_color("black")
                autotext.set_fontweight("bold")

            plt.axis("equal")
            plt.tight_layout()
            self.save_plot("06_room_type_distribution.png")
            plt.close()

        # Chart 7: Property Type Distribution
        if "property_type" in self.listings_df.columns:
            plt.figure(figsize=(14, 8))
            property_counts = self.listings_df["property_type"].value_counts().head(15)

            bars = plt.barh(
                range(len(property_counts)),
                property_counts.values,
                color="darkgreen",
                alpha=0.7,
            )
            plt.yticks(range(len(property_counts)), property_counts.index)
            plt.xlabel("Number of Listings")
            plt.title("Top 15 Property Types", fontweight="bold", pad=20)
            plt.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, count in enumerate(property_counts.values):
                plt.text(count + 100, i, f"{count:,}", va="center", fontsize=10)

            plt.tight_layout()
            self.save_plot("07_property_type_distribution.png")
            plt.close()

        # Chart 8: Accommodates Distribution
        if "accommodates" in self.listings_df.columns:
            plt.figure(figsize=(12, 6))
            accommodates_data = pd.to_numeric(
                self.listings_df["accommodates"], errors="coerce"
            ).dropna()

            plt.hist(
                accommodates_data,
                bins=range(1, int(accommodates_data.max()) + 2),
                alpha=0.7,
                color="purple",
                edgecolor="black",
                linewidth=0.5,
            )
            plt.xlabel("Number of Guests Accommodated")
            plt.ylabel("Number of Listings")
            plt.title("Distribution of Guest Capacity", fontweight="bold", pad=20)
            plt.grid(alpha=0.3)

            # Add statistics
            mean_acc = accommodates_data.mean()
            median_acc = accommodates_data.median()
            plt.axvline(
                mean_acc, color="red", linestyle="--", label=f"Mean: {mean_acc:.1f}"
            )
            plt.axvline(
                median_acc,
                color="blue",
                linestyle="--",
                label=f"Median: {median_acc:.0f}",
            )
            plt.legend()

            plt.tight_layout()
            self.save_plot("08_accommodates_distribution.png")
            plt.close()

    def generate_review_scores_analysis(self):
        """Generate review scores analysis"""
        print("\n=== Generating Review Scores Analysis ===")

        # Chart 9: Overall Review Scores Distribution
        if "review_scores_rating" in self.listings_df.columns:
            plt.figure(figsize=(12, 6))
            review_scores = pd.to_numeric(
                self.listings_df["review_scores_rating"], errors="coerce"
            ).dropna()

            plt.hist(
                review_scores,
                bins=30,
                alpha=0.7,
                color="gold",
                edgecolor="black",
                linewidth=0.5,
            )
            plt.xlabel("Review Score Rating")
            plt.ylabel("Number of Listings")
            plt.title(
                "Distribution of Overall Review Scores", fontweight="bold", pad=20
            )
            plt.grid(alpha=0.3)

            # Add statistics
            mean_score = review_scores.mean()
            median_score = review_scores.median()
            plt.axvline(
                mean_score, color="red", linestyle="--", label=f"Mean: {mean_score:.2f}"
            )
            plt.axvline(
                median_score,
                color="blue",
                linestyle="--",
                label=f"Median: {median_score:.2f}",
            )
            plt.legend()

            plt.tight_layout()
            self.save_plot("09_review_scores_distribution.png")
            plt.close()

        # Chart 10: Number of Reviews Distribution
        if "number_of_reviews" in self.listings_df.columns:
            plt.figure(figsize=(12, 6))
            review_counts = pd.to_numeric(
                self.listings_df["number_of_reviews"], errors="coerce"
            ).dropna()

            # Focus on 95th percentile to avoid extreme outliers
            upper_limit = review_counts.quantile(0.95)
            filtered_reviews = review_counts[review_counts <= upper_limit]

            plt.hist(
                filtered_reviews,
                bins=50,
                alpha=0.7,
                color="teal",
                edgecolor="black",
                linewidth=0.5,
            )
            plt.xlabel("Number of Reviews")
            plt.ylabel("Number of Listings")
            plt.title(
                "Distribution of Number of Reviews (up to 95th percentile)",
                fontweight="bold",
                pad=20,
            )
            plt.grid(alpha=0.3)

            # Add statistics
            mean_reviews = review_counts.mean()
            median_reviews = review_counts.median()
            plt.axvline(
                mean_reviews,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_reviews:.1f}",
            )
            plt.axvline(
                median_reviews,
                color="blue",
                linestyle="--",
                label=f"Median: {median_reviews:.0f}",
            )
            plt.legend()

            plt.tight_layout()
            self.save_plot("10_number_of_reviews_distribution.png")
            plt.close()

    def generate_calendar_price_analysis(self):
        """Generate calendar and price analysis"""
        print("\n=== Generating Calendar and Price Analysis ===")

        # Chart 11: Price Distribution from Calendar Data
        if "price_cleaned" in self.calendar_df.columns:
            plt.figure(figsize=(12, 6))
            valid_prices = self.calendar_df["price_cleaned"].dropna()

            # Focus on reasonable price range (up to 95th percentile)
            upper_limit = valid_prices.quantile(0.95)
            filtered_prices = valid_prices[valid_prices <= upper_limit]

            plt.hist(
                filtered_prices,
                bins=50,
                alpha=0.7,
                color="lightgreen",
                edgecolor="black",
                linewidth=0.5,
            )
            plt.xlabel("Price (£)")
            plt.ylabel("Frequency")
            plt.title(
                "Distribution of Daily Prices (up to 95th percentile)",
                fontweight="bold",
                pad=20,
            )
            plt.grid(alpha=0.3)

            # Add statistics
            mean_price = valid_prices.mean()
            median_price = valid_prices.median()
            plt.axvline(
                mean_price,
                color="red",
                linestyle="--",
                label=f"Mean: £{mean_price:.0f}",
            )
            plt.axvline(
                median_price,
                color="blue",
                linestyle="--",
                label=f"Median: £{median_price:.0f}",
            )
            plt.legend()

            plt.tight_layout()
            self.save_plot("11_daily_price_distribution.png")
            plt.close()

        # Chart 12: Availability Pattern
        if "available" in self.calendar_df.columns:
            plt.figure(figsize=(10, 8))
            availability_counts = self.calendar_df["available"].value_counts()

            labels = ["Not Available (Booked)", "Available"]
            colors = ["#FF6B6B", "#4ECDC4"]
            explode = (0.05, 0)  # Slightly separate the "booked" slice

            wedges, texts, autotexts = plt.pie(
                availability_counts.values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                explode=explode,
                startangle=90,
                shadow=True,
            )
            plt.title(
                "Overall Availability Pattern Across All Listings",
                fontweight="bold",
                pad=20,
            )

            # Enhance text
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(12)

            plt.axis("equal")
            plt.tight_layout()
            self.save_plot("12_availability_pattern.png")
            plt.close()

    def generate_revenue_analysis(self):
        """Generate revenue analysis if data is available"""
        print("\n=== Generating Revenue Analysis ===")

        if (
            not self.has_reliable_data
            or "annual_revenue_estimate" not in self.reliable_listings.columns
        ):
            print("Revenue data not available - skipping revenue analysis")
            return

        revenue_data = self.reliable_listings["annual_revenue_estimate"].dropna()

        # Chart 13: Annual Revenue Distribution
        plt.figure(figsize=(12, 6))
        upper_limit = revenue_data.quantile(0.95)

        plt.hist(
            revenue_data,
            bins=50,
            range=(0, upper_limit),
            alpha=0.7,
            color="darkblue",
            edgecolor="black",
            linewidth=0.5,
        )
        plt.xlabel("Annual Revenue Estimate (£)")
        plt.ylabel("Number of Listings")
        plt.title(
            "Distribution of Annual Revenue Estimates (View up to 95th Percentile)",
            fontweight="bold",
            pad=20,
        )
        plt.grid(alpha=0.3)
        plt.xlim(0, upper_limit)

        # Calculate statistics for the data visible in the plot's range
        revenue_data_in_view = revenue_data[revenue_data <= upper_limit]
        mean_revenue_in_view = revenue_data_in_view.mean()
        median_revenue_in_view = revenue_data_in_view.median()

        # Add statistics lines
        plt.axvline(
            mean_revenue_in_view,
            color="red",
            linestyle="--",
            label=f"Mean (View): £{mean_revenue_in_view:,.0f}",
        )
        plt.axvline(
            median_revenue_in_view,
            color="blue",
            linestyle="--",
            label=f"Median (View): £{median_revenue_in_view:,.0f}",
        )
        plt.legend()

        plt.tight_layout()
        self.save_plot("13_annual_revenue_distribution.png")
        plt.close()

        # Chart 14: Revenue Analysis Subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Box plot
        axes[0, 0].boxplot(revenue_data, vert=True)
        axes[0, 0].set_ylabel("Annual Revenue Estimate (£)")
        axes[0, 0].set_title("Box Plot of Revenue Distribution")
        axes[0, 0].grid(alpha=0.3)

        # 2. Histogram with fewer bins
        axes[0, 1].hist(
            revenue_data, bins=20, alpha=0.7, color="darkgreen", edgecolor="black"
        )
        axes[0, 1].set_xlabel("Annual Revenue Estimate (£)")
        axes[0, 1].set_ylabel("Number of Listings")
        axes[0, 1].set_title("Histogram with 20 bins (full range)")
        axes[0, 1].grid(alpha=0.3)

        # 3. Histogram focusing on lower values
        median_val = revenue_data.median()
        low_range_data = revenue_data[revenue_data <= median_val * 2]
        axes[1, 0].hist(
            low_range_data, bins=30, alpha=0.7, color="orange", edgecolor="black"
        )
        axes[1, 0].set_xlabel("Annual Revenue Estimate (£)")
        axes[1, 0].set_ylabel("Number of Listings")
        axes[1, 0].set_title(f"Revenue up to £{median_val * 2:,.0f} (2x median)")
        axes[1, 0].grid(alpha=0.3)

        # 4. Log scale histogram
        log_data = revenue_data[revenue_data > 0]
        axes[1, 1].hist(log_data, bins=50, alpha=0.7, color="purple", edgecolor="black")
        axes[1, 1].set_xlabel("Annual Revenue Estimate (£)")
        axes[1, 1].set_ylabel("Number of Listings")
        axes[1, 1].set_title("Histogram with Log Scale")
        axes[1, 1].set_xscale("log")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        self.save_plot("14_revenue_analysis_subplots.png")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("=== Starting Professional Visualization Generation ===")
        print(f"Data loaded successfully. Output directory: {self.output_dir}")

        try:
            self.generate_missing_data_analysis()
            self.generate_neighborhood_analysis()
            self.generate_geographic_distribution()
            self.generate_property_features()
            self.generate_review_scores_analysis()
            self.generate_calendar_price_analysis()
            self.generate_revenue_analysis()

            print("\n=== All Visualizations Generated Successfully ===")
            print(f"Check the output directory: {self.output_dir}")

        except Exception as e:
            print(f"Error generating visualizations: {e}")
            raise


def main():
    """Main function to run the visualization generator"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate professional Airbnb data visualizations"
    )
    parser.add_argument(
        "--data_dir",
        default="data/processed/etap2/",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output_dir",
        default="reports/figures/etap2/",
        help="Path to output directory for plots",
    )

    args = parser.parse_args()

    # Initialize and run visualization generator
    generator = AirbnbVisualizationGenerator(
        data_dir=args.data_dir, output_dir=args.output_dir
    )

    generator.generate_all_visualizations()


if __name__ == "__main__":
    main()
