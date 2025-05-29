"""
Feature-Engineered Data Visualization Script

This script provides comprehensive visualization of the feature-engineered Airbnb dataset,
analyzing the effectiveness of transformations and exploring relationships between new features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_palette("husl")


class FeatureEngineeredVisualizer:
    """
    Comprehensive visualization suite for feature-engineered Airbnb data.
    """

    def __init__(self, data_path: str, output_dir: str = None):
        """
        Initialize the visualizer.

        Args:
            data_path: Path to the feature-engineered data file
            output_dir: Directory to save plots (optional)
        """
        self.data_path = data_path
        self.output_dir = output_dir or "reports/figures/feature_engineered/"
        self.df = None

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Load the feature-engineered dataset."""
        print("Loading feature-engineered data...")

        if self.data_path.endswith(".pkl"):
            self.df = pd.read_pickle(self.data_path)
        elif self.data_path.endswith(".csv"):
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Data file must be .pkl or .csv format")

        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    def data_overview(self):
        """Display comprehensive data overview."""
        print("\n" + "=" * 80)
        print("FEATURE-ENGINEERED DATA OVERVIEW")
        print("=" * 80)

        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nData Types Summary:")
        print(self.df.dtypes.value_counts())

        print("\nNumerical Features Summary:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"Total numerical features: {len(numerical_cols)}")
        print(self.df[numerical_cols].describe())

        print("\nCategorical Features Summary:")
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        print(f"Total categorical features: {len(categorical_cols)}")

        print("\nMissing Values Check:")
        missing_summary = self.df.isnull().sum()
        missing_features = missing_summary[missing_summary > 0]
        if len(missing_features) > 0:
            print("Features with missing values:")
            for col, count in missing_features.items():
                pct = (count / len(self.df)) * 100
                print(f"  {col}: {count} ({pct:.2f}%)")
        else:
            print("No missing values found!")

    def plot_numerical_distributions(self):
        """Plot distributions of key numerical features."""
        print("\nGenerating numerical feature distributions...")

        key_features = [
            "price_cleaned_log",
            "beds_per_person",
            "bedrooms_per_person",
            "review_scores_average",
            "review_scores_std",
            "availability_rate",
            "distance_to_center",
            "name_length",
            "description_length",
            "amenities_count",
            "luxury_amenities_count",
            "basic_amenities_count",
        ]

        available_features = [col for col in key_features if col in self.df.columns]

        if not available_features:
            print("No key numerical features found for visualization.")
            return

        n_cols = 3
        n_rows = (len(available_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(available_features):
            if i < len(axes):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
                axes[i].tick_params(axis="x", rotation=45)

        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "numerical_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_scaled_features_validation(self):
        """Validate that scaled features have proper distributions."""
        print("\nValidating scaled feature distributions...")

        scaled_cols = [col for col in self.df.columns if col.endswith("_scaled")]

        if not scaled_cols:
            print("No scaled features found.")
            return

        sample_scaled_cols = scaled_cols[:12]

        n_cols = 4
        n_rows = (len(sample_scaled_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(sample_scaled_cols):
            if i < len(axes):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(
                    f"{col}\nMean: {self.df[col].mean():.3f}, Std: {self.df[col].std():.3f}"
                )
                axes[i].axvline(0, color="red", linestyle="--", alpha=0.7)

        for i in range(len(sample_scaled_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "scaled_features_validation.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_categorical_distributions(self):
        """Plot distributions of key categorical features."""
        print("\nGenerating categorical feature distributions...")

        key_categorical = [
            "availability_category",
            "host_experience_level",
            "property_size",
            "review_intensity",
            "price_competitiveness",
            "geographic_cluster",
        ]

        available_categorical = [
            col for col in key_categorical if col in self.df.columns
        ]

        if not available_categorical:
            print("No key categorical features found.")
            return

        n_cols = 2
        n_rows = (len(available_categorical) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(available_categorical):
            if i < len(axes):
                value_counts = self.df[col].value_counts()
                sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
                axes[i].set_xlabel("Count")

        for i in range(len(available_categorical), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "categorical_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_boolean_features(self):
        """Plot distributions of boolean features."""
        print("\nGenerating boolean feature distributions...")

        boolean_features = []
        for col in self.df.columns:
            if col.startswith("has_") or col.endswith("_sentiment"):
                boolean_features.append(col)

        if not boolean_features:
            print("No boolean features found.")
            return

        sample_boolean = boolean_features[:12]

        n_cols = 4
        n_rows = (len(sample_boolean) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(sample_boolean):
            if i < len(axes):
                value_counts = self.df[col].value_counts()
                colors = ["lightcoral", "lightblue"]
                axes[i].pie(
                    value_counts.values,
                    labels=value_counts.index,
                    autopct="%1.1f%%",
                    colors=colors,
                )
                axes[i].set_title(col.replace("_", " ").title())

        for i in range(len(sample_boolean), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "boolean_features.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_target_relationships(self):
        """Plot relationships between features and target variable."""
        print("\nAnalyzing feature-target relationships...")

        target_col = None
        potential_targets = ["price_cleaned_log", "price_cleaned", "price"]
        for col in potential_targets:
            if col in self.df.columns:
                target_col = col
                break

        if not target_col:
            print("No suitable target variable found.")
            return

        print(f"Using {target_col} as target variable.")

        numerical_features = [
            "distance_to_center",
            "amenities_count",
            "review_scores_average",
            "beds_per_person",
            "availability_rate",
        ]
        available_numerical = [
            col for col in numerical_features if col in self.df.columns
        ]

        if available_numerical:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, col in enumerate(available_numerical[:6]):
                if i < len(axes):
                    sns.scatterplot(
                        data=self.df, x=col, y=target_col, alpha=0.6, ax=axes[i]
                    )
                    axes[i].set_title(f"{col} vs {target_col}")

            for i in range(len(available_numerical), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "numerical_vs_target.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        categorical_features = [
            "property_size",
            "host_experience_level",
            "price_competitiveness",
            "geographic_cluster",
        ]
        available_categorical = [
            col for col in categorical_features if col in self.df.columns
        ]

        if available_categorical:
            n_cols = 2
            n_rows = (len(available_categorical) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]

            for i, col in enumerate(available_categorical):
                if i < len(axes):
                    sns.boxplot(data=self.df, x=col, y=target_col, ax=axes[i])
                    axes[i].set_title(f"{col} vs {target_col}")
                    axes[i].tick_params(axis="x", rotation=45)

            for i in range(len(available_categorical), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "categorical_vs_target.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def plot_transformation_effectiveness(self):
        """Compare pre and post transformation distributions."""
        print("\nAnalyzing transformation effectiveness...")

        if (
            "price_cleaned" in self.df.columns
            and "price_cleaned_log" in self.df.columns
        ):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            sns.histplot(data=self.df, x="price_cleaned", kde=True, ax=axes[0])
            axes[0].set_title("Original Price Distribution")
            axes[0].set_xlabel("Price (Original Scale)")

            sns.histplot(data=self.df, x="price_cleaned_log", kde=True, ax=axes[1])
            axes[1].set_title("Log-Transformed Price Distribution")
            axes[1].set_xlabel("Price (Log Scale)")

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "price_transformation.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def plot_correlation_analysis(self):
        """Generate correlation heatmap for numerical features."""
        print("\nGenerating correlation analysis...")

        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        exclude_patterns = ["id", "listing_id", "host_id"]
        filtered_cols = [
            col
            for col in numerical_cols
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]

        if len(filtered_cols) > 50:
            priority_features = [
                col
                for col in filtered_cols
                if any(
                    keyword in col.lower()
                    for keyword in [
                        "price",
                        "review",
                        "distance",
                        "amenities",
                        "beds",
                        "bedrooms",
                    ]
                )
            ]
            other_features = [
                col for col in filtered_cols if col not in priority_features
            ]
            filtered_cols = (
                priority_features + other_features[: 50 - len(priority_features)]
            )

        if len(filtered_cols) > 2:
            plt.figure(figsize=(20, 16))
            correlation_matrix = self.df[filtered_cols].corr()

            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                annot=False,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.1,
                cbar_kws={"shrink": 0.8},
            )

            plt.title("Correlation Matrix of Feature-Engineered Numerical Data")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "correlation_heatmap.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_corr_pairs.append(
                            (
                                correlation_matrix.columns[i],
                                correlation_matrix.columns[j],
                                corr_value,
                            )
                        )

            if high_corr_pairs:
                print("\nHigh correlation pairs (|correlation| > 0.8):")
                for col1, col2, corr in high_corr_pairs:
                    print(f"  {col1} - {col2}: {corr:.3f}")

    def plot_geographic_clusters(self):
        """Visualize geographic clusters."""
        print("\nVisualizing geographic clusters...")

        required_cols = ["latitude", "longitude", "geographic_cluster"]
        if all(col in self.df.columns for col in required_cols):
            plt.figure(figsize=(12, 10))

            sns.scatterplot(
                data=self.df,
                x="longitude",
                y="latitude",
                hue="geographic_cluster",
                palette="viridis",
                s=30,
                alpha=0.7,
            )

            plt.title("Geographic Clusters of Listings")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend(
                title="Geographic Cluster", bbox_to_anchor=(1.05, 1), loc="upper left"
            )

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "geographic_clusters.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            print(
                "Geographic cluster visualization requires latitude, longitude, and geographic_cluster columns."
            )

    def analyze_text_features(self):
        """Analyze text-based feature insights."""
        print("\nAnalyzing text feature insights...")

        text_features = [
            "name_length",
            "description_length",
            "name_positive_sentiment",
            "name_negative_sentiment",
        ]
        available_text = [col for col in text_features if col in self.df.columns]

        if not available_text:
            print("No text features found.")
            return

        target_col = None
        for col in ["price_cleaned_log", "price_cleaned", "price"]:
            if col in self.df.columns:
                target_col = col
                break

        if not target_col:
            print("No target variable found for text feature analysis.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        plot_idx = 0

        for col in ["name_length", "description_length"]:
            if col in self.df.columns and plot_idx < len(axes):
                sns.scatterplot(
                    data=self.df, x=col, y=target_col, alpha=0.6, ax=axes[plot_idx]
                )
                axes[plot_idx].set_title(f"{col} vs {target_col}")
                plot_idx += 1

        for col in ["name_positive_sentiment", "name_negative_sentiment"]:
            if col in self.df.columns and plot_idx < len(axes):
                sns.boxplot(data=self.df, x=col, y=target_col, ax=axes[plot_idx])
                axes[plot_idx].set_title(f"{col} vs {target_col}")
                plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "text_features_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def analyze_temporal_features(self):
        """Analyze temporal features if available."""
        print("\nAnalyzing temporal features...")

        temporal_features = [
            col
            for col in self.df.columns
            if any(
                keyword in col.lower()
                for keyword in ["available_", "price_cleaned_mean", "weekend_"]
            )
        ]

        if not temporal_features:
            print("No temporal features found.")
            return

        print(f"Found temporal features: {temporal_features}")

        if len(temporal_features) > 0:
            n_cols = 2
            n_rows = (len(temporal_features) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]

            for i, col in enumerate(temporal_features[: len(axes)]):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")

            for i in range(len(temporal_features), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "temporal_features.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def generate_summary_report(self):
        """Generate a summary report of the analysis."""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING ANALYSIS SUMMARY")
        print("=" * 80)

        print("\nDataset Overview:")
        print(f"  Total samples: {len(self.df):,}")
        print(f"  Total features: {len(self.df.columns)}")

        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns

        print(f"  Numerical features: {len(numerical_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")

        engineered_features = []
        feature_patterns = [
            "_per_",
            "_count",
            "_rate",
            "_sentiment",
            "_cluster",
            "_scaled",
            "_log",
            "has_",
            "distance_",
            "_average",
            "_std",
        ]

        for col in self.df.columns:
            if any(pattern in col for pattern in feature_patterns):
                engineered_features.append(col)

        print(f"  Engineered features identified: {len(engineered_features)}")

        missing_count = self.df.isnull().sum().sum()
        print(f"  Total missing values: {missing_count}")

        memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory usage: {memory_mb:.2f} MB")

        print(f"\nVisualizations saved to: {self.output_dir}")

        print("\nRecommendations for Next Steps:")
        print("  1. Perform feature selection to identify most important features")
        print("  2. Check for multicollinearity among engineered features")
        print("  3. Validate feature engineering effectiveness with model performance")
        print("  4. Consider interaction terms between important features")

    def run_complete_analysis(self):
        """Run the complete visualization analysis."""
        print("Starting comprehensive feature-engineered data analysis...")

        self.load_data()

        self.data_overview()
        self.plot_numerical_distributions()
        self.plot_scaled_features_validation()
        self.plot_categorical_distributions()
        self.plot_boolean_features()
        self.plot_target_relationships()
        self.plot_transformation_effectiveness()
        self.plot_correlation_analysis()
        self.plot_geographic_clusters()
        self.analyze_text_features()
        self.analyze_temporal_features()
        self.generate_summary_report()

        print("\nAnalysis complete!")


def main():
    """Main execution function."""

    data_dir = "/home/karolito/IUM/data/processed/etap2/feature_engineered/"
    data_file = os.path.join(data_dir, "listings_feature_engineered.pkl")

    csv_file = os.path.join(data_dir, "listings_feature_engineered.csv")

    if os.path.exists(data_file):
        input_file = data_file
    elif os.path.exists(csv_file):
        input_file = csv_file
    else:
        print(f"Error: No feature-engineered data found at {data_dir}")
        print("Please run the feature engineering pipeline first.")
        return

    visualizer = FeatureEngineeredVisualizer(
        data_path=input_file,
        output_dir="/home/karolito/IUM/reports/figures/feature_engineered/",
    )

    try:
        visualizer.run_complete_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your data file and try again.")


if __name__ == "__main__":
    main()
