import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

# Add the parent directory to the path to import from models
sys.path.append('/home/karolito/IUM')


class FeatureAnalyzer:

    def __init__(self, data_path, output_dir="logs/feature_analysis"):
        """
        Initialize the Feature Analyzer

        Args:
            data_path: Path to the dataset
            output_dir: Directory to save analysis results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.features_df = None
        self.target_col = "annual_revenue_adj"

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def load_and_preprocess_data(self):
        """Load and perform basic preprocessing similar to the model scripts"""
        print("Loading dataset...")

        if self.data_path.endswith(".pkl"):
            self.df = pd.read_pickle(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)

        print(f"Dataset shape: {self.df.shape}")
        print(f"Target column: {self.target_col}")

        # Separate features and target
        if self.target_col not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in dataset")

        # Remove ID columns that shouldn't be used as features
        id_cols = ["listing_id", "id", "host_id"]
        features_to_drop = [col for col in id_cols if col in self.df.columns]
        features_to_drop.append(self.target_col)

        self.features_df = self.df.drop(columns=features_to_drop,
                                        errors="ignore")

        print(
            f"Features dataset shape after removing ID columns: {self.features_df.shape}"
        )

        return self.features_df

    def analyze_feature_types(self):
        """Analyze and categorize all features by data type"""
        print("\n" + "=" * 60)
        print("FEATURE TYPE ANALYSIS")
        print("=" * 60)

        categorical_features = self.features_df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        numerical_features = self.features_df.select_dtypes(
            exclude=['object', 'category']).columns.tolist()

        print(f"\nTotal Features: {len(self.features_df.columns)}")
        print(f"Categorical Features: {len(categorical_features)}")
        print(f"Numerical Features: {len(numerical_features)}")

        # Detailed breakdown
        print(f"\nCATEGORICAL FEATURES ({len(categorical_features)}):")
        print("-" * 40)
        for i, feature in enumerate(categorical_features, 1):
            unique_count = self.features_df[feature].nunique()
            print(f"{i:2d}. {feature:<30} (unique values: {unique_count})")

        print(f"\nNUMERICAL FEATURES ({len(numerical_features)}):")
        print("-" * 40)
        for i, feature in enumerate(numerical_features, 1):
            dtype = str(self.features_df[feature].dtype)
            null_count = self.features_df[feature].isnull().sum()
            print(
                f"{i:2d}. {feature:<30} (dtype: {dtype}, nulls: {null_count})")

        # Save feature lists to file
        feature_summary = {
            'total_features': len(self.features_df.columns),
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'categorical_count': len(categorical_features),
            'numerical_count': len(numerical_features)
        }

        # Save as text file
        with open(os.path.join(self.output_dir, "feature_summary.txt"),
                  'w') as f:
            f.write("FEATURE TYPE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Features: {feature_summary['total_features']}\n")
            f.write(
                f"Categorical Features: {feature_summary['categorical_count']}\n"
            )
            f.write(
                f"Numerical Features: {feature_summary['numerical_count']}\n\n"
            )

            f.write(f"CATEGORICAL FEATURES ({len(categorical_features)}):\n")
            f.write("-" * 40 + "\n")
            for i, feature in enumerate(categorical_features, 1):
                unique_count = self.features_df[feature].nunique()
                f.write(
                    f"{i:2d}. {feature:<30} (unique values: {unique_count})\n")

            f.write(f"\nNUMERICAL FEATURES ({len(numerical_features)}):\n")
            f.write("-" * 40 + "\n")
            for i, feature in enumerate(numerical_features, 1):
                dtype = str(self.features_df[feature].dtype)
                null_count = self.features_df[feature].isnull().sum()
                f.write(
                    f"{i:2d}. {feature:<30} (dtype: {dtype}, nulls: {null_count})\n"
                )

        return feature_summary

    def preprocess_for_correlation_analysis(self):
        """Preprocess features for correlation analysis (encode categorical variables)"""
        print("\n" + "=" * 60)
        print("PREPROCESSING FOR CORRELATION ANALYSIS")
        print("=" * 60)

        # Get categorical and numerical columns
        categorical_columns = self.features_df.select_dtypes(
            include=['object', 'category']).columns
        numerical_columns = self.features_df.select_dtypes(
            exclude=['object', 'category']).columns

        processed_df = self.features_df.copy()

        # One-hot encode categorical variables
        if len(categorical_columns) > 0:
            print(
                f"One-hot encoding {len(categorical_columns)} categorical features..."
            )

            # Handle categorical columns
            for col in categorical_columns:
                processed_df[col] = processed_df[col].astype(str)

            # Create dummy variables
            categorical_encoded = pd.get_dummies(
                processed_df[categorical_columns],
                drop_first=True,
                dtype=np.float32)

            # Combine with numerical features
            numerical_df = processed_df[numerical_columns].astype(np.float32)
            processed_df = pd.concat([numerical_df, categorical_encoded],
                                     axis=1)

        print(f"Final processed shape: {processed_df.shape}")
        print(f"Features after encoding: {len(processed_df.columns)}")

        return processed_df

    def correlation_analysis(self, processed_df, threshold=0.8):
        """Perform comprehensive correlation analysis"""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

        # Calculate correlation matrix
        correlation_matrix = processed_df.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        n_features = len(correlation_matrix.columns)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'feature1':
                        correlation_matrix.columns[i],
                        'feature2':
                        correlation_matrix.columns[j],
                        'correlation':
                        corr_val
                    })

        # Sort by absolute correlation value
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        print(
            f"Found {len(high_corr_pairs)} feature pairs with |correlation| >= {threshold}"
        )

        if high_corr_pairs:
            print(f"\nTop 20 highly correlated feature pairs:")
            print("-" * 80)
            print(f"{'Feature 1':<25} {'Feature 2':<25} {'Correlation':<12}")
            print("-" * 80)

            for i, pair in enumerate(high_corr_pairs[:20]):
                print(
                    f"{pair['feature1']:<25} {pair['feature2']:<25} {pair['correlation']:<12.4f}"
                )

        # Create correlation heatmap for top features
        self._plot_correlation_heatmap(correlation_matrix, processed_df)

        # Save correlation analysis
        self._save_correlation_analysis(high_corr_pairs, threshold)

        return correlation_matrix, high_corr_pairs

    def vif_analysis(self, processed_df, vif_threshold=10.0):
        """Calculate Variance Inflation Factor for multicollinearity detection"""
        print("\n" + "=" * 60)
        print("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
        print("=" * 60)

        # Handle missing values
        processed_df_clean = processed_df.dropna()

        if processed_df_clean.empty:
            print("No data available after removing missing values")
            return None

        # If too many features, sample a subset for VIF analysis
        max_features_for_vif = 50  # VIF calculation can be computationally expensive

        if len(processed_df_clean.columns) > max_features_for_vif:
            print(
                f"Too many features ({len(processed_df_clean.columns)}) for VIF analysis."
            )
            print(
                f"Selecting top {max_features_for_vif} features by correlation with target..."
            )

            # Calculate correlation with target
            target_corr = abs(
                processed_df_clean.corrwith(
                    self.df[self.target_col])).sort_values(ascending=False)
            top_features = target_corr.head(
                max_features_for_vif).index.tolist()
            processed_df_clean = processed_df_clean[top_features]

        print(
            f"Calculating VIF for {len(processed_df_clean.columns)} features..."
        )

        # Calculate VIF
        vif_data = []

        try:
            for i, feature in enumerate(processed_df_clean.columns):
                try:
                    vif_value = variance_inflation_factor(
                        processed_df_clean.values, i)
                    vif_data.append({'feature': feature, 'vif': vif_value})
                except:
                    # Handle cases where VIF can't be calculated
                    vif_data.append({'feature': feature, 'vif': np.nan})

            # Create VIF DataFrame
            vif_df = pd.DataFrame(vif_data)
            vif_df = vif_df.sort_values('vif', ascending=False)

            # Filter out infinite and NaN values for display
            vif_df_clean = vif_df[vif_df['vif'].notna()
                                  & np.isfinite(vif_df['vif'])]

            print(f"\nVIF Analysis Results (threshold = {vif_threshold}):")
            print("-" * 50)
            print(f"{'Feature':<30} {'VIF':<10}")
            print("-" * 50)

            high_vif_features = []
            for _, row in vif_df_clean.head(20).iterrows():
                vif_status = "HIGH" if row['vif'] > vif_threshold else "OK"
                print(f"{row['feature']:<30} {row['vif']:<10.2f} {vif_status}")

                if row['vif'] > vif_threshold:
                    high_vif_features.append(row['feature'])

            print(
                f"\nFeatures with VIF > {vif_threshold}: {len(high_vif_features)}"
            )

            # Save VIF analysis
            vif_df.to_csv(os.path.join(self.output_dir, "vif_analysis.csv"),
                          index=False)

            return vif_df, high_vif_features

        except Exception as e:
            print(f"Error in VIF calculation: {e}")
            return None, []

    def target_correlation_analysis(self, processed_df):
        """Analyze correlation of features with target variable"""
        print("\n" + "=" * 60)
        print("TARGET CORRELATION ANALYSIS")
        print("=" * 60)

        # Calculate correlation with target
        target_correlations = processed_df.corrwith(self.df[self.target_col])
        target_correlations = target_correlations.dropna().sort_values(
            key=abs, ascending=False)

        print(f"Top 20 features most correlated with {self.target_col}:")
        print("-" * 60)
        print(f"{'Feature':<35} {'Correlation':<12}")
        print("-" * 60)

        for feature, corr in target_correlations.head(20).items():
            print(f"{feature:<35} {corr:<12.4f}")

        # Plot target correlations
        self._plot_target_correlations(target_correlations)

        return target_correlations

    def generate_feature_recommendations(self, high_corr_pairs,
                                         high_vif_features,
                                         target_correlations):
        """Generate recommendations for feature selection"""
        print("\n" + "=" * 60)
        print("FEATURE SELECTION RECOMMENDATIONS")
        print("=" * 60)

        # Features to potentially drop based on high correlation
        corr_drop_candidates = set()
        for pair in high_corr_pairs:
            # Keep the feature with higher absolute correlation with target
            feat1_target_corr = abs(
                target_correlations.get(pair['feature1'], 0))
            feat2_target_corr = abs(
                target_correlations.get(pair['feature2'], 0))

            if feat1_target_corr > feat2_target_corr:
                corr_drop_candidates.add(pair['feature2'])
            else:
                corr_drop_candidates.add(pair['feature1'])

        # Combine recommendations
        all_drop_candidates = corr_drop_candidates.union(
            set(high_vif_features))

        print(f"FEATURES TO CONSIDER DROPPING:")
        print("-" * 40)
        print(f"Due to high correlation: {len(corr_drop_candidates)} features")
        print(f"Due to high VIF: {len(high_vif_features)} features")
        print(f"Total unique candidates: {len(all_drop_candidates)} features")

        if all_drop_candidates:
            print(f"\nDetailed drop candidates:")
            for i, feature in enumerate(sorted(all_drop_candidates), 1):
                reasons = []
                if feature in corr_drop_candidates:
                    reasons.append("high correlation")
                if feature in high_vif_features:
                    reasons.append("high VIF")
                print(f"{i:2d}. {feature:<35} ({', '.join(reasons)})")

        # Generate recommended feature list
        all_features = set(target_correlations.index)
        recommended_features = all_features - all_drop_candidates

        print(f"\nRECOMMENDED FEATURES TO KEEP:")
        print("-" * 40)
        print(
            f"Total recommended: {len(recommended_features)} out of {len(all_features)} features"
        )

        # Sort recommended features by target correlation
        recommended_sorted = [(feat, target_correlations[feat])
                              for feat in recommended_features
                              if feat in target_correlations.index]
        recommended_sorted.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\nTop 30 recommended features (by target correlation):")
        for i, (feature, corr) in enumerate(recommended_sorted[:30], 1):
            print(f"{i:2d}. {feature:<35} (corr: {corr:.4f})")

        # Save recommendations
        self._save_recommendations(all_drop_candidates, recommended_features,
                                   recommended_sorted)

        return {
            'drop_candidates': all_drop_candidates,
            'recommended_features': recommended_features,
            'recommended_sorted': recommended_sorted
        }

    def _plot_correlation_heatmap(self, correlation_matrix, processed_df):
        """Plot correlation heatmap for top features"""
        # Select top features by variance or target correlation
        n_features_to_plot = min(50, len(correlation_matrix.columns))

        if hasattr(self, 'df') and self.target_col in self.df.columns:
            # Use target correlation to select features
            target_corr = abs(processed_df.corrwith(
                self.df[self.target_col])).sort_values(ascending=False)
            top_features = target_corr.head(n_features_to_plot).index.tolist()
        else:
            # Use variance to select features
            variances = processed_df.var().sort_values(ascending=False)
            top_features = variances.head(n_features_to_plot).index.tolist()

        # Create heatmap
        plt.figure(figsize=(20, 16))
        correlation_subset = correlation_matrix.loc[top_features, top_features]

        mask = np.triu(np.ones_like(correlation_subset, dtype=bool))
        sns.heatmap(correlation_subset,
                    mask=mask,
                    annot=False,
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    cbar_kws={"shrink": .8})

        plt.title(f'Correlation Heatmap - Top {n_features_to_plot} Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()

        print(
            f"Correlation heatmap saved to: {os.path.join(self.output_dir, 'correlation_heatmap.png')}"
        )

    def _plot_target_correlations(self, target_correlations):
        """Plot feature correlations with target"""
        top_features = target_correlations.head(30)

        plt.figure(figsize=(12, 10))
        colors = ['red' if x < 0 else 'blue' for x in top_features.values]
        plt.barh(range(len(top_features)),
                 top_features.values,
                 color=colors,
                 alpha=0.7)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Correlation with Target')
        plt.title('Top 30 Features - Correlation with Target Variable')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "target_correlations.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()

        print(
            f"Target correlations plot saved to: {os.path.join(self.output_dir, 'target_correlations.png')}"
        )

    def _save_correlation_analysis(self, high_corr_pairs, threshold):
        """Save correlation analysis results"""
        with open(os.path.join(self.output_dir, "correlation_analysis.txt"),
                  'w') as f:
            f.write("HIGH CORRELATION PAIRS ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Threshold: |correlation| >= {threshold}\n")
            f.write(
                f"Found {len(high_corr_pairs)} highly correlated pairs\n\n")

            if high_corr_pairs:
                f.write(
                    f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':<12}\n"
                )
                f.write("-" * 75 + "\n")

                for pair in high_corr_pairs:
                    f.write(
                        f"{pair['feature1']:<30} {pair['feature2']:<30} {pair['correlation']:<12.4f}\n"
                    )

        # Save as CSV
        if high_corr_pairs:
            pd.DataFrame(high_corr_pairs).to_csv(os.path.join(
                self.output_dir, "high_correlation_pairs.csv"),
                                                 index=False)

    def _save_recommendations(self, drop_candidates, recommended_features,
                              recommended_sorted):
        """Save feature selection recommendations"""
        with open(os.path.join(self.output_dir, "feature_recommendations.txt"),
                  'w') as f:
            f.write("FEATURE SELECTION RECOMMENDATIONS\n")
            f.write("=" * 60 + "\n\n")

            f.write("FEATURES TO CONSIDER DROPPING:\n")
            f.write("-" * 40 + "\n")
            for feature in sorted(drop_candidates):
                f.write(f"- {feature}\n")

            f.write(
                f"\nRECOMMENDED FEATURES TO KEEP ({len(recommended_features)}):\n"
            )
            f.write("-" * 40 + "\n")
            for feature, corr in recommended_sorted:
                f.write(f"- {feature:<40} (target_corr: {corr:.4f})\n")

        # Save as Python list for easy import
        with open(
                os.path.join(self.output_dir, "recommended_features_list.py"),
                'w') as f:
            f.write(
                "# Recommended features list generated by feature analysis\n\n"
            )
            f.write("RECOMMENDED_FEATURES = [\n")
            for feature, _ in recommended_sorted:
                f.write(f"    '{feature}',\n")
            f.write("]\n\n")
            f.write("# Features to drop\n")
            f.write("DROP_CANDIDATES = [\n")
            for feature in sorted(drop_candidates):
                f.write(f"    '{feature}',\n")
            f.write("]\n")

    def run_complete_analysis(self,
                              correlation_threshold=0.8,
                              vif_threshold=10.0):
        """Run the complete feature analysis pipeline"""
        print("=" * 80)
        print("COMPREHENSIVE FEATURE ANALYSIS")
        print("=" * 80)

        # Load and preprocess data
        self.load_and_preprocess_data()

        # Analyze feature types
        feature_summary = self.analyze_feature_types()

        # Preprocess for correlation analysis
        processed_df = self.preprocess_for_correlation_analysis()

        # Correlation analysis
        correlation_matrix, high_corr_pairs = self.correlation_analysis(
            processed_df, threshold=correlation_threshold)

        # VIF analysis
        vif_df, high_vif_features = self.vif_analysis(
            processed_df, vif_threshold=vif_threshold)

        # Target correlation analysis
        target_correlations = self.target_correlation_analysis(processed_df)

        # Generate recommendations
        recommendations = self.generate_feature_recommendations(
            high_corr_pairs, high_vif_features, target_correlations)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("- feature_summary.txt")
        print("- correlation_analysis.txt")
        print("- high_correlation_pairs.csv")
        print("- vif_analysis.csv")
        print("- feature_recommendations.txt")
        print("- recommended_features_list.py")
        print("- correlation_heatmap.png")
        print("- target_correlations.png")

        return {
            'feature_summary': feature_summary,
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'vif_results': vif_df,
            'high_vif_features': high_vif_features,
            'target_correlations': target_correlations,
            'recommendations': recommendations
        }


if __name__ == "__main__":
    # Configure the analysis
    data_path = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"
    output_dir = "logs/feature_analysis"

    # Initialize analyzer
    analyzer = FeatureAnalyzer(data_path=data_path, output_dir=output_dir)

    # Run complete analysis
    try:
        results = analyzer.run_complete_analysis(
            correlation_threshold=0.8,  # Features with |correlation| >= 0.8
            vif_threshold=10.0  # Features with VIF >= 10.0
        )

        print("\nFeature analysis completed successfully!")

        # Print quick summary
        print(f"\nQUICK SUMMARY:")
        print(
            f"- Total features: {results['feature_summary']['total_features']}"
        )
        print(f"- High correlation pairs: {len(results['high_corr_pairs'])}")
        print(f"- High VIF features: {len(results['high_vif_features'])}")
        print(
            f"- Recommended features: {len(results['recommendations']['recommended_features'])}"
        )

    except Exception as e:
        print(f"Error in feature analysis: {e}")
        import traceback
        traceback.print_exc()
