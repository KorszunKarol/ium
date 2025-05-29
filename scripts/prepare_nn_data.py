#!/usr/bin/env python3
"""
Data Preprocessing Script for Neural Network Model

This script prepares the modeling dataset for use with the NeuralNetworkModel by:
1. Removing redundant features (scaled, pre-encoded columns)
2. Transforming complex categorical features
3. Converting date columns to numerical features
4. Handling collinearity
5. Ensuring proper data types for the neural network pipeline

Author: Data Science Team
Created: 2025-05-29
"""

import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, correlation_threshold=0.9):
        self.correlation_threshold = correlation_threshold
        self.dropped_features = {
            'id_columns': [],
            'scaled_columns': [],
            'pre_encoded_columns': [],
            'high_cardinality': [],
            'low_variance': [],
            'collinear': [],
            'date_columns': []
        }
        self.created_features = []

    def load_data(self, file_path):
        """Load the dataset from pickle file"""
        print(f"Loading data from {file_path}...")
        df = pd.read_pickle(file_path)
        print(f"Original dataset shape: {df.shape}")
        print(f"Original columns: {len(df.columns)}")
        return df

    def drop_initial_features(self, df):
        """Drop ID columns, scaled columns, and pre-encoded columns"""
        print("\n" + "="*50)
        print("STEP 1: Dropping Initial Features")
        print("="*50)

        # ID columns to drop
        id_cols = ['scrape_id']  # neural_net.py already handles listing_id, id, host_id
        id_cols_present = [col for col in id_cols if col in df.columns]
        df = df.drop(columns=id_cols_present)
        self.dropped_features['id_columns'].extend(id_cols_present)
        print(f"Dropped ID columns ({len(id_cols_present)}): {id_cols_present}")

        # Pre-scaled columns (ending with _scaled)
        scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
        df = df.drop(columns=scaled_cols)
        self.dropped_features['scaled_columns'].extend(scaled_cols)
        print(f"Dropped scaled columns ({len(scaled_cols)}): {scaled_cols[:10]}{'...' if len(scaled_cols) > 10 else ''}")

        # Pre-one-hot encoded columns (grouped and individual categorical dummies)
        pre_encoded_patterns = [
            'neighbourhood_cleansed_grouped_',
            'property_type_grouped_',
            'room_type_',
            'host_response_time_within'
        ]

        pre_encoded_cols = []
        for pattern in pre_encoded_patterns:
            cols = [col for col in df.columns if pattern in col and not col.endswith('_scaled')]
            pre_encoded_cols.extend(cols)

        # Also drop individual room type and response time columns if they're already encoded
        individual_encoded = [col for col in df.columns if col in [
            'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room',
            'host_response_time_within a day', 'host_response_time_within a few hours',
            'host_response_time_within an hour'
        ]]
        pre_encoded_cols.extend(individual_encoded)

        df = df.drop(columns=pre_encoded_cols)
        self.dropped_features['pre_encoded_columns'].extend(pre_encoded_cols)
        print(f"Dropped pre-encoded columns ({len(pre_encoded_cols)}): {pre_encoded_cols[:10]}{'...' if len(pre_encoded_cols) > 10 else ''}")

        # High cardinality text/URL columns
        high_cardinality_cols = [
            'listing_url', 'name', 'description', 'neighborhood_overview',
            'picture_url', 'host_url', 'host_name', 'host_about',
            'host_thumbnail_url', 'host_picture_url', 'amenities'
        ]
        high_cardinality_present = [col for col in high_cardinality_cols if col in df.columns]
        df = df.drop(columns=high_cardinality_present)
        self.dropped_features['high_cardinality'].extend(high_cardinality_present)
        print(f"Dropped high cardinality columns ({len(high_cardinality_present)}): {high_cardinality_present}")

        # Low variance columns
        low_variance_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if unique_count <= 1:
                    low_variance_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() <= 1:
                    low_variance_cols.append(col)

        # Special handling for has_availability (convert 't'/'f' to 1/0 if it has variance)
        if 'has_availability' in df.columns:
            unique_vals = df['has_availability'].dropna().unique()
            if len(unique_vals) <= 1:
                low_variance_cols.append('has_availability')
            else:
                df['has_availability'] = df['has_availability'].map({'t': 1, 'f': 0}).fillna(0)
                print("Converted 'has_availability' to binary (t->1, f->0)")

        # Drop neighbourhood if it only has one unique value
        if 'neighbourhood' in df.columns:
            if df['neighbourhood'].nunique() <= 1:
                low_variance_cols.append('neighbourhood')

        df = df.drop(columns=low_variance_cols)
        self.dropped_features['low_variance'].extend(low_variance_cols)
        print(f"Dropped low variance columns ({len(low_variance_cols)}): {low_variance_cols}")

        print(f"Dataset shape after initial dropping: {df.shape}")
        return df

    def transform_features(self, df):
        """Transform complex features into neural network-friendly formats"""
        print("\n" + "="*50)
        print("STEP 2: Feature Transformation")
        print("="*50)

        # Date columns transformation
        date_columns = ['last_scraped', 'host_since', 'first_review', 'last_review', 'calendar_last_scraped']
        date_features_created = []

        # Get reference date (latest scrape date)
        if 'last_scraped' in df.columns:
            ref_dates = pd.to_datetime(df['last_scraped'], errors='coerce')
            reference_date = ref_dates.max()
            print(f"Using reference date: {reference_date}")
        else:
            reference_date = pd.Timestamp('2024-12-23')  # Based on your data
            print(f"Using default reference date: {reference_date}")

        for col in date_columns:
            if col in df.columns:
                # Convert to datetime
                dates = pd.to_datetime(df[col], errors='coerce')

                if col == 'host_since':
                    # Host tenure in days
                    df['host_days_active'] = (reference_date - dates).dt.days
                    df['host_days_active'] = df['host_days_active'].fillna(df['host_days_active'].median())
                    date_features_created.append('host_days_active')

                elif col == 'first_review':
                    # Days since first review (listing age proxy)
                    df['days_since_first_review'] = (reference_date - dates).dt.days
                    df['days_since_first_review'] = df['days_since_first_review'].fillna(df['days_since_first_review'].median())
                    date_features_created.append('days_since_first_review')

                elif col == 'last_review':
                    # Days since last review (recency)
                    df['days_since_last_review'] = (reference_date - dates).dt.days
                    df['days_since_last_review'] = df['days_since_last_review'].fillna(df['days_since_last_review'].median())
                    date_features_created.append('days_since_last_review')

                # Drop original date column
                df = df.drop(columns=[col])
                self.dropped_features['date_columns'].append(col)

        self.created_features.extend(date_features_created)
        print(f"Created date features ({len(date_features_created)}): {date_features_created}")

        # Host verifications transformation
        if 'host_verifications' in df.columns:
            verification_types = ['email', 'phone', 'work_email']

            for v_type in verification_types:
                col_name = f'has_verification_{v_type}'
                df[col_name] = 0

                # Parse verification strings
                for idx, verif_str in df['host_verifications'].items():
                    if pd.isna(verif_str) or verif_str in ['None', '[]']:
                        continue
                    try:
                        # Clean the string and parse
                        clean_str = verif_str.replace("'", '"')
                        verif_list = ast.literal_eval(clean_str)
                        if v_type in verif_list:
                            df.at[idx, col_name] = 1
                    except:
                        continue

                self.created_features.append(col_name)

            df = df.drop(columns=['host_verifications'])
            self.dropped_features['high_cardinality'].append('host_verifications')
            print(f"Created verification features: {[f'has_verification_{v}' for v in verification_types]}")

        # Bathrooms text transformation
        if 'bathrooms_text' in df.columns:
            # Extract number of bathrooms
            df['bathrooms_from_text'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
            df['bathrooms_from_text'] = df['bathrooms_from_text'].fillna(df['bathrooms_from_text'].median())

            # Extract if shared
            df['bathroom_is_shared'] = df['bathrooms_text'].str.contains('shared', case=False, na=False).astype(int)

            # Compare with existing bathrooms column and use the more complete one
            if 'bathrooms' in df.columns:
                # Fill missing values in original bathrooms with parsed values
                df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms_from_text'])
                df = df.drop(columns=['bathrooms_from_text'])
            else:
                df['bathrooms'] = df['bathrooms_from_text']
                df = df.drop(columns=['bathrooms_from_text'])

            df = df.drop(columns=['bathrooms_text'])
            self.created_features.extend(['bathroom_is_shared'])
            self.dropped_features['high_cardinality'].append('bathrooms_text')
            print("Transformed bathrooms_text to numerical and shared indicator")

        # Binary t/f columns transformation
        binary_tf_columns = ['host_has_profile_pic']
        for col in binary_tf_columns:
            if col in df.columns:
                df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0)
                print(f"Converted {col} to binary (t->1, f->0)")

        # Ensure categorical columns are object type and handle missing values
        categorical_cols_to_keep = [
            'neighbourhood_cleansed', 'property_type', 'source',
            'review_intensity', 'confidence_level', 'host_location'
        ]

        for col in categorical_cols_to_keep:
            if col in df.columns:
                # Fill missing values with 'Unknown'
                df[col] = df[col].fillna('Unknown').astype('object')
                print(f"Processed categorical column: {col} ({df[col].nunique()} unique values)")

        # Handle host_location separately due to high cardinality
        if 'host_location' in df.columns:
            # Group by country or major regions
            df['host_location_grouped'] = 'Other'

            # UK-based locations
            uk_patterns = ['London', 'England', 'UK', 'United Kingdom', 'Scotland', 'Wales', 'Ireland']
            for pattern in uk_patterns:
                mask = df['host_location'].str.contains(pattern, case=False, na=False)
                df.loc[mask, 'host_location_grouped'] = 'UK'

            # European locations
            eu_patterns = ['France', 'Germany', 'Spain', 'Italy', 'Netherlands', 'Europe']
            for pattern in eu_patterns:
                mask = df['host_location'].str.contains(pattern, case=False, na=False)
                df.loc[mask, 'host_location_grouped'] = 'Europe'

            # US locations
            us_patterns = ['United States', 'USA', 'US', 'New York', 'California', 'America']
            for pattern in us_patterns:
                mask = df['host_location'].str.contains(pattern, case=False, na=False)
                df.loc[mask, 'host_location_grouped'] = 'US'

            df = df.drop(columns=['host_location'])
            df['host_location_grouped'] = df['host_location_grouped'].astype('object')
            self.created_features.append('host_location_grouped')
            self.dropped_features['high_cardinality'].append('host_location')
            print("Grouped host_location into major regions")

        # Handle host_neighbourhood (group or encode)
        if 'host_neighbourhood' in df.columns:
            # If too many unique values, group into top N + Other
            unique_count = df['host_neighbourhood'].nunique()
            if unique_count > 50:  # Arbitrary threshold
                top_neighbourhoods = df['host_neighbourhood'].value_counts().head(20).index
                df['host_neighbourhood_grouped'] = df['host_neighbourhood'].apply(
                    lambda x: x if x in top_neighbourhoods else 'Other'
                )
                df = df.drop(columns=['host_neighbourhood'])
                df['host_neighbourhood_grouped'] = df['host_neighbourhood_grouped'].fillna('Unknown').astype('object')
                self.created_features.append('host_neighbourhood_grouped')
                self.dropped_features['high_cardinality'].append('host_neighbourhood')
                print(f"Grouped host_neighbourhood into top 20 + Other (was {unique_count} unique)")
            else:
                df['host_neighbourhood'] = df['host_neighbourhood'].fillna('Unknown').astype('object')

        print(f"Dataset shape after feature transformation: {df.shape}")
        return df

    def reduce_collinearity(self, df, target_col='annual_revenue_adj'):
        """Identify and remove highly correlated features"""
        print("\n" + "="*50)
        print("STEP 3: Collinearity Analysis and Reduction")
        print("="*50)

        # Select only numerical columns (excluding target)
        if target_col in df.columns:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
        else:
            print(f"Warning: Target column '{target_col}' not found!")
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"Analyzing {len(numerical_cols)} numerical features for collinearity...")

        if len(numerical_cols) < 2:
            print("Not enough numerical features for collinearity analysis.")
            return df

        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]

                if corr_val > self.correlation_threshold:
                    high_corr_pairs.append((col1, col2, corr_val))

                    # Decide which column to drop based on predefined rules
                    to_drop_col = self._decide_column_to_drop(col1, col2)
                    to_drop.add(to_drop_col)

        print(f"Found {len(high_corr_pairs)} highly correlated pairs (threshold: {self.correlation_threshold})")
        for col1, col2, corr_val in high_corr_pairs[:10]:  # Show first 10
            print(f"  {col1} <-> {col2}: {corr_val:.3f}")
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs) - 10} more pairs")

        # Drop the identified columns
        to_drop_list = list(to_drop)
        to_drop_present = [col for col in to_drop_list if col in df.columns]

        if to_drop_present:
            df = df.drop(columns=to_drop_present)
            self.dropped_features['collinear'].extend(to_drop_present)
            print(f"Dropped {len(to_drop_present)} collinear features: {to_drop_present[:10]}{'...' if len(to_drop_present) > 10 else ''}")
        else:
            print("No features were dropped due to collinearity.")

        print(f"Dataset shape after collinearity reduction: {df.shape}")
        return df

    def _decide_column_to_drop(self, col1, col2):
        """Decide which of two correlated columns to drop based on predefined rules"""
        # Rules for dropping columns (prefer keeping more interpretable/engineered features)

        # Prefer log-transformed over original
        if 'log' in col1.lower() and 'log' not in col2.lower():
            return col2
        elif 'log' in col2.lower() and 'log' not in col1.lower():
            return col1

        # Prefer total over individual counts
        if 'total' in col1.lower() and 'total' not in col2.lower():
            return col2
        elif 'total' in col2.lower() and 'total' not in col1.lower():
            return col1

        # Prefer engineered features over raw features
        engineered_indicators = ['per_person', 'average', 'count', 'rank', 'distance']
        col1_engineered = any(indicator in col1.lower() for indicator in engineered_indicators)
        col2_engineered = any(indicator in col2.lower() for indicator in engineered_indicators)

        if col1_engineered and not col2_engineered:
            return col2
        elif col2_engineered and not col1_engineered:
            return col1

        # Default: drop the one that comes later alphabetically
        return col2 if col1 < col2 else col1

    def finalize_dataset(self, df, target_col='annual_revenue_adj'):
        """Final cleanup and validation"""
        print("\n" + "="*50)
        print("STEP 4: Final Dataset Preparation")
        print("="*50)

        # Ensure target column is present
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in final dataset!")

        # Handle any remaining missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Fill missing values in numerical columns with median
        for col in numerical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].median())
                print(f"Filled {missing_count} missing values in {col} with median")

        # Fill missing values in categorical columns with 'Unknown'
        for col in categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col] = df[col].fillna('Unknown')
                print(f"Filled {missing_count} missing values in {col} with 'Unknown'")

        # Final dataset summary
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Target column: {target_col}")
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Total features (excluding target): {len(numerical_cols) + len(categorical_cols)}")

        # Show sample of final columns
        print(f"\nFinal columns ({len(df.columns)}):")
        print(f"  Numerical: {numerical_cols[:15]}{'...' if len(numerical_cols) > 15 else ''}")
        print(f"  Categorical: {categorical_cols}")

        return df

    def print_summary(self):
        """Print summary of all transformations performed"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING SUMMARY")
        print("="*60)

        total_dropped = sum(len(v) for v in self.dropped_features.values())
        print(f"Total features dropped: {total_dropped}")
        print(f"Total features created: {len(self.created_features)}")

        for category, features in self.dropped_features.items():
            if features:
                print(f"\n{category.replace('_', ' ').title()} ({len(features)}):")
                print(f"  {features[:10]}{'...' if len(features) > 10 else ''}")

        if self.created_features:
            print(f"\nCreated Features ({len(self.created_features)}):")
            print(f"  {self.created_features}")

def main():
    """Main execution function"""
    # Configuration
    input_path = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset.pkl"
    output_path = "/home/karolito/IUM/data/processed/etap2/modeling/modeling_dataset_nn_ready.pkl"

    # Initialize preprocessor
    preprocessor = DataPreprocessor(correlation_threshold=0.9)

    try:
        # Load data
        df = preprocessor.load_data(input_path)

        # Apply preprocessing steps
        df = preprocessor.drop_initial_features(df)
        df = preprocessor.transform_features(df)
        df = preprocessor.reduce_collinearity(df)
        df = preprocessor.finalize_dataset(df)

        # Save processed dataset
        print(f"\nSaving processed dataset to: {output_path}")
        df.to_pickle(output_path)
        print("Dataset saved successfully!")

        # Print summary
        preprocessor.print_summary()

        # Validation check - try to load with neural network to ensure compatibility
        print("\n" + "="*60)
        print("COMPATIBILITY CHECK")
        print("="*60)

        # Quick check of data types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if 'annual_revenue_adj' in numerical_cols:
            numerical_cols.remove('annual_revenue_adj')

        print(f"✓ Target column 'annual_revenue_adj' present: {'annual_revenue_adj' in df.columns}")
        print(f"✓ Numerical features ready for scaling: {len(numerical_cols)}")
        print(f"✓ Categorical features ready for one-hot encoding: {len(categorical_cols)}")
        print(f"✓ No missing values in target: {df['annual_revenue_adj'].isnull().sum() == 0}")
        print(f"✓ Dataset ready for NeuralNetworkModel.load_and_preprocess_data()")

        print(f"\nDataset is ready! Use this path in your neural network:")
        print(f"'{output_path}'")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
