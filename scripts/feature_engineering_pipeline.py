import json
import os
import re
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for Airbnb listings data.

    This pipeline implements the strategy derived from extensive EDA analysis,
    handling missing values, feature transformations, and creating new features.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the feature engineering pipeline.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.feature_stats = {}
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

        self.log_messages = []

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration or use defaults."""
        default_config = {
            "missing_value_thresholds": {
                "drop_feature": 80,
                "high_priority": 30,
                "medium_priority": 10,
            },
            "outlier_treatment": {
                "price_upper_percentile": 99,
                "price_lower_percentile": 1,
                "apply_log_transform": True,
            },
            "categorical_encoding": {
                "neighbourhood_min_frequency": 10,
                "property_type_min_frequency": 50,
                "max_categories": 50,
            },
            "feature_creation": {
                "create_amenity_clusters": True,
                "create_location_clusters": True,
                "create_text_features": True,
                "create_temporal_features": True,
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        print(log_entry)

    def fit_transform(
        self,
        listings_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame] = None,
        reviews_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Main method to perform complete feature engineering.

        Args:
            listings_df: Main listings dataframe
            calendar_df: Calendar data (optional, for temporal features)
            reviews_df: Reviews data (optional, for review-based features)

        Returns:
            Transformed dataframe ready for modeling
        """
        self.log("🚀 Starting Feature Engineering Pipeline")
        self.log(f"Input data shape: {listings_df.shape}")

        df = listings_df.copy()

        df = self._initial_data_assessment(df)

        df = self._handle_missing_values(df)

        df = self._handle_outliers(df)

        df = self._apply_feature_transformations(df)

        df = self._encode_categorical_features(df)

        df = self._create_derived_features(df)

        if calendar_df is not None:
            df = self._create_temporal_features(df, calendar_df)

        df = self._create_text_features(df)

        df = self._create_geographic_features(df)

        df = self._scale_features(df)

        df = self._final_feature_validation(df)

        self.log(f"✅ Feature engineering complete. Final shape: {df.shape}")
        return df

    def _initial_data_assessment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform initial data assessment and basic cleaning."""
        self.log("📊 Performing initial data assessment")

        self.feature_stats["original_shape"] = df.shape
        self.feature_stats["original_columns"] = list(df.columns)

        df = self._fix_data_types(df)

        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            self.log(f"Removed {len(empty_cols)} completely empty columns")

        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            self.log(f"Removed {initial_rows - len(df)} duplicate rows")

        return df

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix common data type issues."""
        self.log("🔧 Fixing data types")

        price_columns = ["price", "price_cleaned"]
        for col in price_columns:
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = (
                        df[col].astype(str).str.replace(r"[\$,£€]", "", regex=True)
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        percentage_columns = ["host_response_rate", "host_acceptance_rate"]
        for col in percentage_columns:
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.replace("%", "")
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        boolean_columns = [
            "host_is_superhost",
            "host_identity_verified",
            "instant_bookable",
        ]
        for col in boolean_columns:
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].map({"t": True, "f": False})

        numeric_columns = [
            "accommodates",
            "bedrooms",
            "beds",
            "minimum_nights",
            "maximum_nights",
            "availability_365",
            "number_of_reviews",
            "calculated_host_listings_count",
            "reviews_per_month",
            "latitude",
            "longitude",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        review_columns = [col for col in df.columns if "review_scores" in col]
        for col in review_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on EDA insights."""
        self.log("🔄 Handling missing values")

        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_stats[col] = {"count": missing_count, "percentage": missing_pct}

        if "price" in df.columns or "price_cleaned" in df.columns:
            price_col = "price_cleaned" if "price_cleaned" in df.columns else "price"
            df = self._impute_price(df, price_col)

        if "beds" in df.columns:
            df = self._impute_beds(df)

        host_features = ["host_response_rate", "host_acceptance_rate"]
        for col in host_features:
            if col in df.columns:
                df = self._impute_host_features(df, col)

        review_columns = [col for col in df.columns if "review_scores" in col]
        if review_columns:
            df = self._impute_review_scores(df, review_columns)

        if "bedrooms" in df.columns:
            df = self._impute_bedrooms(df)

        low_missing_features = {
            "host_is_superhost": "mode",
            "bathrooms_text": "mode",
            "host_identity_verified": "mode",
        }

        for col, strategy in low_missing_features.items():
            if col in df.columns and df[col].isnull().any():
                if strategy == "mode":
                    mode_value = (
                        df[col].mode().iloc[0] if not df[col].mode().empty else False
                    )
                    df[col] = df[col].fillna(mode_value)
                    self.log(f"Filled {col} missing values with mode: {mode_value}")

        return df

    def _impute_price(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Sophisticated price imputation based on similar listings."""
        self.log(f"🏷️ Imputing {price_col} using KNN based on similar listings")

        price_features = [
            "accommodates",
            "bedrooms",
            "beds",
            "room_type",
            "property_type",
            "neighbourhood_cleansed",
            "latitude",
            "longitude",
        ]

        available_features = [col for col in price_features if col in df.columns]

        if not available_features:
            df[price_col] = df.groupby("room_type")[price_col].transform(
                lambda x: x.fillna(x.median())
            )
            return df

        impute_df = df[available_features + [price_col]].copy()

        categorical_cols = impute_df.select_dtypes(include=["object"]).columns
        le_dict = {}

        for col in categorical_cols:
            if col != price_col:
                le = LabelEncoder()
                impute_df[col] = le.fit_transform(impute_df[col].astype(str))
                le_dict[col] = le

        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(impute_df)

        price_col_idx = impute_df.columns.get_loc(price_col)
        df[price_col] = imputed_data[:, price_col_idx]

        self.imputers[price_col] = {
            "imputer": imputer,
            "encoders": le_dict,
            "features": available_features,
        }

        return df

    def _impute_beds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute beds based on accommodates and bedrooms."""
        self.log("🛏️ Imputing beds using accommodates and bedrooms relationship")

        missing_mask = df["beds"].isnull()

        if "accommodates" in df.columns and "bedrooms" in df.columns:
            estimated_beds = np.maximum(
                df.loc[missing_mask, "accommodates"].fillna(2),
                df.loc[missing_mask, "bedrooms"].fillna(1) * 2,
            )
            df.loc[missing_mask, "beds"] = estimated_beds

        elif "accommodates" in df.columns:
            df.loc[missing_mask, "beds"] = df.loc[missing_mask, "accommodates"]

        else:
            df["beds"] = df["beds"].fillna(df["beds"].median())

        return df

    def _impute_host_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Impute host features based on host characteristics."""
        self.log(f"👤 Imputing {col} based on host characteristics")

        missing_mask = df[col].isnull()

        if "host_is_superhost" in df.columns and "property_type" in df.columns:
            df[col] = df.groupby(["host_is_superhost", "property_type"])[col].transform(
                lambda x: x.fillna(x.median())
            )
        elif "host_is_superhost" in df.columns:
            df[col] = df.groupby("host_is_superhost")[col].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            df[col] = df[col].fillna(df[col].median())

        return df

    def _impute_review_scores(
        self, df: pd.DataFrame, review_columns: List[str]
    ) -> pd.DataFrame:
        """Multivariate imputation for review scores (they're missing together)."""
        self.log("⭐ Imputing review scores using multivariate approach")

        review_df = df[review_columns].copy()

        imputer = KNNImputer(n_neighbors=10)
        imputed_reviews = imputer.fit_transform(review_df)

        for i, col in enumerate(review_columns):
            df[col] = imputed_reviews[:, i]

        self.imputers["review_scores"] = imputer

        return df

    def _impute_bedrooms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute bedrooms based on accommodates and room_type."""
        self.log("🏠 Imputing bedrooms using accommodates and room_type")

        missing_mask = df["bedrooms"].isnull()

        if "accommodates" in df.columns and "room_type" in df.columns:
            for room_type in df["room_type"].unique():
                room_mask = (df["room_type"] == room_type) & missing_mask

                if room_type == "Shared room":
                    df.loc[room_mask, "bedrooms"] = 0
                elif room_type == "Private room":
                    df.loc[room_mask, "bedrooms"] = 1
                else:
                    df.loc[room_mask, "bedrooms"] = np.maximum(
                        1, (df.loc[room_mask, "accommodates"] / 2).round()
                    )

        df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers based on EDA insights."""
        self.log("📊 Handling outliers")

        price_cols = ["price", "price_cleaned"]
        for col in price_cols:
            if col in df.columns:
                df = self._handle_price_outliers(df, col)

        accommodation_cols = [
            "accommodates",
            "bedrooms",
            "beds",
            "minimum_nights",
            "maximum_nights",
        ]
        for col in accommodation_cols:
            if col in df.columns:
                df = self._cap_outliers(
                    df, col, lower_percentile=1, upper_percentile=99
                )

        return df

    def _handle_price_outliers(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Handle price outliers with sophisticated approach."""
        self.log(f"💰 Handling {price_col} outliers")

        df = df[df[price_col] > 0]

        lower_bound = df[price_col].quantile(
            self.config["outlier_treatment"]["price_lower_percentile"] / 100
        )
        upper_bound = df[price_col].quantile(
            self.config["outlier_treatment"]["price_upper_percentile"] / 100
        )

        df[price_col] = df[price_col].clip(lower=lower_bound, upper=upper_bound)

        if self.config["outlier_treatment"]["apply_log_transform"]:
            df[f"{price_col}_log"] = np.log1p(df[price_col])

        return df

    def _cap_outliers(
        self,
        df: pd.DataFrame,
        col: str,
        lower_percentile: int = 1,
        upper_percentile: int = 99,
    ) -> pd.DataFrame:
        """Cap outliers using percentile method."""
        lower_bound = df[col].quantile(lower_percentile / 100)
        upper_bound = df[col].quantile(upper_percentile / 100)

        original_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        if original_outliers > 0:
            self.log(f"Capped {original_outliers} outliers in {col}")

        return df

    def _apply_feature_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply mathematical transformations to features."""
        self.log("🔄 Applying feature transformations")

        if "beds" in df.columns and "accommodates" in df.columns:
            df["beds_per_person"] = df["beds"] / df["accommodates"].replace(0, 1)

        if "bedrooms" in df.columns and "accommodates" in df.columns:
            df["bedrooms_per_person"] = df["bedrooms"] / df["accommodates"].replace(
                0, 1
            )

        if "number_of_reviews" in df.columns and "reviews_per_month" in df.columns:
            df["total_review_months"] = df["number_of_reviews"] / df[
                "reviews_per_month"
            ].replace(0, 1)

        review_cols = [
            col
            for col in df.columns
            if "review_scores" in col and col != "review_scores_rating"
        ]
        if len(review_cols) > 0:
            df["review_scores_average"] = df[review_cols].mean(axis=1)
            df["review_scores_std"] = df[review_cols].std(axis=1)

        if "availability_365" in df.columns:
            df["availability_rate"] = df["availability_365"] / 365
            df["availability_category"] = pd.cut(
                df["availability_365"],
                bins=[0, 30, 90, 180, 365],
                labels=["Low", "Medium", "High", "Very High"],
            )

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features with smart strategies."""
        self.log("🏷️ Encoding categorical features")

        high_cardinality_features = {
            "neighbourhood_cleansed": self.config["categorical_encoding"][
                "neighbourhood_min_frequency"
            ],
            "property_type": self.config["categorical_encoding"][
                "property_type_min_frequency"
            ],
        }

        for col, min_freq in high_cardinality_features.items():
            if col in df.columns:
                df = self._encode_high_cardinality(df, col, min_freq)

        medium_cardinality_features = ["room_type", "host_response_time"]
        for col in medium_cardinality_features:
            if col in df.columns:
                df = self._one_hot_encode(df, col)

        binary_features = [
            "host_is_superhost",
            "host_identity_verified",
            "instant_bookable",
        ]
        for col in binary_features:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    def _encode_high_cardinality(
        self, df: pd.DataFrame, col: str, min_frequency: int
    ) -> pd.DataFrame:
        """Encode high cardinality categorical features."""
        self.log(f"🔢 Encoding high cardinality feature: {col}")

        value_counts = df[col].value_counts()
        df[f"{col}_frequency"] = df[col].map(value_counts)

        top_categories = value_counts[value_counts >= min_frequency].index.tolist()
        df[f"{col}_grouped"] = df[col].apply(
            lambda x: x if x in top_categories else "Other"
        )

        df = self._one_hot_encode(df, f"{col}_grouped", drop_original=True)

        self.encoders[col] = {
            "top_categories": top_categories,
            "value_counts": value_counts.to_dict(),
        }

        return df

    def _one_hot_encode(
        self, df: pd.DataFrame, col: str, drop_original: bool = True
    ) -> pd.DataFrame:
        """Apply one-hot encoding to a categorical column."""
        if col not in df.columns:
            return df

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)

        if drop_original:
            df = df.drop(columns=[col])

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new derived features based on domain knowledge."""
        self.log("✨ Creating derived features")

        if "calculated_host_listings_count" in df.columns:
            df["host_experience_level"] = pd.cut(
                df["calculated_host_listings_count"],
                bins=[0, 1, 3, 10, float("inf")],
                labels=["New", "Casual", "Experienced", "Professional"],
            )

        if "accommodates" in df.columns:
            df["property_size"] = pd.cut(
                df["accommodates"],
                bins=[0, 2, 4, 8, float("inf")],
                labels=["Small", "Medium", "Large", "XLarge"],
            )

        if "number_of_reviews" in df.columns and "reviews_per_month" in df.columns:
            df["review_intensity"] = np.where(
                df["reviews_per_month"] > df["reviews_per_month"].median(),
                "High",
                "Low",
            )

        if "neighbourhood_cleansed" in df.columns and "price" in df.columns:
            price_col = "price_cleaned" if "price_cleaned" in df.columns else "price"
            df["neighbourhood_price_rank"] = df.groupby("neighbourhood_cleansed")[
                price_col
            ].rank(pct=True)
            df["price_competitiveness"] = pd.cut(
                df["neighbourhood_price_rank"],
                bins=[0, 0.33, 0.67, 1.0],
                labels=["Budget", "Mid-range", "Premium"],
            )

        return df

    def _create_temporal_features(
        self, df: pd.DataFrame, calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create temporal features from calendar data."""
        self.log("📅 Creating temporal features from calendar data")

        calendar_features = (
            calendar_df.groupby("listing_id")
            .agg(
                {
                    "available": ["mean", "sum"],
                    "price_cleaned": ["mean", "median", "std"]
                    if "price_cleaned" in calendar_df.columns
                    else "mean",
                }
            )
            .round(2)
        )

        calendar_features.columns = [
            "_".join(col).strip() for col in calendar_features.columns
        ]
        calendar_features = calendar_features.reset_index()

        if "date" in calendar_df.columns:
            calendar_df["day_of_week"] = pd.to_datetime(
                calendar_df["date"]
            ).dt.day_name()
            calendar_df["is_weekend"] = (
                pd.to_datetime(calendar_df["date"]).dt.weekday >= 5
            )

            weekend_stats = (
                calendar_df.groupby("listing_id")
                .agg({"is_weekend": "mean"})
                .reset_index()
            )
            weekend_stats.columns = ["listing_id", "weekend_availability_rate"]

            calendar_features = calendar_features.merge(
                weekend_stats, on="listing_id", how="left"
            )

        if "id" in df.columns:
            df = df.merge(
                calendar_features, left_on="id", right_on="listing_id", how="left"
            )

        return df

    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from text columns."""
        self.log("📝 Creating text-based features")

        if "name" in df.columns:
            df["name_length"] = df["name"].str.len().fillna(0)
            df["name_word_count"] = df["name"].str.split().str.len().fillna(0)

            positive_words = [
                "amazing",
                "beautiful",
                "perfect",
                "luxury",
                "stunning",
                "cozy",
            ]
            negative_words = ["basic", "simple", "budget"]

            df["name_positive_sentiment"] = (
                df["name"].str.lower().str.contains("|".join(positive_words), na=False)
            )
            df["name_negative_sentiment"] = (
                df["name"].str.lower().str.contains("|".join(negative_words), na=False)
            )

        if "description" in df.columns:
            df["description_length"] = df["description"].str.len().fillna(0)
            df["description_word_count"] = (
                df["description"].str.split().str.len().fillna(0)
            )

        if "amenities" in df.columns:
            df = self._extract_amenities_features(df)

        return df

    def _extract_amenities_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract detailed amenities features."""
        self.log("🛋️ Extracting amenities features")

        df["amenities_count"] = df["amenities"].str.count(",") + 1
        df["amenities_count"] = df["amenities_count"].fillna(0)

        key_amenities = {
            "wifi": ["WiFi", "Wifi", "wifi", "Internet"],
            "kitchen": ["Kitchen", "kitchen"],
            "air_conditioning": ["Air conditioning", "AC", "air conditioning"],
            "pool": ["Pool", "pool", "Swimming pool"],
            "parking": ["Free parking", "Parking", "parking"],
            "tv": ["TV", "television", "Cable TV"],
            "washer": ["Washer", "washer", "Washing machine"],
            "dryer": ["Dryer", "dryer"],
            "elevator": ["Elevator", "elevator", "Lift"],
            "gym": ["Gym", "gym", "Fitness"],
        }

        for amenity_name, amenity_terms in key_amenities.items():
            pattern = "|".join(amenity_terms)
            df[f"has_{amenity_name}"] = df["amenities"].str.contains(
                pattern, case=False, na=False
            )

        luxury_amenities = ["pool", "air_conditioning", "gym", "elevator"]
        basic_amenities = ["wifi", "kitchen", "tv"]

        df["luxury_amenities_count"] = sum(
            df[f"has_{amenity}"] for amenity in luxury_amenities
        )
        df["basic_amenities_count"] = sum(
            df[f"has_{amenity}"] for amenity in basic_amenities
        )

        return df

    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic and location-based features."""
        self.log("🗺️ Creating geographic features")

        if "latitude" in df.columns and "longitude" in df.columns:
            city_center_lat, city_center_lon = 51.5074, -0.1278

            df["distance_to_center"] = (
                np.sqrt(
                    (df["latitude"] - city_center_lat) ** 2
                    + (df["longitude"] - city_center_lon) ** 2
                )
                * 111
            )

            if len(df) > 50:
                coords = df[["latitude", "longitude"]].dropna()
                if len(coords) > 10:
                    kmeans = KMeans(
                        n_clusters=min(20, len(coords) // 50), random_state=42
                    )
                    clusters = kmeans.fit_predict(coords)

                    df.loc[coords.index, "geographic_cluster"] = clusters
                    df["geographic_cluster"] = df["geographic_cluster"].fillna(-1)

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features for modeling."""
        self.log("⚖️ Scaling numerical features")

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        exclude_cols = ["id", "listing_id", "host_id"] + [
            col for col in numerical_cols if "log" in col.lower()
        ]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if numerical_cols:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[numerical_cols].fillna(0))

            for i, col in enumerate(numerical_cols):
                df[f"{col}_scaled"] = scaled_features[:, i]

            self.scalers["numerical"] = {"scaler": scaler, "columns": numerical_cols}

        return df

    def _final_feature_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup of features."""
        self.log("✅ Final feature validation")

        missing_threshold = self.config["missing_value_thresholds"]["drop_feature"]

        cols_to_drop = []
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > missing_threshold:
                cols_to_drop.append(col)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.log(
                f"Dropped {len(cols_to_drop)} features with >{missing_threshold}% missing values"
            )

        constant_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() == 0:
                constant_cols.append(col)

        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.log(f"Dropped {len(constant_cols)} constant features")

        self.feature_stats["final_shape"] = df.shape
        self.feature_stats["final_columns"] = list(df.columns)
        self.feature_stats["numerical_features"] = list(
            df.select_dtypes(include=[np.number]).columns
        )
        self.feature_stats["categorical_features"] = list(
            df.select_dtypes(include=["object", "category"]).columns
        )

        return df

    def save_pipeline_artifacts(self, output_dir: str):
        """Save pipeline artifacts for future use."""
        os.makedirs(output_dir, exist_ok=True)

        with open(
            os.path.join(output_dir, "feature_engineering_config.json"), "w"
        ) as f:
            json.dump(self.config, f, indent=2)

        with open(os.path.join(output_dir, "feature_statistics.json"), "w") as f:
            json.dump(self.feature_stats, f, indent=2, default=str)

        with open(os.path.join(output_dir, "feature_engineering_log.txt"), "w") as f:
            f.write("\n".join(self.log_messages))

        encoder_info = {}
        for name, encoder_data in self.encoders.items():
            if isinstance(encoder_data, dict):
                encoder_info[name] = {
                    k: v for k, v in encoder_data.items() if not callable(v)
                }

        with open(os.path.join(output_dir, "encoders_info.json"), "w") as f:
            json.dump(encoder_info, f, indent=2, default=str)

        self.log(f"Pipeline artifacts saved to {output_dir}")

    def generate_feature_report(self) -> Dict[str, Any]:
        """Generate a comprehensive feature engineering report."""
        report = {
            "pipeline_summary": {
                "original_shape": self.feature_stats.get("original_shape"),
                "final_shape": self.feature_stats.get("final_shape"),
                "features_created": self.feature_stats.get("final_shape", [0])[1]
                - self.feature_stats.get("original_shape", [0])[1],
                "processing_steps": len(self.log_messages),
            },
            "feature_categories": {
                "numerical_features": len(
                    self.feature_stats.get("numerical_features", [])
                ),
                "categorical_features": len(
                    self.feature_stats.get("categorical_features", [])
                ),
                "total_features": len(self.feature_stats.get("final_columns", [])),
            },
            "missing_value_handling": {
                "imputers_used": list(self.imputers.keys()),
                "encoding_strategies": list(self.encoders.keys()),
            },
            "new_features_created": [
                "Feature ratios (beds_per_person, bedrooms_per_person)",
                "Review score composites (average, std)",
                "Availability categories",
                "Host experience levels",
                "Property size categories",
                "Geographic clusters",
                "Distance to city center",
                "Amenity features",
                "Text-based features",
                "Temporal features (if calendar data provided)",
            ],
            "recommendations": [
                "Monitor feature importance in downstream modeling",
                "Consider additional domain-specific features",
                "Validate geographic clustering results",
                "Review text feature effectiveness",
                "Consider interaction terms for important features",
            ],
        }

        return report


def main():
    """Example usage of the feature engineering pipeline."""

    pipeline = FeatureEngineeringPipeline()

    print("Loading data...")
    data_dir = "data/processed/etap2/"

    try:
        listings_df = pd.read_pickle(os.path.join(data_dir, "listings_e2_df.pkl"))
        calendar_df = pd.read_pickle(os.path.join(data_dir, "calendar_e2_df.pkl"))
        print(f"Loaded listings: {listings_df.shape}")
        print(f"Loaded calendar: {calendar_df.shape}")

        print("\nApplying feature engineering pipeline...")
        processed_df = pipeline.fit_transform(listings_df, calendar_df)

        output_dir = "data/processed/etap2/feature_engineered/"
        os.makedirs(output_dir, exist_ok=True)

        processed_df.to_pickle(
            os.path.join(output_dir, "listings_feature_engineered.pkl")
        )
        processed_df.to_csv(
            os.path.join(output_dir, "listings_feature_engineered.csv"), index=False
        )

        pipeline.save_pipeline_artifacts(output_dir)

        report = pipeline.generate_feature_report()
        with open(
            os.path.join(output_dir, "feature_engineering_report.json"), "w"
        ) as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✅ Feature engineering complete!")
        print(f"Original shape: {report['pipeline_summary']['original_shape']}")
        print(f"Final shape: {report['pipeline_summary']['final_shape']}")
        print(f"Features created: {report['pipeline_summary']['features_created']}")
        print(f"Output saved to: {output_dir}")

        print(f"\nSample of processed features:")
        print(processed_df.head())

        print(f"\nFeature types:")
        print(f"Numerical: {report['feature_categories']['numerical_features']}")
        print(f"Categorical: {report['feature_categories']['categorical_features']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure data files exist in the specified directory.")


if __name__ == "__main__":
    main()
