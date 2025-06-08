from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from datetime import datetime
import uuid
import pandas as pd
import numpy as np

class ABTestEnum(str, Enum):
    A = "A"
    B = "B"

class PredictionRequest(BaseModel):
    """Request model for revenue prediction"""

    ab_test: ABTestEnum = Field(..., description="A/B test variant ('A' or 'B')")
    latitude: float = Field(...,
                            description="Latitude of the listing",
                            ge=-90,
                            le=90)
    longitude: float = Field(...,
                             description="Longitude of the listing",
                             ge=-180,
                             le=180)
    distance_to_center: Optional[float] = Field(
        None, description="Distance to city center in km")

    accommodates: int = Field(
        ..., description="Number of guests the property accommodates", ge=1)
    bedrooms: Optional[float] = Field(None,
                                      description="Number of bedrooms",
                                      ge=0)
    bathrooms: Optional[float] = Field(None,
                                       description="Number of bathrooms",
                                       ge=0)

    price_log: Optional[float] = Field(
        None, description="Log-transformed nightly price")

    amenities_count: Optional[int] = Field(None,
                                           description="Number of amenities",
                                           ge=0)

    review_scores_rating: Optional[float] = Field(
        None, description="Overall review rating", ge=0, le=5)
    number_of_reviews: Optional[int] = Field(
        None, description="Total number of reviews", ge=0)
    review_scores_location: Optional[float] = Field(
        None, description="Location review score", ge=0, le=5)

    host_is_superhost: Optional[bool] = Field(
        None, description="Whether host is a superhost")
    host_days_active: Optional[float] = Field(
        None, description="Number of days host has been active", ge=0)
    host_response_rate: Optional[float] = Field(
        None, description="Host response rate", ge=0, le=1)
    calculated_host_listings_count: Optional[int] = Field(
        None, description="Number of listings by host", ge=1)

    instant_bookable: Optional[bool] = Field(
        None, description="Whether property is instantly bookable")
    minimum_nights: Optional[int] = Field(
        None, description="Minimum nights for booking", ge=1)

    neighbourhood_price_rank: Optional[float] = Field(
        None, description="Price rank within neighborhood")
    property_type_frequency: Optional[float] = Field(
        None, description="Frequency of property type in area")
    name_positive_sentiment: Optional[float] = Field(
        None, description="Sentiment score of listing name")

    neighbourhood_cleansed: Optional[str] = Field(
        None, description="Cleaned neighborhood name")

    property_type: Optional[str] = Field(
        None, description="The type of the property, e.g., 'Entire home/apt'")

    room_type: Optional[str] = Field(
        None, description="The type of the room, e.g., 'Entire home/apt'")

    class Config:
        schema_extra = {
            "example": {
                "latitude": 51.5074,
                "longitude": -0.1278,
                "distance_to_center": 2.5,
                "accommodates": 4,
                "bedrooms": 2.0,
                "bathrooms": 1.0,
                "price_log": 4.6,
                "amenities_count": 15,
                "review_scores_rating": 4.8,
                "number_of_reviews": 25,
                "review_scores_location": 4.9,
                "host_is_superhost": True,
                "host_days_active": 365.0,
                "host_response_rate": 0.95,
                "calculated_host_listings_count": 3,
                "instant_bookable": True,
                "minimum_nights": 2,
                "neighbourhood_price_rank": 0.7,
                "property_type_frequency": 0.3,
                "name_positive_sentiment": 0.8,
                "neighbourhood_cleansed": "Westminster",
                "property_type": "Entire home/apt",
                "room_type": "Entire home/apt"
            }
        }

    @validator('host_is_superhost', 'instant_bookable', pre=True)
    def convert_bool_strings(cls, v):
        """Convert string boolean values to actual booleans"""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 't')
        return v

    def to_dataframe(self, model_columns: List[str]) -> pd.DataFrame:
        """
        Converts request to a preprocessed DataFrame ready for the model.

        This method handles missing value imputation, one-hot encoding of
        categorical features, and alignment of columns to match the
        training data's structure.

        Args:
            model_columns: The list of feature names the model was trained on.

        Returns:
            A pandas DataFrame ready for scaling and prediction.
        """
        data = self.dict()

        defaults = {
            'distance_to_center': 10.0,
            'bedrooms': 1.0,
            'bathrooms': 1.0,
            'price_log': 4.5,
            'amenities_count': 10,
            'review_scores_rating': 4.5,
            'number_of_reviews': 5,
            'review_scores_location': 4.5,
            'host_is_superhost': False,
            'host_days_active': 100.0,
            'host_response_rate': 0.9,
            'calculated_host_listings_count': 1,
            'instant_bookable': True,
            'minimum_nights': 3,
            'neighbourhood_price_rank': 0.5,
            'property_type_frequency': 0.2,
            'name_positive_sentiment': 0.5,
            'neighbourhood_cleansed': 'Unknown',
            'property_type': 'Unknown',
            'room_type': 'Entire home/apt'
        }

        for key, default_value in defaults.items():
            if data.get(key) is None:
                data[key] = default_value

        # Create an empty DataFrame with all expected columns from the model
        aligned_df = pd.DataFrame(columns=model_columns,
                                  index=[0],
                                  dtype=np.float32).fillna(0)

        # Process numerical features directly
        numerical_features = [
            'latitude', 'longitude', 'distance_to_center', 'accommodates',
            'bedrooms', 'bathrooms', 'price_log', 'amenities_count',
            'review_scores_rating', 'number_of_reviews', 'host_days_active',
            'host_response_rate', 'review_scores_location', 'minimum_nights',
            'calculated_host_listings_count', 'name_positive_sentiment',
            'neighbourhood_price_rank', 'property_type_frequency'
        ]

        for col in numerical_features:
            if col in data and col in aligned_df.columns:
                aligned_df[col] = data[col]

        # Handle boolean features
        bool_cols = ['host_is_superhost', 'instant_bookable']
        for col in bool_cols:
            if col in data and col in aligned_df.columns:
                aligned_df[col] = float(data[col])

        # Handle categorical features by setting the appropriate one-hot encoded column
        categorical_mappings = {
            'neighbourhood_cleansed': 'neighbourhood_cleansed_',
            'property_type': 'property_type_',
            'room_type': 'room_type_'
        }

        for cat_col, prefix in categorical_mappings.items():
            if cat_col in data and data[cat_col]:
                encoded_col = f"{prefix}{data[cat_col]}"
                if encoded_col in aligned_df.columns:
                    aligned_df[encoded_col] = 1.0
                # If this specific encoding isn't in the model's columns, it's fine
                # The DataFrame already has zeros for all columns

        # Debug log to see shape before returning
        print(
            f"Prepared DataFrame with {len(aligned_df.columns)} columns. Model expects {len(model_columns)}"
        )
        print(f"DataFrame shape: {aligned_df.shape}")

        return aligned_df


class PredictionResponse(BaseModel):
    """Response model for revenue prediction"""

    predicted_revenue: float = Field(
        ..., description="Predicted annual revenue in USD")
    model_used: str = Field(
        ..., description="Model used for prediction (neural_net or baseline)")
    confidence_interval: Optional[Tuple[float, float]] = Field(
        None, description="95% confidence interval")
    prediction_id: str = Field(
        ..., description="Unique identifier for this prediction")
    timestamp: datetime = Field(..., description="Timestamp of prediction")

    class Config:
        schema_extra = {
            "example": {
                "predicted_revenue": 25000.50,
                "model_used": "neural_net",
                "confidence_interval": [20000.0, 30000.0],
                "prediction_id": "pred_123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-05-31T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Health status (healthy/unhealthy)")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    neural_net_available: bool = Field(
        ..., description="Whether neural network model is available")
    baseline_available: bool = Field(
        ..., description="Whether baseline model is available")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-05-31T10:30:00Z",
                "models_loaded": True,
                "neural_net_available": True,
                "baseline_available": True
            }
        }


class ABTestResponse(BaseModel):
    """Response model for A/B testing status"""

    current_split: Dict[str, float] = Field(...,
                                            description="Current split ratios")
    total_predictions: int = Field(
        ..., description="Total number of predictions made")
    neural_net_count: int = Field(
        ..., description="Number of neural net predictions")
    baseline_count: int = Field(...,
                                description="Number of baseline predictions")
    neural_net_percentage: float = Field(
        ..., description="Percentage of neural net predictions")
    baseline_percentage: float = Field(
        ..., description="Percentage of baseline predictions")

    class Config:
        schema_extra = {
            "example": {
                "current_split": {
                    "neural_net": 0.5,
                    "baseline": 0.5
                },
                "total_predictions": 1000,
                "neural_net_count": 500,
                "baseline_count": 500,
                "neural_net_percentage": 50.0,
                "baseline_percentage": 50.0
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None,
                                  description="Detailed error information")
    timestamp: datetime = Field(..., description="Timestamp of error")

    class Config:
        schema_extra = {
            "example": {
                "error": "Prediction failed",
                "detail": "Invalid input data format",
                "timestamp": "2025-05-31T10:30:00Z"
            }
        }
