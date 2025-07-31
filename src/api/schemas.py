"""
Pydantic schemas for API request/response validation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import time

class CaliforniaHousing(BaseModel):
    """Housing features for prediction."""
    bedrooms: int = Field(..., ge=1, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0.5, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=300, le=15000, description="Living area square footage")
    sqft_lot: int = Field(..., ge=500, le=100000, description="Lot size square footage")
    floors: float = Field(..., ge=1, le=4, description="Number of floors")
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront property (0/1)")
    view: int = Field(..., ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Property condition (1-5)")
    grade: int = Field(..., ge=3, le=13, description="Building grade (3-13)")
    sqft_above: int = Field(..., ge=300, le=15000, description="Above ground square footage")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Basement square footage")
    yr_built: int = Field(..., ge=1900, le=2025, description="Year built")
    yr_renovated: int = Field(..., ge=0, le=2025, description="Year renovated (0 if never)")
    zipcode: int = Field(..., ge=98001, le=98199, description="Property zipcode")
    lat: float = Field(..., ge=47.1, le=47.8, description="Latitude")
    long: float = Field(..., ge=-122.6, le=-121.3, description="Longitude")
    sqft_living15: int = Field(..., ge=300, le=15000, description="Living area of 15 nearest neighbors")
    sqft_lot15: int = Field(..., ge=500, le=100000, description="Lot size of 15 nearest neighbors")

    median_income: confloat(gt=0) = Field(..., description="Median income in block group")
    housing_median_age: conint(ge=0, le=100) = Field(..., description="Median house age in block group")
    avg_rooms_per_household: confloat(gt=0) = Field(..., description="Average number of rooms per household")
    avg_num_bedrooms_per_house: confloat(gt=0) = Field(..., description="Average number of bedrooms per household")
    Population: conint(ge=0) = Field(..., description="Block group population")
    avg_household_members: confloat(gt=0) = Field(..., description="Average number of household members")
    Latitude: confloat(ge=32.0, le=42.0) = Field(..., description="LatBlock group latitudeitude (California range)")
    Longitude: confloat(ge=-124.0, le=-114.0) = Field(..., description="Block group longitude (California range)")
      

    @validator('yr_renovated')
    def validate_renovation_year(cls, v, values):
        """Validate renovation year is after build year."""
        if v > 0 and 'yr_built' in values and v < values['yr_built']:
            raise ValueError('Renovation year cannot be before build year')
        return v

class PredictionRequest(BaseModel):
    """Request schema for housing price prediction."""
    features: CaliforniaHousing = Field(..., description="Housing features for prediction")
    model_preference: Optional[str] = Field(None, description="Preferred model name")
    return_confidence: bool = Field(True, description="Whether to return confidence score")
    
class PredictionResponse(BaseModel):
    """Response schema for housing price prediction."""
    prediction: float = Field(..., description="Predicted housing price")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Version of the model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: float = Field(default_factory=time.time, description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Overall system status")
    timestamp: float = Field(default_factory=time.time, description="Health check timestamp")
    models_loaded: int = Field(..., description="Number of loaded models")
    system_metrics: Dict[str, float] = Field(..., description="System performance metrics")

class ModelInfo(BaseModel):
    """Model information schema."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    algorithm: str = Field(..., description="ML algorithm used")
    status: str = Field(..., description="Model status (active/inactive)")
    created_at: float = Field(..., description="Model creation timestamp")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")

class RetrainRequest(BaseModel):
    """Request schema for model retraining."""
    model_name: str = Field(..., description="Name of model to retrain")
    data_path: Optional[str] = Field(None, description="Path to training data")
    config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")
    force_retrain: bool = Field(False, description="Force retraining even if recent")

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")

class MetricsResponse(BaseModel):
    """Metrics response schema."""
    model_name: str = Field(..., description="Model name")
    total_predictions: int = Field(..., description="Total number of predictions made")
    average_processing_time: float = Field(..., description="Average processing time")
    prediction_distribution: Dict[str, int] = Field(..., description="Distribution of predictions")
    error_rate: float = Field(..., description="Error rate percentage")
    last_updated: float = Field(..., description="Last metrics update timestamp")