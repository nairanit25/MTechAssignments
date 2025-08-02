"""
Pydantic schemas for API request/response validation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import time

class CaliforniaHousing(BaseModel):
    """Housing features for prediction."""
    median_income: float = Field(..., gt=0, description="Median income in block group")
    housing_median_age: float = Field(..., gt=0, description="Median house age in block group")
    avg_rooms_per_household: float = Field(..., ge=1, le=20, description="Average number of rooms per household")
    avg_num_bedrooms_per_house: float = Field(...,  ge=1, le=20,  description="Average number of bedrooms per household")
    Population: float = Field(..., ge=1, description="Block group population")
    avg_household_members: float = Field(..., ge=32.0, le=42.0, description="Average number of household members")
    Latitude: float = Field(..., description="LatBlock group latitudeitude (California range)")
    Longitude: float = Field(..., ge=-124.0, le=-114.0, description="Block group longitude (California range)")     

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