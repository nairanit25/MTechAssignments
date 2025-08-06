"""
A FastAPI Server for prediction and health checks.
"""

import numpy as np
import psutil
import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.api.app import app, models, logger 

router = APIRouter() 

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.debug("health_check called successfully")

    try:        
        # Check model availability
        models_status = {name: model is not None for name, model in models.items()}
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        status = "healthy" if all(models_status.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=time.time(),
            models_loaded= len([k for k, v in models_status.items() if v]),
            system_metrics={
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts the housing price based on input features.
    """

    start_time = time.time()
    logger.info(f"predict called successfully: Request payload {request.features} ") 

    if "main_model" not in models or models["main_model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    try:
        features_list = [
            request.features.median_income,
            request.features.housing_median_age,
            request.features.avg_rooms_per_household,
            request.features.avg_num_bedrooms_per_house,
            request.features.Population,
            request.features.avg_household_members,
            request.features.Latitude,
            request.features.Longitude
        ]

        # Convert request data to a numpy array for the model
        features = np.array(features_list).reshape(1, -1)
        prediction = models["main_model"].predict(features)[0]
        
        logger.info(f"predict Response: {prediction}")

        # Total request execution time
        execution_time = time.time() - start_time

        response = PredictionResponse(
            prediction=float(prediction), 
            processing_time=execution_time,
            timestamp=time.time()
        )

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    
app.include_router(router)
