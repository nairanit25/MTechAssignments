"""
Main Application setup.
"""
import time 
import mlflow
 
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from src.utils.config import Settings
from src.utils.logger import setup_logger
import uvicorn
 
# Initialize components
settings = Settings()
logger = setup_logger(__name__ , log_file="./logs/api.log")

# Global variables for models and metrics
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting MLOps Housing Price Prediction Inference App")
    
    # Load models on startup
    try:
        # Load the best model from the MLflow Model Registry
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("linear_regression_housing_price_predictor", stages=["Production"])[0]
        model_uri = f"models:/{latest_version.name}/{latest_version.version}"
        
        # Load model using the appropriate flavor
        models["main_model"] = mlflow.sklearn.load_model(model_uri)
        models["main_model_info"] = {
            "name": latest_version.name,
            "version": latest_version.version,
            "uri": model_uri
        }
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    
    yield
    
    logger.info("Shutting down MLOps Housing Price Prediction API Server")

# Create FastAPI application
app = FastAPI(  
    title="Housing Price Prediction API Server",
    description="API Server for housing price prediction with monitoring",
    version="1.0",
    docs_url="/docs",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware to collect request metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time  
   
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Root endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
   
    return {
        "name": "MLOps Housing Price Prediction API Server",
        "version": "1.0",
        "status": "healthy",
        "docs": "/docs"
    }

from . import fast_server
if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )