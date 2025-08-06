"""
Main Application setup.
"""
import time 
 
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware 
from src.utils.config import Settings
from src.utils.logger import setup_logger
from src.models.model_registry import ModelRegistry
from src.api.metrics_registry import prediction_counter, prediction_latency
from prometheus_client import make_asgi_app

import uvicorn
 

logger = setup_logger(__name__ , log_file="./logs/api.log")

# Initialize components
settings = Settings()
model_registry = ModelRegistry()


# Global variables for models and metrics
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting MLOps Housing Price Prediction Inference App")
    
    logger.info("Starting to load model from registry")
    load_model_for_inferencing()

    yield
    
    logger.info("Shutting down MLOps Housing Price Prediction API Server")

def load_model_for_inferencing():
    # Load models on app startup
    try:
        MODEL_NAME = 'housing_price_predictor'
        # list all the models from registry
        list_models = model_registry.list_mlflow_registered_models()
        
        if not list_models:
            logger.info(f"No Available models found in the registry: {list_models}")
        else:
            logger.info(f"Available models in the registry: {list_models}")
            
            # Load the best model from the MLflow Model Registry
            latest_model_ver = model_registry.get_lastest_model(model_name=MODEL_NAME)
            logger.info(f"latest model version available in registry: {latest_model_ver}")
            
            model_loaded = model_registry.load_model(model_name=MODEL_NAME, model_version=latest_model_ver.version) 
            logger.info(f" loading model: {MODEL_NAME} from registry: {model_loaded}")

            models["main_model"] = model_loaded 
            models["main_model_info"] = {
                "name": latest_model_ver.name,
                "version": latest_model_ver.version,
                "uri": "model_uri"
            } 
            logger.info(f"Model {MODEL_NAME} loaded successfully")
            logger.info(f"loaded model info : {models["main_model_info"]}")
            
    except Exception as e:
        logger.error(f"Failed to load models: {e}")


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

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    prediction_counter.inc()
    with prediction_latency.time():
        response = await call_next(request)
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
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

from . import fast_server
if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )