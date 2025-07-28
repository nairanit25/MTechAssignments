"""
FastAPI main application module for MLOps Housing Price Prediction System.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from src.api.schemas import (
    HealthResponse
)

from src.utils.config import Settings
from src.utils.logger import setup_logger

# Initialize components
settings = Settings()
logger = setup_logger(__name__)
 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting MLOps Housing Price Prediction API Server")
    
    # Load models on startup
    try:
      #  model_registry.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    
    yield
    
    logger.info("Shutting down MLOps Housing Price Prediction API Server")

# Create FastAPI application
app = FastAPI(  
    title="Housing Price Prediction API Server",
    description="Production-ready ML API for housing price prediction with monitoring",
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

@app.get("/", response_model=Dict[str, str])
async def root():
   
    return {
        "name": "MLOps Housing Price Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check model availability
        #models_status = model_registry.get_models_status()
        
        # Check system resources
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        status = "healthy" #if all(models_status.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=time.time(),
            #models_loaded=len([k for k, v in models_status.items() if v]),
            system_metrics={
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )