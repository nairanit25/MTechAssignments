"""
Configuration management for the Housing Price Prediction system.
"""

import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./mlops_housing_price.db"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "housing_price_prediction"
    
    # Model Storage
    MODELS_DIR: str = "./models"
    
    # Data Configuration
    DATA_DIR: str = "./data"
    DEFAULT_DATA_FILE: str = "Housing.csv"
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    
    # Redis (for caching and queuing)
    REDIS_URL: str = "redis://localhost:6379"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/mlops_housing_price.log"
    
    # Model Training
    MAX_TRAINING_TIME: int = 3600  # 1 hour
    RETRAIN_THRESHOLD: float = 0.05  # 5% performance drop
    
    # API Keys (for external services)
    API_KEY: str = ""
    SECRET_KEY: str = "your-secret-key-here"
    
    # Feature Flags
    ENABLE_MODEL_SERVING: bool = True
    ENABLE_EXPERIMENT_TRACKING: bool = True
    ENABLE_MONITORING: bool = True
    ENABLE_AUTO_RETRAINING: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()