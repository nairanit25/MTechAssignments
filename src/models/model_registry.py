

import os
import time
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient

from src.models.linear_regression import LinearRegressionModel 
  
from src.utils.config import Settings
from src.utils.logger import setup_logger


# Setup logging
logger = setup_logger(__name__, log_file="./logs/models.log")

class ModelRegistry:
    """Model registry for managing ML models."""
    
    def __init__(self):
        self.settings = Settings()
        self.models: Dict[str, Any] = {}        
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.settings.MLFLOW_TRACKING_URI)
        self.mlflow_client = MlflowClient()
        
        # Model classes mapping
        self.model_classes = {
            'linear_regression': LinearRegressionModel
        }

    def load_models(self) -> None:
         self._load_best_model_from_mlflow()

    def _load_best_model_from_mlflow(self) -> None:
        """Load the best model from MLflow registry."""
        try:
            registered_models = self.mlflow_client.search_registered_models()

            if not registered_models:
                print("No registered models found.")
                return
 

            # Get the latest version by querying the model versions directly.
            model_versions = self.mlflow_client.search_model_versions(f"name='housing_price_predictor'")

            # Sort the versions to find the one with the highest version number.
            latest_version = max(model_versions, key=lambda mv: mv.version)
            model_uri = f"models:/{latest_version.name}/{latest_version.version}"
            print(f"Loading latest version: {latest_version.version} from model_uri {model_uri}")

            '''
            # Get the latest version of the best model
            latest_versions = self.mlflow_client.get_latest_versions(
                name="housing_price_predictor", stages=None
            )
            print(latest_versions)
            
            if latest_versions:
                # Use the production model if available, otherwise staging
                best_version = None
                for version in latest_versions:
                    if version.current_stage == "Production":
                        best_version = version
                        break
                    elif version.current_stage == "Staging" and not best_version:
                        best_version = version
                
                if best_version:
                    # Load model from MLflow
                    model_uri = f"models:/housing_price_predictor/{best_version.version}"
                    sklearn_model = mlflow.sklearn.load_model(model_uri)
                    
                    # Create wrapper model
                    wrapper_model = MLflowModelWrapper(
                        name="best",
                        sklearn_model=sklearn_model,
                        version=best_version.version
                    )
                    
                    self.models["best"] = wrapper_model
                    logger.info(f"Loaded best model from MLflow: v{best_version.version}")
            '''
        except Exception as e:
            logger.warning(f"Could not load model from MLflow: {e}")
    
    def list_mlflow_registered_models(self, max_results=100):
        """List all MLFlow registered models."""
        try:
            models = self.mlflow_client.search_registered_models(max_results=max_results)
            if not models:
                print("No registered models found.")
                return []
            
            return [
                {
                    "name": model.name,
                    "description": model.description,
                    "tags": model.tags,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "creation_timestamp": version.creation_timestamp,
                            "status": version.status
                        }
                        for version in model.latest_versions
                    ]
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Error listing registered models: {str(e)}")
            return []

    def get_model_version(self, model_name: str, version: int):
        try:
            model_version = self.mlflow_client.get_model_version(name=model_name, version=version)
            print(f"Successfully retrieved details for model '{model_name}' version {version}.")
            return model_version
            
        except Exception as e:
            print(f"Error retrieving model version: {e}")
            return None
        
    def get_lastest_model(self, model_name: str):
        try:
            model_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                print(f"No versions found for model '{model_name}'.")
                return None

            latest_version = max(model_versions, key=lambda mv: int(mv.version))
            print(f"Found latest_version.name = {latest_version.name} : latest version: {latest_version.version} for model '{model_name}'.")
           
            return latest_version
            
        except Exception as e:
            print(f"Error retrieving model version: {e}")
            return None
   

    def load_model(self, model_name, model_version: Optional[int] = None):

        try:
            # Search for all model versions with the given name
            model_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                print(f"No versions found for model '{model_name}'.")
                return None
                   
            if model_version is None:
                latest_version = max(model_versions, key=lambda mv: int(mv.version))
                print(f"Found latest_version.name = {latest_version.name} : latest version: {latest_version.version} for model '{model_name}'.")
                model_uri = f"models:/{latest_version.name}/{latest_version.version}"
            else:
                spec_model_ver = self.get_model_version(model_name, model_version)
                model_uri = f"models:/{spec_model_ver.name}/{model_version}"
            
            print(f"model_uri : {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None