

import os
from typing import Dict, Optional, Any, List
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient

from src.models.linear_regression import LinearRegressionModel
from src.models.decision_tree import DecisionTreeModel

from src.utils.config import Settings
from src.utils.logger import setup_logger
from datetime import datetime

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
            'linear_regression': LinearRegressionModel,
            'decision_tree': DecisionTreeModel
        }
    
    def list_mlflow_registered_models(self, max_results=100):
        """List all MLFlow registered models."""
        try:
            models = self.mlflow_client.search_registered_models(max_results=max_results)
            if not models:
                logger.info("No registered models found.")
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
            logger.info(f"Successfully retrieved details for model '{model_name}' version {version}.")
            return model_version
            
        except Exception as e:
            logger.error(f"Error retrieving model version: {e}")
            return None
        
    def get_lastest_model(self, model_name: str):
        try:
            model_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                logger.info(f"No versions found for model '{model_name}'.")
                return None

            latest_version = max(model_versions, key=lambda mv: int(mv.version))
            logger.info(f"Found latest_version name = {latest_version.name} : latest version: {latest_version.version} for model '{model_name}'.")
           
            return latest_version
            
        except Exception as e:
            logger.error(f"Error retrieving model version: {e}")
            return None
   

    def load_model(self, model_name, model_version: Optional[int] = None):

        try:
            # Search for all model versions with the given name
            model_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            if not model_versions:
                logger.info(f"No versions found for model '{model_name}'.")
                return None
                   
            if model_version is None:
                latest_version = max(model_versions, key=lambda mv: int(mv.version))
                logger.info(f"Found latest_version.name = {latest_version.name} : latest version: {latest_version.version} for model '{model_name}'.")
                model_uri = f"models:/{latest_version.name}/{latest_version.version}"
            else:
                spec_model_ver = self.get_model_version(model_name, model_version)
                model_uri = f"models:/{spec_model_ver.name}/{model_version}"
            
                logger.info(f"model_uri : {model_uri}")

            model = mlflow.sklearn.load_model(model_uri)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def  get_model_info_by_model_metrics(self, model_name, no_recent_versions_to_consider: int = 20) :
            recent_models_info: List[Dict[str, Any]] = []
            
            try:                
                
                all_models = self.mlflow_client.search_model_versions(f"name='{model_name}'")

                if not all_models:
                    logger.info("No registered models found.")
                    return recent_models_info
                
                # 2. Sort the versions by their version number in descending order        
                sorted_versions = sorted(all_models, key=lambda mv: int(mv.version), reverse=True)
                
                # 3. Get the top '10' most recent versions
                top_n_versions = sorted_versions[:no_recent_versions_to_consider]

                 # 4. For each of the top 'top_n_versions' versions, retrieve its run details and metrics
                for mv in top_n_versions:
                    if mv.run_id:
                        try:
                            run = self.mlflow_client.get_run(mv.run_id)  
                            recent_models_info.append({
                                "model_name": mv.name,
                                "model_version": mv.version,
                                "run_id": mv.run_id,
                                "metrics": run.data.metrics  
                            })
                        except Exception as error:
                            logger.error(f"Could not retrieve run {mv.run_id} or its metrics for version {mv.version}: {error}")
                    else:
                        logger.info(f"Model version {mv.version} has no associated run_id.")

                if recent_models_info:
                    logger.info(f"\nSuccessfully retrieved {len(recent_models_info)} most recent versions for '{model_name}'.")
                    return recent_models_info    

            except Exception as e:
                logger.error(f"Error fetching model details: {e}")
                return recent_models_info
    

    def register_model(self, target_model_name, ml_model_to_be_registered, model_data: Dict[str, Any]):
        artifact_path = "model"
        registered_target_model_name = target_model_name
        mlflow.set_experiment("MLOps_Housing_Price_Prediction_Best_Model_Registration_Experiment")

        with mlflow.start_run(run_name=f"Registering_{registered_target_model_name}") as run:
            logger.info(f"Logging and registering model under new name '{registered_target_model_name}'...")
        
            try:
                # Log the model and register it under the new name 
                model_registered = mlflow.sklearn.log_model(
                    sk_model=ml_model_to_be_registered,
                    artifact_path=artifact_path,
                    registered_model_name=registered_target_model_name,
                )

                logger.info(f"Registered Model Name: {registered_target_model_name}, Experiment Run Info (model-run-id): {model_registered.run_id}")
                logger.info(f"Registered Model version: {model_registered.registered_model_version}, model_uri Info: {model_registered.model_uri}") 
                logger.info(f"Registered Model Param: {model_registered.params}")

                tags = {
                    "dataset": "california-housing",
                    "optimization_framework": "optuna",
                    "best_model_type": model_data.get("model_name"),
                    "run_id": run.info.run_id,
                    "registration_date": datetime.now().isoformat(),
                }

                 # Iterate through the dictionary and add each tag individually
                for key, value in tags.items():
                    self.mlflow_client.set_model_version_tag(
                        name=registered_target_model_name,
                        version=model_registered.registered_model_version,
                        key=key,
                        value=value
                )

            except Exception as e:
                logger.error(f"Error registering model with new name '{registered_target_model_name}': {e}")
                return None
        
       
        
        

        