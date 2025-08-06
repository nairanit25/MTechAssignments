 
from typing import Dict, Optional, Any, List 
from src.utils.config import Settings
from src.utils.logger import setup_logger
from src.models.model_registry import ModelRegistry
 
 

logger = setup_logger(__name__ , log_file="./logs/model.log")

# Initialize components
settings = Settings()
model_registry = ModelRegistry()


def find_best_regression_model(model_name, no_recent_versions_to_consider: int = 20) -> Dict[str, Any]: 
    

    models_data = model_registry.get_model_info_by_model_metrics(model_name, no_recent_versions_to_consider)

    best_model_data = None
    best_r2 = -float('inf') # Initialize with negative infinity for R2 (maximize)
    best_rmse = float('inf') # Initialize with infinity for RMSE (minimize)
    
    if not models_data:
        logger.info(f"No Available models found in the registry: {model_name}")
        return None
    
    for model_info in models_data:
        metrics = model_info.get('metrics')
        if not metrics:
            print(f"Skipping model version {model_info.get('model_version', 'N/A')} due to missing metrics.")
            continue

        current_r2 = metrics.get('val_r2')
        current_rmse = metrics.get('val_rmse')

        # Ensure both metrics are present and are numbers
        if current_r2 is None or current_rmse is None: 
        #or not isinstance(current_r2, (int, float)) or \
        #not isinstance(current_rmse, (int, float)):
            print(f"Skipping model version {model_info.get('model_version', 'N/A')} due to invalid or missing val_r2/val_rmse.")
            continue

        # Prioritize higher R2, then lower RMSE for ties
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_rmse = current_rmse
            best_model = model_info
        elif current_r2 == best_r2:
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_model = model_info

        if best_model:
            print(f"\nBest model found:")
            print(f"  Model Name: {best_model.get('model_name', 'N/A')}")
            print(f"  Version: {best_model.get('model_version', 'N/A')}")
            print(f"  R2: {best_model['metrics'].get('val_r2', 'N/A')}")
            print(f"  RMSE: {best_model['metrics'].get('val_rmse', 'N/A')}")
        else:
            print("No best model could be determined from the provided data.")

    return best_model 
 
def register_best_model_info(lr_best_model: Dict[str, Any], dt_best_model: Dict[str, Any]):
    
    if dt_best_model is None and lr_best_model is not None:
        return lr_best_model
    elif dt_best_model is not None and lr_best_model is None:
        return dt_best_model
    elif dt_best_model is not None and lr_best_model is not None:
        lr_val_metrics = lr_best_model.metrics
        dt_best_model = dt_best_model.metrics
        
        lr_r2 = lr_val_metrics.get('val_r2', -float('inf'))
        lr_rmse = lr_val_metrics.get('val_rmse', float('inf'))

        dt_r2 = dt_best_model.get('val_r2', -float('inf'))
        dt_rmse = dt_best_model.get('val_rmse', float('inf'))

        if lr_r2 > dt_r2:
            return lr_best_model
        elif lr_r2 < dt_r2:
            return dt_best_model
        elif lr_r2 == dt_r2:
            if lr_rmse < dt_best_model:
                 return lr_best_model
            if lr_rmse > dt_best_model:
                return dt_best_model
            else:
                return lr_best_model


if __name__ == "__main__":
    MODEL_NAME = 'housing_price_predictor'
    best_model = find_best_regression_model(MODEL_NAME, no_recent_versions_to_consider=20)
    register_best_model_info(best_model, None)
    print(best_model)