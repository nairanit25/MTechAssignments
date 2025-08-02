"""
Base model class for ML models in the MLOps system.
"""

import abc
import time
import joblib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BaseModel(abc.ABC):
    """Abstract base class for ML models."""
    
    def __init__(
        self,
        name: str,
        version: str = "1.0",
        algorithm: str = "BaseModel"
    ):
        self.name = name
        self.version = version
        self.algorithm = algorithm
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.created_at = time.time()
        self.metrics = {}
        self.config = {}
        
    @abc.abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **args) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    '''
    @abc.abstractmethod
    def predict(self, features: Dict[str, Any]) -> float:
        """Make a prediction."""
        pass  
    '''
    
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make a prediction."""
        pass

    def evaluate_model(self, X_eval: pd.DataFrame, y_actual: pd.DataFrame, type='test') -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        y_pred = self.predict(X_eval)
                
        metrics = {
            type+'_mse': mean_squared_error(y_actual, y_pred),
            type+'_rmse': np.sqrt(mean_squared_error(y_actual, y_pred)),
            type+'_mae': mean_absolute_error(y_actual, y_pred),
            type+'_r2': r2_score(y_actual, y_pred),
            type+'_mape': np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        }
        
        self.metrics.update(metrics)
        return metrics

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        predictions = []
        for _, row in X.iterrows():
            pred = self.predict(row.to_dict())
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def save(self, model_path: str) -> None:
        """Save model to disk."""
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model:
            joblib.dump(self.model, model_path / self.name+"_model.pkl")
        
        # Save preprocessor
        if self.preprocessor:
            joblib.dump(self.preprocessor, model_path / self.name+"_preprocessor.pkl")
        
        # Save metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'algorithm': self.algorithm,
            'created_at': self.created_at,
            'metrics': self.metrics,
            'config': self.config
        }
        
        with open(model_path / self.name+"_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, model_path: str) -> None:
        """Load model from disk."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        # Load model
        model_file = model_path / self.name+"_model.pkl"
        if model_file.exists():
            self.model = joblib.load(model_file)
        
        # Load preprocessor
        preprocessor_file = model_path / self.name+"_preprocessor.pkl"
        if preprocessor_file.exists():
            self.preprocessor = joblib.load(preprocessor_file)
        
        # Load metadata
        metadata_file = model_path / self.name+"_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.name = metadata.get('name', self.name)
            self.version = metadata.get('version', self.version)
            self.algorithm = metadata.get('algorithm', self.algorithm)
            self.created_at = metadata.get('created_at', self.created_at)
            self.metrics = metadata.get('metrics', {})
            self.config = metadata.get('config', {})
        
        self.is_loaded = True
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': self.name,
            'version': self.version,
            'algorithm': self.algorithm,
            'is_loaded': self.is_loaded,
            'created_at': self.created_at,
            'metrics': self.metrics,
            'config': self.config
        }
    
    def _preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for prediction."""
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Apply preprocessor if available
        if self.preprocessor:
            return self.preprocessor.transform(df)
        
        # Basic preprocessing - convert to numpy array
        feature_order = [
            'median_income', 'housing_median_age', 'avg_rooms_per_household', 'avg_num_bedrooms_per_house', 
            'Population', 'avg_household_members', 'Latitude', 'Longitude'
        ]
        
        return df[feature_order].values
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update model configuration."""
        self.config.update(config)
