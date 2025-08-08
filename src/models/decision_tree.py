"""
Decision Tree model implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """Decision Tree model for housing price prediction."""
    
    def __init__(self, name: str = "decision_tree", version: str = "1.0.0"):
        super().__init__(name, version, "decision_tree")
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the decision tree model."""
        # Get hyperparameters
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        max_features = kwargs.get('max_features', None)
        random_state = kwargs.get('random_state', 42)
        
        # Create and train model
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        
        self.model.fit(X, y)
        self.is_loaded = True
        
        # Update configuration
        self.config = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'actual_max_depth': self.model.get_depth(),
            'n_leaves': self.model.get_n_leaves()
        }
        
        # Calculate training metrics
        train_predictions = self.model.predict(X)
        training_metrics = {
            'train_mse': np.mean((y - train_predictions) ** 2),
            'train_rmse': np.sqrt(np.mean((y - train_predictions) ** 2)),
            'train_mae': np.mean(np.abs(y - train_predictions)),
            'train_r2': self.model.score(X, y)
        }
        
        self.metrics.update(training_metrics)
        
        return training_metrics
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Make a prediction using the decision tree model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Preprocess features
        X = self._preprocess_features(features)
        
        # Make prediction
        prediction = self.model.predict(X)
        
        return prediction
    
    def predict_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on leaf statistics."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Preprocess features
        X = self._preprocess_features(features)
        
        # Get the leaf that the sample falls into
        leaf_id = self.model.apply(X)[0]
        
        # Calculate confidence based on number of samples in leaf
        # This is a simplified approach - in practice you might want to
        # store training statistics for each leaf
        base_confidence = self.metrics.get('train_r2', 0.0)
        
        # Adjust confidence (simple heuristic)
        confidence = min(max(base_confidence * 0.85, 0.1), 0.90)
        
        return float(confidence)