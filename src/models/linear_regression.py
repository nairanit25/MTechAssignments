"""
Linear Regression model for housing price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel

class LinearRegressionModel(BaseModel):
        
    def __init__(self, name: str = "linear_regression", version: str = "1.0"):
        super().__init__(name, version, "linear_regression")
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **args) -> Dict[str, Any]:
        """Train the linear regression model."""
        # Get configuration
        regularization = args.get('regularization', 'none')
        alpha = args.get('alpha_value', 1.0)
        
        # Choose model based on regularization
        if regularization == 'ridge':
            model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            model = Lasso(alpha=alpha)
        else:
            model = LinearRegression()
        
        # Create pipeline with scaling
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Train the model
        self.model.fit(X, y)
        self.is_loaded = True
        
        # Update configuration
        self.config = {
            'regularization': regularization,
            'alpha': alpha,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # Calculate training metrics
        train_predictions = self.model.predict(X)

        training_metrics = {
            'train_mse': np.mean((y - train_predictions) ** 2),
            'train_mae': np.mean(np.abs(y - train_predictions)),
            'train_rmse': np.sqrt(np.mean((y - train_predictions) ** 2)),            
            'train_r2': self.model.score(X, y)
        }
        
        self.metrics.update(training_metrics)        
        return training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make a prediction using the linear regression model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Make prediction
        return self.model.predict(X)       
   
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")       
                
        # Get coefficients from the pipeline
        regressor = self.model.named_steps['regressor']
        coefficients = regressor.coef_
        
        return dict(zip(self.get_features(), coefficients))
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        regressor = self.model.named_steps['regressor']
        return float(regressor.intercept_)