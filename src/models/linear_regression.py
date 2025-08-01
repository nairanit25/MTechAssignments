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
    
    def train(self, X: pd.DataFrame, y: pd.Series, **args) -> Dict[str, Any]:
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
            'train_rmse': np.sqrt(np.mean((y - train_predictions) ** 2)),
            'train_mae': np.mean(np.abs(y - train_predictions)),
            'train_r2': self.model.score(X, y)
        }
        
        self.metrics.update(training_metrics)
        
        return training_metrics
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Make a prediction using the linear regression model."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Preprocess features
        X = self._preprocess_features(features)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return float(prediction)
    
    def predict_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on model certainty."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # For linear regression, we can use the R² score as a proxy for confidence
        # In a real scenario, you might want to use prediction intervals
        base_confidence = self.metrics.get('train_r2', 0.0)
        
        # Adjust confidence based on feature values (simple heuristic)
        X = self._preprocess_features(features)
        
        # Calculate distance from training data mean (if available)
        # This is a simplified confidence calculation
        confidence = min(max(base_confidence * 0.9, 0.1), 0.95)
        
        return float(confidence)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        feature_names = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
            'lat', 'long', 'sqft_living15', 'sqft_lot15'
        ]
        
        # Get coefficients from the pipeline
        regressor = self.model.named_steps['regressor']
        coefficients = regressor.coef_
        
        return dict(zip(feature_names, coefficients))
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        regressor = self.model.named_steps['regressor']
        return float(regressor.intercept_)