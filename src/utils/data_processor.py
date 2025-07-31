"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logging

logger = logging.getLogger(__name__)

class DataLoader:
        
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def load_data(self, data_file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from file."""
        try:
            data_path = Path(data_file_path) 
            if data_path.suffix == '.csv':
                df = pd.read_csv(data_file_path)                
           
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            logger.info(f"Loaded data: {X.shape[0]} samples in Housing dataset with , {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to load Housing data from {data_path}: {e}")
            raise
   
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the housing data."""
        try:
            # Make a copy to avoid modifying original data
            housing_df = df.copy()
            
            logger.info(f" data: {X.shape[0]} samples in Housing dataset with , {X.shape[1]} features")

            logger.info("Null values in the Housing dataset: {housing_df.isnull().sum()}")
            logger.info("NA  records in the Housingdataset: {housing_df.isna().sum()}")

            housing_df = df.copy()
            housing_df.rename(columns={
                'HouseAge': 'housing_median_age',
                'MedInc': 'median_income',
                'AveRooms': 'avg_rooms_per_household',
                'AveBedrms': 'avg_num_bedrooms_per_house',
                'MedHouseVal': 'median_house_value',
                'AveOccup': 'avg_household_members'
            }, inplace=True)

            housing_df.head(2)
                       
            # Handle missing values
            housing_df = housing_df.fillna(housing_df.median(numeric_only=True))
            housing_df = housing_df.dropna()

            X = housing_df.drop(columns=['median_house_value'])
            y = housing_df['median_house_value']
            
            
            logger.info(f"Features present in the origional housing dataset: {X.columns}")
            logger.info(f"Target variable in the housing dataset: median_house_value")
            
            logger.info(f"Feature statistics: {get_feature_statistics(X)}")

            # Feature engineering
            '''
            
            # Select features for modeling
            feature_columns = [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                'lat', 'long', 'sqft_living15', 'sqft_lot15'
            ]
            
            # Ensure all feature columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    df[col] = 0
            
            X = df[feature_columns]
            
            # Handle target variable
            if 'price' in df.columns:
                y = df['price']
            else:
                # If no price column, create dummy target
                logger.warning("No price column found, creating dummy target")
                y = pd.Series(np.random.uniform(100000, 1000000, len(df)))
            
            # Handle outliers (simple approach)
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove extreme outliers
            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Preprocessed data: {len(X)} samples remaining after outlier removal")
            '''
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
    
    def validate_features(self, features: dict) -> bool:
        """Validate input features."""
        required_features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
            'lat', 'long', 'sqft_living15', 'sqft_lot15'
        ]
        
        # Check if all required features are present
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False
        
        # Basic validation
        try:
            if features['bedrooms'] < 1 or features['bedrooms'] > 20:
                return False
            if features['bathrooms'] < 0.5 or features['bathrooms'] > 10:
                return False
            if features['sqft_living'] < 300 or features['sqft_living'] > 15000:
                return False
            # Add more validations as needed
            
            return True
            
        except (TypeError, ValueError, KeyError):
            return False
    
    def get_feature_statistics(self, X: pd.DataFrame) -> dict:
        """Get basic statistics about features."""
        return {
            'shape': X.shape,
            'dtypes': X.dtypes.to_dict(),
            'missing_values': X.isnull().sum().to_dict(),
            'statistics': X.describe().to_dict()
        }