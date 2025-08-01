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
            
            self.perform_eda(df)

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
            
            logger.info(f"Feature statistics: {self.get_feature_statistics(X)}")

            #Handle Outliers

            # Possible Feature engineering
                #Avg_Rooms_ per_Person
                #Population_Density = Population / avg_household_members -> Higher values may indicate urban areas
                #Urban_proximity
            
            '''
            
                     
            
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

    def get_feature_statistics(self, X: pd.DataFrame) -> dict:
        """Get basic statistics about features."""
        return {
            'shape': X.shape,
            'dtypes': X.dtypes.to_dict(),
            'missing_values': X.isnull().sum().to_dict(),
            'statistics': X.describe().to_dict()
        }
    
    def perform_eda(self, df: pd.DataFrame):
        from ydata_profiling import ProfileReport
        housing_report=ProfileReport(df)
        housing_report.to_file("./logs/housing_report.html")
