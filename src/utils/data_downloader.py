import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

import logging

logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Fetch the California Housing dataset
housing = fetch_california_housing()

# Convert to a pandas DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Save the DataFrame to a CSV file
df.to_csv('data1/california_housing.csv', index=False)

logger.info("Dataset downloaded and saved to data/california_housing.csv")