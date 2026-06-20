"""
extract.py - Data extraction module
"""

import pandas as pd
import logging
import os
from datetime import datetime

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'pipeline.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract(input_path):
    """
    Extract raw opioid prescribing rate data from CSV file.
    
    Args:
        input_path (str): Path to the raw CSV file
        
    Returns:
        pd.DataFrame: DataFrame with raw data
    """
    try:
        logger.info(f"Starting extraction from: {input_path}")
        
        # Check if file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Load data with proper parsing
        df = pd.read_csv(input_path, low_memory=False)
        
        logger.info(f"Successfully extracted {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Date range: {df['Year'].min()} to {df['Year'].max()}")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise


if __name__ == "__main__":
    # Test extraction
    try:
        # Construct path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'data', 'opioid_rates.csv')
        df = extract(data_path)
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nData info:")
        print(df.info())
    except Exception as e:
        print(f"Error: {e}")