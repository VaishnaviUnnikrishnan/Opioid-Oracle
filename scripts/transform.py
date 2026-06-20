"""
transform.py - Data transformation and feature engineering
"""

import pandas as pd
import numpy as np
import logging
import os

# Setup logging
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


def clean_data(df):
    """Clean raw data."""
    logger.info("Cleaning data...")
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicates")
    
    return df


def handle_missing_values(df):
    """Handle missing values."""
    logger.info("Handling missing values...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Fill numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {col} with median: {median_val:.4f}")
    
    return df


def encode_categorical(df):
    """Encode categorical columns."""
    logger.info("Encoding categorical columns...")
    
    categorical_cols = ['Geo_Lvl', 'Plan_Type', 'Geo_Cd']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
            logger.info(f"Encoded {col}")
    
    return df


def filter_national_data(df):
    """Filter to National level and All Plans."""
    logger.info("Filtering to National + All Plans...")
    initial_rows = len(df)
    
    # Filter for National level (Geo_Lvl = 0) and All Plans (Plan_Type = 0)
    df_filtered = df[
        (df["Geo_Lvl"].astype(str) == "0") &
        (df["Plan_Type"].astype(str) == "0")
    ].copy()
    
    logger.info(f"Filtered from {initial_rows} to {len(df_filtered)} rows")
    return df_filtered


def engineer_features(df):
    """
    Engineer time series features for Prophet.
    """
    logger.info("Engineering features...")
    
    # Create datetime column
    df["ds"] = pd.to_datetime(df["Year"].astype(str))
    logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # Set target variable
    df["y"] = df["LA_Opioid_Prscrbng_Rate"]
    
    # Calculate opioid claim ratio
    df["Raw_Opioid_Ratio"] = df["Tot_Opioid_Clms"] / df["Tot_Clms"]
    
    # Handle infinite values
    df["Raw_Opioid_Ratio"] = df["Raw_Opioid_Ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Impute ratio
    ratio_median = df["Raw_Opioid_Ratio"].median()
    df["Raw_Opioid_Ratio"] = df["Raw_Opioid_Ratio"].fillna(ratio_median)
    logger.info(f"Imputed Raw_Opioid_Ratio with median: {ratio_median:.6f}")
    
    # Sort for time series operations
    df = df.sort_values(["Geo_Cd", "Year"])
    
    # Create lag features
    for lag in [1, 2, 3]:
        df[f"LA_Lag_{lag}"] = (
            df.groupby("Geo_Cd")["LA_Opioid_Prscrbng_Rate"].shift(lag)
        )
    
    # Create rolling mean
    df["LA_Rolling_Mean_3Y"] = (
        df.groupby("Geo_Cd")["LA_Opioid_Prscrbng_Rate"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    
    # Fill lag nulls
    lag_cols = ["LA_Lag_1", "LA_Lag_2", "LA_Lag_3"]
    for col in lag_cols:
        df[col] = df[col].fillna(df[col].median())
        logger.info(f"Imputed {col} with median: {df[col].median():.4f}")
    
    # Calculate growth rate
    df["Growth"] = df["LA_Opioid_Prscrbng_Rate"].pct_change() * 100
    
    # Additional features
    df["Year_Num"] = df["Year"]
    
    logger.info(f"Final feature set: {list(df.columns)}")
    return df


def transform(input_path, output_path):
    """
    Main transform function.
    """
    logger.info(f"Starting transformation from {input_path}")
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Apply transformations
        df = clean_data(df)
        df = handle_missing_values(df)
        df = encode_categorical(df)
        df = filter_national_data(df)
        df = engineer_features(df)
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"Final shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Transform failed: {e}")
        raise


if __name__ == "__main__":
    # Test transformation
    try:
        # Construct paths relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, '..', 'data', 'opioid_rates.csv')
        output_path = os.path.join(script_dir, '..', 'data', 'processed_opioid_rates.csv')
        df = transform(input_path, output_path)
        print(f"\nTransformed data shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df[['ds', 'y', 'Raw_Opioid_Ratio']].head())
    except Exception as e:
        print(f"Error: {e}")