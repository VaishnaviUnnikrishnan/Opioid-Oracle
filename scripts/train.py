"""
train.py - Model training module
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import json
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from datetime import datetime

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


def prepare_prophet_data(df):
    """Prepare data for Prophet model."""
    logger.info("Preparing data for Prophet...")
    
    # Select required columns
    prophet_df = df[["ds", "y", "Raw_Opioid_Ratio"]].copy()
    
    # Fill any remaining missing values
    prophet_df = prophet_df.fillna(prophet_df.median())
    
    # Sort by date
    prophet_df = prophet_df.sort_values("ds")
    
    logger.info(f"Prophet data shape: {prophet_df.shape}")
    logger.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    
    return prophet_df


def train_test_split(df, test_year=2023):
    """Split data into train and test sets."""
    train = df[df["ds"].dt.year < test_year]
    test = df[df["ds"].dt.year >= test_year]
    
    logger.info(f"Train size: {len(train)} (pre-{test_year})")
    logger.info(f"Test size: {len(test)} ({test_year}+)")
    
    return train, test


def build_model():
    """Build and configure Prophet model."""
    logger.info("Building Prophet model...")
    
    model = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        interval_width=0.95
    )
    
    # Add regressor
    model.add_regressor('Raw_Opioid_Ratio', standardize='auto')
    
    logger.info("Model configured with Raw_Opioid_Ratio regressor")
    return model


def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'r2': r2_score(y_true, y_pred)
    }
    
    metrics['accuracy'] = 100 - metrics['mape']
    
    return metrics


def train_model(data_path, model_path):
    """Main training function."""
    logger.info(f"Loading data from {data_path}")
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        df['ds'] = pd.to_datetime(df['ds'])
        logger.info(f"Loaded {len(df)} rows")
        
        # Prepare data
        prophet_df = prepare_prophet_data(df)
        
        # Split data
        train, test = train_test_split(prophet_df)
        
        # Build and train model
        model = build_model()
        logger.info("Training Prophet model...")
        model.fit(train)
        logger.info("Model training completed")
        
        # Make predictions
        logger.info("Making predictions on test set...")
        forecast = model.predict(test)
        
        # Evaluate
        y_true = test["y"]
        y_pred = forecast["yhat"]
        metrics = evaluate_model(y_true, y_pred)
        
        # Log metrics
        logger.info("\n" + "=" * 50)
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info("=" * 50)
        for key, value in metrics.items():
            if key == 'accuracy':
                logger.info(f"{key:>15}: {value:.2f}%")
            elif 'mape' in key:
                logger.info(f"{key:>15}: {value:.2f}%")
            else:
                logger.info(f"{key:>15}: {value:.4f}")
        logger.info("=" * 50)
        
        # Create results
        results = pd.DataFrame({
            'ds': test['ds'],
            'actual': y_true,
            'predicted': y_pred,
            'error': y_true - y_pred,
            'abs_error': abs(y_true - y_pred)
        })
        
        # Save results
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = model_path.replace('.pkl', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save results
        results_path = model_path.replace('.pkl', '_results.csv')
        results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Save training info
        info = {
            'train_size': len(train),
            'test_size': len(test),
            'train_start': train['ds'].min().isoformat(),
            'train_end': train['ds'].max().isoformat(),
            'test_start': test['ds'].min().isoformat(),
            'test_end': test['ds'].max().isoformat(),
            'features': ['Raw_Opioid_Ratio'],
            'model_type': 'Prophet',
            'training_date': datetime.now().isoformat()
        }
        
        info_path = model_path.replace('.pkl', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4, default=str)
        logger.info(f"Training info saved to {info_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    try:
        # Construct paths relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'data', 'processed_opioid_rates.csv')
        model_path = os.path.join(script_dir, '..', 'models', 'prophet_model.pkl')
        model = train_model(data_path, model_path)
        print("\n Model training completed successfully!")
    except Exception as e:
        print(f" Training failed: {e}")