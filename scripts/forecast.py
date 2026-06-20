"""
forecast.py - Forecasting module
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import json
from datetime import datetime, timedelta

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


def load_model(model_path):
    """Load trained Prophet model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {model_path}")
    return model


def load_data(data_path):
    """Load processed data."""
    df = pd.read_csv(data_path)
    df['ds'] = pd.to_datetime(df['ds'])
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    return df


def prepare_future_data(df, periods=12):
    """Prepare future dataframe for prediction."""
    logger.info(f"Preparing future data for {periods} periods...")
    
    # Get last date
    last_date = df['ds'].max()
    logger.info(f"Last historical date: {last_date}")
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + timedelta(days=365),
        periods=periods,
        freq='YS'
    )
    
    # Create future dataframe
    future = pd.DataFrame({'ds': future_dates})
    
    # Use last known ratio or calculate from historical pattern
    last_ratio = df['Raw_Opioid_Ratio'].iloc[-1]
    future['Raw_Opioid_Ratio'] = last_ratio
    
    logger.info(f"Created future data with {len(future)} rows")
    logger.info(f"Future date range: {future['ds'].min()} to {future['ds'].max()}")
    
    return future


def generate_forecast(model, future):
    """Generate forecasts."""
    logger.info("Generating forecast...")
    forecast = model.predict(future)
    logger.info(f"Generated {len(forecast)} forecasts")
    return forecast


def save_forecast(forecast, output_path):
    """Save forecast results."""
    # Select columns
    output_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    output = forecast[output_cols].copy()
    output['year'] = output['ds'].dt.year
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)
    logger.info(f"Saved forecasts to {output_path}")
    
    return output


def forecast(model_path, data_path, output_path, periods=12):
    """Main forecasting function."""
    logger.info("Starting forecast generation...")
    
    try:
        # Load model and data
        model = load_model(model_path)
        df = load_data(data_path)
        
        # Prepare future data
        future = prepare_future_data(df, periods)
        
        # Generate forecast
        forecast_df = generate_forecast(model, future)
        
        # Save results
        output = save_forecast(forecast_df, output_path)
        
        # Save forecast report
        report = {
            'generated_at': datetime.now().isoformat(),
            'forecast_periods': len(forecast_df),
            'forecast_start': forecast_df['ds'].min().isoformat(),
            'forecast_end': forecast_df['ds'].max().isoformat(),
            'summary': {
                'mean_forecast': forecast_df['yhat'].mean(),
                'min_forecast': forecast_df['yhat'].min(),
                'max_forecast': forecast_df['yhat'].max(),
                'mean_upper': forecast_df['yhat_upper'].mean(),
                'mean_lower': forecast_df['yhat_lower'].mean()
            }
        }
        
        report_path = model_path.replace('.pkl', '_forecast_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        logger.info(f"Saved forecast report to {report_path}")
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("FORECAST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Forecast period: {forecast_df['ds'].min().year} - {forecast_df['ds'].max().year}")
        logger.info(f"Number of forecasts: {len(forecast_df)}")
        logger.info(f"Mean forecast: {forecast_df['yhat'].mean():.4f}")
        logger.info(f"Min forecast: {forecast_df['yhat'].min():.4f}")
        logger.info(f"Max forecast: {forecast_df['yhat'].max():.4f}")
        logger.info("=" * 50)
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise


if __name__ == "__main__":
    try:
        # Construct paths relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        forecasts = forecast(
            model_path=os.path.join(script_dir, '..', 'models', 'prophet_model.pkl'),
            data_path=os.path.join(script_dir, '..', 'data', 'processed_opioid_rates.csv'),
            output_path=os.path.join(script_dir, '..', 'data', 'forecasts.csv'),
            periods=12
        )
        print("\n Forecasting completed successfully!")
        print(f"\nLatest forecast:")
        print(forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    except Exception as e:
        print(f" Forecasting failed: {e}")