#!/usr/bin/env python3
"""
run_pipeline.py - Complete pipeline runner
"""

import sys
import os
import logging
from datetime import datetime
import argparse

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import modules
from extract import extract
from transform import transform
from train import train_model
from forecast import forecast

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline(force=False):
    """Run the complete pipeline."""
    
    logger.info("=" * 60)
    logger.info("OPIOID FORECASTING PIPELINE")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    # Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    INPUT_FILE = os.path.join(DATA_DIR, 'opioid_rates.csv')
    PROCESSED_FILE = os.path.join(DATA_DIR, 'processed_opioid_rates.csv')
    MODEL_FILE = os.path.join(MODELS_DIR, 'prophet_model.pkl')
    FORECAST_FILE = os.path.join(DATA_DIR, 'forecasts.csv')
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Please place your data file at: data/opioid_rates.csv")
        logger.info("\nCreating sample data...")
        
        # Create sample data
        create_sample_data(INPUT_FILE)
    
    try:
        # Step 1: Extract
        logger.info("\n" + "-" * 40)
        logger.info("STEP 1: EXTRACTING DATA")
        logger.info("-" * 40)
        df = extract(INPUT_FILE)
        logger.info(f"✓ Extracted {len(df)} rows")
        
        # Step 2: Transform
        logger.info("\n" + "-" * 40)
        logger.info("STEP 2: TRANSFORMING DATA")
        logger.info("-" * 40)
        transformed_df = transform(INPUT_FILE, PROCESSED_FILE)
        logger.info(f"✓ Transformed data saved to {PROCESSED_FILE}")
        
        # Step 3: Train Model
        logger.info("\n" + "-" * 40)
        logger.info("STEP 3: TRAINING MODEL")
        logger.info("-" * 40)
        model = train_model(PROCESSED_FILE, MODEL_FILE)
        logger.info(f"✓ Model saved to {MODEL_FILE}")
        
        # Step 4: Generate Forecast
        logger.info("\n" + "-" * 40)
        logger.info("STEP 4: GENERATING FORECAST")
        logger.info("-" * 40)
        forecast_df = forecast(
            model_path=MODEL_FILE,
            data_path=PROCESSED_FILE,
            output_path=FORECAST_FILE,
            periods=12
        )
        logger.info(f"✓ Forecast saved to {FORECAST_FILE}")
        
        # Step 5: Summary
        logger.info("\n" + "=" * 60)
        logger.info(" PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Data file: {PROCESSED_FILE}")
        logger.info(f"Model file: {MODEL_FILE}")
        logger.info(f"Forecast file: {FORECAST_FILE}")
        logger.info(f"Completed at: {datetime.now()}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f" Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_data(output_file):
    """Create sample data if no data file exists."""
    import numpy as np
    import pandas as pd
    
    logger.info("Creating sample data...")
    
    years = list(range(2010, 2025))
    geo_codes = range(10)  # 10 regions
    
    data = []
    for year in years:
        for geo in geo_codes:
            # Generate realistic opioid rates
            base_rate = 5 + np.random.normal(0, 0.5)
            trend = (year - 2010) * 0.1
            seasonal = 0.3 * np.sin(year / 2)
            rate = max(0, base_rate + trend + seasonal + np.random.normal(0, 0.3))
            
            data.append({
                'Year': year,
                'Geo_Lvl': 0,  # National
                'Geo_Cd': geo,
                'Geo_Desc': f'Region_{geo}',
                'Plan_Type': 0,  # All Plans
                'Tot_Opioid_Clms': int(rate * 1000 + np.random.normal(0, 50)),
                'Tot_Clms': int(rate * 5000 + np.random.normal(0, 200)),
                'Opioid_Prscrbng_Rate': rate,
                'Opioid_Prscrbng_Rate_5Y_Chg': np.random.normal(0, 0.3),
                'Opioid_Prscrbng_Rate_1Y_Chg': np.random.normal(0, 0.2),
                'LA_Opioid_Prscrbng_Rate': rate * 1.1 + np.random.normal(0, 0.3),
                'LA_Opioid_Prscrbng_Rate_5Y_Chg': np.random.normal(0, 0.2),
                'LA_Opioid_Prscrbng_Rate_1Y_Chg': np.random.normal(0, 0.2),
            })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Created sample data with {len(df)} rows at {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run opioid forecasting pipeline')
    parser.add_argument('--force', action='store_true', help='Force re-run even if files exist')
    args = parser.parse_args()
    
    success = run_pipeline(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()