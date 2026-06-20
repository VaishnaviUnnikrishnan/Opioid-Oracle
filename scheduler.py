"""
scheduler.py - Simple scheduler for production
"""

import schedule
import time
import subprocess
import logging
import os
import sys
from datetime import datetime

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'scheduler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Run the pipeline."""
    logger.info("=" * 50)
    logger.info(f"Starting scheduled pipeline run at {datetime.now()}")
    logger.info("=" * 50)
    
    try:
        # Get the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        pipeline_script = os.path.join(project_root, 'run_pipeline.py')
        
        # Check if pipeline script exists
        if not os.path.exists(pipeline_script):
            logger.error(f"Pipeline script not found: {pipeline_script}")
            return False
        
        # Run the pipeline
        result = subprocess.run(
            [sys.executable, pipeline_script],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(" Pipeline completed successfully")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f" Pipeline failed with code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            return False
        
    except subprocess.TimeoutExpired:
        logger.error(" Pipeline timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f" Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline_alternative():
    """
    Alternative: Import and run pipeline functions directly.
    This avoids subprocess issues.
    """
    logger.info("=" * 50)
    logger.info(f"Starting scheduled pipeline run (direct) at {datetime.now()}")
    logger.info("=" * 50)
    
    try:
        # Add scripts to path
        scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
        sys.path.append(scripts_dir)
        
        # Import modules
        from extract import extract
        from transform import transform
        from train import train_model
        from forecast import forecast
        
        # Configuration
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        
        INPUT_FILE = os.path.join(DATA_DIR, 'opioid_rates.csv')
        PROCESSED_FILE = os.path.join(DATA_DIR, 'processed_opioid_rates.csv')
        MODEL_FILE = os.path.join(MODELS_DIR, 'prophet_model.pkl')
        FORECAST_FILE = os.path.join(DATA_DIR, 'forecasts.csv')
        
        # Check if input file exists
        if not os.path.exists(INPUT_FILE):
            logger.error(f"Input file not found: {INPUT_FILE}")
            return False
        
        # Run pipeline steps
        logger.info("Step 1: Extracting data...")
        df = extract(INPUT_FILE)
        logger.info(f"✓ Extracted {len(df)} rows")
        
        logger.info("Step 2: Transforming data...")
        transform(INPUT_FILE, PROCESSED_FILE)
        logger.info(f"✓ Transformed data saved")
        
        logger.info("Step 3: Training model...")
        train_model(PROCESSED_FILE, MODEL_FILE)
        logger.info(f"✓ Model saved")
        
        logger.info("Step 4: Generating forecast...")
        forecast(MODEL_FILE, PROCESSED_FILE, FORECAST_FILE)
        logger.info(f"✓ Forecast saved")
        
        logger.info(" Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f" Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main scheduler loop."""
    logger.info("=" * 60)
    logger.info("OPIOID FORECASTING SCHEDULER")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    # Choose which method to use
    use_direct = True  # Set to False to use subprocess method
    
    # Clear any existing schedules
    schedule.clear()
    
    if use_direct:
        # Schedule: Run every 30 days (approximately monthly)
        # schedule library doesn't support .month, so we use days instead
        schedule.every(30).days.at("02:00").do(run_pipeline_alternative)
        logger.info("Using direct method (run_pipeline_alternative)")
        logger.info("Scheduled time: Every 30 days at 2:00 AM")
    else:
        # Schedule: Run every 30 days (approximately monthly)
        schedule.every(30).days.at("02:00").do(run_pipeline)
        logger.info("Using subprocess method (run_pipeline)")
        logger.info("Scheduled time: Every 30 days at 2:00 AM")
    
    # For testing, you can uncomment these:
    # schedule.every(1).minutes.do(run_pipeline_alternative)
    # logger.info("TEST MODE: Running every 1 minute")
    
    # Run once immediately for testing
    logger.info("-" * 60)
    logger.info("Running pipeline immediately for testing...")
    if use_direct:
        run_pipeline_alternative()
    else:
        run_pipeline()
    
    logger.info("-" * 60)
    logger.info("Scheduler is now running. Press Ctrl+C to stop")
    
    # Keep running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\nScheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()