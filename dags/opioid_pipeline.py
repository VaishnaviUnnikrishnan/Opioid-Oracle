"""
Opioid Prescribing Rate ETL Pipeline DAG
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import functions from scripts
from scripts.extract import extract
from scripts.transform import transform
from scripts.train import train_model
from scripts.forecast import forecast

# Default arguments
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG definition
with DAG(
    dag_id='opioid_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for opioid prescribing rate forecasting',
    schedule='@monthly',  # or '0 0 1 * *' for first day of month
    start_date=datetime(2025, 1, 1),
    tags=['opioid', 'forecast', 'etl'],
    catchup=False,
    max_active_runs=1,
) as dag:

    # Task 1: Extract Data
    extract_task = PythonOperator(
        task_id='extract',
        python_callable=extract,
        op_kwargs={
            'input_path': '/home/airflow/data/opioid_rates.csv'
        },
        dag=dag
    )

    # Task 2: Transform Data
    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform,
        op_kwargs={
            'input_path': '/home/airflow/data/opioid_rates.csv',
            'output_path': '/home/airflow/data/processed_opioid_rates.csv'
        },
        dag=dag
    )

    # Task 3: Train Model
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={
            'data_path': '/home/airflow/data/processed_opioid_rates.csv',
            'model_path': '/home/airflow/models/prophet_model.pkl'
        },
        dag=dag
    )

    # Task 4: Generate Forecast
    forecast_task = PythonOperator(
        task_id='forecast',
        python_callable=forecast,
        op_kwargs={
            'model_path': '/home/airflow/models/prophet_model.pkl',
            'data_path': '/home/airflow/data/processed_opioid_rates.csv',
            'output_path': '/home/airflow/data/forecasts.csv',
            'periods': 12
        },
        dag=dag
    )

    # Task 5: Log Completion (Optional)
    def log_completion():
        print("=" * 50)
        print("OPIOID ETL PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Time: {datetime.now()}")
        print("=" * 50)
        return "Pipeline completed"

    log_task = PythonOperator(
        task_id='log_completion',
        python_callable=log_completion,
        dag=dag
    )

    # Define dependencies
    extract_task >> transform_task >> train_task >> forecast_task >> log_task