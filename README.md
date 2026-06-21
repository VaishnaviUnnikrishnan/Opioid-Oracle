# OpioidOracle

## AI-Powered Forecasting and Workflow Automation for Medicaid Long-Acting Opioid Prescribing Rates

OpioidOracle is an end-to-end healthcare analytics platform that forecasts Medicaid Long-Acting (LA) opioid prescribing rates using Meta's Prophet time-series forecasting framework. The project combines data engineering, automated workflow orchestration, feature engineering, predictive analytics, and model deployment principles to generate actionable forecasts from historical Medicaid prescription data.

The solution is orchestrated using Apache Airflow, enabling automated data ingestion, preprocessing, model training, forecasting, and performance evaluation through a reproducible machine learning pipeline.

---

## Key Features

* Forecasts Long-Acting Opioid Prescribing Rates using Facebook Prophet
* Automated ETL and forecasting workflow using Apache Airflow
* Feature engineering with lag and rolling-window statistics
* Time-series trend analysis and changepoint detection
* Forecast evaluation using MAE, MSE, RMSE, and MAPE
* Serialized forecasting model using Pickle
* Reproducible machine learning pipeline architecture
* Healthcare analytics use case based on CMS Medicaid datasets

---

## System Architecture

```text
CMS Medicaid Dataset
          │
          ▼
Data Ingestion
          │
          ▼
Data Cleaning
          │
          ▼
Feature Engineering
          │
          ▼
Prophet Formatting
(ds, y)
          │
          ▼
Model Training
(Facebook Prophet)
          │
          ▼
Forecast Generation
          │
          ▼
Model Evaluation
          │
          ▼
Forecast Outputs
```

---

## Workflow Orchestration

The forecasting pipeline is automated using Apache Airflow.

### DAG Workflow

```text
extract_data
      │
      ▼
preprocess_data
      │
      ▼
train_prophet_model
      │
      ▼
generate_forecast
      │
      ▼
evaluate_model
```

Each stage executes independently and can be monitored, scheduled, retried, and audited through the Airflow UI.

---

## Machine Learning Methodology

### Forecasting Model

Meta Prophet was selected due to its ability to:

* Capture long-term trends
* Detect structural changes through changepoints
* Handle healthcare time-series data effectively
* Provide interpretable forecasts
* Require minimal hyperparameter tuning

### Growth Strategy Selection

Two forecasting strategies were evaluated:

#### Logistic Growth

Suitable for saturation-based growth patterns.

```text
S-Shaped Trend
```

#### Piecewise Linear Growth

Suitable for policy-driven or behavior-driven trend changes.

```text
Trend + Automatic Changepoints
```

Exploratory analysis showed no significant saturation behavior. Therefore, the final model uses Prophet's Piecewise Linear Growth framework.

---

## Data Engineering Pipeline

### Data Cleaning

* Duplicate removal
* Missing value treatment
* Infinite value handling
* Data type validation

### Feature Engineering

#### Temporal Features

* One-Year Rate Change
* Five-Year Rate Change
* Lag Features
* Rolling Window Statistics

#### Utilization Features

* Total Opioid Claims
* Total Prescription Claims
* Raw Opioid Ratio

#### Geographic Features

* Geographic Level
* Geographic Code
* Geographic Description

#### Plan Features

* Medicaid Plan Type

---

## Model Performance

| Metric            | Value  |
| ----------------- | ------ |
| MAE               | 3.36   |
| MSE               | 12.54  |
| RMSE              | 3.54   |
| MAPE              | 12.95% |
| Forecast Accuracy | 87.05% |

### Performance Summary

The model achieved approximately 87.05% forecasting accuracy based on Mean Absolute Percentage Error (MAPE).

Results indicate that Prophet effectively captures long-term opioid prescribing trends despite the limited number of annual observations available within the dataset.

---

## Technologies Used

| Category               | Technology          |
| ---------------------- | ------------------- |
| Programming Language   | Python              |
| Data Processing        | Pandas, NumPy       |
| Forecasting            | Facebook Prophet    |
| Machine Learning       | Scikit-Learn        |
| Workflow Orchestration | Apache Airflow      |
| Data Visualization     | Matplotlib, Seaborn |
| Model Serialization    | Pickle              |
| Containerization       | Docker              |
| Scheduling             | Airflow Scheduler   |
| Version Control        | Git & GitHub        |

---

## Repository Structure

```text
OpioidOracle/
│
├── dags/
│   └── opioid_pipeline.py
│
├── data/
│   └── OMT_MDCD_RY26_P02_V10_YTD24_GEO.csv
│
├── scripts/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── forecast.py
│   └── evaluate.py
│
├── models/
│   └── opioidoracle_prophet.pkl
│
├── forecasts/
│   └── opioidoracle_forecasts.csv
│
├── images/
│   └── performance_metrics.png
│
├── requirements.txt
│
└── README.md
```

---

## Future Enhancements

* State-Level Forecasting Models
* County-Level Forecasting Models
* Interactive Streamlit Dashboard
* SHAP-Based Forecast Explainability
* Comparative Analysis with ARIMA, XGBoost, and LSTM
* Automated Hyperparameter Optimization
* Geospatial Opioid Trend Visualization



---

## Disclaimer

This project is intended for educational, research, and analytical purposes only. Forecasts generated by the model should not be interpreted as medical advice, prescribing guidance, or public policy recommendations without appropriate clinical and domain-specific validation.
