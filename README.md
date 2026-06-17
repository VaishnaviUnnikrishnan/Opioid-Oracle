# OpioidOracle

## AI-Powered Forecasting of Medicaid Long-Acting Opioid Prescribing Rates

OpioidOracle is a predictive analytics and time-series forecasting project that leverages Meta's Prophet framework to forecast Long-Acting (LA) opioid prescribing rates using historical Medicaid prescription data. The project analyzes prescribing behavior across geographic regions and healthcare plan types to identify long-term trends, forecast future opioid utilization, and support evidence-based healthcare decision-making.

---

## Project Overview

The opioid epidemic remains a major public health challenge in the United States. Understanding how opioid prescribing patterns evolve over time is critical for healthcare policymakers, researchers, and public health organizations.

OpioidOracle applies machine learning-driven forecasting techniques to Medicaid prescription data and generates interpretable forecasts of Long-Acting Opioid Prescribing Rates. The model captures historical trends and structural changes in prescribing behavior to estimate future prescribing patterns.

---

## Problem Statement

Long-acting opioid prescriptions represent a significant subset of overall opioid utilization. Identifying future trends in prescribing rates can help:

- Monitor opioid usage patterns
- Support healthcare policy development
- Evaluate the effectiveness of intervention programs
- Identify regions with increasing prescribing trends
- Enable proactive public health planning

---

## Dataset

### Medicaid Opioid Prescribing Rates by Geography (2013–2024)

The dataset contains opioid prescribing statistics aggregated across multiple geographic levels and healthcare plan types.

### Features Used

#### Target Variable

| Feature | Description |
|----------|-------------|
| LA_Opioid_Prscrbng_Rate | Long-Acting Opioid Prescribing Rate |

---

#### Temporal Features

| Feature | Description |
|----------|-------------|
| Year | Observation Year |
| Opioid_Prscrbng_Rate_1Y_Chg | One-Year Change in Opioid Prescribing Rate |
| Opioid_Prscrbng_Rate_5Y_Chg | Five-Year Change in Opioid Prescribing Rate |
| LA_Lag_1 | Previous Year's LA Prescribing Rate |
| LA_Rolling_Mean_3Y | Three-Year Rolling Average |

---

#### Opioid Context Features

| Feature | Description |
|----------|-------------|
| Opioid_Prscrbng_Rate | Overall Opioid Prescribing Rate |
| Tot_Opioid_Clms | Total Opioid Claims |
| Tot_Clms | Total Prescription Claims |
| Raw_Opioid_Ratio | Opioid Claims / Total Claims |

---

#### Geographic Features

| Feature | Description |
|----------|-------------|
| Geo_Lvl | Geographic Aggregation Level |
| Geo_Cd | Geographic Identifier |
| Geo_Desc | Geographic Description |

---

#### Plan Features

| Feature | Description |
|----------|-------------|
| Plan_Type | Medicaid Plan Type (FFS / MC / All) |

---

## Data Source

**Centers for Medicare & Medicaid Services (CMS)**

Dataset:
**Medicaid Opioid Prescribing Rates by Geography**

Source:
https://data.medicaid.gov

The dataset provides annual opioid prescribing statistics at National, State, County, and ZIP-code levels across Medicaid beneficiaries.

---

## Machine Learning Approach

### Forecasting Model

The project utilizes **Facebook Prophet**, a decomposable time-series forecasting model developed by Meta.

Prophet was selected because it:

- Handles trend changes automatically
- Detects changepoints in historical data
- Produces interpretable forecasts
- Works well on healthcare and policy datasets
- Requires minimal parameter tuning

---

## Trend Selection

Two Prophet growth models were evaluated:

### Logistic Growth

Suitable when data exhibits saturation and approaches a carrying capacity.

```text
S-shaped growth
```

### Piecewise Linear Growth

Suitable when trends change due to policy shifts, behavioral changes, or external interventions.

```text
Trend + Changepoints
```

Exploratory trend analysis showed no strong saturation pattern; therefore, the final model uses Prophet's **piecewise linear growth** strategy.

---

## Data Preprocessing

The preprocessing pipeline includes:

### Data Cleaning

- Removal of duplicate records
- Missing value treatment
- Handling infinite values

### Feature Engineering

- Raw Opioid Ratio
- Lag Features
- Rolling Window Features

### Time-Series Formatting

Prophet-compatible schema:

```python
ds → Timestamp
y  → Target Variable
```

Example:

```python
ds = 2024-01-01
y  = 26.01
```

---

## Model Pipeline

```text
Raw Medicaid Dataset
            │
            ▼
Data Cleaning
            │
            ▼
Feature Engineering
            │
            ▼
Prophet Data Formatting
            │
            ▼
Train/Test Split
            │
            ▼
Facebook Prophet
            │
            ▼
Forecast Generation
            │
            ▼
Performance Evaluation
```

---

## Model Performance

### Evaluation Metrics

| Metric | Value |
|----------|----------|
| MAE | 3.36 |
| MSE | 12.54 |
| RMSE | 3.54 |
| MAPE | 12.95% |
| Forecast Accuracy | 87.05% |

### Interpretation

The model achieved approximately **87.05% forecasting accuracy** based on Mean Absolute Percentage Error (MAPE).

The results indicate that Prophet successfully captures long-term opioid prescribing trends despite the limited number of annual observations available.

---

## Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Forecasting | Facebook Prophet |
| Machine Learning | Scikit-Learn |
| Visualization | Matplotlib |
| Model Serialization | Pickle |

---

## Repository Structure

```text
OpioidOracle/
│
├── data/
│   └── OMT_MDCD_RY26_P02_V10_YTD24_GEO.csv
│
├── notebooks/
│   └── opioid_forecasting.ipynb
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
├── README.md
│
└── requirements.txt
```

---

## Future Enhancements

- State-Level Forecasting Models
- County-Level Forecasting Models
- Interactive Streamlit Dashboard
- Forecast Explainability
- Comparative Analysis with ARIMA and XGBoost
- Automated Cross-Validation
- Geospatial Opioid Trend Visualization

---

## Disclaimer

This project is intended for educational, research, and analytical purposes only. Forecasts generated by the model should not be interpreted as medical advice or policy recommendations without additional domain-specific evaluation.
