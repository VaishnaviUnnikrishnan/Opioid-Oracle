"""
Utility functions for the opioid forecasting project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_forecast(forecast: pd.DataFrame, actual: pd.DataFrame = None, 
                  title: str = "Opioid Prescribing Rate Forecast", 
                  save_path: str = None):
    """
    Plot forecast results with optional actual values.
    
    Args:
        forecast: Forecast DataFrame with ds and yhat columns
        actual: Optional DataFrame with historical actual values
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot forecast
    plt.plot(forecast['ds'], forecast['yhat'], 
             label='Forecast', color='blue', linewidth=2)
    
    # Plot confidence intervals
    if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        plt.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            alpha=0.2,
            color='blue',
            label='Confidence Interval'
        )
    
    # Plot actual historical data
    if actual is not None:
        plt.plot(actual['ds'], actual['y'], 
                 label='Actual', color='green', marker='o', linestyle='-')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Prescribing Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_components(forecast: pd.DataFrame, model, save_path: str = None):
    """
    Plot forecast components.
    
    Args:
        forecast: Forecast DataFrame
        model: Trained Prophet model
        save_path: Optional path to save the plot
    """
    fig = model.plot_components(forecast)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved components plot to {save_path}")
    
    plt.show()


def load_metrics(model_path: str) -> dict:
    """
    Load evaluation metrics from JSON file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary of metrics
    """
    metrics_path = model_path.replace('.pkl', '_metrics.json')
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics from {metrics_path}")
        return metrics
    except FileNotFoundError:
        logger.warning(f"Metrics file not found: {metrics_path}")
        return {}


def load_training_info(model_path: str) -> dict:
    """
    Load training information from JSON file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary of training info
    """
    info_path = model_path.replace('.pkl', '_info.json')
    
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        logger.info(f"Loaded training info from {info_path}")
        return info
    except FileNotFoundError:
        logger.warning(f"Training info file not found: {info_path}")
        return {}


def load_forecast_report(model_path: str) -> dict:
    """
    Load forecast report from JSON file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary of forecast report
    """
    report_path = model_path.replace('.pkl', '_forecast_report.json')
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        logger.info(f"Loaded forecast report from {report_path}")
        return report
    except FileNotFoundError:
        logger.warning(f"Forecast report not found: {report_path}")
        return {}


def get_feature_importance(model) -> dict:
    """
    Extract feature importance from Prophet model.
    
    Args:
        model: Trained Prophet model
        
    Returns:
        Dictionary of feature importances
    """
    importance = {}
    
    # Get regressor coefficients if available
    if hasattr(model, 'params') and model.params:
        for key in model.params:
            if 'regressor' in key:
                importance[key] = model.params[key]
    
    return importance


def compare_forecasts(forecast1: pd.DataFrame, forecast2: pd.DataFrame,
                      name1: str = "Model 1", name2: str = "Model 2",
                      save_path: str = None):
    """
    Compare two forecasts visually.
    
    Args:
        forecast1: First forecast DataFrame
        forecast2: Second forecast DataFrame
        name1: Name of first model
        name2: Name of second model
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(forecast1['ds'], forecast1['yhat'], 
             label=name1, color='blue', linewidth=2)
    plt.plot(forecast2['ds'], forecast2['yhat'], 
             label=name2, color='red', linewidth=2, linestyle='--')
    
    plt.title(f'Forecast Comparison: {name1} vs {name2}')
    plt.xlabel('Date')
    plt.ylabel('Prescribing Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully")