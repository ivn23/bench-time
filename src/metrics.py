"""
Centralized metrics calculation module for the M5 Time Series Benchmarking Framework.

This module provides a unified interface for calculating evaluation metrics,
eliminating code duplication and ensuring consistent metric computation across
the framework.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Centralized calculator for evaluation metrics."""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing standard regression metrics
            
        Raises:
            ValueError: If input arrays are invalid
        """
        # Validate inputs
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length")
        
        # Core regression metrics
        metrics = {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred))
        }

        return metrics

    @staticmethod
    def calculate_quantile_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  quantile_alpha: float) -> Dict[str, float]:
        """
        Calculate quantile-specific evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted quantile values
            quantile_alpha: Target quantile level (e.g., 0.7 for 70% quantile)
            
        Returns:
            Dictionary containing quantile-specific metrics
            
        Raises:
            ValueError: If input arrays are invalid or quantile_alpha is out of range
        """
        # Validate inputs
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Input arrays cannot be empty")
            
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length")
            
        if not 0 < quantile_alpha < 1:
            raise ValueError("quantile_alpha must be between 0 and 1")
        
        quantile_metrics = {}
        
        # Quantile loss (pinball loss)
        quantile_loss = np.where(y_true >= y_pred.round(), quantile_alpha * (y_true - y_pred.round()), (quantile_alpha - 1) * (y_true - y_pred.round()))

        quantile_metrics["quantile_losses"] = quantile_loss
        quantile_metrics["predictions"] = y_pred.round()
        quantile_metrics["mean_quantile_loss"] = quantile_loss.mean()

        return quantile_metrics
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            quantile_alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate all relevant metrics (standard + quantile if applicable).
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            quantile_alpha: Optional quantile level for quantile-specific metrics
            
        Returns:
            Dictionary containing all applicable metrics
        """
        # Always calculate standard regression metrics
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        # Add quantile-specific metrics if quantile_alpha is provided
        if quantile_alpha is not None:
            quantile_metrics = MetricsCalculator.calculate_quantile_metrics(
                y_true, y_pred, quantile_alpha
            )
            metrics.update(quantile_metrics)
        
        return metrics