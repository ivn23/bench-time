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
        try:
            # Validate inputs
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("Input arrays cannot be empty")
            
            if len(y_true) != len(y_pred):
                raise ValueError("Input arrays must have the same length")
            
            # Core regression metrics
            metrics = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred))
            }
            
            # Additional error analysis
            residuals = y_true - y_pred
            metrics.update({
                "max_error": float(np.max(np.abs(residuals))),
                "mean_error": float(np.mean(residuals)),  # Bias measurement
                "std_error": float(np.std(residuals))
            })
            
            # Calculate MAPE (Mean Absolute Percentage Error) with zero-division protection
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics["mape"] = float(mape)
            else:
                metrics["mape"] = float('inf')
                
            # Calculate accuracy within error bands
            abs_errors = np.abs(residuals)
            metrics.update({
                "within_1_unit": float(np.mean(abs_errors <= 1.0)),
                "within_2_units": float(np.mean(abs_errors <= 2.0)),
                "within_5_units": float(np.mean(abs_errors <= 5.0))
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate regression metrics: {str(e)}")
            raise ValueError(f"Failed to calculate evaluation metrics: {str(e)}")
    
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
        try:
            # Validate inputs
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("Input arrays cannot be empty")
                
            if len(y_true) != len(y_pred):
                raise ValueError("Input arrays must have the same length")
                
            if not 0 < quantile_alpha < 1:
                raise ValueError("quantile_alpha must be between 0 and 1")
            
            quantile_metrics = {}
            
            # Quantile loss (pinball loss)
            residuals = y_true - y_pred
            quantile_loss = np.where(
                residuals >= 0,
                quantile_alpha * residuals,
                (quantile_alpha - 1) * residuals
            )
            quantile_metrics["quantile_score"] = float(np.mean(quantile_loss))
            
            # Coverage probability - proportion of actual values below predicted quantile
            coverage = np.mean(y_true <= y_pred)
            quantile_metrics["coverage_probability"] = float(coverage)
            
            # Coverage error - difference between actual and target coverage
            coverage_error = abs(coverage - quantile_alpha)
            quantile_metrics["coverage_error"] = float(coverage_error)
            
            # Store quantile alpha for reference
            quantile_metrics["quantile_alpha"] = float(quantile_alpha)
            
            return quantile_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate quantile metrics: {str(e)}")
            raise ValueError(f"Failed to calculate quantile metrics: {str(e)}")
    
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