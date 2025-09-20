"""
Model evaluation and benchmarking module.
Provides comprehensive evaluation capabilities across different modeling strategies.
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path

from .structures import TrainingResult, ModelingStrategy
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for benchmark models with comprehensive metrics and visualizations."""
    
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
    
    def evaluate_training_result(self, 
                             training_result: TrainingResult,
                             X_test: pl.DataFrame,
                             y_test: pl.DataFrame) -> TrainingResult:
        """
        Evaluate a TrainingResult and add computed performance metrics.
        
        This method takes a TrainingResult from training and adds performance metrics
        by evaluating the model on test data.
        
        Args:
            training_result: TrainingResult from training (without test metrics)
            X_test: Test features DataFrame with bdID column
            y_test: Test targets DataFrame with bdID column
            
        Returns:
            TrainingResult with computed performance metrics added
        """
        
        logger.info(f"Evaluating trained model: {training_result.model_type}")
        
        # Use test bdIDs from training result's split info
        bdids_to_use = training_result.split_info.test_bdIDs
        
        # Filter to test data based on bdIDs from split info
        test_features = X_test.filter(pl.col("bdID").is_in(bdids_to_use))
        test_targets = y_test.filter(pl.col("bdID").is_in(bdids_to_use))
        
        if len(test_features) == 0:
            logger.warning("No test data found for evaluation")
            raise ValueError("No test data available for evaluation")
        
        # Prepare features for prediction using training result's feature columns
        # XGBoost models need pandas DataFrames for feature names, others use numpy
        if "xgboost" in training_result.model_type.lower():
            X_test_filtered = test_features.select(training_result.feature_columns).to_pandas()
        else:
            X_test_filtered = test_features.select(training_result.feature_columns).to_numpy()
        y_test_filtered = test_targets.select(training_result.target_column).to_numpy().flatten()
        
        # Make predictions using the trained model
        y_pred = training_result.model.predict(X_test_filtered)
        y_pred = np.clip(np.round(y_pred).astype(int), 0, None)
        
        # Calculate metrics using centralized metrics calculator
        if training_result.is_quantile_model():
            # Calculate quantile-specific metrics
            metrics = self.metrics_calc.calculate_quantile_metrics(
                y_true=y_test_filtered,
                y_pred=y_pred,
                quantile_alpha=training_result.quantile_level
            )
        else:
            # Calculate standard regression metrics
            metrics = self.metrics_calc.calculate_regression_metrics(
                y_true=y_test_filtered,
                y_pred=y_pred
            )
        
        # Update the TrainingResult with performance metrics
        training_result.performance_metrics = metrics
        
        rmse_val = metrics.get('rmse', 'N/A')
        rmse_str = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else rmse_val
        logger.info(f"Updated TrainingResult with metrics: RMSE={rmse_str}")
        if training_result.quantile_level is not None:
            coverage_val = metrics.get('coverage_probability', 'N/A')
            coverage_str = f"{coverage_val:.4f}" if isinstance(coverage_val, (int, float)) else coverage_val
            logger.info(f"Quantile α={training_result.quantile_level}: Coverage={coverage_str}")
        
        return training_result
    
    def evaluate_multiple_results(self, 
                                 training_results: List[TrainingResult],
                                 X_test: pl.DataFrame,
                                 y_test: pl.DataFrame) -> List[TrainingResult]:
        """
        Evaluate multiple TrainingResult objects.
        
        Args:
            training_results: List of TrainingResult objects to evaluate
            X_test: Test features DataFrame
            y_test: Test targets DataFrame
            
        Returns:
            List of TrainingResult objects with metrics added
        """
        logger.info(f"Evaluating {len(training_results)} models")
        
        evaluated_results = []
        for i, result in enumerate(training_results):
            logger.info(f"Evaluating model {i+1}/{len(training_results)}")
            evaluated_result = self.evaluate_training_result(result, X_test, y_test)
            evaluated_results.append(evaluated_result)
        
        logger.info(f"Completed evaluation of {len(evaluated_results)} models")
        return evaluated_results
    
    def get_evaluation_summary(self, training_results: List[TrainingResult]) -> Dict[str, Any]:
        """
        Get summary statistics across multiple evaluated models.
        
        Args:
            training_results: List of evaluated TrainingResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not training_results:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for result in training_results:
            metrics = result.performance_metrics
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate summary statistics
        summary = {
            "num_models": len(training_results),
            "model_types": list(set(r.model_type for r in training_results)),
            "strategies": list(set(r.modeling_strategy.value for r in training_results))
        }
        
        # Add metric summaries
        for metric_name, values in all_metrics.items():
            if len(values) == 1:
                summary[f"{metric_name}"] = values[0]
            else:
                summary[f"{metric_name}_mean"] = sum(values) / len(values)
                summary[f"{metric_name}_min"] = min(values)
                summary[f"{metric_name}_max"] = max(values)
                summary[f"{metric_name}_std"] = np.std(values)
        
        return summary