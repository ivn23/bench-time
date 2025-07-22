"""
Model evaluation and benchmarking module.
Provides comprehensive evaluation capabilities across different granularity levels.
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path

from .data_structures import BenchmarkModel, GranularityLevel, ModelRegistry
from .data_loading import DataLoader

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for benchmark models with comprehensive metrics and visualizations."""
    
    def __init__(self, data_loader: DataLoader, model_registry: ModelRegistry):
        self.data_loader = data_loader
        self.model_registry = model_registry
    
    def evaluate_model(self, 
                      model: BenchmarkModel,
                      test_bdIDs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.
        
        Args:
            model: BenchmarkModel to evaluate
            test_bdIDs: Optional test set bdIDs. If None, uses validation set.
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating model: {model.get_identifier()}")
        
        # Reconstruct test data based on stored split information
        if test_bdIDs is None:
            bdids_to_use = model.data_split.validation_bdIDs
            data_split_name = "validation"
        else:
            bdids_to_use = test_bdIDs
            data_split_name = "test"
        
        # For now, return a message indicating this needs the engineered data
        logger.warning("Use evaluate_model_with_data() method instead for notebook usage")
        return {"error": "Use evaluate_model_with_data() method with pre-engineered features"}
    
    def evaluate_model_with_data(self, model: BenchmarkModel, X_data: pl.DataFrame, y_data: pl.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a model using pre-engineered X and y data.
        This is more suitable for notebook usage.
        
        Args:
            model: BenchmarkModel to evaluate
            X_data: Pre-engineered features DataFrame with bdID column
            y_data: Target DataFrame with bdID column
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating model with provided data: {model.get_identifier()}")
        
        # Use validation bdIDs from the model's data split
        bdids_to_use = model.data_split.validation_bdIDs
        
        # Filter to validation data
        test_features = X_data.filter(pl.col("bdID").is_in(bdids_to_use))
        test_target = y_data.filter(pl.col("bdID").is_in(bdids_to_use))
        
        if len(test_features) == 0:
            logger.warning("No test data found for evaluation")
            return {"error": "No test data available"}
        
        # Prepare features for prediction
        X_test = test_features.select(model.metadata.feature_columns).to_numpy()
        y_test = test_target.select(model.metadata.target_column).to_numpy().flatten()
        
        # Make predictions
        y_pred = model.model.predict(X_test)
        y_pred = np.round(y_pred).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(y_test, y_pred)
        
        # Create evaluation result
        evaluation_result = {
            "model_id": model.get_identifier(),
            "granularity": model.metadata.granularity.value,
            "entity_ids": model.metadata.entity_ids,
            "n_samples": len(y_test),
            "predictions": y_pred.tolist(),
            "actuals": y_test.tolist(),
            "prediction_errors": (y_test - y_pred).tolist(),
            "metrics": metrics,
            "data_split_name": "validation"
        }
        
        return evaluation_result
    
    def compare_models(self, 
                      model_ids: List[str],
                      test_bdIDs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compare multiple models on the same test data.
        
        Args:
            model_ids: List of model IDs to compare
            test_bdIDs: Optional test set bdIDs
            
        Returns:
            Comparison results dictionary
        """
        logger.info(f"Comparing {len(model_ids)} models")
        
        comparison_results = {
            "model_evaluations": {},
            "metrics_comparison": {},
            "rankings": {}
        }
        
        # Evaluate each model
        all_metrics = {}
        for model_id in model_ids:
            model = self.model_registry.get_model(model_id)
            if model is None:
                logger.warning(f"Model {model_id} not found in registry")
                continue
            
            eval_result = self.evaluate_model(model, test_bdIDs)
            comparison_results["model_evaluations"][model_id] = eval_result
            
            if "metrics" in eval_result:
                all_metrics[model_id] = eval_result["metrics"]
        
        # Compare metrics
        if all_metrics:
            comparison_results["metrics_comparison"] = self._create_metrics_comparison(all_metrics)
            comparison_results["rankings"] = self._rank_models(all_metrics)
        
        return comparison_results
    
    def evaluate_by_granularity(self, 
                               granularity: GranularityLevel,
                               test_bdIDs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate all models of a specific granularity level.
        
        Args:
            granularity: Granularity level to evaluate
            test_bdIDs: Optional test set bdIDs
            
        Returns:
            Granularity-level evaluation results
        """
        logger.info(f"Evaluating all models at {granularity.value} level")
        
        # Get all models for this granularity
        model_ids = self.model_registry.list_models(granularity)
        
        if not model_ids:
            logger.warning(f"No models found for granularity {granularity.value}")
            return {"error": f"No models found for granularity {granularity.value}"}
        
        # Compare all models at this granularity
        return self.compare_models(model_ids, test_bdIDs)
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any],
                                 output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_model or compare_models
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("# Model Evaluation Report\n")
        
        if "model_evaluations" in evaluation_results:
            # Multiple model comparison report
            report_lines.append("## Model Comparison\n")
            
            # Metrics summary table
            if "metrics_comparison" in evaluation_results:
                report_lines.append("### Performance Metrics\n")
                metrics_df = pd.DataFrame(evaluation_results["metrics_comparison"]).T
                report_lines.append(metrics_df.to_string())
                report_lines.append("\n")
            
            # Rankings
            if "rankings" in evaluation_results:
                report_lines.append("### Model Rankings\n")
                for metric, ranking in evaluation_results["rankings"].items():
                    report_lines.append(f"**{metric.upper()}:**\n")
                    for i, (model_id, value) in enumerate(ranking, 1):
                        report_lines.append(f"{i}. {model_id}: {value:.4f}\n")
                    report_lines.append("\n")
        
        else:
            # Single model report
            model_id = evaluation_results.get("model_id", "Unknown")
            report_lines.append(f"## Model: {model_id}\n")
            
            # Basic info
            report_lines.append(f"**Granularity:** {evaluation_results.get('granularity', 'Unknown')}\n")
            report_lines.append(f"**Entity IDs:** {evaluation_results.get('entity_ids', {})}\n")
            report_lines.append(f"**Test Samples:** {evaluation_results.get('n_samples', 0)}\n\n")
            
            # Metrics
            if "metrics" in evaluation_results:
                report_lines.append("### Performance Metrics\n")
                for metric, value in evaluation_results["metrics"].items():
                    report_lines.append(f"- **{metric.upper()}:** {value:.4f}\n")
                report_lines.append("\n")
        
        report_text = "".join(report_lines)
        
        # Save to file if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # MAPE (Mean Absolute Percentage Error)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            metrics['mape'] = float('inf')
        
        # Additional metrics
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['mean_error'] = np.mean(y_true - y_pred)  # Bias
        metrics['std_error'] = np.std(y_true - y_pred)
        
        # Percentage of predictions within certain error ranges
        abs_errors = np.abs(y_true - y_pred)
        metrics['within_1_unit'] = np.mean(abs_errors <= 1.0) * 100
        metrics['within_2_units'] = np.mean(abs_errors <= 2.0) * 100
        metrics['within_5_units'] = np.mean(abs_errors <= 5.0) * 100
        
        return metrics
    
    def _get_feature_importance(self, model: BenchmarkModel) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        try:
            if model.metadata.model_type == "xgboost":
                if hasattr(model.model, 'feature_importances_'):
                    importance_dict = dict(zip(
                        model.metadata.feature_columns,
                        model.model.feature_importances_
                    ))
                    # Sort by importance
                    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return None
    
    def _create_metrics_comparison(self, all_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Create comparison table of metrics across models."""
        comparison = {}
        
        # Get all unique metrics
        all_metric_names = set()
        for metrics in all_metrics.values():
            all_metric_names.update(metrics.keys())
        
        # Create comparison table
        for metric_name in all_metric_names:
            comparison[metric_name] = {}
            for model_id, metrics in all_metrics.items():
                comparison[metric_name][model_id] = metrics.get(metric_name, float('nan'))
        
        return comparison
    
    def _rank_models(self, all_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """Rank models by different metrics."""
        rankings = {}
        
        # Metrics where lower is better
        lower_is_better = ['mse', 'rmse', 'mae', 'mape', 'max_error', 'std_error'] # mase # msse
        # Metrics where higher is better  
        higher_is_better = ['r2', 'within_1_unit', 'within_2_units', 'within_5_units']
        
        for metric_name in set().union(*[metrics.keys() for metrics in all_metrics.values()]):
            metric_values = []
            for model_id, metrics in all_metrics.items():
                if metric_name in metrics and not np.isnan(metrics[metric_name]):
                    metric_values.append((model_id, metrics[metric_name]))
            
            if metric_values:
                if metric_name in lower_is_better:
                    # Sort ascending (lower is better)
                    rankings[metric_name] = sorted(metric_values, key=lambda x: x[1])
                elif metric_name in higher_is_better:
                    # Sort descending (higher is better)
                    rankings[metric_name] = sorted(metric_values, key=lambda x: x[1], reverse=True)
                else:
                    # Default to descending
                    rankings[metric_name] = sorted(metric_values, key=lambda x: x[1], reverse=True)
        
        return rankings


class VisualizationGenerator:
    """Generate evaluation visualizations using lets-plot (as in original notebook)."""
    
    def __init__(self):
        try:
            import lets_plot as lp
            from lets_plot import ggplot, aes, geom_point, geom_histogram, geom_bar, labs, theme_minimal, ggtitle, xlab, ylab
            lp.LetsPlot.setup_html()
            self.lets_plot = lp
            self.ggplot = ggplot
            self.aes = aes
            self.geom_point = geom_point
            self.geom_histogram = geom_histogram
            self.geom_bar = geom_bar
            self.labs = labs
            self.theme_minimal = theme_minimal
            self.ggtitle = ggtitle
            self.xlab = xlab
            self.ylab = ylab
            self.lets_plot_available = True
        except ImportError:
            logger.warning("lets-plot not available. Visualizations will be limited.")
            self.lets_plot_available = False
    
    def create_prediction_plot(self, 
                              evaluation_result: Dict[str, Any],
                              time_data: Optional[pl.DataFrame] = None) -> Any:
        """Create prediction vs actual plot."""
        if not self.lets_plot_available:
            logger.warning("lets-plot not available for visualization")
            return None
        
        # Prepare data for plotting
        actuals = evaluation_result["actuals"]
        predictions = evaluation_result["predictions"]
        
        plot_data = pd.DataFrame({
            "actual": actuals,
            "predicted": predictions
        })
        
        # Create scatter plot
        p = self.ggplot(plot_data, self.aes(x='actual', y='predicted')) + \
            self.geom_point(alpha=0.6) + \
            self.labs(title=f'Predictions vs Actuals: {evaluation_result["model_id"]}',
                 x='Actual Values', 
                 y='Predicted Values') + \
            self.theme_minimal()
        
        return p
    
    def create_error_distribution_plot(self, evaluation_result: Dict[str, Any]) -> Any:
        """Create error distribution plot."""
        if not self.lets_plot_available:
            return None
        
        errors = evaluation_result["prediction_errors"]
        
        plot_data = pd.DataFrame({"errors": errors})
        
        p = self.ggplot(plot_data, self.aes(x='errors')) + \
            self.geom_histogram(bins=50, alpha=0.7) + \
            self.labs(title=f'Prediction Error Distribution: {evaluation_result["model_id"]}',
                 x='Prediction Error',
                 y='Frequency') + \
            self.theme_minimal()
        
        return p
    
    def create_model_comparison_plot(self, comparison_results: Dict[str, Any], metric: str = 'rmse') -> Any:
        """Create model comparison plot."""
        if not self.lets_plot_available:
            return None
        
        if "metrics_comparison" not in comparison_results or metric not in comparison_results["metrics_comparison"]:
            logger.warning(f"Metric {metric} not found in comparison results")
            return None
        
        metric_data = comparison_results["metrics_comparison"][metric]
        
        plot_data = pd.DataFrame({
            "model": list(metric_data.keys()),
            "value": list(metric_data.values())
        })
        
        p = self.ggplot(plot_data, self.aes(x='model', y='value')) + \
            self.geom_bar() + \
            self.labs(title=f'Model Comparison: {metric.upper()}',
                 x='Model',
                 y=metric.upper()) + \
            self.theme_minimal()
        
        return p