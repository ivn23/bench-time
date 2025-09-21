"""
Release Manager for consolidating complete experiment outputs.
Simplified version without HierarchicalStorageManager dependency.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..structures import TrainingResult, ExperimentResults

logger = logging.getLogger(__name__)


class ReleaseManager:
    """
    Release manager that consolidates experiment outputs into
    a simple, functional directory structure.

    Creates:
    - bundle.json with model metadata and configuration
    - metrics.json with performance metrics organized by quantile and SKU
    - model_params/ directory with saved models
    """

    def __init__(self):
        pass

    def create_complete_release(
        self,
        experiment_results: ExperimentResults,
        base_output_dir: Path
    ) -> Path:
        """
        Create a complete release package from experiment results.

        Args:
            experiment_results: Complete experiment results container
            base_output_dir: Base directory for release creation

        Returns:
            Path to created release directory
        """
        # Create release directory using experiment name
        release_dir = base_output_dir / experiment_results.experiment_name

        logger.info(f"Creating release: {experiment_results.experiment_name}")

        # Ensure release directory exists
        release_dir.mkdir(parents=True, exist_ok=True)

        # Create release components
        self._create_bundle_json(experiment_results, release_dir)
        self._create_models_directory(experiment_results.training_results, release_dir)
        self._create_metrics_json(experiment_results.training_results, release_dir)

        logger.info(f"Release created successfully at: {release_dir}")
        return release_dir

    def _create_bundle_json(self, experiment_results: ExperimentResults, release_dir: Path):
        """
        Create bundle.json with model type, SKU information, and hyperparameters.

        According to spec:
        - Model type used for the experiment
        - Complete list of SKU tuples
        - Complete list of product IDs
        - Complete list of store IDs
        - List of quantiles present
        - Hyperparameters (same for all models)
        """
        logger.debug("Creating bundle.json")

        # Get model type from the first training result (all use the same type)
        model_type = experiment_results.training_results[0].model_type if experiment_results.training_results else "unknown"

        # Collect all unique SKU tuples, product IDs, and store IDs
        all_sku_tuples = []
        product_ids = set()
        store_ids = set()

        for result in experiment_results.training_results:
            for sku_tuple in result.sku_tuples:
                if sku_tuple not in all_sku_tuples:
                    all_sku_tuples.append(sku_tuple)
                product_ids.add(sku_tuple[0])
                store_ids.add(sku_tuple[1])

        # Collect all unique quantile levels
        quantile_levels = []
        for result in experiment_results.training_results:
            if result.quantile_level is not None and result.quantile_level not in quantile_levels:
                quantile_levels.append(result.quantile_level)

        # Get hyperparameters (same for all models)
        hyperparameters = {}
        if experiment_results.training_results:
            hyperparameters = experiment_results.training_results[0].hyperparameters.copy()
            # Remove quantile-specific params if present
            hyperparameters.pop('quantile_alphas', None)

        # Build bundle data according to specification
        bundle_data = {
            "model_type": model_type,
            "sku_tuples": all_sku_tuples,
            "product_ids": sorted(list(product_ids)),
            "store_ids": sorted(list(store_ids)),
            "quantiles": sorted(quantile_levels) if quantile_levels else [],
            "hyperparameters": hyperparameters,
            "experiment_name": experiment_results.experiment_name,
            "num_models": len(experiment_results.training_results)
        }

        # Write bundle.json
        bundle_path = release_dir / "bundle.json"
        with open(bundle_path, 'w') as f:
            json.dump(bundle_data, f, indent=2, default=str)

        logger.debug(f"Bundle created: {bundle_path}")

    def _create_models_directory(self, training_results: List[TrainingResult], release_dir: Path):
        """
        Create model_params directory with saved models.

        Naming convention: <quantile_level>_<sku_tuple>_<model_type>.[ext]
        For non-quantile models, use "standard" as quantile_level.
        """
        logger.debug("Creating model_params directory")

        models_dir = release_dir / "model_params"
        models_dir.mkdir(parents=True, exist_ok=True)

        for result in training_results:
            # Determine quantile level string
            if result.quantile_level is not None:
                quantile_str = f"{result.quantile_level}"
            else:
                quantile_str = "standard"

            # Create SKU tuple string (use first SKU for naming)
            if result.sku_tuples:
                sku = result.sku_tuples[0]
                sku_str = f"{sku[0]}_{sku[1]}"  # product_id_store_id
            else:
                sku_str = "combined"

            # Create model filename
            model_filename = f"{quantile_str}_{sku_str}_{result.model_type}"

            # Use the model's save_model method
            try:
                result.model.save_model(str(models_dir), model_filename)
                logger.debug(f"Saved model: {model_filename}")
            except Exception as e:
                logger.error(f"Failed to save model {model_filename}: {e}")

        logger.debug(f"Models saved to: {models_dir}")

    def _create_metrics_json(self, training_results: List[TrainingResult], release_dir: Path):
        """
        Create metrics.json organized by quantile level and SKU tuple.

        Structure:
        {
            "<quantile_level>": {
                "<sku_tuple>": <metric_value>
            }
        }

        For non-quantile models, use "standard" as the outer key.
        """
        logger.debug("Creating metrics.json")

        metrics_data = {}

        for result in training_results:
            # Determine quantile level key
            if result.quantile_level is not None:
                quantile_key = str(result.quantile_level)
            else:
                quantile_key = "standard"

            # Initialize quantile level dict if not exists
            if quantile_key not in metrics_data:
                metrics_data[quantile_key] = {}

            # Create SKU tuple string
            if result.sku_tuples:
                # For multiple SKUs (combined strategy), create a compound key
                if len(result.sku_tuples) > 1:
                    sku_key = "combined_" + "_".join([f"{sku[0]}_{sku[1]}" for sku in result.sku_tuples[:2]]) + ("_etc" if len(result.sku_tuples) > 2 else "")
                else:
                    sku = result.sku_tuples[0]
                    sku_key = f"({sku[0]}, {sku[1]})"
            else:
                sku_key = "all"

            # Extract the appropriate metric
            if result.performance_metrics:
                # For quantile models, prefer quantile_loss if available
                if result.quantile_level is not None and 'quantile_loss' in result.performance_metrics:
                    metric_value = result.performance_metrics['quantile_loss']
                elif 'rmse' in result.performance_metrics:
                    metric_value = result.performance_metrics['rmse']
                elif 'mse' in result.performance_metrics:
                    # If RMSE not available, use MSE
                    metric_value = result.performance_metrics['mse']
                else:
                    # Use first available metric
                    metric_value = list(result.performance_metrics.values())[0] if result.performance_metrics else None
            else:
                metric_value = None

            # Add to metrics data
            metrics_data[quantile_key][sku_key] = metric_value

        # Write metrics.json (only if we have metrics)
        if metrics_data:
            metrics_path = release_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            logger.debug(f"Metrics created: {metrics_path}")
        else:
            logger.debug("No metrics available - skipping metrics.json")