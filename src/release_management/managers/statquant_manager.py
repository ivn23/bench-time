"""
StatQuant Release Manager.

This module provides release management functionality for Statsmodels
quantile regression models in the benchmarking framework.
"""

import typing as t
from pathlib import Path

from ..base import BaseReleaseManager

if t.TYPE_CHECKING:
    from ...data_structures import BenchmarkModel


class StatQuantReleaseManager(BaseReleaseManager):
    """
    Release manager for Statsmodels quantile regression models.
    
    Handles StatQuant models with quantile-specific bundle metadata
    and statsmodels-specific information.
    """

    @property
    def family_name(self) -> str:
        """Return the unique identifier for StatQuant family."""
        return "statquant"

    def create_bundle_metadata(self, benchmark_model: "BenchmarkModel") -> dict:
        """
        Create bundle metadata from StatQuant BenchmarkModel.

        Args:
            benchmark_model: BenchmarkModel to extract metadata from

        Returns:
            Dictionary containing all model-relevant details for bundle.json
        """
        metadata = benchmark_model.metadata
        
        # Base bundle structure
        bundle = {
            # Model identification
            "model_id": metadata.model_id,
            "model_type": metadata.model_type,
            "model_family": self.family_name,
            
            # Model strategy and SKUs
            "modeling_strategy": metadata.modeling_strategy.value,
            "sku_tuples": metadata.sku_tuples,
            
            # Model parameters
            "hyperparameters": metadata.hyperparameters,
            "training_config": metadata.training_config,
            
            # Model features and target
            "feature_columns": metadata.feature_columns,
            "target_column": metadata.target_column,
            
            # Training information
            "training_date_range": metadata.training_date_range,
            
            
            # Quantile-specific information
            "quantile_level": getattr(metadata, 'quantile_level', None),
            
            # StatQuant-specific information
            "statsmodels_model_class": "StatQuantModel",
            "training_method": "statsmodels_quantreg",
            "framework": "statsmodels",
            "regression_type": "quantile_regression",
        }
        
        # Add StatQuant-specific parameters to bundle
        statquant_params = {
            "method": metadata.hyperparameters.get("method", "interior-point"),
            "max_iter": metadata.hyperparameters.get("max_iter", 1000),
            "p_tol": metadata.hyperparameters.get("p_tol", 1e-6),
        }
        bundle["statquant_parameters"] = statquant_params
        
        # Add additional metadata if available
        if hasattr(metadata, 'store_id') and metadata.store_id is not None:
            bundle["store_id"] = metadata.store_id
        if hasattr(metadata, 'product_id') and metadata.product_id is not None:
            bundle["product_id"] = metadata.product_id
        if hasattr(metadata, 'model_instance') and metadata.model_instance is not None:
            bundle["model_instance"] = metadata.model_instance
        if hasattr(metadata, 'storage_location') and metadata.storage_location is not None:
            bundle["storage_location"] = metadata.storage_location
        
        return bundle