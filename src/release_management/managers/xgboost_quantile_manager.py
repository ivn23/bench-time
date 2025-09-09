"""
XGBoost Quantile Release Manager.

This module provides release management functionality for XGBoost quantile
models in the benchmarking framework.
"""

import typing as t
from pathlib import Path

from ..base import BaseReleaseManager

if t.TYPE_CHECKING:
    from ...data_structures import BenchmarkModel


class XGBoostQuantileReleaseManager(BaseReleaseManager):
    """
    Release manager for XGBoost quantile models.
    
    Handles XGBoost quantile models with quantile-specific bundle metadata
    and appropriate model file organization.
    """

    @property
    def family_name(self) -> str:
        """Return the unique identifier for XGBoost quantile family."""
        return "xgboost_quantile"

    def create_bundle_metadata(self, benchmark_model: "BenchmarkModel") -> dict:
        """
        Create bundle metadata from XGBoost quantile BenchmarkModel.

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
            
            # XGBoost quantile-specific information
            "xgboost_model_class": "XGBQuantileModel",
            "training_method": "xgb_train_custom_objective",
            "objective_type": "quantile_loss",
        }
        
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