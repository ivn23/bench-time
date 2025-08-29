"""
Model implementations for the M5 benchmarking framework.

This module provides an extensible model architecture using the factory pattern.
Different model types can be easily added by implementing the BaseModel interface.
"""

from .base import BaseModel
from .xgboost_standard import XGBoostStandardModel
from .xgboost_quantile import XGBoostQuantileModel
from .lightning_standard import LightningStandardModel
from .lightning_quantile import LightningQuantileModel

# Model registry mapping
MODEL_REGISTRY = {
    "xgboost": XGBoostStandardModel,
    "xgboost_quantile": XGBoostQuantileModel,
    "lightning_standard": LightningStandardModel,
    "lightning_quantile": LightningQuantileModel,
}


def get_model_class(model_type: str):
    """
    Factory function to get model class by type.
    
    Args:
        model_type: String identifier for model type
        
    Returns:
        Model class implementing BaseModel interface
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_REGISTRY:
        available_types = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available types: {available_types}")
    
    return MODEL_REGISTRY[model_type]


def list_available_models():
    """Get list of all available model types."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "BaseModel",
    "XGBoostStandardModel", 
    "XGBoostQuantileModel",
    "LightningStandardModel",
    "LightningQuantileModel",
    "get_model_class",
    "list_available_models"
]