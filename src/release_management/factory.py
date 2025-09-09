"""
Release Manager Factory for the benchmarking framework.

This module provides a factory pattern for selecting the appropriate
BaseReleaseManager implementation based on model type.
"""

import typing as t
from typing import Dict, Type, Optional

from .base import BaseReleaseManager

if t.TYPE_CHECKING:
    from ..data_structures import BenchmarkModel


class ReleaseManagerFactory:
    """
    Factory for creating model-specific release managers.
    
    This factory maps model_type strings to their corresponding
    BaseReleaseManager implementations.
    """
    
    _managers: Dict[str, Type[BaseReleaseManager]] = {}
    
    @classmethod
    def register_manager(cls, model_type: str, manager_class: Type[BaseReleaseManager]) -> None:
        """
        Register a release manager for a specific model type.
        
        Args:
            model_type: Model type identifier (e.g., "xgboost_standard")
            manager_class: BaseReleaseManager implementation class
        """
        cls._managers[model_type] = manager_class
    
    @classmethod
    def get_manager(cls, model_type: str) -> BaseReleaseManager:
        """
        Get appropriate release manager for model type.
        
        Args:
            model_type: Model type identifier from BenchmarkModel.metadata.model_type
            
        Returns:
            Instance of the appropriate BaseReleaseManager implementation
            
        Raises:
            ValueError: If no manager is registered for the model type
        """
        if model_type not in cls._managers:
            available_types = list(cls._managers.keys())
            raise ValueError(
                f"No release manager registered for model type '{model_type}'. "
                f"Available types: {available_types}"
            )
        
        manager_class = cls._managers[model_type]
        return manager_class()
    
    @classmethod
    def get_manager_for_model(cls, benchmark_model: "BenchmarkModel") -> BaseReleaseManager:
        """
        Get appropriate release manager for a BenchmarkModel.
        
        Args:
            benchmark_model: BenchmarkModel to get manager for
            
        Returns:
            Instance of the appropriate BaseReleaseManager implementation
        """
        return cls.get_manager(benchmark_model.metadata.model_type)
    
    @classmethod
    def list_supported_types(cls) -> list[str]:
        """
        List all registered model types.
        
        Returns:
            List of supported model type identifiers
        """
        return list(cls._managers.keys())
    
    @classmethod
    def is_supported(cls, model_type: str) -> bool:
        """
        Check if a model type is supported.
        
        Args:
            model_type: Model type identifier to check
            
        Returns:
            True if the model type is supported, False otherwise
        """
        return model_type in cls._managers