"""
Model Type Registry and Factory System

This module provides dynamic discovery and registration of available model types
in the framework, enabling extensible model support.
"""

import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Type, Any, Optional
from dataclasses import dataclass
from abc import ABC

from .models.base import BaseModel


@dataclass
class ModelTypeInfo:
    """Information about a registered model type."""
    name: str
    model_class: Type[BaseModel]
    description: str
    default_hyperparameters: Dict[str, Any]
    requires_quantile: bool = False


class ModelTypeRegistry:
    """Registry for discovering and managing available model types."""
    
    _instance: Optional['ModelTypeRegistry'] = None
    _model_types: Dict[str, ModelTypeInfo] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._discover_models()
        return cls._instance
    
    def _discover_models(self):
        """Dynamically discover model types from the models package."""
        models_dir = Path(__file__).parent / "models"
        
        # Import all model modules
        for model_file in models_dir.glob("*.py"):
            if model_file.name.startswith("_") or model_file.name == "base.py":
                continue
                
            module_name = f"src.models.{model_file.stem}"
            try:
                module = importlib.import_module(module_name)
                self._register_models_from_module(module)
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
    
    def _register_models_from_module(self, module):
        """Register all BaseModel subclasses found in a module."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj != BaseModel and 
                issubclass(obj, BaseModel) and 
                hasattr(obj, 'MODEL_TYPE')):
                
                model_info = ModelTypeInfo(
                    name=obj.MODEL_TYPE,
                    model_class=obj,
                    description=getattr(obj, 'DESCRIPTION', f"{name} model"),
                    default_hyperparameters=getattr(obj, 'DEFAULT_HYPERPARAMETERS', {}),
                    requires_quantile=getattr(obj, 'REQUIRES_QUANTILE', False)
                )
                
                self._model_types[obj.MODEL_TYPE] = model_info
    
    def get_model_info(self, model_type: str) -> Optional[ModelTypeInfo]:
        """Get information about a specific model type."""
        return self._model_types.get(model_type)
    
    def get_model_class(self, model_type: str) -> Optional[Type[BaseModel]]:
        """Get the model class for a specific model type."""
        info = self.get_model_info(model_type)
        return info.model_class if info else None
    
    def list_available_types(self) -> List[str]:
        """List all available model types."""
        return list(self._model_types.keys())
    
    def create_model(self, model_type: str, **kwargs) -> Optional[BaseModel]:
        """Create a model instance of the specified type."""
        model_class = self.get_model_class(model_type)
        if model_class:
            return model_class(**kwargs)
        return None
    
    def get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        info = self.get_model_info(model_type)
        return info.default_hyperparameters.copy() if info else {}
    
    def requires_quantile(self, model_type: str) -> bool:
        """Check if a model type requires quantile parameters."""
        info = self.get_model_info(model_type)
        return info.requires_quantile if info else False


# Global registry instance
model_registry = ModelTypeRegistry()