"""
Core data structures for the benchmarking framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
from pathlib import Path
from .storage_utils import ModelStorageLocation


class ModelingStrategy(Enum):
    """Enumeration for different modeling strategies."""
    COMBINED = "combined"    # One model for all specified SKUs
    INDIVIDUAL = "individual"  # Separate model for each SKU


# Helper types for better code clarity
SkuTuple = Tuple[int, int]  # (product_id, store_id)
SkuList = List[SkuTuple]


@dataclass
class ModelMetadata:
    """Enhanced metadata for a trained model with hierarchical storage support."""
    model_id: str
    modeling_strategy: ModelingStrategy
    sku_tuples: SkuList  # List of (product_id, store_id) tuples
    model_type: str  # "xgboost_standard", "xgboost_quantile", etc.
    
    # Explicit store/product identification for hierarchical storage
    store_id: int
    product_id: int
    
    # Training configuration and performance
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""
    training_date_range: Tuple[str, str] = ("", "")
    
    # Additional metadata for hierarchical storage
    model_instance: str = "default"
    storage_location: Optional[str] = None  # Full storage path
    quantile_level: Optional[float] = None  # Quantile level for quantile models
    
    @classmethod
    def from_sku_tuple(cls, sku_tuple: SkuTuple, model_type: str, **kwargs) -> 'ModelMetadata':
        """Create ModelMetadata from a single SKU tuple."""
        product_id, store_id = sku_tuple
        return cls(
            store_id=store_id,
            product_id=product_id,
            model_type=model_type,
            sku_tuples=[sku_tuple],
            **kwargs
        )
    
    def get_sku_tuple(self) -> SkuTuple:
        """Get the primary SKU tuple for this model."""
        return (self.product_id, self.store_id)


@dataclass
class DataSplit:
    """Information about train/validation splits without storing actual data."""
    train_bdIDs: np.ndarray
    validation_bdIDs: np.ndarray
    test_bdIDs: Optional[np.ndarray] = None
    split_date: Optional[str] = None  # Date where train/validation split occurs


@dataclass 
class BenchmarkModel:
    """Container for a trained model with all its metadata."""
    metadata: ModelMetadata
    model: Any  # The actual trained model object
    data_split: DataSplit
    
    def get_identifier(self) -> str:
        """Generate unique identifier for this model using hierarchical naming."""
        # Use the model_id from metadata if it exists and is not empty
        if self.metadata.model_id and self.metadata.model_id.strip():
            return self.metadata.model_id
        
        # Generate hierarchical identifier: strategy_store_product_modeltype_instance
        strategy = self.metadata.modeling_strategy.value
        store_id = self.metadata.store_id
        product_id = self.metadata.product_id
        model_type = self.metadata.model_type
        instance = self.metadata.model_instance
        
        return f"{strategy}_{store_id}_{product_id}_{model_type}_{instance}"
    
    def get_storage_location(self) -> 'ModelStorageLocation':
        """Get the storage location for this model."""
        from .storage_utils import ModelStorageLocation
        return ModelStorageLocation(
            store_id=self.metadata.store_id,
            product_id=self.metadata.product_id,
            model_type=self.metadata.model_type,
            model_instance=self.metadata.model_instance,
            quantile_level=self.metadata.quantile_level  # Include quantile level in storage location
        )


class ModelRegistry:
    """Simplified registry for in-memory storage and management of benchmark models.
    
    Note: Model persistence is now handled by the release management system,
    not by this registry. This registry only manages in-memory model storage
    and retrieval.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.models: Dict[str, BenchmarkModel] = {}
        # Note: storage_path kept for compatibility but not used for persistence
        # Persistence is now handled by release management system
        self.storage_path = storage_path or Path("benchmark_results")
        self.storage_path.mkdir(exist_ok=True)
    
    def register_model(self, model: BenchmarkModel) -> str:
        """Register a new model in the registry."""
        model_id = model.get_identifier()
        self.models[model_id] = model
        return model_id
    
    def get_model(self, model_id: str) -> Optional[BenchmarkModel]:
        """Retrieve a model by ID."""
        return self.models.get(model_id)
    
    def list_models(self, modeling_strategy: Optional[ModelingStrategy] = None) -> List[str]:
        """List all model IDs, optionally filtered by modeling strategy."""
        if modeling_strategy is None:
            return list(self.models.keys())
        return [
            mid for mid, model in self.models.items() 
            if model.metadata.modeling_strategy == modeling_strategy
        ]


@dataclass
class DataConfig:
    """Enhanced configuration for data loading and processing."""
    features_path: str
    target_path: str
    mapping_path: str
    date_column: str = "date"
    target_column: str = "target"
    bdid_column: str = "bdID"
    
    # Filter configurations
    remove_not_for_sale: bool = True
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    
    # Default store/product filtering options
    default_stores: Optional[List[int]] = None
    default_products: Optional[List[int]] = None
    
    # Temporal split configuration - consolidated in DataConfig
    validation_split: float = 0.2          # Default 80/20 split
    split_date: Optional[str] = None       # If provided, overrides validation_split
    
    def get_split_strategy(self) -> Tuple[str, Union[float, str]]:
        """Return split strategy and parameter."""
        if self.split_date:
            return "date", self.split_date
        return "percentage", self.validation_split
    
    def validate_paths(self) -> List[str]:
        """Validate that required data files exist."""
        from pathlib import Path
        errors = []
        
        for path_name, path_value in [
            ("features_path", self.features_path),
            ("target_path", self.target_path),
            ("mapping_path", self.mapping_path)
        ]:
            if not Path(path_value).exists():
                errors.append(f"{path_name} does not exist: {path_value}")
        
        return errors
    
    def validate_split_config(self) -> List[str]:
        """Validate split configuration."""
        errors = []
        
        if self.validation_split <= 0 or self.validation_split >= 1:
            errors.append("validation_split must be between 0 and 1")
        
        if self.split_date:
            try:
                from datetime import datetime
                datetime.strptime(self.split_date, "%Y-%m-%d")
            except ValueError:
                errors.append("split_date must be in YYYY-MM-DD format")
        
        return errors


@dataclass
class ModelSelectionConfig:
    """Configuration for specifying which models to train."""
    model_types: List[str] = field(default_factory=lambda: ["xgboost_standard"])
    train_all_available: bool = False  # If True, ignore model_types and train all available
    
    def get_models_to_train(self) -> List[str]:
        """Get list of model types to train."""
        if self.train_all_available:
            from .model_types import model_registry
            return model_registry.list_available_types()
        return self.model_types


@dataclass
class ModelTypeConfig:
    """Configuration for a specific model type."""
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_specific_params: Dict[str, Any] = field(default_factory=dict)
    quantile_alpha: Optional[float] = None  # For single quantile 
    quantile_alphas: Optional[List[float]] = None  # For multiple quantile levels
    
    def __post_init__(self):
        """Validate quantile configuration after initialization."""
        # Validate quantile ranges
        if self.quantile_alphas is not None:
            for alpha in self.quantile_alphas:
                if not (0 < alpha < 1):
                    raise ValueError(f"quantile_alpha must be between 0 and 1, got {alpha}")
        
        if self.quantile_alpha is not None:
            if not (0 < self.quantile_alpha < 1):
                raise ValueError(f"quantile_alpha must be between 0 and 1, got {self.quantile_alpha}")
        
        # Handle quantile compatibility
        if self.quantile_alpha is not None and self.quantile_alphas is not None:
            raise ValueError("Cannot specify both quantile_alpha and quantile_alphas")
    
    @property
    def effective_quantile_alphas(self) -> Optional[List[float]]:
        """Get the effective list of quantile alphas (backward compatibility)."""
        if self.quantile_alphas is not None:
            return self.quantile_alphas
        elif self.quantile_alpha is not None:
            return [self.quantile_alpha]
        return None
    
    @property
    def is_quantile_model(self) -> bool:
        """Check if this configuration represents a quantile model."""
        return self.quantile_alpha is not None or self.quantile_alphas is not None
    
    def merge_with_defaults(self) -> Dict[str, Any]:
        """Merge config hyperparameters with model type defaults."""
        from .model_types import model_registry
        defaults = model_registry.get_default_hyperparameters(self.model_type)
        merged = defaults.copy()
        merged.update(self.hyperparameters)
        return merged


@dataclass
class TrainingConfig:
    """Simplified configuration for model training."""
    random_state: int = 42
    
    # Model selection configuration
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    
    # Per-model-type configurations
    model_configs: Dict[str, ModelTypeConfig] = field(default_factory=dict)
    
    def get_model_config(self, model_type: str) -> ModelTypeConfig:
        """Get configuration for a specific model type."""
        if model_type not in self.model_configs:
            # Create default config for this model type
            self.model_configs[model_type] = ModelTypeConfig(model_type=model_type)
        return self.model_configs[model_type]
    
    def add_model_config(self, model_type: str, **kwargs) -> 'TrainingConfig':
        """Add or update configuration for a specific model type."""
        config_dict = {'model_type': model_type}
        config_dict.update(kwargs)
        
        # Handle quantile lists for backward compatibility
        if 'quantile_alphas' in kwargs and 'quantile_alpha' in kwargs:
            raise ValueError("Cannot specify both quantile_alpha and quantile_alphas")
        
        self.model_configs[model_type] = ModelTypeConfig(**config_dict)
        
        # Automatically add model type to model selection if not already present
        if model_type not in self.model_selection.model_types:
            # If we're adding the first non-default model, replace the defaults
            if self.model_selection.model_types == ["xgboost_standard"] and model_type != "xgboost_standard":
                self.model_selection.model_types = [model_type]
            else:
                self.model_selection.model_types.append(model_type)
        
        return self
    
    def validate_configuration(self) -> List[str]:
        """Validate the training configuration and return any errors."""
        errors = []
        
        # Validate model types
        from .model_types import model_registry
        available_types = set(model_registry.list_available_types())
        
        for model_type in self.model_selection.get_models_to_train():
            if model_type not in available_types:
                errors.append(f"Unknown model type: {model_type}")
        
        # Validate quantile configurations
        for model_type, config in self.model_configs.items():
            if (config.quantile_alpha is not None and 
                not model_registry.requires_quantile(model_type)):
                errors.append(f"Model type {model_type} does not support quantile parameters")
        
        return errors