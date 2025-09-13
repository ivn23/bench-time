"""
Core data structures for the benchmarking framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
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


@dataclass
class ExperimentResults:
    """Container for complete experiment results before release packaging."""
    models: List[BenchmarkModel]
    evaluation_results: Optional[Dict[str, Any]]  # Optional metrics from ModelEvaluator
    experiment_log: Dict[str, Any]
    configurations: Dict[str, Any]  # DataConfig, TrainingConfig
    experiment_name: str
    timestamp: datetime


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
        # Removed directory creation - ComprehensiveReleaseManager handles all persistence
    
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
class ModelingDataset:
    """Complete dataset ready for model training."""
    X_train: pl.DataFrame
    y_train: pl.DataFrame
    X_test: pl.DataFrame
    y_test: pl.DataFrame
    feature_cols: List[str]
    target_col: str
    split_info: Dict[str, Any]  # Contains split_date, train_bdids, test_bdids
    dataset_stats: Dict[str, Any]  # Contains n_samples, n_features
    sku_tuples: SkuList
    modeling_strategy: ModelingStrategy


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



@dataclass
class ModelTypeConfig:
    """Configuration for a specific model type."""
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_specific_params: Dict[str, Any] = field(default_factory=dict)
    quantile_alphas: Optional[List[float]] = None  # For quantile levels
    
    def __post_init__(self):
        """Validate quantile configuration after initialization."""
        # Validate quantile ranges
        if self.quantile_alphas is not None:
            for alpha in self.quantile_alphas:
                if not (0 < alpha < 1):
                    raise ValueError(f"quantile_alphas values must be between 0 and 1, got {alpha}")
    
    @property
    def is_quantile_model(self) -> bool:
        """Check if this configuration represents a quantile model."""
        return self.quantile_alphas is not None
    
    def merge_with_defaults(self) -> Dict[str, Any]:
        """Merge config hyperparameters with model type defaults."""
        from .model_types import model_registry
        defaults = model_registry.get_default_hyperparameters(self.model_type)
        merged = defaults.copy()
        merged.update(self.hyperparameters)
        return merged


@dataclass
class TrainingConfig:
    """Simplified configuration for single model type training."""
    random_state: int = 42
    
    # Single model configuration (replaces multi-model complexity)
    model_config: ModelTypeConfig = field(default_factory=lambda: ModelTypeConfig(model_type="xgboost_standard"))
    
    def set_model_config(self, model_type: str, **kwargs) -> 'TrainingConfig':
        """Set the single model configuration."""
        config_dict = {'model_type': model_type}
        config_dict.update(kwargs)
        
        
        self.model_config = ModelTypeConfig(**config_dict)
        return self
    
    @property
    def model_type(self) -> str:
        """Get the configured model type."""
        return self.model_config.model_type