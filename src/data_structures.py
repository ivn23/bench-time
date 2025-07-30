"""
Core data structures for the benchmarking framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
import polars as pl
from pathlib import Path
import pickle
import json


class GranularityLevel(Enum):
    """Enumeration for different granularity levels."""
    SKU = "sku"          # Individual product at specific store
    PRODUCT = "product"   # Same product across multiple stores
    STORE = "store"      # All products within specific store
    GLOBAL = "global"      # All products across all stores


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    granularity: GranularityLevel
    entity_ids: Dict[str, Union[int, List[int]]]  # SKU/Product/Store IDs
    model_type: str  # "xgboost", "linear", etc.
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""
    training_date_range: Tuple[str, str] = ("", "")
    validation_date_range: Tuple[str, str] = ("", "")


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
        """Generate unique identifier for this model."""
        granularity = self.metadata.granularity.value
        entity_str = "_".join(str(v) for v in self.metadata.entity_ids.values())
        return f"{granularity}_{entity_str}_{self.metadata.model_type}"


class ModelRegistry:
    """Registry for storing and managing benchmark models."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.models: Dict[str, BenchmarkModel] = {}
        self.storage_path = storage_path or Path("models")
        self.storage_path.mkdir(exist_ok=True)
    
    def register_model(self, model: BenchmarkModel) -> str:
        """Register a new model in the registry."""
        model_id = model.get_identifier()
        self.models[model_id] = model
        return model_id
    
    def get_model(self, model_id: str) -> Optional[BenchmarkModel]:
        """Retrieve a model by ID."""
        return self.models.get(model_id)
    
    def list_models(self, granularity: Optional[GranularityLevel] = None) -> List[str]:
        """List all model IDs, optionally filtered by granularity."""
        if granularity is None:
            return list(self.models.keys())
        return [
            mid for mid, model in self.models.items() 
            if model.metadata.granularity == granularity
        ]
    
    def save_model(self, model_id: str, save_data_splits: bool = True):
        """Save a model to disk."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model = self.models[model_id]
        model_dir = self.storage_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model object
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model.model, f)
        
        # Save metadata as JSON
        metadata_dict = {
            "model_id": model.metadata.model_id,
            "granularity": model.metadata.granularity.value,
            "entity_ids": model.metadata.entity_ids,
            "model_type": model.metadata.model_type,
            "hyperparameters": model.metadata.hyperparameters,
            "training_config": model.metadata.training_config,
            "performance_metrics": model.metadata.performance_metrics,
            "feature_columns": model.metadata.feature_columns,
            "target_column": model.metadata.target_column,
            "training_date_range": model.metadata.training_date_range,
            "validation_date_range": model.metadata.validation_date_range
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Save data splits
        if save_data_splits:
            splits_dict = {
                "train_bdIDs": model.data_split.train_bdIDs.tolist(),
                "validation_bdIDs": model.data_split.validation_bdIDs.tolist(),
                "test_bdIDs": model.data_split.test_bdIDs.tolist() if model.data_split.test_bdIDs is not None else None,
                "split_date": model.data_split.split_date
            }
            
            with open(model_dir / "data_splits.json", "w") as f:
                json.dump(splits_dict, f)
    
    def load_model(self, model_id: str) -> BenchmarkModel:
        """Load a model from disk."""
        model_dir = self.storage_path / model_id
        
        if not model_dir.exists():
            raise ValueError(f"Model directory {model_dir} not found")
        
        # Load model object
        with open(model_dir / "model.pkl", "rb") as f:
            model_obj = pickle.load(f)
        
        # Load metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata(
            model_id=metadata_dict["model_id"],
            granularity=GranularityLevel(metadata_dict["granularity"]),
            entity_ids=metadata_dict["entity_ids"],
            model_type=metadata_dict["model_type"],
            hyperparameters=metadata_dict["hyperparameters"],
            training_config=metadata_dict["training_config"],
            performance_metrics=metadata_dict["performance_metrics"],
            feature_columns=metadata_dict["feature_columns"],
            target_column=metadata_dict["target_column"],
            training_date_range=tuple(metadata_dict["training_date_range"]),
            validation_date_range=tuple(metadata_dict["validation_date_range"])
        )
        
        # Load data splits
        with open(model_dir / "data_splits.json", "r") as f:
            splits_dict = json.load(f)
        
        data_split = DataSplit(
            train_bdIDs=np.array(splits_dict["train_bdIDs"]),
            validation_bdIDs=np.array(splits_dict["validation_bdIDs"]),
            test_bdIDs=np.array(splits_dict["test_bdIDs"]) if splits_dict["test_bdIDs"] is not None else None,
            split_date=splits_dict["split_date"]
        )
        
        benchmark_model = BenchmarkModel(
            metadata=metadata,
            model=model_obj,
            data_split=data_split
        )
        
        # Register in memory
        self.models[model_id] = benchmark_model
        
        return benchmark_model


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
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
    
    # Feature engineering configurations
    lag_features: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    calendric_features: bool = True
    trend_features: bool = True


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    validation_split: float = 0.2
    random_state: int = 42
    n_trials: int = 50  # For Optuna optimization
    cv_folds: int = 5   # For TimeSeriesSplit
    
    # Model-specific configurations
    model_type: str = "xgboost"
    model_params: Dict[str, Any] = field(default_factory=dict)