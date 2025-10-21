"""
Simplified data structures for the benchmarking framework.

This module provides clean, minimal data structures focused on the user's actual needs:
- Data split information
- SKU information  
- Model parameters
- Training loss
- Optional test performance metrics

Replaces the overengineered 11+ dataclass system with 3 essential classes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import polars as pl
from datetime import datetime

# Keep these from original - they're well designed
class ModelingStrategy(Enum):
    """Enumeration for different modeling strategies."""
    COMBINED = "combined"    # One model for all specified SKUs
    INDIVIDUAL = "individual"  # Separate model for each SKU

# Helper types for better code clarity
SkuTuple = Tuple[int, int]  # (product_id, store_id)
SkuList = List[SkuTuple]


@dataclass
class DataConfig:
    """
    Configuration for data loading and processing.
    
    Moved from data_structures.py - this class is well-designed and used
    by both legacy and simplified systems.
    """
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
    validation_split: float = 0.2          # Default 80/20 train/test split (creates test data, not validation)
    split_date: Optional[str] = None       # If provided, overrides validation_split


@dataclass
class ModelingDataset:
    """
    Dataset container for model training.
    
    Simplified version of ModelingDataset that works directly with the
    simplified structures without legacy dependencies.
    """
    X_train: pl.DataFrame
    y_train: pl.DataFrame
    X_test: pl.DataFrame
    y_test: pl.DataFrame
    feature_cols: List[str]
    target_col: str
    sku_tuples: SkuList
    modeling_strategy: ModelingStrategy

    # Feature normalization scaler (fitted on training data)
    scaler: Optional[Any] = None

    # Split information embedded directly (replaces separate DataSplit)
    train_bdids: np.ndarray = field(default_factory=lambda: np.array([]))
    test_bdids: np.ndarray = field(default_factory=lambda: np.array([]))
    split_date: Optional[str] = None
    
    # Basic dataset statistics
    n_train_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    
    def __post_init__(self):
        """Calculate dataset statistics after initialization."""
        self.n_train_samples = len(self.X_train)
        self.n_test_samples = len(self.X_test)
        self.n_features = len(self.feature_cols)
        
        # Extract bdIDs if not provided
        if len(self.train_bdids) == 0:
            if 'bdID' in self.X_train.columns:
                self.train_bdids = self.X_train['bdID'].to_numpy()
        
        if len(self.test_bdids) == 0:
            if 'bdID' in self.X_test.columns:
                self.test_bdids = self.X_test['bdID'].to_numpy()
    
    def get_split_info(self) -> SplitInfo:
        """Convert to SplitInfo object for TrainingResult."""
        return SplitInfo(
            train_bdIDs=self.train_bdids,
            test_bdIDs=self.test_bdids,
            split_date=self.split_date
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary information."""
        return {
            "n_train_samples": self.n_train_samples,
            "n_test_samples": self.n_test_samples,
            "n_features": self.n_features,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "sku_tuples": self.sku_tuples,
            "modeling_strategy": self.modeling_strategy.value,
            "split_date": self.split_date
        }


@dataclass
class ModelConfig:
    """
    Unified configuration for model training experiments.
    
    Combines the functionality of ExperimentConfig and ModelTypeConfig
    into a single, simple configuration class.
    """
    model_type: str
    hyperparameters: Dict[str, Any]
    quantile_alphas: Optional[List[float]] = None
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.model_type:
            raise ValueError("model_type cannot be empty")
        
        if not isinstance(self.hyperparameters, dict):
            raise TypeError("hyperparameters must be a dictionary")
        
        # Validate quantile_alphas if provided
        if self.quantile_alphas is not None:
            if not isinstance(self.quantile_alphas, list):
                raise TypeError("quantile_alphas must be a list of floats")
            
            for alpha in self.quantile_alphas:
                if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
                    raise ValueError(f"quantile_alphas must be between 0 and 1, got {alpha}")
        
        # Add random_state to hyperparameters if not present
        if 'random_state' not in self.hyperparameters:
            self.hyperparameters['random_state'] = self.random_state

    @property
    def is_quantile_model(self) -> bool:
        """Check if this configuration represents a quantile model."""
        return self.quantile_alphas is not None


@dataclass
class SplitInfo:
    """
    Information about train/test data splits.
    
    Embedded directly in TrainingResult instead of being a separate class.
    Note: This framework uses fixed hyperparameters, so the holdout data is test data,
    not validation data (which would be used for hyperparameter tuning).
    """
    train_bdIDs: np.ndarray
    test_bdIDs: np.ndarray
    split_date: Optional[str] = None
    
    def __post_init__(self):
        """Validate split arrays."""
        if len(self.train_bdIDs) == 0:
            raise ValueError("train_bdIDs cannot be empty")
        if len(self.test_bdIDs) == 0:
            raise ValueError("test_bdIDs cannot be empty")


@dataclass
class TrainingResult:
    """
    Complete result from training a single model.
    
    This class consolidates TrainedModel + BenchmarkModel + ModelMetadata
    into a single result object containing exactly what the user needs:
    - The trained model object
    - Data split information
    - SKU information
    - Model parameters
    - Test performance metrics (optional)
    """
    # Core model information
    model: Any  # The actual trained model object
    model_type: str
    modeling_strategy: ModelingStrategy
    sku_tuples: SkuList
    
    # Training configuration
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    
    # Data split information (embedded)
    split_info: SplitInfo
    
    # Results and metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Optional metadata
    quantile_level: Optional[float] = None  # For quantile models
    model_id: Optional[str] = None
    training_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set defaults and validate."""
        # Validate SKU tuples first (before generating model_id)
        if not self.sku_tuples:
            raise ValueError("sku_tuples cannot be empty")
            
        for sku in self.sku_tuples:
            if not isinstance(sku, tuple) or len(sku) != 2:
                raise ValueError(f"Each SKU must be a 2-tuple (product_id, store_id), got {sku}")
        
        if self.training_timestamp is None:
            self.training_timestamp = datetime.now()
            
        if self.model_id is None:
            self.model_id = self._generate_model_id()
    
    def _generate_model_id(self) -> str:
        """Generate a simple, readable model identifier."""
        strategy = self.modeling_strategy.value
        sku_part = f"{len(self.sku_tuples)}skus" if len(self.sku_tuples) > 1 else f"{self.sku_tuples[0][0]}x{self.sku_tuples[0][1]}"
        quantile_part = f"_q{self.quantile_level}" if self.quantile_level else ""
        timestamp = self.training_timestamp.strftime("%H%M%S") if self.training_timestamp else "unknown"
        
        return f"{self.model_type}_{strategy}_{sku_part}{quantile_part}_{timestamp}"
    
    def get_identifier(self) -> str:
        """Return the model identifier."""
        return self.model_id
    
    def has_test_metrics(self) -> bool:
        """Check if test performance metrics are available."""
        return bool(self.performance_metrics)
    
    def get_primary_sku(self) -> SkuTuple:
        """Get the primary SKU (first one in the list)."""
        return self.sku_tuples[0]
    
    def is_quantile_model(self) -> bool:
        """Check if this is a quantile model result."""
        return self.quantile_level is not None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this training result."""
        summary = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "strategy": self.modeling_strategy.value,
            "num_skus": len(self.sku_tuples),
            "skus": self.sku_tuples,
            "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
            "has_test_metrics": self.has_test_metrics()
        }
        
        if self.quantile_level is not None:
            summary["quantile_level"] = self.quantile_level
            
        if self.performance_metrics:
            summary["test_metrics"] = self.performance_metrics
            
        return summary


@dataclass
class ExperimentResults:
    """
    Results from a complete experiment (potentially multiple models).
    
    Simplified version of the original ExperimentResults that works with
    the new TrainingResult objects.
    """
    training_results: List[TrainingResult]
    experiment_name: str
    config: ModelConfig
    experiment_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate results structure."""
        if not self.training_results:
            raise ValueError("ExperimentResults must contain at least one training result")
    
    @property
    def num_models(self) -> int:
        """Number of trained models."""
        return len(self.training_results)
    
    @property
    def model_identifiers(self) -> List[str]:
        """List of model identifiers."""
        return [result.get_identifier() for result in self.training_results]
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of model performance metrics."""
        if not self.training_results:
            return {}
        
        # Collect all metrics from all models
        all_metrics = {}
        for result in self.training_results:
            metrics = result.performance_metrics
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in all_metrics.items():
            if len(values) == 1:
                summary[f"{metric_name}"] = values[0]
            else:
                summary[f"{metric_name}_mean"] = sum(values) / len(values)
                summary[f"{metric_name}_min"] = min(values)
                summary[f"{metric_name}_max"] = max(values)
        
        return summary
    
    def get_results_by_strategy(self, strategy: ModelingStrategy) -> List[TrainingResult]:
        """Get training results filtered by modeling strategy."""
        return [result for result in self.training_results 
                if result.modeling_strategy == strategy]
    
    def get_quantile_results(self) -> List[TrainingResult]:
        """Get only quantile model results."""
        return [result for result in self.training_results 
                if result.is_quantile_model()]


# Convenience functions
def create_config(model_type: str, 
                        hyperparameters: Dict[str, Any],
                        quantile_alphas: Optional[List[float]] = None,
                        random_state: int = 42) -> ModelConfig:
    """
    Convenience function to create ModelConfig with validation.
    """
    return ModelConfig(
        model_type=model_type,
        hyperparameters=hyperparameters.copy(),  # Copy to avoid mutation
        quantile_alphas=quantile_alphas,
        random_state=random_state
    )


def validate_sku_tuples(sku_tuples: SkuList) -> None:
    """
    Validate SKU tuples input for experiments.
    """
    if not sku_tuples:
        raise ValueError("At least one SKU tuple must be provided")
    
    if not isinstance(sku_tuples, list):
        raise TypeError("sku_tuples must be a list")
    
    for i, sku_tuple in enumerate(sku_tuples):
        if not isinstance(sku_tuple, tuple) or len(sku_tuple) != 2:
            raise ValueError(f"SKU tuple {i} must be a 2-tuple (product_id, store_id), got {sku_tuple}")
        
        product_id, store_id = sku_tuple
        if not isinstance(product_id, int) or not isinstance(store_id, int):
            raise ValueError(f"SKU tuple {i} must contain integers, got ({type(product_id)}, {type(store_id)})")
        
        if product_id <= 0 or store_id <= 0:
            raise ValueError(f"SKU tuple {i} must contain positive integers, got {sku_tuple}")


def validate_modeling_strategy(modeling_strategy: ModelingStrategy,
                             sku_tuples: SkuList) -> None:
    """
    Validate modeling strategy against SKU tuples.
    """
    if modeling_strategy == ModelingStrategy.COMBINED and len(sku_tuples) < 1:
        raise ValueError("COMBINED strategy requires at least one SKU tuple")
    
    if modeling_strategy == ModelingStrategy.INDIVIDUAL and len(sku_tuples) < 1:
        raise ValueError("INDIVIDUAL strategy requires at least one SKU tuple")