"""
Simplified configuration structures for the new BenchmarkPipeline interface.

This module provides clean, simple configuration classes that replace the complex
TrainingConfig system for single model type, single strategy experiments.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from .data_structures import ModelingStrategy, SkuList

if TYPE_CHECKING:
    from .data_structures import BenchmarkModel


@dataclass
class ExperimentConfig:
    """
    Simple configuration for a single model type experiment.
    
    This replaces the complex TrainingConfig system with a clean, direct
    parameter specification approach for single experiments.
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


@dataclass  
class ExperimentResults:
    """
    Complete results from a benchmark experiment.
    
    Contains all trained models, evaluation results, and metadata from
    a single experimental run.
    """
    models: List['BenchmarkModel']
    experiment_name: str
    model_type: str
    modeling_strategy: ModelingStrategy
    sku_tuples: SkuList
    experiment_config: ExperimentConfig
    evaluation_summary: Dict[str, Any]
    output_directory: str
    
    def __post_init__(self):
        """Validate results structure."""
        if not self.models:
            raise ValueError("ExperimentResults must contain at least one model")
        
        # Verify all models match the experiment configuration
        for model in self.models:
            if model.metadata.model_type != self.model_type:
                raise ValueError(f"Model type mismatch: expected {self.model_type}, got {model.metadata.model_type}")
            
            if model.metadata.modeling_strategy != self.modeling_strategy:
                raise ValueError(f"Modeling strategy mismatch: expected {self.modeling_strategy}, got {model.metadata.modeling_strategy}")
    
    @property
    def num_models(self) -> int:
        """Number of trained models."""
        return len(self.models)
    
    @property
    def model_identifiers(self) -> List[str]:
        """List of model identifiers."""
        return [model.get_identifier() for model in self.models]
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of model performance metrics."""
        if not self.models:
            return {}
        
        # Collect all metrics from all models
        all_metrics = {}
        for model in self.models:
            metrics = model.metadata.performance_metrics
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


def create_experiment_config(model_type: str, 
                           hyperparameters: Dict[str, Any],
                           quantile_alphas: Optional[List[float]] = None,
                           random_state: int = 42) -> ExperimentConfig:
    """
    Convenience function to create ExperimentConfig with validation.
    
    Args:
        model_type: Type of model to train (e.g., 'xgboost_standard')
        hyperparameters: Model-specific hyperparameters
        quantile_alphas: List of quantile levels for quantile models
        random_state: Random seed for reproducibility
    
    Returns:
        Validated ExperimentConfig instance
    """
    return ExperimentConfig(
        model_type=model_type,
        hyperparameters=hyperparameters.copy(),  # Copy to avoid mutation
        quantile_alphas=quantile_alphas,
        random_state=random_state
    )


def validate_sku_tuples(sku_tuples: SkuList) -> None:
    """
    Validate SKU tuples input for experiments.
    
    Args:
        sku_tuples: List of (product_id, store_id) tuples
        
    Raises:
        ValueError: If SKU tuples are invalid
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
    
    Args:
        modeling_strategy: COMBINED or INDIVIDUAL strategy
        sku_tuples: List of SKU tuples
        
    Raises:
        ValueError: If strategy is incompatible with SKU tuples
    """
    if modeling_strategy == ModelingStrategy.COMBINED and len(sku_tuples) < 1:
        raise ValueError("COMBINED strategy requires at least one SKU tuple")
    
    if modeling_strategy == ModelingStrategy.INDIVIDUAL and len(sku_tuples) < 1:
        raise ValueError("INDIVIDUAL strategy requires at least one SKU tuple")