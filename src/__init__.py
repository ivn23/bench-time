"""
M5 Benchmarking Framework

A simplified framework for multi-SKU time series forecasting benchmarking
using the M5 competition dataset.

This version uses clean, focused data structures for streamlined functionality.
"""

# Data structures and core functionality
from .structures import (
    ModelingStrategy, SkuTuple, SkuList, DataConfig, ModelingDataset,
    ModelConfig, SplitInfo, TrainingResult, ExperimentResults,
    create_config, validate_sku_tuples, validate_modeling_strategy
)

# Core modules
from .data_loading import DataLoader
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator
from .pipeline import BenchmarkPipeline
from .release_management import ReleaseManager
from .hyperparameter_tuning import HyperparameterTuner, TuningResult

__version__ = "2.0.0"
__author__ = "M5 Benchmarking Team"

__all__ = [
    # Core types
    "ModelingStrategy",
    "SkuTuple",
    "SkuList",

    # Data structures
    "DataConfig",
    "ModelingDataset",
    "ModelConfig",
    "SplitInfo",
    "TrainingResult",
    "ExperimentResults",
    
    # Factory functions
    "create_config",
    "validate_sku_tuples",
    "validate_modeling_strategy",
    
    # Core modules
    "DataLoader",
    "ModelTrainer",
    "ModelEvaluator",
    "BenchmarkPipeline",
    "ReleaseManager",
    "HyperparameterTuner",
    "TuningResult"
]