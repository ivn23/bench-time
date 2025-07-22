"""
M5 Benchmarking Framework

A comprehensive framework for multi-granularity time series forecasting benchmarking
using the M5 competition dataset.
"""

from .data_structures import (
    GranularityLevel, ModelMetadata, DataSplit, BenchmarkModel, 
    ModelRegistry, DataConfig, TrainingConfig
)

from .data_loading import DataLoader
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer, EnsembleTrainer
from .evaluation import ModelEvaluator, VisualizationGenerator
from .benchmark_pipeline import BenchmarkPipeline

__version__ = "1.0.0"
__author__ = "M5 Benchmarking Team"

__all__ = [
    "GranularityLevel",
    "ModelMetadata", 
    "DataSplit",
    "BenchmarkModel",
    "ModelRegistry",
    "DataConfig",
    "TrainingConfig", 
    "DataLoader",
    "FeatureEngineer",
    "ModelTrainer",
    "EnsembleTrainer",
    "ModelEvaluator",
    "VisualizationGenerator",
    "BenchmarkPipeline"
]