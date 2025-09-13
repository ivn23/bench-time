"""
M5 Benchmarking Framework

A comprehensive framework for multi-SKU time series forecasting benchmarking
using the M5 competition dataset.
"""

from .data_structures import (
    ModelingStrategy, SkuTuple, SkuList, ModelMetadata, DataSplit, BenchmarkModel, 
    ModelRegistry, DataConfig, TrainingConfig, ModelTypeConfig, ExperimentResults, ModelingDataset
)

from .data_loading import DataLoader
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator, VisualizationGenerator
from .benchmark_pipeline import BenchmarkPipeline
from .release_management import ComprehensiveReleaseManager

__version__ = "1.0.0"
__author__ = "M5 Benchmarking Team"

__all__ = [
    "ModelingStrategy",
    "SkuTuple",
    "SkuList",
    "ModelMetadata", 
    "DataSplit",
    "BenchmarkModel",
    "ModelRegistry",
    "DataConfig",
    "TrainingConfig",
    "ModelTypeConfig",
    "ExperimentResults",
    "ModelingDataset",
    "DataLoader",
    "ModelTrainer",
    "ModelEvaluator",
    "VisualizationGenerator",
    "BenchmarkPipeline",
    "ComprehensiveReleaseManager"
]