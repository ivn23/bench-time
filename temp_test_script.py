#!/usr/bin/env python
"""
Updated test script using new BenchmarkPipeline API without legacy TrainingConfig.
"""

import sys
from pathlib import Path
import numpy as np

# Add current directory to Python path for imports
current_dir = Path.cwd()
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from src import (
    DataConfig, ModelingStrategy,
    BenchmarkPipeline
)

np.random.seed(42)

# Configuration
quantiles = [0.5, 0.7, 0.9, 0.95, 0.99]
sku_tuples = [(81054, 1334)]

data_config = DataConfig(
    features_path="../data/processed/train_data_features.feather",
    target_path="../data/train_data_target.feather",
    mapping_path="../data/feature_mapping_train.pkl"
)

# Create pipeline with simplified API
pipeline = BenchmarkPipeline(
    data_config=data_config,
    output_dir=Path("model_test")
)

print("=== Statistical Quantile Model Test ===")

# StatQuant experiment using new API
statquant_results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="statquant_model",
    hyperparameters={
        'method': 'interior-point',
        'max_iter': 1000,
        'p_tol': 1e-6,
    },
    quantile_alphas=[0.7],
    experiment_name="statquant_test_exp"
)

print(f"StatQuant experiment completed. Results saved to: {statquant_results.output_directory}")
print(f"Trained {statquant_results.num_models} model(s)")
print(f"Model identifiers: {statquant_results.model_identifiers}")

print("\n=== XGBoost Quantile Model Test ===")

# XGBoost quantile experiment  
xgb_results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="xgboost_quantile",
    hyperparameters={
        "tree_method": "hist",
        "n_estimators": 50,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
    },
    quantile_alphas=[0.7],
    experiment_name="xgb_quantile_test_exp"
)

print(f"XGBoost quantile experiment completed. Results saved to: {xgb_results.output_directory}")
print(f"Trained {xgb_results.num_models} model(s)")
print(f"Model identifiers: {xgb_results.model_identifiers}")

print("\n=== Lightning Standard Model Test ===")

# Lightning standard model experiment
lightning_results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="lightning_standard",
    hyperparameters={
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "max_epochs": 10,
        "batch_size": 32,
        "dropout": 0.1,
    },
    experiment_name="lightning_standard_test_exp"
)

print(f"Lightning standard experiment completed. Results saved to: {lightning_results.output_directory}")
print(f"Trained {lightning_results.num_models} model(s)")
print(f"Model identifiers: {lightning_results.model_identifiers}")

print("\n=== Multi-Model Comparison Experiment ===")

# XGBoost standard model
xgb_standard_results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="xgboost_standard",
    hyperparameters={
        "n_estimators": 50,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    experiment_name="xgb_standard_comparison"
)

# Multi-quantile XGBoost
xgb_multiquantile_results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="xgboost_quantile",
    hyperparameters={
        "n_estimators": 50,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    quantile_alphas=[0.1, 0.5, 0.9],
    experiment_name="xgb_multiquantile_comparison"
)

print(f"XGBoost standard: {xgb_standard_results.num_models} model(s)")
print(f"XGBoost multi-quantile: {xgb_multiquantile_results.num_models} model(s)")

print("\n=== All Tests Completed Successfully! ===")
print("New BenchmarkPipeline API working correctly without legacy TrainingConfig.")