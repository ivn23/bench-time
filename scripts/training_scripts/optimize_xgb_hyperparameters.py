#!/usr/bin/env python3
"""
XGBoost Quantile Hyperparameter Optimization with Optuna

This script optimizes hyperparameters for XGBoost quantile models using a sample
of SKUs and Optuna-based optimization. Results can be directly used in the pipeline.
"""

import sys
import json
import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('../../')

from src.models.xgboost_quantile import XGBoostQuantileModel
from src.structures import ModelingStrategy
from src.structures import DataConfig
from src.data_loading import DataLoader


import optuna


# Global variables for objective function
GLOBAL_X_TRAIN = None
GLOBAL_Y_TRAIN = None
GLOBAL_X_VAL = None
GLOBAL_Y_VAL = None
GLOBAL_QUANTILE_ALPHA = None

def load_and_sample_skus(data_path: str, sample_size: int = 100, random_seed: int = 42) -> List[Tuple[int, int]]:
    """
    Load processed data and sample SKU combinations.

    Args:
        data_path: Path to processed feather file
        sample_size: Number of SKUs to sample
        random_seed: Random seed for reproducible sampling

    Returns:
        List of (productID, storeID) tuples
    """
    print(f"Loading data from {data_path}")
    df = pl.read_ipc(data_path)

    # Get unique SKU combinations
    unique_skus = df.select([
        pl.col("productID"),
        pl.col("storeID")
    ]).unique()

    print(f"Found {len(unique_skus)} unique SKU combinations")

    # Sample the requested number
    if len(unique_skus) < sample_size:
        print(f"Warning: Only {len(unique_skus)} SKUs available, using all")
        sample_size = len(unique_skus)

    sampled = unique_skus.sample(n=sample_size, seed=random_seed)
    sku_tuples = [(row['productID'], row['storeID']) for row in sampled.to_dicts()]

    print(f"Sampled {len(sku_tuples)} SKUs for optimization")
    return sku_tuples

def prepare_data_for_optimization(sku_tuples: List[Tuple[int, int]],
                                 data_config: DataConfig,
                                 val_split: float = 0.2,
                                 random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training and validation data for the sampled SKUs.

    Args:
        sku_tuples: List of SKU tuples to include
        data_config: Data configuration
        val_split: Fraction of data to use for validation
        random_seed: Random seed for split

    Returns:
        X_train, y_train, X_val, y_val arrays
    """
    print("Preparing data for optimization...")

    # Initialize data loader
    data_loader = DataLoader(data_config)
    data_loader.load_data()

    # Prepare combined dataset for all SKUs
    dataset = data_loader.prepare_modeling_dataset(sku_tuples, ModelingStrategy.COMBINED)

    # Get training data
    X, y = DataLoader.prepare_training_data(dataset, "xgboost_quantile")

    print(f"Combined dataset shape: X={X.shape}, y={y.shape}")

    # Create train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_seed, shuffle=True
    )

    print(f"Train split: X={X_train.shape}, y={y_train.shape}")
    print(f"Val split: X={X_val.shape}, y={y_val.shape}")

    return X_train, y_train, X_val, y_val

def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate pinball loss (quantile loss).

    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile level (e.g., 0.7)

    Returns:
        Mean quantile loss
    """
    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object

    Returns:
        Validation loss to minimize
    """
    # Suggest hyperparameters
    params = {
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'random_state': 42
    }


    # Create and train model
    model = XGBoostQuantileModel(quantile_alphas=[GLOBAL_QUANTILE_ALPHA], **params)
    model.train(GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN)

    # Make predictions on validation set
    y_pred = model.predict(GLOBAL_X_VAL)

    # Calculate quantile loss
    loss = quantile_loss(GLOBAL_Y_VAL, y_pred, GLOBAL_QUANTILE_ALPHA)

    return loss


def run_optimization(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    quantile_alpha: float = 0.7,
                    n_trials: int = 50,
                    random_seed: int = 42) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        quantile_alpha: Target quantile level
        n_trials: Number of optimization trials
        random_seed: Random seed

    Returns:
        Dictionary with best parameters and optimization results
    """
    global GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN, GLOBAL_X_VAL, GLOBAL_Y_VAL, GLOBAL_QUANTILE_ALPHA

    # Set global variables for objective function
    GLOBAL_X_TRAIN = X_train
    GLOBAL_Y_TRAIN = y_train
    GLOBAL_X_VAL = X_val
    GLOBAL_Y_VAL = y_val
    GLOBAL_QUANTILE_ALPHA = quantile_alpha

    print(f"Starting optimization with {n_trials} trials for quantile α={quantile_alpha}")

    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_seed)
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get results
    best_params = study.best_params
    best_value = study.best_value

    print(f"\nOptimization completed!")
    print(f"Best validation loss: {best_value:.6f}")
    print(f"Best parameters: {best_params}")

    return {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'quantile_alpha': quantile_alpha,
        'study': study
    }

def save_results(results: Dict[str, Any], output_path: str = "optimized_hyperparameters.json"):
    """
    Save optimization results to JSON file.

    Args:
        results: Optimization results dictionary
        output_path: Output file path
    """
    # Prepare results for JSON serialization
    output_data = {
        'best_hyperparameters': results['best_params'],
        'best_validation_loss': results['best_value'],
        'quantile_alpha': results['quantile_alpha'],
        'n_trials': results['n_trials'],
        'optimization_info': {
            'model_type': 'xgboost_quantile',
            'optimization_method': 'optuna_tpe',
            'metric': 'pinball_loss'
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("These parameters can be used directly in the pipeline:")
    print(f"hyperparameters = {json.dumps(results['best_params'], indent=2)}")

def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="Optimize XGBoost quantile hyperparameters")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of SKUs to sample for optimization")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of Optuna trials")
    parser.add_argument("--quantile-alpha", type=float, default=0.7,
                       help="Target quantile level")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split fraction")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="optimized_hyperparameters.json",
                       help="Output file for optimized parameters")

    args = parser.parse_args()

    print("XGBoost Quantile Hyperparameter Optimization")
    print("=" * 50)

    # Data paths (same as other scripts)
    data_path = "../../data/db_snapshot_offsite/train_data/processed/train_data_features.feather"

    # Check if data file exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Make sure you've run the data processing pipeline first.")
        sys.exit(1)


    # Step 1: Sample SKUs
    sku_tuples = load_and_sample_skus(
        data_path,
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )

    # Step 2: Prepare data
    data_config = DataConfig(
        features_path=data_path,
        target_path="../../data/db_snapshot_offsite/train_data/train_data_target.feather",
        mapping_path="../../data/feature_mapping_train.pkl",
        split_date="2016-01-01"
    )

    X_train, y_train, X_val, y_val = prepare_data_for_optimization(
        sku_tuples,
        data_config,
        val_split=args.val_split,
        random_seed=args.random_seed
    )

    # Step 3: Run optimization
    results = run_optimization(
        X_train, y_train, X_val, y_val,
        quantile_alpha=args.quantile_alpha,
        n_trials=args.n_trials,
        random_seed=args.random_seed
    )

    # Step 4: Save results
    save_results(results, args.output)

    print("\nOptimization completed successfully!")

if __name__ == "__main__":
    main()