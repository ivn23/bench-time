"""
Utility functions for the M5 Benchmarking Framework.

This module provides helper functions for:
- Saving and loading hyperparameter tuning results
- SKU data loading and filtering
- File I/O operations
- Data formatting and conversion
"""

import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional, Union, List, Tuple


def save_hp_tuning_results(
    tune_result: Any,
    model_type: str,
    quantile_alpha: Optional[float],
    tune_on: int,
    n_trials: int,
    n_folds: int,
    execution_time: float,
    output_dir: Union[str, Path] = "HP_RESULTS"
) -> Path:
    """
    Save hyperparameter tuning results to a CSV file with metadata.

    Args:
        tune_result: TuningResult object from hyperparameter optimization
        model_type: Type of model (e.g., 'xgboost_quantile', 'lightning_standard')
        quantile_alpha: Quantile level for quantile models (None for standard models)
        tune_on: Number of SKUs used for tuning
        n_trials: Number of Optuna trials executed
        n_folds: Number of cross-validation folds
        execution_time: Total optimization time in seconds
        output_dir: Base directory for saving results (default: 'HP_RESULTS')

    Returns:
        Path to the saved CSV file

    Example:
        >>> save_hp_tuning_results(
        ...     tune_result=tune_result,
        ...     model_type="xgboost_quantile",
        ...     quantile_alpha=0.7,
        ...     tune_on=100,
        ...     n_trials=100,
        ...     n_folds=5,
        ...     execution_time=3600.5
        ... )
        Path('HP_RESULTS/xgb/xgboost_quantile_q0.7_tuned100_trials100_20251106_143022.csv')
    """
    # Create subdirectory based on model type
    model_subdir = model_type.split('_')[0]  # e.g., 'xgboost' -> 'xgb', 'lightning' -> 'lightning'
    if model_subdir == 'xgboost':
        model_subdir = 'xgb'
    elif model_subdir == 'lightning':
        model_subdir = 'mlp'

    output_path = Path(output_dir) / model_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quantile_str = f"_q{quantile_alpha}" if quantile_alpha is not None else ""
    filename = f"{model_type}{quantile_str}_tuned{tune_on}_trials{n_trials}_{timestamp}.csv"
    filepath = output_path / filename

    # Create DataFrame with two sections: metadata and hyperparameters
    rows = []

    # Metadata section
    rows.append({
        'section': 'metadata',
        'key': 'model_type',
        'value': model_type,
        'dtype': 'str'
    })
    rows.append({
        'section': 'metadata',
        'key': 'quantile_alpha',
        'value': quantile_alpha if quantile_alpha is not None else 'N/A',
        'dtype': 'float' if quantile_alpha is not None else 'str'
    })
    rows.append({
        'section': 'metadata',
        'key': 'n_skus_sampled',
        'value': tune_on,
        'dtype': 'int'
    })
    rows.append({
        'section': 'metadata',
        'key': 'n_trials',
        'value': n_trials,
        'dtype': 'int'
    })
    rows.append({
        'section': 'metadata',
        'key': 'n_folds',
        'value': n_folds,
        'dtype': 'int'
    })
    rows.append({
        'section': 'metadata',
        'key': 'best_score',
        'value': tune_result.best_score,
        'dtype': 'float'
    })
    rows.append({
        'section': 'metadata',
        'key': 'execution_time_seconds',
        'value': execution_time,
        'dtype': 'float'
    })
    rows.append({
        'section': 'metadata',
        'key': 'timestamp',
        'value': timestamp,
        'dtype': 'str'
    })

    # Hyperparameters section
    for param_name, param_value in tune_result.best_params.items():
        # Infer dtype from value
        if isinstance(param_value, bool):
            dtype = 'bool'
        elif isinstance(param_value, int):
            dtype = 'int'
        elif isinstance(param_value, float):
            dtype = 'float'
        else:
            dtype = 'str'

        rows.append({
            'section': 'hyperparameter',
            'key': param_name,
            'value': param_value,
            'dtype': dtype
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

    return filepath


def load_hp_tuning_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load hyperparameter tuning results from CSV and convert to dict.

    Args:
        filepath: Path to the CSV file saved by save_hp_tuning_results()

    Returns:
        Dictionary of hyperparameters ready for pipeline.run_experiment()

    Example:
        >>> best_params = load_hp_tuning_results("HP_RESULTS/xgb/xgboost_quantile_q0.7_tuned100_trials100_20251106_143022.csv")
        >>> pipeline.run_experiment(
        ...     sku_tuples=all_skus,
        ...     modeling_strategy=ModelingStrategy.COMBINED,
        ...     model_type="xgboost_quantile",
        ...     quantile_alphas=[0.7],
        ...     hyperparameters=best_params
        ... )
    """
    # Read CSV
    df = pd.read_csv(filepath)

    # Filter for hyperparameter rows only
    hp_df = df[df['section'] == 'hyperparameter']

    # Convert to dictionary with correct types
    params = {}
    for _, row in hp_df.iterrows():
        key = row['key']
        value = row['value']
        dtype = row['dtype']

        # Convert value to correct type
        if dtype == 'int':
            params[key] = int(value)
        elif dtype == 'float':
            params[key] = float(value)
        elif dtype == 'bool':
            params[key] = bool(value) if isinstance(value, bool) else str(value).lower() == 'true'
        else:
            params[key] = str(value)

    return params


def get_skus(
    data_path: Union[str, Path],
    exclude_after_date: str = "2016-01-01"
) -> List[Tuple[int, int]]:
    """
    Load and filter SKU tuples from time series data.

    This function loads a dataset and returns a list of (productID, storeID) tuples,
    excluding SKUs that first appear on or after a specified date. This is useful for
    filtering out incomplete time series that don't have sufficient historical data.

    Args:
        data_path: Path to the features data file (Feather/Arrow format)
        exclude_after_date: ISO format date string (YYYY-MM-DD). SKUs that first
                           appear on or after this date will be excluded.
                           Default: "2016-01-01"

    Returns:
        List of (productID, storeID) tuples for SKUs with complete time series

    Example:
        >>> # Get all SKUs with data before 2016-01-01
        >>> sku_tuples = get_skus("data/train_data_features.feather")
        >>> print(f"Found {len(sku_tuples)} complete SKUs")
        Found 3049 complete SKUs

        >>> # Use custom date cutoff
        >>> sku_tuples = get_skus(
        ...     "data/train_data_features.feather",
        ...     exclude_after_date="2015-06-01"
        ... )
    """
    # Load data
    df_clean = pl.read_ipc(data_path)

    # Extract all unique SKU tuples
    sku_tuples_all = [
        (d['productID'], d['storeID'])
        for d in df_clean.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()
    ]

    # Parse the date string
    cutoff_date = datetime.strptime(exclude_after_date, "%Y-%m-%d").date()

    # Find SKUs to exclude (those that start on or after the cutoff date)
    sku_exclude = (
        df_clean
        .group_by("storeID", "productID")
        .agg(pl.col("date").first())
        .filter(pl.col("date") >= cutoff_date)
        .select("productID", "storeID")
    )

    sku_exclude_list = [
        (d['productID'], d['storeID'])
        for d in sku_exclude.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()
    ]

    # Return complete SKUs (all SKUs minus excluded ones)
    sku_tuples_complete = [sku for sku in sku_tuples_all if sku not in sku_exclude_list]

    return sku_tuples_complete
