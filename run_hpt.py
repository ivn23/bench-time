import multiprocessing
import sys

# CRITICAL: Set multiprocessing start method BEFORE any other imports
# This must be the first thing in the script to prevent segfaults
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

from pathlib import Path
import numpy as np
import polars as pl
import time
from datetime import date

current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src import (
    DataConfig, ComputeConfig, ModelingStrategy, BenchmarkPipeline
)

# ============================================================================
# COMPUTE RESOURCE CONFIGURATION
# ============================================================================
# Explicitly configure compute resources for hyperparameter tuning.
# All 4 settings are required. No defaults, no magic.
#
# accelerator: Device for PyTorch Lightning ("cpu", "gpu", "auto", "mps")
# dataloader_workers: PyTorch DataLoader num_workers (0 = main process only)
# optuna_n_jobs: Optuna parallel trials (1 = sequential, >1 = parallel)
# torch_threads: torch.set_num_threads (None = no limit, 1 = single-threaded)
# ============================================================================

# macOS Development Configuration (Current)
compute_config = ComputeConfig(
    accelerator="cpu",          # macOS M4: Use CPU (MPS not stable for Lightning)
    dataloader_workers=0,       # Main process only (safest on macOS)
    optuna_n_jobs=1,           # Sequential trials (avoids segfaults)
    torch_threads=1            # Single-threaded (prevents multiprocessing conflicts)
)

# Linux CUDA Configuration (Uncomment for GPU server)
# compute_config = ComputeConfig(
#     accelerator="gpu",          # Use CUDA GPUs
#     dataloader_workers=4,       # Parallel data loading (adjust based on CPU cores)
#     optuna_n_jobs=1,           # Sequential trials (GPU memory constraints)
#     torch_threads=None         # No thread limit (let PyTorch decide)
# )

# Linux CPU Parallelized Configuration (Uncomment for CPU cluster)
# compute_config = ComputeConfig(
#     accelerator="cpu",          # CPU only
#     dataloader_workers=2,       # Some parallel data loading
#     optuna_n_jobs=4,           # Parallel trials (adjust based on CPU cores)
#     torch_threads=2            # Limit threads per trial
# )

# Apply torch threading configuration
import torch
if compute_config.torch_threads is not None:
    torch.set_num_threads(compute_config.torch_threads)
    print(f"PyTorch threads limited to: {compute_config.torch_threads}")
else:
    print("PyTorch threads: unlimited")

# Print compute configuration
print("=" * 60)
print("COMPUTE CONFIGURATION")
print("=" * 60)
print(f"Accelerator: {compute_config.accelerator}")
print(f"DataLoader workers: {compute_config.dataloader_workers}")
print(f"Optuna parallel jobs: {compute_config.optuna_n_jobs}")
print(f"Torch threads: {compute_config.torch_threads if compute_config.torch_threads else 'unlimited'}")
print("=" * 60)

np.random.seed(42)


data_path = "data/db_snapshot_offsite/train_data/processed/train_data_features.feather"
df_clean = pl.read_ipc(data_path)

sku_tuples_all = [(d['productID'], d['storeID']) for d in df_clean.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()]

sku_exclude = (df_clean
 .group_by("storeID","productID")
 .agg(pl.col("date").first())
 .filter(pl.col("date") >= date(2016,1,1))
 .select("productID","storeID")
 )

sku_exclude = [(d['productID'], d['storeID']) for d in sku_exclude.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()]

sku_tuples_complete =  [sku for sku in sku_tuples_all if sku not in sku_exclude]

data_config = DataConfig(
    mapping_path = 'data/feature_mapping_train.pkl',
    features_path = "data/db_snapshot_offsite/train_data/processed/train_data_features.feather",
    target_path = "data/db_snapshot_offsite/train_data/train_data/train_data_target.feather",
    split_date="2016-01-01",
)

pipeline = BenchmarkPipeline(data_config, compute_config)

start_time = time.time()
# Step 1: Tune hyperparameters
tune_result = pipeline.run_experiment(
    sku_tuples=sku_tuples_complete,
    modeling_strategy=ModelingStrategy.COMBINED,
    model_type="lightning_quantile",
    quantile_alphas=[0.5],
    mode="hp_tune",
    tune_on= 100,
    tuning_config={'n_trials': 100, 'n_folds': 5}
)

end_time = time.time()
execution_time = end_time - start_time
print("score: ",tune_result.best_score)
print("params: ",tune_result.best_params)
print(f"Execution time: {execution_time} seconds")

