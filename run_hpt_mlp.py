import sys
from pathlib import Path
import numpy as np
import polars as pl
import time
from datetime import date

current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src import (
    DataConfig, ModelingStrategy, BenchmarkPipeline
)

import torch
import warnings

# Configure TF32 for optimal RTX 4090 performance (new PyTorch API)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Suppress TF32 deprecation warnings
warnings.filterwarnings('ignore', message='.*TF32.*', category=UserWarning)

# GPU detection and resource configuration
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    print(f"GPU detected: {gpu_name}")
    print(f"Number of GPUs: {gpu_count}")
    print(f"CUDA version: {torch.version.cuda}")

    # Optimal resource configuration for GPU training
    accelerator = 'cuda'
    devices = 1  # Single GPU per trial for hyperparameter tuning
    dataloader_workers = 8  # Good default for GPU training with large datasets
    print(f"Resource configuration: accelerator={accelerator}, devices={devices}, dataloader_workers={dataloader_workers}")
else:
    print("WARNING: No GPU detected. Falling back to CPU training.")
    accelerator = 'cpu'
    devices = 1
    dataloader_workers = 4  # Lower for CPU
    print(f"Resource configuration: accelerator={accelerator}, devices={devices}, dataloader_workers={dataloader_workers}")

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

pipeline = BenchmarkPipeline(data_config)

start_time = time.time()
# Step 1: Tune hyperparameters for Lightning quantile model
tune_result = pipeline.run_experiment(
    sku_tuples=sku_tuples_complete,
    modeling_strategy=ModelingStrategy.COMBINED,
    model_type="lightning_quantile",
    quantile_alphas=[0.7],
    mode="hp_tune",
    tune_on=100,  # Reduced sample size for tuning
    tuning_config={
        'n_trials': 100,
        'n_folds': 5,
        'n_jobs': 1,  # Sequential trials (one at a time on GPU)
        'dataloader_workers': dataloader_workers,  # DataLoader worker processes
        'accelerator': accelerator,  # Device type (cuda/cpu)
        'devices': devices  # Number of GPUs per trial
    }
)

end_time = time.time()
execution_time = end_time - start_time
print("score: ",tune_result.best_score)
print("params: ",tune_result.best_params)
print(f"Execution time: {execution_time} seconds")
