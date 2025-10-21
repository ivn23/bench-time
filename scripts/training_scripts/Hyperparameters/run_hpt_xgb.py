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
torch.set_num_threads(1)
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
# Step 1: Tune hyperparameters
tune_result = pipeline.run_experiment(
    sku_tuples=sku_tuples_complete,
    modeling_strategy=ModelingStrategy.COMBINED,
    model_type="xgboost_quantile",
    quantile_alphas=[0.7],
    mode="hp_tune",
    tune_on= 100,
    tuning_config={'n_trials': 100, 'n_folds': 5, 'n_jobs': -1},
)

end_time = time.time()
execution_time = end_time - start_time
print("score: ",tune_result.best_score)
print("params: ",tune_result.best_params)
print(f"Execution time: {execution_time} seconds")