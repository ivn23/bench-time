import sys
from pathlib import Path
import numpy as np
import time 

current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src import (
    DataConfig, ModelingStrategy, BenchmarkPipeline
)
from src.utils import save_hp_tuning_results, get_skus

import torch
torch.set_num_threads(1)
np.random.seed(42)



data_path = "data/db_snapshot_offsite/train_data/processed/train_data_features.feather"
mapping_path = 'data/feature_mapping_train.pkl'
target_path = "data/db_snapshot_offsite/train_data/train_data/train_data_target.feather"
split_date="2016-01-01"

model_type = "xgboost_quantile"
quantile = [0.7]
tune_on = 10

n_trials = 10
n_folds = 5
n_jobs = -1

sku_tuples_complete = get_skus(data_path)

data_config = DataConfig(
    mapping_path = mapping_path,
    features_path = data_path,
    target_path = target_path,
    split_date=split_date
)

pipeline = BenchmarkPipeline(data_config)

start_time = time.time()
# Tune hyperparameters
tune_result = pipeline.run_experiment(
    sku_tuples=sku_tuples_complete,
    modeling_strategy=ModelingStrategy.COMBINED,
    model_type=model_type,
    quantile_alphas=quantile,
    mode="hp_tune",
    tune_on= tune_on,
    tuning_config={'n_trials': n_trials, 'n_folds': n_folds, 'n_jobs': n_jobs},
)

end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = execution_time % 60

# Save results to CSV 
saved_path = save_hp_tuning_results(
    tune_result=tune_result,
    model_type=model_type,
    quantile_alpha=quantile[0] if quantile else None,
    tune_on=tune_on,
    n_trials=n_trials,
    n_folds=5,
    execution_time=execution_time
)

print(f"Results saved to: {saved_path}")
print(f"Best Validation Score: {tune_result.best_score:.6f}")
print(f"Execution Time: {minutes}m {seconds:.1f}s")

