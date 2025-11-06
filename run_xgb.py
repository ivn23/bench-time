import sys
from pathlib import Path
import numpy as np
import polars as pl
import logging

# Configure logging to see output from all modules
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('xgb_hp_experiment.log'),  # Save to file
        logging.StreamHandler()                         # Also print to console
    ]
)
logger = logging.getLogger(__name__)

current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src import (
    DataConfig, ModelingStrategy, BenchmarkPipeline
)
from src.utils import get_skus, load_hp_tuning_results

np.random.seed(42)


def main():

    # Centralized data path definitions
    data_path = "data/db_snapshot_offsite/train_data/processed/train_data_features.feather"
    mapping_path = 'data/feature_mapping_train.pkl'
    target_path = "data/db_snapshot_offsite/train_data/train_data/train_data_target.feather"
    split_date = "2016-01-01"
    param_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/HP_RESULTS/xgb/xgboost_quantile_q0.7_tuned10_trials10_20251106_163806.csv"

    # Load SKUs
    sku_tuples_complete = get_skus(data_path)

    xgb_params = load_hp_tuning_results(param_path)


    hp_list = [xgb_params]
    quantiles = [0.5, 0.7, 0.9, 0.95, 0.99]

    # Configure data
    data_config = DataConfig(
        mapping_path=mapping_path,
        features_path=data_path,
        target_path=target_path,
        split_date=split_date
    )

    experiments= []

    for i, hp in enumerate(hp_list, start=1):

        pipeline = BenchmarkPipeline(data_config)

        results = pipeline.run_experiment(
            sku_tuples= sku_tuples_complete[:100],
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            model_type="xgboost_quantile",
            quantile_alphas=quantiles,
            hyperparameters = hp,
            experiment_name=f"xgb_quantile_{i}",
            data_workers=4,
            evaluate_on_test=True
        )

        experiments.append([hp,results])
        print(f"Completed experiment {i}")


    #give results quick hp_name for later processing
    hp_type = ["test_hp"]
    for i,experiment in enumerate(experiments, start=0):
        experiment.append(hp_type[i])

    #combine results into one dataframe for easier processing
    results_dfs = []
    for experiment in experiments:

        results_df = pl.DataFrame({
            "productID": [result.sku_tuples[0][0] for result in experiment[1].training_results],
            "storeID": [result.sku_tuples[0][1] for result in experiment[1].training_results],
            "quantile_level": [result.quantile_level for result in experiment[1].training_results],
            "mean_quantile_loss" : [result.performance_metrics.get('mean_quantile_loss') for result in experiment[1].training_results],
            "hp_type" : experiment[2]
        })
        results_dfs.append(results_df)


    experiment_results = pl.concat(results_dfs, how="vertical")

    experiment_results.write_csv("xgb_test_params.csv", separator=",", include_header=True)
 

if __name__ == "__main__":
    main()

