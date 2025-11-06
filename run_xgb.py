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
from src.utils import get_skus

np.random.seed(42)


def main():

    # Centralized data path definitions
    data_path = "data/db_snapshot_offsite/train_data/processed/train_data_features.feather"
    mapping_path = 'data/feature_mapping_train.pkl'
    target_path = "data/db_snapshot_offsite/train_data/train_data/train_data_target.feather"
    split_date = "2016-01-01"

    # Load SKUs
    sku_tuples_complete = get_skus(data_path)


    #Parameter from runs over 500 and over 1000 SKUs  each on 100 trials and 5 folds

    # Magnus parameter
    xgb_params = {
        "eta": 0.3,                    # learning rate
        "max_depth": 2,                # tree depth
        "min_child_weight": 5,         # minimum sum of instance weight (hessian) in a child
        "subsample": 0.9,              # fraction of samples per tree
        "colsample_bytree": 0.85,      # fraction of features per tree
        "colsample_bylevel": 0.9,      # fraction of features per tree level
        "colsample_bynode": 0.9,       # fraction of features per split
        "gamma": 1,                    # minimum loss reduction required to make a split
        "reg_alpha": 4,                # L1 regularization term on weights
        "reg_lambda": 0.5,             # L2 regularization term on weights
        "max_bin": 512,                # number of bins for histogram-based algorithms
        "max_leaves": 7,               # maximum number of leaves (used with grow_policy='lossguide')
        "max_delta_step": 5,           # limit step size for each leaf weight update
        "tree_method": "auto",         # can be 'auto', 'hist', 'approx', 'gpu_hist'
        "grow_policy": "depthwise",    # 'depthwise' or 'lossguide'
        "num_parallel_tree": 1,        # used for random forest or DART       
        "sampling_method": "uniform",  # data sampling method
        "refresh_leaf": 1,             # whether to refresh leaf values after each boosting step
        "device": "cpu",               # or 'cuda' if available
        "nthread": 1,                  # number of CPU threads
        "verbosity": 1, 
        "n_estimators": 500              # log level
    }

    hp_100_new_2 = {'eta': 0.11498210032794669,
                  'max_depth': 10,
                  'min_child_weight': 24,
                  'subsample': 0.9457013763068587,
                  'colsample_bytree': 0.736173526683077,
                  'gamma': 0.5391072234743155,
                  'reg_alpha': 1.1594541493594819,
                  'reg_lambda': 9.001210122377953,
                  'n_estimators': 168,
                  'nthread': 1,
                  'tree_method': 'hist'}


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
            sku_tuples= sku_tuples_complete,
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            model_type="xgboost_quantile",
            quantile_alphas=quantiles,
            hyperparameters = hp,
            experiment_name=f"xgb_quantile_{i}",
            data_workers=8,
            evaluate_on_test=True
        )

        experiments.append([hp,results])
        print(f"Completed experiment {i}")


    #give results quick hp_name for later processing
    hp_type = ["100_new_2"]
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

    experiment_results.write_csv("xgb_quantile_hp_all_magnus_params.csv", separator=",", include_header=True)
 

if __name__ == "__main__":
    main()

