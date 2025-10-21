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
    DataConfig, ModelingStrategy, ReleaseManager, BenchmarkPipeline
)
from datetime import date
from lets_plot import *

LetsPlot.setup_html()
np.random.seed(42)


def main():
    """Main execution function for macOS multiprocessing compatibility."""
    logger.info("="*80)
    logger.info("Starting XGBoost Hyperparameter Experiment")
    logger.info("="*80)

    #get list of trainable SKUs
    data_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/data/db_snapshot_offsite/train_data/processed/train_data_features.feather"
    df_clean = pl.read_ipc(data_path)

    sku_tuples_all = [(d['productID'], d['storeID']) for d in df_clean.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()]
    print("total unseen skus: ", len(sku_tuples_all))

    sku_exclude = (df_clean
     .group_by("storeID","productID")
     .agg(pl.col("date").first())
     .filter(pl.col("date") >= date(2016,1,1))
     .select("productID","storeID")
     )

    sku_exclude = [(d['productID'], d['storeID']) for d in sku_exclude.select(pl.col("productID"), pl.col("storeID")).unique().to_dicts()]

    sku_tuples_complete =  [sku for sku in sku_tuples_all if sku not in sku_exclude]
    print("total unseen skus available for training: ", len(sku_tuples_complete))
    logger.info(f"Total SKUs available for training: {len(sku_tuples_complete)}")


    #Parameter from runs over 500 and over 1000 SKUs  each on 100 trials and 5 folds

    # 0.5q 
    hp_100_new_1 = {'eta': 0.027051034952242748,
                  'max_depth': 10,
                    'min_child_weight': 17,
                      'subsample': 0.7445825574640936,
                        'colsample_bytree': 0.8492971371711306,
                          'gamma': 0.959507238466266,
                          'reg_alpha': 2.4319682226789663,
                            'reg_lambda': 6.486895233793623,
                              'n_estimators': 287,
                              'nthread': -1,
                              "tree_method": "hist"}

    hp_100_new_2 = {'eta': 0.11498210032794669,
                  'max_depth': 10,
                  'min_child_weight': 24,
                  'subsample': 0.9457013763068587,
                  'colsample_bytree': 0.736173526683077,
                  'gamma': 0.5391072234743155,
                  'reg_alpha': 1.1594541493594819,
                  'reg_lambda': 9.001210122377953,
                  'n_estimators': 168,
                  'nthread': -1,
                  'tree_method': 'hist'}


    hp_list = [hp_100_new_2]
    quantiles = [0.5, 0.7, 0.9, 0.95, 0.99]

    data_config = DataConfig(
        mapping_path = 'data/feature_mapping_train.pkl',
        features_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/data/db_snapshot_offsite/train_data/processed/train_data_features.feather",
        target_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/data/db_snapshot_offsite/train_data/train_data/train_data_target.feather",
        split_date="2016-01-01",
    )

    experiments= []
    logger.info(f"Starting {len(hp_list)} experiments with {len(quantiles)} quantiles each")
    logger.info(f"Training on {len(sku_tuples_complete)} SKUs with INDIVIDUAL strategy")

    for i, hp in enumerate(hp_list, start=1):
        logger.info("="*80)
        logger.info(f"EXPERIMENT {i}/{len(hp_list)}: Starting training with hyperparameters set {i}")
        logger.info("="*80)

        pipeline = BenchmarkPipeline(data_config)

        results = pipeline.run_experiment(
            sku_tuples= sku_tuples_complete,
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            model_type="xgboost_quantile",
            quantile_alphas=quantiles,
            hyperparameters = hp,
            experiment_name=f"xgb_quantile_{i}",
            evaluate_on_test=True
        )

        experiments.append([hp,results])
        logger.info(f"✓ Completed experiment {i}/{len(hp_list)}")
        logger.info(f"  Trained {results.num_models} models across {len(quantiles)} quantiles")
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

    logger.info("="*80)
    logger.info("Saving results to myexp.csv")
    experiment_results.write_csv("xgb_quantile_hp_all_test1.csv", separator=",", include_header=True)
    logger.info(f"✓ Results saved: {len(experiment_results)} rows")
    logger.info("="*80)
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()

