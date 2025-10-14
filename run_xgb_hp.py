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
logger.info("="*80)
logger.info("Starting XGBoost Hyperparameter Experiment")
logger.info("="*80)

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
hp_random = {
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "gamma": 1.0,   
        "lambda": 10.0,
        "alpha": 1.0,
        "tree_method": "hist",
        "num_boost_round": 100,
        "seed": 42
}
hp_100 = {'eta': 0.299573707733717,
        'max_depth': 8,
        'min_child_weight': 19,
        'subsample': 0.696340395708422,
        'colsample_bytree': 0.6220507570163917,
        'gamma': 1.0789384910762259,
        'reg_alpha': 9.755231227570237,
        'reg_lambda': 8.295423481524228, 
        'n_estimators': 297}

hp_500 = {'eta': 0.2251586017238223,
        'max_depth': 10,
        'min_child_weight': 26,
        'subsample': 0.9639446007829685,
        'colsample_bytree': 0.6507755877543546,
        'gamma': 0.6139963456942783,
        'reg_alpha': 6.768595449859619,
        'reg_lambda': 4.832153974114423,
        'n_estimators': 293}

hp_1000 = {'eta': 0.2657057478526166, 
        'max_depth': 9,
        'min_child_weight': 19,
        'subsample': 0.6897467091557125, 
        'colsample_bytree': 0.9965497359024938,
        'gamma': 1.064228070424531, 
        'reg_alpha': 0.16585154768227728,
        'reg_lambda': 0.17000025072317992,
        'n_estimators': 297} 


hp_list = [hp_random,hp_100,hp_500,hp_1000]
quantiles = [0.5, 0.7, 0.9, 0.95, 0.99]



data_config = DataConfig(
    mapping_path = 'data/feature_mapping_train.pkl',
    features_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/data/db_snapshot_offsite/train_data/processed/train_data_features.feather",
    target_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/data/db_snapshot_offsite/train_data/train_data/train_data_target.feather",
    split_date="2016-01-01",
)

experiments= []
logger.info(f"Starting {len(hp_list)} experiments with {len(quantiles)} quantiles each")
logger.info(f"Training on {len(sku_tuples_complete[0:1000])} SKUs with INDIVIDUAL strategy")

for i, hp in enumerate(hp_list, start=1):
    logger.info("="*80)
    logger.info(f"EXPERIMENT {i}/{len(hp_list)}: Starting training with hyperparameters set {i}")
    logger.info("="*80)

    pipeline = BenchmarkPipeline(data_config)

    results = pipeline.run_experiment(
        sku_tuples= sku_tuples_complete[0:10],
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
hp_type = ["0","100","500","1000"]
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
experiment_results.write_csv("zzz_results/myexp.csv", separator=",", include_header=True)
logger.info(f"✓ Results saved: {len(experiment_results)} rows")
logger.info("="*80)
logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
logger.info("="*80)

