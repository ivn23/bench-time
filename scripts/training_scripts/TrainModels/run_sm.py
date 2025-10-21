import sys
from pathlib import Path
import numpy as np
import polars as pl
import logging

# Configure logging to see output from all modules
logging.basicConfig(
    level=logging.INFO,
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
    DataConfig, ModelingStrategy, BenchmarkPipeline
)
from datetime import date
from lets_plot import *

LetsPlot.setup_html()
np.random.seed(42)

def main():

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
    print(sku_tuples_complete[:1])
    data_config = DataConfig(
        mapping_path = 'data/feature_mapping_train.pkl',
        features_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/data/db_snapshot_offsite/train_data/processed/train_data_features.feather",
        target_path = "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking/data/db_snapshot_offsite/train_data/train_data/train_data_target.feather",
        split_date="2016-01-01",
    )

    quantiles = [0.5, 0.7, 0.9, 0.95, 0.99]
    pipeline = BenchmarkPipeline(data_config)

    results = pipeline.run_experiment(
        sku_tuples= sku_tuples_complete[:1],
        modeling_strategy=ModelingStrategy.INDIVIDUAL,
        model_type="statquant",
        quantile_alphas=quantiles,
        hyperparameters = {},
        experiment_name="sm_quantile",
        evaluate_on_test=True
    )



    logger.info(f"  Trained {results.num_models} models across {len(quantiles)} quantiles")


    #combine results into one dataframe for easier processing

    results_df = pl.DataFrame({
        "productID": [result.sku_tuples[0][0] for result in results.training_results],
        "storeID": [result.sku_tuples[0][1] for result in results.training_results],
        "quantile_level": [result.quantile_level for result in results.training_results],
        "mean_quantile_loss": [result.performance_metrics.get('mean_quantile_loss') for result in results.training_results],
        "model_type": "statquant_regression" 
    })

    results_df.write_csv("statquant_100.csv", separator=",", include_header=True)




if __name__ == "__main__":
    main()
