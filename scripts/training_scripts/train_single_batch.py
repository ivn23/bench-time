import sys
import json
import argparse
from pathlib import Path
import os

# Add src to path
sys.path.append('../../')

from src import BenchmarkPipeline, DataConfig, ModelingStrategy, ReleaseManager

def main():
    parser = argparse.ArgumentParser(description="Train single SKU batch")
    parser.add_argument("batch_file", help="JSON file with SKU batch")
    parser.add_argument("model_type", help="Model type (e.g., xgboost_standard)")
    parser.add_argument("experiment_name", help="Experiment name")

    args = parser.parse_args()

    # Load SKU batch
    with open(args.batch_file, 'r') as f:
        sku_tuples = json.load(f)
    
    sku_tuples = [tuple(sku) for sku in sku_tuples]  # Ensure tuples

    print(f"Training {len(sku_tuples)} SKUs with {args.model_type}")

    # Configure data (same as sku_selection notebook)
    #priunt current dir

    print(os.getcwd())
    data_config = DataConfig(
        features_path="../../data/db_snapshot_offsite/train_data/processed/train_data_features.feather",
        target_path="../../data/db_snapshot_offsite/train_data/train_data_target.feather",
        mapping_path="../../data/feature_mapping_train.pkl",
        split_date="2016-01-01"
        
    )

    # Initialize pipeline
    pipeline = BenchmarkPipeline(data_config=data_config)

    # Run experiment
    results = pipeline.run_experiment(
        sku_tuples=sku_tuples,
        modeling_strategy=ModelingStrategy.INDIVIDUAL,
        model_type="xgboost_quantile",
        quantile_alphas=[0.5, 0.7, 0.9, 0.95, 0.99],
        hyperparameters = {
            "eta": 0.05,
            "max_depth": 2,
            "max_leaves":7,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "colsample_bylevel": 0.9,
            "colsample_bynode": 0.9,
            "gamma": 1.0,   
            "lambda_": 0.5,
            "alpha": 4.0,
            "tree_method": "auto",
            "n_estimators": 500,
            "meta_learn_units": False,
            "multi_strategy": "one_output_per_tree", 
            "n_epochs":500
    },
        evaluate_on_test=True,
        experiment_name=args.experiment_name
        )

    # Create release
    release_manager = ReleaseManager()
    release_dir = release_manager.create_complete_release(
        experiment_results=results,
        base_output_dir=Path("../../xgb_offsite")
    )

    print(f"Experiment complete. Release at: {release_dir}")

if __name__ == "__main__":
    main()