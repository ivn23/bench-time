"""
Main orchestration script for the M5 benchmarking pipeline.
Coordinates data loading, model training, and evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import polars as pl

from .data_structures import (
    DataConfig, TrainingConfig, ModelingStrategy, SkuList, SkuTuple,
    ModelRegistry, BenchmarkModel
)
from .data_loading import DataLoader
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator, VisualizationGenerator

logger = logging.getLogger(__name__)


class BenchmarkPipeline:
    """Main pipeline for running M5 benchmarking experiments."""
    
    def __init__(self, 
                 data_config: DataConfig,
                 training_config: TrainingConfig,
                 output_dir: Path = Path("benchmark_results")):
        self.data_config = data_config
        self.training_config = training_config
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_config)
        self.model_trainer = ModelTrainer(training_config)
        self.model_registry = ModelRegistry(self.output_dir / "models")
        self.evaluator = ModelEvaluator(self.data_loader, self.model_registry)
        self.viz_generator = VisualizationGenerator()
        
        # Track experiment state
        self.experiment_log = []
        
    def load_and_prepare_data(self):
        """Load and prepare the base dataset."""
        logger.info("Loading and preparing M5 dataset...")
        self.data_loader.load_data(lazy=False)
        logger.info("Data loading completed")
    
    def run_experiment(self, 
                     sku_tuples: SkuList,
                     modeling_strategy: ModelingStrategy = ModelingStrategy.COMBINED,
                     experiment_name: Optional[str] = None) -> List[BenchmarkModel]:
        """
        Run experiment(s) for specified SKU tuples.
        
        Args:
            sku_tuples: List of (product_id, store_id) tuples defining SKUs
            modeling_strategy: COMBINED (one model for all) or INDIVIDUAL (model per SKU)
            experiment_name: Optional name for this experiment
            
        Returns:
            List of trained BenchmarkModel(s)
        """
        if not sku_tuples:
            raise ValueError("At least one SKU tuple must be provided")
        
        exp_name = experiment_name or f"{modeling_strategy.value}_{len(sku_tuples)}skus"
        logger.info(f"Running experiment: {exp_name} with {len(sku_tuples)} SKUs")
        
        models = []
        
        if modeling_strategy == ModelingStrategy.COMBINED:
            # Train one model on all SKUs together
            model = self._train_combined_model(sku_tuples, exp_name)
            models.append(model)
            
        elif modeling_strategy == ModelingStrategy.INDIVIDUAL:
            # Train separate model for each SKU
            for i, sku_tuple in enumerate(sku_tuples):
                individual_name = f"{exp_name}_sku_{sku_tuple[0]}x{sku_tuple[1]}"
                model = self._train_individual_model([sku_tuple], individual_name, i+1, len(sku_tuples))
                models.append(model)
        
        logger.info(f"Experiment {exp_name} completed. Trained {len(models)} model(s)")
        return models

    def _train_combined_model(self, sku_tuples: SkuList, exp_name: str) -> BenchmarkModel:
        """Train a single model on all specified SKUs."""
        # Get data for all SKUs
        features_df, target_df = self.data_loader.get_data_for_tuples(
            sku_tuples, ModelingStrategy.COMBINED, collect=True
        )
        
        # Prepare features for modeling
        X, y, feature_cols = self.data_loader.prepare_features_for_modeling(
            features_df, target_df, self.data_config.target_column
        )
        
        logger.info(f"Combined model dataset: {len(X)} samples, {len(feature_cols)} features")
        
        # Create temporal split - use date-based split if specified, otherwise use percentage
        if self.data_config.split_date:
            train_bdids, test_bdids, split_date = self.data_loader.create_temporal_split_by_date(
                X, self.data_config.split_date
            )
        else:
            train_bdids, test_bdids, split_date = self.data_loader.create_temporal_split(
                X, self.training_config.validation_split
            )
        
        # Split data into train and test
        X_train = X.filter(pl.col("bdID").is_in(train_bdids))
        y_train = y.filter(pl.col("bdID").is_in(train_bdids))
        X_test = X.filter(pl.col("bdID").is_in(test_bdids))
        y_test = y.filter(pl.col("bdID").is_in(test_bdids))
        
        # Train model
        model = self.model_trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, self.data_config.target_column,
            ModelingStrategy.COMBINED, sku_tuples
        )
        
        # Update data split info
        model.data_split.split_date = str(split_date)
        
        # Register and save model
        model_id = self.model_registry.register_model(model)
        self.model_registry.save_model(model_id)
        
        # Log experiment
        experiment_record = {
            "experiment_name": exp_name,
            "model_id": model_id,
            "modeling_strategy": ModelingStrategy.COMBINED.value,
            "sku_tuples": sku_tuples,
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "split_date": str(split_date),
            "performance": model.metadata.performance_metrics
        }
        self.experiment_log.append(experiment_record)
        
        logger.info(f"Combined model trained for {len(sku_tuples)} SKUs. Model ID: {model_id}")
        return model

    def _train_individual_model(self, sku_tuples: SkuList, exp_name: str, 
                               current: int, total: int) -> BenchmarkModel:
        """Train an individual model for a single SKU."""
        logger.info(f"Training individual model {current}/{total}: {sku_tuples[0]}")
        
        # Get data for this SKU
        features_df, target_df = self.data_loader.get_data_for_tuples(
            sku_tuples, ModelingStrategy.INDIVIDUAL, collect=True
        )
        
        # Prepare features for modeling
        X, y, feature_cols = self.data_loader.prepare_features_for_modeling(
            features_df, target_df, self.data_config.target_column
        )
        
        logger.info(f"Individual model dataset: {len(X)} samples, {len(feature_cols)} features")
        
        # Create temporal split
        if self.data_config.split_date:
            train_bdids, test_bdids, split_date = self.data_loader.create_temporal_split_by_date(
                X, self.data_config.split_date
            )
        else:
            train_bdids, test_bdids, split_date = self.data_loader.create_temporal_split(
                X, self.training_config.validation_split
            )
        
        # Split data
        X_train = X.filter(pl.col("bdID").is_in(train_bdids))
        y_train = y.filter(pl.col("bdID").is_in(train_bdids))
        X_test = X.filter(pl.col("bdID").is_in(test_bdids))
        y_test = y.filter(pl.col("bdID").is_in(test_bdids))
        
        # Train model
        model = self.model_trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, self.data_config.target_column,
            ModelingStrategy.INDIVIDUAL, sku_tuples
        )
        
        # Update data split info
        model.data_split.split_date = str(split_date)
        
        # Register and save model
        model_id = self.model_registry.register_model(model)
        self.model_registry.save_model(model_id)
        
        # Log experiment
        experiment_record = {
            "experiment_name": exp_name,
            "model_id": model_id,
            "modeling_strategy": ModelingStrategy.INDIVIDUAL.value,
            "sku_tuples": sku_tuples,
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "split_date": str(split_date),
            "performance": model.metadata.performance_metrics
        }
        self.experiment_log.append(experiment_record)
        
        logger.info(f"Individual model trained for SKU {sku_tuples[0]}. Model ID: {model_id}")
        return model

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models in the registry."""
        logger.info("Evaluating all models in registry")
        
        all_model_ids = self.model_registry.list_models()
        if not all_model_ids:
            logger.warning("No models found in registry")
            return {"error": "No models found in registry"}
        
        evaluation_results = {}
        
        # Evaluate models by strategy
        for strategy in ModelingStrategy:
            strategy_results = self.evaluator.evaluate_by_modeling_strategy(strategy)
            if "error" not in strategy_results:
                evaluation_results[strategy.value] = strategy_results
        
        # Overall comparison
        evaluation_results["overall"] = self.evaluator.compare_models(all_model_ids)
        
        logger.info(f"Evaluation completed for {len(all_model_ids)} models")
        return evaluation_results

    def save_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Save evaluation results to files."""
        results_dir = self.output_dir / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save main results
        import json
        with open(results_dir / "evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Generate and save reports for each strategy
        for strategy, results in evaluation_results.items():
            if strategy != "overall" and "error" not in results:
                report = self.evaluator.generate_evaluation_report(
                    results, 
                    results_dir / f"{strategy}_evaluation_report.md"
                )
        
        logger.info(f"Evaluation results saved to {results_dir}")

    def save_experiment_log(self):
        """Save experiment log to file."""
        log_path = self.output_dir / "experiment_log.json"
        import json
        with open(log_path, "w") as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
        logger.info(f"Experiment log saved to {log_path}")

    def _make_json_serializable(self, obj):
        """Helper to make objects JSON serializable."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
        else:
            return str(obj)
    

def create_default_configs(data_dir: Path) -> tuple[DataConfig, TrainingConfig]:
    """Create default configurations for the pipeline."""
    
    data_config = DataConfig(
        features_path=str(data_dir / "train_data_features.feather"),
        target_path=str(data_dir / "train_data_target.feather"),
        mapping_path=str(data_dir / "feature_mapping_train.pkl"),
        date_column="date",
        target_column="target",
        bdid_column="bdID",
        remove_not_for_sale=True
    )
    
    training_config = TrainingConfig(
        validation_split=0.2,
        random_state=42,
        cv_folds=5,
        model_type="xgboost"
    )
    
    return data_config, training_config


def main():
    """Example usage of the benchmark pipeline."""
    import polars as pl
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configurations
    data_dir = Path("../data")
    data_config, training_config = create_default_configs(data_dir)
    
    # Initialize pipeline
    pipeline = BenchmarkPipeline(
        data_config=data_config,
        training_config=training_config,
        output_dir=Path("benchmark_results")
    )
    
    # Load data
    pipeline.load_and_prepare_data()
    
    # Example: Train models for specific SKUs
    # Each tuple is (product_id, store_id)
    example_skus = [
        (80558, 2),  # Product 80558 at Store 2
        (80558, 5),  # Product 80558 at Store 5
        (80651, 2)   # Product 80651 at Store 2
    ]
    
    # Run combined strategy (one model for all SKUs)
    combined_models = pipeline.run_experiment(
        sku_tuples=example_skus,
        modeling_strategy=ModelingStrategy.COMBINED,
        experiment_name="example_combined_model"
    )
    
    # Run individual strategy (separate model per SKU)
    individual_models = pipeline.run_experiment(
        sku_tuples=example_skus[:2],  # Use first 2 SKUs for individual models
        modeling_strategy=ModelingStrategy.INDIVIDUAL,
        experiment_name="example_individual_models"
    )
    
    # Evaluate all models
    evaluation_results = pipeline.evaluate_all_models()
    
    # Save logs
    pipeline.save_experiment_log()
    
    logger.info("Benchmark pipeline completed successfully!")


if __name__ == "__main__":
    main()