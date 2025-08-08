"""
Main orchestration script for the M5 benchmarking pipeline.
Coordinates data loading, feature engineering, model training, and evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import polars as pl

from .data_structures import (
    DataConfig, TrainingConfig, GranularityLevel, 
    ModelRegistry, BenchmarkModel
)
from .data_loading import DataLoader
from .feature_engineering import FeatureEngineer
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
        self.feature_engineer = FeatureEngineer(
            feature_engineering=data_config.feature_engineering,
            feature_engineering_methods=data_config.feature_engineering_methods,
            lag_features=data_config.lag_features,
            calendric_features=data_config.calendric_features,
            trend_features=data_config.trend_features
        )
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
    
    def run_single_model_experiment(self, 
                                   granularity: GranularityLevel,
                                   entity_ids: Dict[str, Any],
                                   experiment_name: Optional[str] = None) -> BenchmarkModel:
        """
        Run a single model experiment for specified granularity and entities.
        
        Args:
            granularity: SKU, PRODUCT, or STORE level
            entity_ids: Dictionary specifying which entities to include
            experiment_name: Optional name for this experiment
            
        Returns:
            Trained BenchmarkModel
        """
        exp_name = experiment_name or f"{granularity.value}_{entity_ids}"
        logger.info(f"Running experiment: {exp_name}")
        
        # Get data for specified granularity
        features_df, target_df = self.data_loader.get_data_for_granularity(
            granularity, entity_ids, collect=True
        )
        
        # Feature engineering
        engineered_df, feature_cols = self.feature_engineer.create_features(
            features_df, target_df, granularity, entity_ids
        )
        
        # Prepare model data
        X, y = self.feature_engineer.prepare_model_data(
            engineered_df, feature_cols, self.data_config.target_column
        )
        
        logger.info(f"Dataset prepared: {len(X)} samples, {len(feature_cols)} features")
        
        # Create temporal split
        train_bdids, val_bdids, split_date = self.data_loader.create_temporal_split(
            X, self.training_config.validation_split
        )
        
        # Split data
        X_train = X.filter(pl.col("bdID").is_in(train_bdids))
        y_train = y.filter(pl.col("bdID").is_in(train_bdids))
        X_val = X.filter(pl.col("bdID").is_in(val_bdids))
        y_val = y.filter(pl.col("bdID").is_in(val_bdids))
        
        # Train model
        model = self.model_trainer.train_model(
            X_train, y_train, X_val, y_val,
            feature_cols, self.data_config.target_column,
            granularity, entity_ids
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
            "granularity": granularity.value,
            "entity_ids": entity_ids,
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "split_date": str(split_date),
            "performance": model.metadata.performance_metrics
        }
        self.experiment_log.append(experiment_record)
        
        logger.info(f"Experiment {exp_name} completed. Model ID: {model_id}")
        
        return model
    
    def run_multi_entity_experiment(self,
                                   granularity: GranularityLevel,
                                   entity_list: List[Any],
                                   max_models: Optional[int] = None) -> List[BenchmarkModel]:
        """
        Run experiments across multiple entities at the same granularity level.
        
        Args:
            granularity: Level of granularity  
            entity_list: List of entity IDs to experiment with
            max_models: Optional limit on number of models to train
            
        Returns:
            List of trained BenchmarkModels
        """
        logger.info(f"Running multi-entity experiment at {granularity.value} level")
        logger.info(f"Training on {len(entity_list)} entities")
        
        if max_models and max_models < len(entity_list):
            entity_list = entity_list[:max_models]
            logger.info(f"Limited to first {max_models} entities")
        
        models = []
        
        for i, entity_id in enumerate(entity_list, 1):
            try:
                logger.info(f"Training model {i}/{len(entity_list)} for entity {entity_id}")
                
                # Create entity_ids dict based on granularity
                if granularity == GranularityLevel.SKU:
                    entity_ids = {"skuID": entity_id}
                elif granularity == GranularityLevel.PRODUCT:
                    entity_ids = {"productID": entity_id}
                elif granularity == GranularityLevel.STORE:
                    entity_ids = {"storeID": entity_id}
                else:
                    raise ValueError(f"Unknown granularity: {granularity}")
                
                model = self.run_single_model_experiment(
                    granularity, 
                    entity_ids,
                    f"{granularity.value}_{entity_id}"
                )
                models.append(model)
                
            except Exception as e:
                logger.error(f"Failed to train model for entity {entity_id}: {e}")
                continue
        
        logger.info(f"Multi-entity experiment completed. Trained {len(models)} models")
        
        return models
    
    def run_full_benchmark_suite(self,
                                sample_entities: Optional[Dict[str, List]] = None,
                                max_models_per_granularity: int = 5) -> Dict[str, List[BenchmarkModel]]:
        """
        Run comprehensive benchmark across all granularity levels.
        
        Args:
            sample_entities: Optional dict with entity samples for each level
            max_models_per_granularity: Max models to train per granularity
            
        Returns:
            Dictionary mapping granularity to list of models
        """
        logger.info("Starting full benchmark suite")
        
        if sample_entities is None:
            # Get sample entities from data
            unique_entities = self.data_loader.get_unique_entities()
            sample_entities = {
                "skuIDs": unique_entities["skuIDs"][:max_models_per_granularity],
                "productIDs": unique_entities["productIDs"][:max_models_per_granularity],
                "storeIDs": unique_entities["storeIDs"][:max_models_per_granularity]
            }
        
        all_models = {}
        
        # SKU level models
        logger.info("Training SKU-level models...")
        sku_models = self.run_multi_entity_experiment(
            GranularityLevel.SKU, 
            sample_entities["skuIDs"],
            max_models_per_granularity
        )
        all_models["sku"] = sku_models
        
        # Product level models
        logger.info("Training Product-level models...")
        product_models = self.run_multi_entity_experiment(
            GranularityLevel.PRODUCT,
            sample_entities["productIDs"],
            max_models_per_granularity
        )
        all_models["product"] = product_models
        
        # Store level models
        logger.info("Training Store-level models...")
        store_models = self.run_multi_entity_experiment(
            GranularityLevel.STORE,
            sample_entities["storeIDs"], 
            max_models_per_granularity
        )
        all_models["store"] = store_models
        
        logger.info(f"Full benchmark suite completed. Trained {sum(len(models) for models in all_models.values())} total models")
        
        return all_models
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models in the registry."""
        logger.info("Evaluating all models in registry")
        
        all_model_ids = self.model_registry.list_models()
        
        if not all_model_ids:
            logger.warning("No models found in registry")
            return {"error": "No models to evaluate"}
        
        # Evaluate models by granularity
        evaluation_results = {}
        
        for granularity in GranularityLevel:
            granularity_results = self.evaluator.evaluate_by_granularity(granularity)
            if "error" not in granularity_results:
                evaluation_results[granularity.value] = granularity_results
        
        # Overall comparison
        if all_model_ids:
            overall_comparison = self.evaluator.compare_models(all_model_ids)
            evaluation_results["overall"] = overall_comparison
        
        # Save evaluation results
        self.save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Save evaluation results to files."""
        results_dir = self.output_dir / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        with open(results_dir / "evaluation_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(evaluation_results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate reports
        for granularity, results in evaluation_results.items():
            if granularity != "overall" and "error" not in results:
                report = self.evaluator.generate_evaluation_report(
                    results, 
                    results_dir / f"{granularity}_evaluation_report.md"
                )
        
        logger.info(f"Evaluation results saved to {results_dir}")
    
    def save_experiment_log(self):
        """Save experiment log to file."""
        log_file = self.output_dir / "experiment_log.json"
        with open(log_file, "w") as f:
            json.dump(self.experiment_log, f, indent=2)
        logger.info(f"Experiment log saved to {log_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # custom objects
            return str(obj)
        else:
            return obj


def create_default_configs(data_dir: Path) -> tuple[DataConfig, TrainingConfig]:
    """Create default configurations for the pipeline."""
    
    data_config = DataConfig(
        features_path=str(data_dir / "train_data_features.feather"),
        target_path=str(data_dir / "train_data_target.feather"),
        mapping_path=str(data_dir / "feature_mapping_train.pkl"),
        date_column="date",
        target_column="target",
        bdid_column="bdID",
        remove_not_for_sale=True,
        lag_features=[1, 2, 3, 4, 5, 6, 7],
        calendric_features=True,
        trend_features=True
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
    
    # Run a single model experiment as example
    model = pipeline.run_single_model_experiment(
        granularity=GranularityLevel.SKU,
        entity_ids={"skuID": 278993},  # Example SKU
        experiment_name="example_sku_model"
    )
    
    # Evaluate the model
    evaluation_results = pipeline.evaluate_all_models()
    
    # Save logs
    pipeline.save_experiment_log()
    
    logger.info("Benchmark pipeline completed successfully!")


if __name__ == "__main__":
    main()