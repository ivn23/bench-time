"""
Main orchestration script for the M5 benchmarking pipeline.
Coordinates data loading, model training, and evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import polars as pl

from .data_structures import (
    DataConfig, TrainingConfig, ModelingStrategy, SkuList, SkuTuple,
    ModelRegistry, BenchmarkModel, ExperimentResults, ModelingDataset
)
from .data_loading import DataLoader
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator
from .release_management import ComprehensiveReleaseManager

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
        self.model_registry = ModelRegistry()  # In-memory only registry
        self.evaluator = ModelEvaluator(self.data_loader, self.model_registry)
        
        # Track experiment state
        self.experiment_log = {
            "experiments": [],
            "pipeline_config": {
                "data_config": str(data_config),
                "training_config": str(training_config),
                "output_dir": str(output_dir)
            }
        }
        
    def load_and_prepare_data(self):
        """Load and prepare the base dataset."""
        logger.info("Loading and preparing M5 dataset...")
        self.data_loader.load_data(lazy=False)
        logger.info("Data loading completed")
    
    def run_experiment(self, 
                     sku_tuples: SkuList,
                     modeling_strategy: ModelingStrategy = ModelingStrategy.COMBINED,
                     experiment_name: Optional[str] = None,
                     ) -> List[BenchmarkModel]:
        """
        Simplified experiment runner using the new clean architecture.
        
        Args:
            sku_tuples: List of (product_id, store_id) tuples defining SKUs
            modeling_strategy: COMBINED (one model for all) or INDIVIDUAL (model per SKU)
            experiment_name: Optional name for this experiment
            
        Returns:
            List of trained BenchmarkModel(s)
        """
        if not sku_tuples:
            raise ValueError("At least one SKU tuple must be provided")
                
        # Use the single configured model type
        model_type = self.training_config.model_type
        
        exp_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Running experiment: {exp_name} with {len(sku_tuples)} SKUs using {model_type}")
        
        # New clean architecture - single unified method
        models = self._train_models_for_strategy(sku_tuples, modeling_strategy, model_type, exp_name)
        
        logger.info(f"Experiment {exp_name} completed. Trained {len(models)} model(s)")
        return models
    
    def run_complete_experiment(
        self,
        sku_tuples: SkuList,
        modeling_strategy: ModelingStrategy = ModelingStrategy.COMBINED,
        experiment_name: Optional[str] = None,
        evaluate: bool = True
    ) -> Path:
        """
        Run complete experiment with training, evaluation, and release packaging.
        
        Args:
            sku_tuples: List of (product_id, store_id) tuples defining SKUs
            modeling_strategy: COMBINED (one model for all) or INDIVIDUAL (model per SKU)
            experiment_name: Optional name for this experiment
            evaluate: Whether to run evaluation (creates metrics.json if True)
            
        Returns:
            Path to created release directory
        """
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Running complete experiment: {experiment_name}")
        
        # Train models (returns models in memory)
        models = self.run_experiment(sku_tuples, modeling_strategy, experiment_name)
        
        # Optionally evaluate models (returns results in memory)
        evaluation_results = None
        if evaluate:
            logger.info("Running evaluation...")
            evaluation_results = self.evaluate_models(models)
        
        # Create ExperimentResults object
        experiment_results = ExperimentResults(
            models=models,
            evaluation_results=evaluation_results,
            experiment_log=self.experiment_log,
            configurations={
                'data_config': self.data_config,
                'training_config': self.training_config
            },
            experiment_name=experiment_name,
            timestamp=datetime.now()
        )
        
        # Delegate to ComprehensiveReleaseManager for consolidation
        release_manager = ComprehensiveReleaseManager()
        release_dir = release_manager.create_complete_release(experiment_results, self.output_dir)
        
        logger.info(f"Complete experiment finished. Release available at: {release_dir}")
        return release_dir
    
    def evaluate_models(self, models: List[BenchmarkModel]) -> Dict[str, Any]:
        """
        Evaluate specific models and return results in memory.
        
        Args:
            models: List of models to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating {len(models)} models")
        
        # Register models temporarily for evaluation
        temp_registry = ModelRegistry()
        model_ids = []
        for model in models:
            model_id = temp_registry.register_model(model)
            model_ids.append(model_id)
        
        # Create temporary evaluator with the models
        temp_evaluator = ModelEvaluator(self.data_loader, temp_registry)
        
        evaluation_results = {}
        
        # Evaluate models by strategy
        for strategy in ModelingStrategy:
            strategy_models = [mid for mid in model_ids 
                             if temp_registry.get_model(mid).metadata.modeling_strategy == strategy]
            if strategy_models:
                strategy_results = temp_evaluator.evaluate_by_modeling_strategy(strategy)
                if "error" not in strategy_results:
                    evaluation_results[strategy.value] = strategy_results
        
        # Overall comparison if we have models
        if model_ids:
            evaluation_results["overall"] = temp_evaluator.compare_models(model_ids)
        
        logger.info(f"Evaluation completed for {len(models)} models")
        return evaluation_results
    
    def _train_models_for_strategy(
        self,
        sku_tuples: SkuList,
        modeling_strategy: ModelingStrategy,
        model_type: str,
        exp_name: str
    ) -> List[BenchmarkModel]:
        """
        Unified training method for both COMBINED and INDIVIDUAL strategies.
        Uses DataLoader for all data operations - no data manipulation here.
        """
        if modeling_strategy == ModelingStrategy.COMBINED:
            # Prepare single dataset for all SKUs
            dataset = self.data_loader.prepare_modeling_dataset(sku_tuples, modeling_strategy)
            models = self.model_trainer.train_model(dataset, model_type)
            return self._process_trained_models(models, dataset, exp_name)
            
        elif modeling_strategy == ModelingStrategy.INDIVIDUAL:
            # Train separate model for each SKU
            all_models = []
            for i, sku_tuple in enumerate(sku_tuples):
                logger.info(f"Training individual model {i+1}/{len(sku_tuples)}: {sku_tuple}")
                
                # Prepare dataset for this single SKU
                dataset = self.data_loader.prepare_modeling_dataset([sku_tuple], modeling_strategy)
                models = self.model_trainer.train_model(dataset, model_type)
                processed_models = self._process_trained_models(models, dataset, f"{exp_name}_sku_{sku_tuple[0]}x{sku_tuple[1]}")
                all_models.extend(processed_models)
                
            return all_models
        else:
            raise ValueError(f"Unknown modeling strategy: {modeling_strategy}")
    
    def _process_trained_models(self, models: List[BenchmarkModel], dataset: ModelingDataset, exp_name: str) -> List[BenchmarkModel]:
        """Process and register trained models with experiment logging."""
        processed_models = []
        
        for model in models:
            # Register model in memory
            model_id = self.model_registry.register_model(model)
            
            # Log experiment using dataset statistics
            experiment_record = {
                "experiment_name": exp_name,
                "model_id": model_id,
                "modeling_strategy": dataset.modeling_strategy.value,
                "sku_tuples": dataset.sku_tuples,
                "model_type": model.metadata.model_type,
                "quantile_level": model.metadata.quantile_level,
                "n_samples": dataset.dataset_stats["n_samples_total"],
                "n_features": dataset.dataset_stats["n_features"],
                "split_date": dataset.split_info["split_date"],
                "performance": model.metadata.performance_metrics
            }
            self.experiment_log["experiments"].append(experiment_record)
            processed_models.append(model)
            
            logger.info(f"Model trained: {model_id} with RMSE: {model.metadata.performance_metrics.get('rmse', 'N/A'):.4f}")
        
        return processed_models

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


    def _make_json_serializable(self, obj):
        """Helper to make objects JSON serializable."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
        else:
            return str(obj)