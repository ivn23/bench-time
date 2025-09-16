"""Clean orchestrator for M5 benchmarking experiments.

This module provides a simplified BenchmarkPipeline that serves as a stateless
orchestrator for single model type, single strategy experiments. The pipeline
delegates all specialized work to appropriate modules and focuses on clean
orchestration rather than state management.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .data_structures import (
    DataConfig, ModelingStrategy, SkuList, BenchmarkModel, ModelingDataset,
    ModelTypeConfig, TrainedModel
)
from .experiment_config import (
    ExperimentConfig, ExperimentResults, create_experiment_config,
    validate_sku_tuples, validate_modeling_strategy
)
from .data_loading import DataLoader
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator
from .release_management import ComprehensiveReleaseManager

logger = logging.getLogger(__name__)


class BenchmarkPipeline:
    """Clean orchestrator for M5 benchmarking experiments.
    
    This pipeline serves as a stateless orchestrator that coordinates specialized
    modules to run single model type, single strategy experiments. It focuses on
    clear delegation rather than state management.
    
    Key principles:
    - Stateless operation: no internal state between experiments
    - Single responsibility: orchestration only, delegates all specialized work
    - Clear interface: one method for complete experiments
    - Parameter-based configuration: no complex config objects
    """
    
    def __init__(self, 
                 data_config: DataConfig,
                 output_dir: Path = Path("benchmark_results")):
        """Initialize the orchestrator.
        
        Args:
            data_config: Configuration for data loading
            output_dir: Directory for experiment outputs
        """
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"BenchmarkPipeline initialized with output directory: {self.output_dir}")

    def run_experiment(self,
                      sku_tuples: SkuList,
                      modeling_strategy: ModelingStrategy,
                      model_type: str,
                      hyperparameters: Dict[str, Any],
                      experiment_name: Optional[str] = None,
                      quantile_alphas: Optional[List[float]] = None,
                      random_state: int = 42) -> ExperimentResults:
        """
        Run a complete single model type, single strategy experiment.
        
        This is the main orchestrator method that coordinates all experiment steps:
        1. Data preparation (delegates to DataLoader)
        2. Model training (delegates to ModelTrainer)
        3. Model evaluation (delegates to ModelEvaluator)
        4. Results packaging (delegates to ReleaseManager)
        
        Args:
            sku_tuples: List of (product_id, store_id) tuples defining SKUs
            modeling_strategy: COMBINED (one model for all) or INDIVIDUAL (model per SKU)
            model_type: Type of model to train (e.g., 'xgboost_standard')
            hyperparameters: Model-specific hyperparameters
            experiment_name: Optional name for this experiment
            quantile_alphas: List of quantile levels for quantile models
            random_state: Random seed for reproducibility
            
        Returns:
            ExperimentResults object containing models, metrics, and metadata
            
        Example:
            >>> results = pipeline.run_experiment(
            ...     sku_tuples=[(80558, 2), (80558, 5)],
            ...     modeling_strategy=ModelingStrategy.INDIVIDUAL,
            ...     model_type="xgboost_standard",
            ...     hyperparameters={"n_estimators": 100, "max_depth": 6}
            ... )
            >>> print(f"Trained {results.num_models} models")
        """
        # Validate inputs
        validate_sku_tuples(sku_tuples)
        validate_modeling_strategy(modeling_strategy, sku_tuples)
        
        # Create experiment configuration
        experiment_config = create_experiment_config(
            model_type=model_type,
            hyperparameters=hyperparameters,
            quantile_alphas=quantile_alphas,
            random_state=random_state
        )
        
        # Generate experiment name
        exp_name = experiment_name or f"{model_type}_{modeling_strategy.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting experiment '{exp_name}': {model_type} with {modeling_strategy.value} strategy for {len(sku_tuples)} SKU(s)")
        
        try:
            # Step 1: Data preparation
            logger.info("Step 1: Preparing data...")
            datasets = self._prepare_datasets(sku_tuples, modeling_strategy)
            
            # Step 2: Model training  
            logger.info(f"Step 2: Training {len(datasets)} model(s)...")
            trained_models = self._train_models(datasets, experiment_config)
            
            # Step 3: Model evaluation
            logger.info(f"Step 3: Evaluating {len(trained_models)} model(s)...")
            evaluated_models = self._evaluate_models(trained_models, datasets)
            
            # Step 4: Results packaging
            logger.info("Step 4: Packaging results...")
            results = self._package_results(
                models=evaluated_models,
                experiment_name=exp_name,
                experiment_config=experiment_config,
                modeling_strategy=modeling_strategy,
                sku_tuples=sku_tuples
            )
            
            logger.info(f"Experiment '{exp_name}' completed successfully. Trained {results.num_models} model(s)")
            return results
            
        except Exception as e:
            logger.error(f"Experiment '{exp_name}' failed: {str(e)}")
            raise

    def _prepare_datasets(self, 
                         sku_tuples: SkuList, 
                         modeling_strategy: ModelingStrategy) -> List[ModelingDataset]:
        """Step 1: Prepare datasets for training (delegates to DataLoader)."""
        data_loader = DataLoader(self.data_config)
        
        if modeling_strategy == ModelingStrategy.COMBINED:
            # Single dataset for all SKUs
            dataset = data_loader.prepare_modeling_dataset(sku_tuples, modeling_strategy)
            return [dataset]
            
        elif modeling_strategy == ModelingStrategy.INDIVIDUAL:
            # Separate dataset for each SKU
            datasets = []
            for sku_tuple in sku_tuples:
                dataset = data_loader.prepare_modeling_dataset([sku_tuple], modeling_strategy)
                datasets.append(dataset)
            return datasets
            
        else:
            raise ValueError(f"Unknown modeling strategy: {modeling_strategy}")

    def _train_models(self, 
                     datasets: List[ModelingDataset], 
                     experiment_config: ExperimentConfig) -> List['TrainedModel']:
        """Step 2: Train models (delegates to ModelTrainer)."""
        # Create trainer with experiment configuration
        trainer = self._create_model_trainer(experiment_config)
        
        all_trained_models = []
        for i, dataset in enumerate(datasets):
            logger.info(f"Training model {i+1}/{len(datasets)} for {len(dataset.sku_tuples)} SKU(s)")
            
            # Train model(s) for this dataset (may return multiple models for quantile)
            trained_models = trainer.train_model(dataset, experiment_config.model_type)
            all_trained_models.extend(trained_models)
            
            # Log training completion
            for _ in trained_models:
                sku_info = f"{len(dataset.sku_tuples)} SKUs" if len(dataset.sku_tuples) > 1 else f"SKU {dataset.sku_tuples[0]}"
                logger.info(f"Trained {experiment_config.model_type} for {sku_info}")
        
        return all_trained_models

    def _evaluate_models(self, 
                        trained_models: List['TrainedModel'], 
                        datasets: List[ModelingDataset]) -> List[BenchmarkModel]:
        """Step 3: Evaluate models (delegates to ModelEvaluator)."""
        # Create data loader and empty registry for evaluator
        data_loader = DataLoader(self.data_config)
        from .data_structures import ModelRegistry
        temp_registry = ModelRegistry()
        evaluator = ModelEvaluator(data_loader, temp_registry)
        
        evaluated_models = []
        for i, trained_model in enumerate(trained_models):
            # Find corresponding dataset for this model
            dataset = self._find_dataset_for_trained_model(trained_model, datasets)
            
            # Use ModelEvaluator to convert TrainedModel to BenchmarkModel with metrics
            benchmark_model = evaluator.evaluate_trained_model(
                trained_model=trained_model,
                X_test=dataset.X_test,
                y_test=dataset.y_test
            )
            
            evaluated_models.append(benchmark_model)
            
            # Log evaluation results
            metrics = benchmark_model.metadata.performance_metrics
            rmse = metrics.get('rmse', 'N/A')
            logger.info(f"Evaluated model {i+1}: RMSE = {rmse}")
            
            # Log quantile-specific metrics if applicable
            if trained_model.quantile_level is not None:
                coverage = metrics.get('coverage_probability', 'N/A')
                logger.info(f"  Quantile α={trained_model.quantile_level}: Coverage = {coverage}")
        
        return evaluated_models

    def _package_results(self,
                        models: List[BenchmarkModel],
                        experiment_name: str,
                        experiment_config: ExperimentConfig,
                        modeling_strategy: ModelingStrategy,
                        sku_tuples: SkuList) -> ExperimentResults:
        """Step 4: Package results (delegates to ReleaseManager)."""
        # Create evaluation summary
        evaluation_summary = self._create_evaluation_summary(models)
        
        # Create experiment results object
        results = ExperimentResults(
            models=models,
            experiment_name=experiment_name,
            model_type=experiment_config.model_type,
            modeling_strategy=modeling_strategy,
            sku_tuples=sku_tuples,
            experiment_config=experiment_config,
            evaluation_summary=evaluation_summary,
            output_directory=str(self.output_dir)
        )
        
        # Delegate to ReleaseManager for file system operations  
        release_manager = ComprehensiveReleaseManager()
        
        # Create temporary old-style results for ReleaseManager compatibility
        # TODO: Update ReleaseManager to accept new ExperimentResults format
        old_format_results = type('TempResults', (), {
            'models': models,
            'evaluation_results': evaluation_summary,
            'experiment_log': {"experiments": [{"experiment_name": experiment_name}]},
            'configurations': {"data_config": self.data_config},
            'experiment_name': experiment_name,
            'timestamp': datetime.now()
        })()
        
        release_dir = release_manager.create_complete_release(old_format_results, self.output_dir)
        
        # Update results with actual output directory
        results.output_directory = str(release_dir)
        logger.info(f"Results saved to: {release_dir}")
        return results

    def _create_model_trainer(self, experiment_config: ExperimentConfig) -> ModelTrainer:
        """Create ModelTrainer with experiment configuration."""
        # Create ModelTypeConfig directly from ExperimentConfig
        model_type_config = ModelTypeConfig(
            model_type=experiment_config.model_type,
            hyperparameters=experiment_config.hyperparameters.copy(),
            quantile_alphas=experiment_config.quantile_alphas
        )
        
        return ModelTrainer(model_type_config, experiment_config.random_state)

    def _find_dataset_for_trained_model(self, 
                                       trained_model: 'TrainedModel', 
                                       datasets: List[ModelingDataset]) -> ModelingDataset:
        """Find the dataset used to train a specific model."""
        model_skus = set(trained_model.sku_tuples)
        
        for dataset in datasets:
            dataset_skus = set(dataset.sku_tuples)
            if model_skus == dataset_skus:
                return dataset
        
        # Fallback: return first dataset if exact match not found
        logger.warning(f"Could not find exact dataset match for model with SKUs {model_skus}")
        return datasets[0]
    
    def _create_evaluation_summary(self, models: List[BenchmarkModel]) -> Dict[str, Any]:
        """Create evaluation summary from trained models."""
        if not models:
            return {}
        
        # Collect metrics from all models
        all_metrics = {}
        for model in models:
            metrics = model.metadata.performance_metrics
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Create summary
        summary = {
            "num_models": len(models),
            "model_identifiers": [model.get_identifier() for model in models]
        }
        
        # Add metric summaries
        for metric_name, values in all_metrics.items():
            if len(values) == 1:
                summary[metric_name] = values[0]
            else:
                summary[f"{metric_name}_mean"] = sum(values) / len(values)
                summary[f"{metric_name}_min"] = min(values)
                summary[f"{metric_name}_max"] = max(values)
        
        return summary