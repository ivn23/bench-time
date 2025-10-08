"""
Simplified benchmark pipeline using the new simplified data structures.

This pipeline provides a clean, direct interface for the user's workflow:
DataConfig → Train Models → Get TrainingResults

No overengineered abstractions, just the essential functionality.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .structures import (
    ModelConfig, TrainingResult, ExperimentResults, SplitInfo,
    ModelingStrategy, SkuList, DataConfig, ModelingDataset,
    create_config, validate_sku_tuples, validate_modeling_strategy
)
from .data_loading import DataLoader
from .evaluation import ModelEvaluator
from .model_types import model_registry
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BenchmarkPipeline:

    def __init__(self, data_config: DataConfig):

        self.data_config = data_config
        logger.info("BenchmarkPipeline initialized")

    def run_experiment(self,
                      sku_tuples: SkuList,
                      modeling_strategy: ModelingStrategy,
                      model_type: str,
                      hyperparameters: Dict[str, Any],
                      quantile_alphas: Optional[List[float]] = None,
                      experiment_name: Optional[str] = None,
                      evaluate_on_test: bool = True,
                      random_state: int = 42) -> ExperimentResults:

        # Validate inputs
        validate_sku_tuples(sku_tuples)
        validate_modeling_strategy(modeling_strategy, sku_tuples)
        
        # Create simplified configuration
        config = create_config(
            model_type=model_type,
            hyperparameters=hyperparameters,
            quantile_alphas=quantile_alphas,
            random_state=random_state
        )
        
        # Generate experiment name
        exp_name = experiment_name 
        
        logger.info(f"Starting experiment '{exp_name}': {model_type} with {modeling_strategy.value} strategy for {len(sku_tuples)} SKU(s)")
        
        # Step 1: Prepare data
        logger.info("Preparing data...")
        datasets = self._prepare_datasets(sku_tuples, modeling_strategy)
        
        # Step 2: Train models
        logger.info(f"Training {len(datasets)} model(s)...")
        training_results = self._train_models(datasets, config)
        
        # Step 3: Optionally evaluate on test data
        if evaluate_on_test:
            logger.info(f"Evaluating {len(training_results)} model(s) on test data...")
            training_results = self._evaluate_models(training_results, datasets)
        
        # Step 4: Package results
        results = ExperimentResults(
            training_results=training_results,
            experiment_name=exp_name,
            config=config
        )
        
        logger.info(f"Experiment '{exp_name}' completed successfully. Trained {results.num_models} model(s)")
        return results
            
    def _prepare_datasets(self, sku_tuples: SkuList, modeling_strategy: ModelingStrategy) -> List[ModelingDataset]:
        
        data_loader = DataLoader(self.data_config)
        
        if modeling_strategy == ModelingStrategy.COMBINED:
            # Single dataset for all SKUs
            dataset = data_loader.prepare_modeling_dataset(sku_tuples, modeling_strategy)
            return [dataset]
            
        elif modeling_strategy == ModelingStrategy.INDIVIDUAL:
            # Separate dataset for each SKU
            datasets = []
            for sku_tuple in tqdm(sku_tuples):
                dataset = data_loader.prepare_modeling_dataset([sku_tuple], modeling_strategy)
                datasets.append(dataset)
            return datasets
            
        else:
            raise ValueError(f"Unknown modeling strategy: {modeling_strategy}")

    def _train_models(self, datasets: List[ModelingDataset], config: ModelConfig) -> List[TrainingResult]:
        """Train models and return TrainingResult objects""" 

        all_results = []
        
        for i, dataset in enumerate(tqdm(datasets, desc="Training models")):
            logger.info(f"Training model {i+1}/{len(datasets)} for {len(dataset.sku_tuples)} SKU(s)")
            
            if config.is_quantile_model:
                # Train multiple quantile models
                for quantile_alpha in config.quantile_alphas:
                    result = self._train_single_model(dataset, config, quantile_alpha)
                    all_results.append(result)
            else:
                # Train single standard model
                result = self._train_single_model(dataset, config)
                all_results.append(result)
        
        return all_results

    def _train_single_model(self, dataset: ModelingDataset, config: ModelConfig, 
                           quantile_alpha: Optional[float] = None) -> TrainingResult:
        """Train a single model and return TrainingResult."""
        
        # Get model class
        model_class = model_registry.get_model_class(config.model_type)
        
        # Prepare hyperparameters
        model_params = config.hyperparameters.copy()
        if quantile_alpha is not None:
            model_params['quantile_alphas'] = [quantile_alpha]
        
        # Create and train model
        model_instance = model_class(**model_params)
        
        # Use DataLoader for centralized data preparation
        X_train, y_train = DataLoader.prepare_training_data(dataset, config.model_type)
        
        # Train the model 
        model_instance.train(X_train, y_train)
        
        # Get split info from dataset
        split_info = dataset.get_split_info()
        
        # Create and return TrainingResult
        result = TrainingResult(
            model=model_instance,
            model_type=config.model_type,
            modeling_strategy=dataset.modeling_strategy,
            sku_tuples=dataset.sku_tuples,
            hyperparameters=model_params,
            feature_columns=dataset.feature_cols,
            target_column=dataset.target_col,
            split_info=split_info,
            quantile_level=quantile_alpha
        )
        
        sku_info = f"{len(dataset.sku_tuples)} SKUs" if len(dataset.sku_tuples) > 1 else f"SKU {dataset.sku_tuples[0]}"
        logger.info(f"Trained {config.model_type} for {sku_info}")
        
        return result

    def _evaluate_models(self, training_results: List[TrainingResult], 
                        datasets: List[ModelingDataset]) -> List[TrainingResult]:
        """
        Evaluate models on test data and add performance metrics.
        
        This directly updates the TrainingResult objects instead of creating
        a separate BenchmarkModel wrapper.
        """
        evaluator = ModelEvaluator()
        
        evaluated_results = []
        for i, result in enumerate(training_results):
            # Find corresponding dataset
            dataset = self._find_dataset_for_result(result, datasets)
            
            # Use the simplified evaluator with entire ModelingDataset
            evaluated_result = evaluator.evaluate_training_result(
                result, dataset
            )
            evaluated_results.append(evaluated_result)
            
            # Log evaluation results
            rmse = evaluated_result.performance_metrics.get('rmse', 'N/A')
            logger.info(f"Evaluated model {i+1}: RMSE = {rmse}")
            
            if evaluated_result.is_quantile_model():
                coverage = evaluated_result.performance_metrics.get('coverage_probability', 'N/A')
                logger.info(f"  Quantile α={evaluated_result.quantile_level}: Coverage = {coverage}")
        
        return evaluated_results

    def _find_dataset_for_result(self, result: TrainingResult, 
                                datasets: List[ModelingDataset]) -> ModelingDataset:
        """Find the dataset used to train a specific model."""
        result_skus = set(result.sku_tuples)
        
        for dataset in datasets:
            dataset_skus = set(dataset.sku_tuples)
            if result_skus == dataset_skus:
                return dataset
        
        # Fallback: return first dataset if exact match not found
        logger.warning(f"Could not find exact dataset match for model with SKUs {result_skus}")
        return datasets[0]