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

from typing import Dict, List, Any, Optional, Union
import random

from .hyperparameter_tuning import HyperparameterTuner, TuningResult

logger = logging.getLogger(__name__)


class BenchmarkPipeline:

    def __init__(self, data_config: DataConfig):

        self.data_config = data_config
        logger.info("BenchmarkPipeline initialized")

    def run_experiment(self,
                      sku_tuples: SkuList,
                      modeling_strategy: ModelingStrategy,
                      model_type: str,
                      hyperparameters: Optional[Dict[str, Any]] = None,
                      quantile_alphas: Optional[List[float]] = None,
                      experiment_name: Optional[str] = None,
                      evaluate_on_test: bool = True,
                      mode: str = "train",
                      tune_on: Optional[int] = None,
                      tuning_config: Optional[Dict[str, Any]] = None,
                      random_state: int = 42) -> Union[ExperimentResults, TuningResult]:
        """
        Run benchmark experiment with mode-based execution.
        
        Args:
            sku_tuples: List of (product_id, store_id) tuples
            modeling_strategy: COMBINED or INDIVIDUAL strategy
            model_type: Type of model (e.g., 'xgboost_quantile')
            hyperparameters: Model hyperparameters (required for mode='train')
            quantile_alphas: Quantile levels for quantile models
            experiment_name: Optional name for the experiment
            evaluate_on_test: Whether to evaluate on test data (train mode only)
            mode: Execution mode - Options:
                - "train": Normal training workflow (default)
                - "hp_tune": Hyperparameter tuning mode
            tune_on: Number of SKUs to sample for tuning (required for mode='hp_tune')
            tuning_config: Tuning configuration dict with keys:
                - 'n_trials': Number of Optuna trials (default: 50)
                - 'n_folds': Number of CV folds (default: 3)
            random_state: Random seed for reproducibility
            
        Returns:
            ExperimentResults: If mode="train"
            TuningResult: If mode="hp_tune"
        """
        
        # Validate mode parameter
        if mode not in ["train", "hp_tune"]:
            raise ValueError(f"mode must be 'train' or 'hp_tune', got '{mode}'")
        
        # Mode-specific validation
        if mode == "hp_tune":
            if tune_on is None:
                raise ValueError("tune_on must be specified when mode='hp_tune'")
            logger.info(f"HYPERPARAMETER TUNING MODE: Sampling {tune_on} SKUs for optimization")
            return self._run_hyperparameter_tuning(
                sku_tuples, model_type, quantile_alphas,
                tune_on, tuning_config, random_state
            )
        
        elif mode == "train":
            if hyperparameters is None:
                raise ValueError("hyperparameters must be provided when mode='train'")
            
            # EXISTING TRAINING WORKFLOW (unchanged)
            logger.info(f"TRAINING MODE: {len(sku_tuples)} SKUs with {modeling_strategy.value} strategy")
            
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

    def _run_hyperparameter_tuning(self,
                                    sku_tuples: SkuList,
                                    model_type: str,
                                    quantile_alphas: Optional[List[float]],
                                    tune_on: int,
                                    tuning_config: Optional[Dict[str, Any]],
                                    random_state: int) -> TuningResult:
        """
        Run hyperparameter tuning on sampled SKUs.
        
        Steps:
        1. Sample `tune_on` SKUs from provided sku_tuples
        2. Create COMBINED dataset from sampled SKUs
        3. Run hyperparameter optimization using CV
        4. Return TuningResult with best params and validation loss
        
        Args:
            sku_tuples: Available SKUs to sample from
            model_type: Type of model to tune
            quantile_alphas: Quantile levels (tunes on first only)
            tune_on: Number of SKUs to sample
            tuning_config: Tuning configuration (n_trials, n_folds)
            random_state: Random seed
            
        Returns:
            TuningResult with optimized hyperparameters
        """
        # Step 1: Sample SKUs
        random.seed(random_state)
        sampled_skus = random.sample(sku_tuples, min(tune_on, len(sku_tuples)))
        logger.info(f"Sampled {len(sampled_skus)} SKUs for tuning: {sampled_skus[:5]}{'...' if len(sampled_skus) > 5 else ''}")
        
        # Step 2: Create COMBINED dataset from sampled SKUs
        data_loader = DataLoader(self.data_config)
        tuning_dataset = data_loader.prepare_modeling_dataset(
            sampled_skus,
            ModelingStrategy.COMBINED  # Always COMBINED for tuning
        )
        logger.info(f"Created tuning dataset: {tuning_dataset.n_train_samples} train samples, {tuning_dataset.n_features} features")
        print(f"Tuning dataset has {tuning_dataset.n_train_samples} training samples and {tuning_dataset.n_features} features.")


        # Step 3: Setup tuning configuration
        config = tuning_config or {}
        n_trials = config.get('n_trials', 50)
        n_folds = config.get('n_folds', 3)
        logger.info(f"Tuning configuration: {n_trials} trials, {n_folds} CV folds")
        
        # Step 4: Handle quantile models (tune for first quantile only)
        quantile_alpha = quantile_alphas[0] if quantile_alphas else None
        if quantile_alphas and len(quantile_alphas) > 1:
            logger.warning(
                f"Multiple quantiles specified {quantile_alphas}, "
                f"tuning for alpha={quantile_alpha} only. "
                f"Hyperparameters are quantile-agnostic and will work for all quantiles."
            )
        
        # Step 5: Run hyperparameter optimization
        tuner = HyperparameterTuner(random_state=random_state)
        result = tuner.tune(
            dataset=tuning_dataset,
            model_type=model_type,
            quantile_alpha=quantile_alpha,
            n_trials=n_trials,
            n_folds=n_folds
        )
        
        # Step 6: Log results
        logger.info(f"Tuning complete! Best validation loss: {result.best_score:.6f}")
        logger.info(f"Best hyperparameters: {result.best_params}")
        logger.info(f"Optimization time: {result.optimization_time:.2f}s")
        
        return result
            
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