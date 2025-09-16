"""
Model training module with fixed hyperparameters.
Supports XGBoost and extensible to other algorithms.
"""

import xgboost as xgb
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple, Optional
import logging

from .model_types import model_registry
from .data_structures import (
    TrainedModel, DataSplit, ModelingStrategy, SkuList,
    ModelTypeConfig, ModelingDataset
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training with fixed hyperparameters."""
    
    def __init__(self, model_config: ModelTypeConfig, random_state: int = 42):
        self.model_config = model_config
        self.random_state = random_state
    
    def train_model(self, dataset: ModelingDataset, model_type: str) -> List['TrainedModel']:
        """
        Train model(s) using ModelingDataset with simplified interface.
        Supports multiple quantile levels for quantile models.
        
        PURE TRAINING: Returns TrainedModel objects without any evaluation logic.
        All metrics calculation is handled by ModelEvaluator.
        
        Args:
            dataset: Complete modeling dataset with all necessary data
            model_type: Type of model to train (should match configured type)
            
        Returns:
            List of trained TrainedModel(s) - multiple models for multi-quantile configs
        """
        # Validate that the provided model_type matches the configured type
        if model_type != self.model_config.model_type:
            raise ValueError(f"Model type mismatch: expected {self.model_config.model_type}, got {model_type}")
            
        logger.info(f"Training {model_type} model with {dataset.modeling_strategy.value} strategy for {len(dataset.sku_tuples)} SKU(s)")
        
        # Check if this is a multi-quantile configuration
        quantile_alphas = self.model_config.quantile_alphas
        
        if quantile_alphas is not None:
            # Train multiple quantile models
            return self._train_quantile_models(dataset, model_type, quantile_alphas)
        else:
            # Train single standard model
            model = self._train_single_model(dataset, model_type, quantile_alpha=None)
            return [model]

    
    def _train_quantile_models(self, dataset: ModelingDataset, model_type: str, quantile_alphas: List[float]) -> List['TrainedModel']:
        """Train multiple quantile models for different quantile levels."""
        models = []
        
        logger.info(f"Training {len(quantile_alphas)} quantile models for levels: {quantile_alphas}")
        
        for quantile_alpha in quantile_alphas:
            logger.info(f"Training quantile model for alpha={quantile_alpha}")
            
            model = self._train_single_model(dataset, model_type, quantile_alpha)
            models.append(model)
        
        logger.info(f"Completed training {len(models)} quantile models")
        return models
    
    def _train_single_model(self, dataset: ModelingDataset, model_type: str, quantile_alpha: Optional[float] = None) -> 'TrainedModel':
        """
        Train a single model (quantile or standard) using ModelingDataset.
        
        PURE TRAINING: No evaluation, metrics calculation, or test predictions.
        Returns TrainedModel with only training results and basic metadata.
        """
        
        # Convert to numpy arrays for training
        X_train_np = dataset.X_train.select(dataset.feature_cols).to_numpy()
        y_train_np = dataset.y_train.select(dataset.target_col).to_numpy().flatten()
        X_test_np = dataset.X_test.select(dataset.feature_cols).to_numpy()
        y_test_np = dataset.y_test.select(dataset.target_col).to_numpy().flatten()
        
        # Get the single model configuration
        hyperparameters = self.model_config.merge_with_defaults()
        
        # Ensure random_state is set
        hyperparameters['random_state'] = self.random_state
        
        logger.info(f"Using hyperparameters: {hyperparameters}")
        
        # Log model-specific parameters
        if quantile_alpha is not None:
            logger.info(f"Quantile alpha: {quantile_alpha}")
        
        # Train model with fixed hyperparameters using model factory
        final_model = self._train_model_with_params(
            X_train_np, y_train_np, hyperparameters, model_type,
            X_val=X_test_np, y_val=y_test_np, quantile_alpha=quantile_alpha
        )
        
        # Create data split info from dataset split_info
        data_split = DataSplit(
            train_bdIDs=dataset.split_info["train_bdids"],
            validation_bdIDs=dataset.split_info["test_bdids"],  # Using test as validation for consistency
            split_date=dataset.split_info["split_date"]
        )
        
        # Create TrainedModel with pure training results (NO METRICS)
        trained_model = TrainedModel(
            model=final_model,
            model_type=model_type,
            modeling_strategy=dataset.modeling_strategy,
            sku_tuples=dataset.sku_tuples,
            hyperparameters=hyperparameters,
            training_config={"model_config": self.model_config.__dict__, "random_state": self.random_state},
            feature_columns=dataset.feature_cols,
            target_column=dataset.target_col,
            data_split=data_split,
            quantile_level=quantile_alpha,
            model_instance="default"
        )
        
        logger.info(f"Model training completed (no evaluation performed)")
        
        return trained_model
    
    def _train_model_with_params(self, X_train: np.ndarray, y_train: np.ndarray, 
                                hyperparameters: Dict[str, Any], model_type: str,
                                X_val: np.ndarray = None, y_val: np.ndarray = None,
                                quantile_alpha: float = None) -> Any:
        """Train model with specified hyperparameters using model factory."""
        try:           
            # Get the appropriate model class
            model_class = model_registry.get_model_class(model_type)
            if model_class is None:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Prepare model parameters
            model_params = hyperparameters.copy()
            
            # Add quantile parameter for quantile models if required
            if model_registry.requires_quantile(model_type) and quantile_alpha is not None:
                model_instance = model_class(quantile_alphas=[quantile_alpha], **model_params)
            else:
                model_instance = model_class(**model_params)
            
            # Train the model
            if hasattr(model_instance, 'train'):
                model_instance.train(X_train, y_train, X_val, y_val)
            else:
                # Fallback to fit method for scikit-learn compatible models
                model_instance.fit(X_train, y_train)
            
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to train model type '{model_type}': {str(e)}")
            raise
    

