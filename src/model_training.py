"""
Model training module with fixed hyperparameters.
Supports XGBoost and extensible to other algorithms.
"""

import xgboost as xgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List, Tuple, Optional
import logging

from .data_structures import (
    BenchmarkModel, ModelMetadata, DataSplit, ModelingStrategy, SkuList,
    TrainingConfig
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training with fixed hyperparameters."""
    
    def __init__(self, training_config: TrainingConfig):
        self.config = training_config
    
    def train_model(self,
                   X_train: pl.DataFrame,
                   y_train: pl.DataFrame, 
                   X_test: pl.DataFrame,
                   y_test: pl.DataFrame,
                   feature_cols: List[str],
                   target_col: str,
                   modeling_strategy: ModelingStrategy,
                   sku_tuples: SkuList,
                   model_type: str) -> BenchmarkModel:
        """
        Train a model with fixed hyperparameters using model factory pattern.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data for evaluation
            feature_cols: List of feature column names
            target_col: Target column name
            modeling_strategy: How SKUs are being modeled (combined vs individual)
            sku_tuples: List of (product_id, store_id) tuples being modeled
            model_type: Type of model to train
            
        Returns:
            Trained BenchmarkModel
        """
        logger.info(f"Training {model_type} model with {modeling_strategy.value} strategy for {len(sku_tuples)} SKU(s)")
        
        # Convert to numpy arrays for training
        X_train_np = X_train.select(feature_cols).to_numpy()
        y_train_np = y_train.select(target_col).to_numpy().flatten()
        X_test_np = X_test.select(feature_cols).to_numpy()
        y_test_np = y_test.select(target_col).to_numpy().flatten()
        
        # Get model-specific configuration
        model_config = self.config.get_model_config(model_type)
        hyperparameters = model_config.merge_with_defaults()
        
        # Ensure random_state is set
        hyperparameters['random_state'] = self.config.random_state
        
        logger.info(f"Using hyperparameters: {hyperparameters}")
        
        # Log model-specific parameters
        if model_config.quantile_alpha is not None:
            logger.info(f"Quantile alpha: {model_config.quantile_alpha}")
        
        # Train model with fixed hyperparameters using model factory
        final_model = self._train_model_with_params(
            X_train_np, y_train_np, hyperparameters, model_type,
            X_val=X_test_np, y_val=y_test_np, quantile_alpha=model_config.quantile_alpha
        )
        
        # Make predictions on test set
        test_predictions = final_model.predict(X_test_np)
        
        # Apply prediction clipping for integer targets
        test_predictions = np.clip(np.round(test_predictions).astype(int), 0, None)
        
        # Calculate metrics using centralized metrics calculator
        from .metrics import MetricsCalculator
        metrics = MetricsCalculator.calculate_all_metrics(
            y_test_np, test_predictions, model_config.quantile_alpha
        )
        
        # Extract primary SKU for hierarchical storage
        primary_sku = sku_tuples[0]  # Use first SKU as primary
        product_id, store_id = primary_sku
        
        # Create model metadata with hierarchical storage support
        model_id_parts = [modeling_strategy.value, f"{len(sku_tuples)}skus", model_type]
        if model_config.quantile_alpha is not None:
            model_id_parts.append(f"q{model_config.quantile_alpha}")
        model_id = "_".join(model_id_parts)
        
        metadata = ModelMetadata(
            model_id=model_id,
            modeling_strategy=modeling_strategy,
            sku_tuples=sku_tuples,
            model_type=model_type,
            store_id=store_id,
            product_id=product_id,
            hyperparameters=hyperparameters,
            training_config=self.config.__dict__,
            performance_metrics=metrics,
            feature_columns=feature_cols,
            target_column=target_col
        )
        
        # Create data split info
        train_bdids = X_train.select("bdID").to_numpy().flatten()
        test_bdids = X_test.select("bdID").to_numpy().flatten()
        
        data_split = DataSplit(
            train_bdIDs=train_bdids,
            validation_bdIDs=test_bdids  # Using test as validation for consistency
        )
        
        # Create benchmark model
        benchmark_model = BenchmarkModel(
            metadata=metadata,
            model=final_model,
            data_split=data_split
        )
        
        logger.info(f"Model training completed. Primary metric: {metrics.get('rmse', metrics.get('mse', 'N/A'))}")
        
        return benchmark_model
    
    def _train_model_with_params(self, X_train: np.ndarray, y_train: np.ndarray, 
                                hyperparameters: Dict[str, Any], model_type: str,
                                X_val: np.ndarray = None, y_val: np.ndarray = None,
                                quantile_alpha: float = None) -> Any:
        """Train model with specified hyperparameters using model factory."""
        try:
            # Import the model type registry
            from .model_types import model_registry
            
            # Get the appropriate model class
            model_class = model_registry.get_model_class(model_type)
            if model_class is None:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Prepare model parameters
            model_params = hyperparameters.copy()
            
            # Add quantile parameter for quantile models if required
            if model_registry.requires_quantile(model_type) and quantile_alpha is not None:
                model_instance = model_class(quantile_alpha=quantile_alpha, **model_params)
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
    

