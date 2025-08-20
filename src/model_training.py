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
                   sku_tuples: SkuList) -> BenchmarkModel:
        """
        Train a model with fixed hyperparameters.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data for evaluation
            feature_cols: List of feature column names
            target_col: Target column name
            modeling_strategy: How SKUs are being modeled (combined vs individual)
            sku_tuples: List of (product_id, store_id) tuples being modeled
            
        Returns:
            Trained BenchmarkModel
        """
        logger.info(f"Training {self.config.model_type} model with {modeling_strategy.value} strategy for {len(sku_tuples)} SKU(s)")
        
        # Convert to numpy arrays for training
        X_train_np = X_train.select(feature_cols).to_numpy()
        y_train_np = y_train.select(target_col).to_numpy().flatten()
        X_test_np = X_test.select(feature_cols).to_numpy()
        y_test_np = y_test.select(target_col).to_numpy().flatten()
        
        # Get hyperparameters from config
        hyperparameters = self.config.hyperparameters.copy()
        # Merge with any additional model_params
        hyperparameters.update(self.config.model_params)
        # Ensure random_state is set
        hyperparameters['random_state'] = self.config.random_state
        
        logger.info(f"Using hyperparameters: {hyperparameters}")
        
        # Train model with fixed hyperparameters
        final_model = self._train_model_with_params(
            X_train_np, y_train_np, hyperparameters, self.config.model_type
        )
        
        # Evaluate on test set
        test_predictions = final_model.predict(X_test_np)
        test_predictions = np.clip(np.round(test_predictions).astype(int), 0, None)
        
        metrics = self._calculate_metrics(y_test_np, test_predictions)
        
        # Create model metadata
        metadata = ModelMetadata(
            model_id=f"{modeling_strategy.value}_{len(sku_tuples)}skus_{self.config.model_type}",
            modeling_strategy=modeling_strategy,
            sku_tuples=sku_tuples,
            model_type=self.config.model_type,
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
        
        logger.info(f"Model training completed. Test MSE: {metrics['mse']:.4f}")
        
        return benchmark_model
    
    def _train_model_with_params(self, X_train: np.ndarray, y_train: np.ndarray, 
                                hyperparameters: Dict[str, Any], model_type: str):
        """Train model with specified hyperparameters."""
        if model_type == "xgboost":
            model = xgb.XGBRegressor(**hyperparameters)
            model.fit(X_train, y_train)
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # MAPE (Mean Absolute Percentage Error)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            metrics['mape'] = float('inf')
            
        return metrics

