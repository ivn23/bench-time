"""
Model training module with fixed hyperparameters.
Supports XGBoost and extensible to other algorithms.
"""

import xgboost as xgb
import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List, Tuple, Optional
import logging

from .data_structures import (
    BenchmarkModel, ModelMetadata, DataSplit, GranularityLevel,
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
                   X_val: pl.DataFrame,
                   y_val: pl.DataFrame,
                   feature_cols: List[str],
                   target_col: str,
                   granularity: GranularityLevel,
                   entity_ids: Dict[str, Any]) -> BenchmarkModel:
        """
        Train a model with fixed hyperparameters.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            feature_cols: List of feature column names
            target_col: Target column name
            granularity: Model granularity level
            entity_ids: Entity identifiers
            
        Returns:
            Trained BenchmarkModel
        """
        logger.info(f"Training {self.config.model_type} model for {granularity.value} level")
        
        # Convert to numpy arrays for training
        X_train_np = X_train.select(feature_cols).to_numpy()
        y_train_np = y_train.select(target_col).to_numpy().flatten()
        X_val_np = X_val.select(feature_cols).to_numpy()
        y_val_np = y_val.select(target_col).to_numpy().flatten()
        
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
        
        # Evaluate on validation set
        val_predictions = final_model.predict(X_val_np)
        val_predictions = np.round(val_predictions).astype(int)
        
        metrics = self._calculate_metrics(y_val_np, val_predictions)
        
        # Create model metadata
        metadata = ModelMetadata(
            model_id=f"{granularity.value}_{entity_ids}_{self.config.model_type}",
            granularity=granularity,
            entity_ids=entity_ids,
            model_type=self.config.model_type,
            hyperparameters=hyperparameters,
            training_config=self.config.__dict__,
            performance_metrics=metrics,
            feature_columns=feature_cols,
            target_column=target_col
        )
        
        # Create data split info
        train_bdids = X_train.select("bdID").to_numpy().flatten()
        val_bdids = X_val.select("bdID").to_numpy().flatten()
        
        data_split = DataSplit(
            train_bdIDs=train_bdids,
            validation_bdIDs=val_bdids
        )
        
        # Create benchmark model
        benchmark_model = BenchmarkModel(
            metadata=metadata,
            model=final_model,
            data_split=data_split
        )
        
        logger.info(f"Model training completed. Validation MSE: {metrics['mse']:.4f}")
        
        return benchmark_model
    
    def train_with_cross_validation(self,
                                   X: pl.DataFrame,
                                   y: pl.DataFrame,
                                   feature_cols: List[str],
                                   target_col: str,
                                   granularity: GranularityLevel,
                                   entity_ids: Dict[str, Any]) -> BenchmarkModel:
        """
        Train model using time series cross-validation for evaluation.
        
        Args:
            X, y: Full dataset
            feature_cols: Feature column names
            target_col: Target column name  
            granularity: Model granularity level
            entity_ids: Entity identifiers
            
        Returns:
            Trained BenchmarkModel with CV performance
        """
        logger.info(f"Training with {self.config.cv_folds}-fold time series CV")
        
        # Convert to numpy for CV
        X_np = X.select(feature_cols).to_numpy()
        y_np = y.select(target_col).to_numpy().flatten()
        
        # Get hyperparameters from config
        hyperparameters = self.config.hyperparameters.copy()
        hyperparameters.update(self.config.model_params)
        hyperparameters['random_state'] = self.config.random_state
        
        logger.info(f"Using hyperparameters: {hyperparameters}")
        
        # Time series cross-validation for performance estimation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_np):
            X_fold_train, X_fold_val = X_np[train_idx], X_np[val_idx]
            y_fold_train, y_fold_val = y_np[train_idx], y_np[val_idx]
            
            model = self._train_model_with_params(
                X_fold_train, y_fold_train, hyperparameters, self.config.model_type
            )
            predictions = model.predict(X_fold_val)
            predictions = np.round(predictions).astype(int)
            
            mse = mean_squared_error(y_fold_val, predictions)
            cv_scores.append(mse)
        
        cv_mse = np.mean(cv_scores)
        logger.info(f"CV MSE: {cv_mse:.4f}")
        
        # Train final model on full dataset
        final_model = self._train_model_with_params(
            X_np, y_np, hyperparameters, self.config.model_type
        )
        
        # Create model metadata
        metadata = ModelMetadata(
            model_id=f"cv_{granularity.value}_{entity_ids}_{self.config.model_type}",
            granularity=granularity,
            entity_ids=entity_ids,
            model_type=self.config.model_type,
            hyperparameters=hyperparameters,
            training_config=self.config.__dict__,
            performance_metrics={"cv_mse": cv_mse},
            feature_columns=feature_cols,
            target_column=target_col
        )
        
        # Create data split (full dataset used for training)
        all_bdids = X.select("bdID").to_numpy().flatten()
        data_split = DataSplit(
            train_bdIDs=all_bdids,
            validation_bdIDs=np.array([])  # Empty for CV training
        )
        
        benchmark_model = BenchmarkModel(
            metadata=metadata,
            model=final_model,
            data_split=data_split
        )
        
        return benchmark_model
    
    def _train_model_with_params(self, X_train: np.ndarray, y_train: np.ndarray,
                                params: Dict[str, Any], model_type: str) -> Any:
        """Train model with given parameters."""
        if model_type == "xgboost":
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = float('inf')
        
        return metrics


class EnsembleTrainer:
    """Trainer for ensemble models."""
    
    def __init__(self, base_trainers: List[ModelTrainer]):
        self.base_trainers = base_trainers
    
    def train_ensemble(self,
                      X_train: pl.DataFrame,
                      y_train: pl.DataFrame,
                      X_val: pl.DataFrame, 
                      y_val: pl.DataFrame,
                      feature_cols: List[str],
                      target_col: str,
                      granularity: GranularityLevel,
                      entity_ids: Dict[str, Any]) -> List[BenchmarkModel]:
        """Train ensemble of models."""
        models = []
        
        for i, trainer in enumerate(self.base_trainers):
            logger.info(f"Training ensemble member {i+1}/{len(self.base_trainers)}")
            
            model = trainer.train_model(
                X_train, y_train, X_val, y_val,
                feature_cols, target_col, granularity, entity_ids
            )
            
            # Update model ID to indicate ensemble membership
            model.metadata.model_id = f"ensemble_{i}_{model.metadata.model_id}"
            models.append(model)
        
        return models