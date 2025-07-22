"""
Model training module with Optuna hyperparameter optimization.
Supports XGBoost and extensible to other algorithms.
"""

import xgboost as xgb
import optuna
import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging

from .data_structures import (
    BenchmarkModel, ModelMetadata, DataSplit, GranularityLevel,
    TrainingConfig
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training with hyperparameter optimization."""
    
    def __init__(self, training_config: TrainingConfig):
        self.config = training_config
        self.model_factories = {
            "xgboost": self._create_xgboost_objective
        }
    
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
        Train a model with hyperparameter optimization.
        
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
        
        # Hyperparameter optimization
        if self.config.model_type not in self.model_factories:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        objective_func = self.model_factories[self.config.model_type](
            X_train_np, y_train_np, X_val_np, y_val_np
        )
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_func, n_trials=self.config.n_trials)
        
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation MSE: {study.best_value:.4f}")
        
        # Train final model with best parameters
        final_model = self._train_final_model(
            X_train_np, y_train_np, best_params, self.config.model_type
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
            hyperparameters=best_params,
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
        Train model using time series cross-validation.
        
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
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        if self.config.model_type not in self.model_factories:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Run hyperparameter optimization with CV
        def cv_objective(trial):
            params = self._suggest_hyperparameters(trial, self.config.model_type)
            fold_scores = []
            
            for train_idx, val_idx in tscv.split(X_np):
                X_fold_train, X_fold_val = X_np[train_idx], X_np[val_idx]
                y_fold_train, y_fold_val = y_np[train_idx], y_np[val_idx]
                
                model = self._train_final_model(X_fold_train, y_fold_train, params, self.config.model_type)
                predictions = model.predict(X_fold_val)
                predictions = np.round(predictions).astype(int)
                
                mse = mean_squared_error(y_fold_val, predictions)
                fold_scores.append(mse)
            
            return np.mean(fold_scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(cv_objective, n_trials=self.config.n_trials)
        
        best_params = study.best_params
        logger.info(f"Best CV parameters: {best_params}")
        logger.info(f"Best CV MSE: {study.best_value:.4f}")
        
        # Train final model on full dataset
        final_model = self._train_final_model(X_np, y_np, best_params, self.config.model_type)
        
        # Create model metadata
        metadata = ModelMetadata(
            model_id=f"cv_{granularity.value}_{entity_ids}_{self.config.model_type}",
            granularity=granularity,
            entity_ids=entity_ids,
            model_type=self.config.model_type,
            hyperparameters=best_params,
            training_config=self.config.__dict__,
            performance_metrics={"cv_mse": study.best_value},
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
    
    def _create_xgboost_objective(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Callable:
        """Create Optuna objective function for XGBoost."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'random_state': self.config.random_state
            }
            
            # Override with user-specified params
            params.update(self.config.model_params)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_val)
            predictions = np.round(predictions).astype(int)
            
            mse = mean_squared_error(y_val, predictions)
            return mse
        
        return objective
    
    def _suggest_hyperparameters(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for given model type."""
        if model_type == "xgboost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'random_state': self.config.random_state
            }
            params.update(self.config.model_params)
            return params
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          params: Dict[str, Any], model_type: str) -> Any:
        """Train final model with given parameters."""
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
    
    def add_model_factory(self, model_type: str, objective_factory: Callable):
        """Add support for new model types."""
        self.model_factories[model_type] = objective_factory
        logger.info(f"Added support for model type: {model_type}")


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