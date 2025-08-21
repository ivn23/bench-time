"""
Quantile XGBoost regression model implementation.

This module implements quantile regression using XGBoost with a custom objective function,
based on the approach from the quantile_xgboost_simple.ipynb notebook.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from .base import BaseModel, ModelTrainingError, ModelPredictionError


class XGBoostQuantileModel(BaseModel):
    """
    Quantile XGBoost regression model using custom objective function.
    
    This implementation uses xgb.train() with a custom pinball loss objective
    to perform quantile regression, following the approach from the notebook.
    """
    
    def __init__(self, quantile_alpha: float = 0.7, **model_params):
        """
        Initialize XGBoost quantile model.
        
        Args:
            quantile_alpha: Target quantile level (e.g., 0.7 for 70% quantile)
            **model_params: XGBoost hyperparameters
        """
        super().__init__(**model_params)
        self.quantile_alpha = quantile_alpha
        self.model_type = "xgboost_quantile"
        
        # Set default XGBoost parameters if not provided
        default_params = {
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        # Update with provided parameters
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def _create_quantile_objective(self):
        """
        Create quantile loss objective function for XGBoost.
        
        Returns:
            Objective function compatible with XGBoost obj parameter
        """
        def objective(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            """
            Quantile loss objective function (pinball loss).
            
            Returns gradient and hessian for XGBoost optimization.
            """
            y_true = dtrain.get_label()
            residual = y_true - predt
            
            # Gradient of quantile loss (pinball loss derivative)
            gradient = np.where(residual >= 0, self.quantile_alpha - 1, self.quantile_alpha)
            
            # Hessian approximation (constant for stability)
            hessian = np.ones_like(gradient) * 0.1
            
            return gradient, hessian
        
        return objective
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              num_boost_round: int = 100, **training_kwargs) -> None:
        """
        Train the quantile XGBoost model using custom objective.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            num_boost_round: Number of boosting rounds
            **training_kwargs: Additional training parameters
        """
        try:
            # Create DMatrix for training data
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # Create quantile objective function
            quantile_objective = self._create_quantile_objective()
            
            # Train model with custom objective
            self.model = xgb.train(
                params=self.model_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                obj=quantile_objective,
                verbose_eval=False
            )
            
            self.is_trained = True
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train XGBoost quantile model: {str(e)}")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make quantile predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of quantile predictions
            
        Raises:
            ModelPredictionError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise ModelPredictionError("Model must be trained before making predictions")
            
        try:
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest)
        except Exception as e:
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model type, parameters, and quantile-specific info
        """
        info = {
            "model_type": self.model_type,
            "parameters": self.model_params,
            "quantile_alpha": self.quantile_alpha,
            "is_trained": self.is_trained,
            "model_class": "XGBQuantileModel",
            "training_method": "xgb_train_custom_objective"
        }
        
        return info
        
    def get_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate quantile-specific and standard evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted quantile values
            
        Returns:
            Dictionary containing both standard and quantile-specific metrics
        """
        try:
            # Standard regression metrics
            metrics = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred))
            }
            
            # Additional standard metrics
            residuals = y_true - y_pred
            metrics.update({
                "max_error": float(np.max(np.abs(residuals))),
                "mean_error": float(np.mean(residuals)),
                "std_error": float(np.std(residuals))
            })
            
            # Calculate MAPE (avoiding division by zero)
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics["mape"] = float(mape)
            else:
                metrics["mape"] = float('inf')
                
            # Standard accuracy bands
            abs_errors = np.abs(residuals)
            metrics.update({
                "within_1_unit": float(np.mean(abs_errors <= 1.0)),
                "within_2_units": float(np.mean(abs_errors <= 2.0)),
                "within_5_units": float(np.mean(abs_errors <= 5.0))
            })
            
            # Quantile-specific metrics
            quantile_metrics = self._calculate_quantile_metrics(y_true, y_pred)
            metrics.update(quantile_metrics)
            
            return metrics
            
        except Exception as e:
            raise ValueError(f"Failed to calculate evaluation metrics: {str(e)}")
            
    def _calculate_quantile_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate quantile-specific evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted quantile values
            
        Returns:
            Dictionary of quantile-specific metrics
        """
        # Quantile score (pinball loss)
        residual = y_true - y_pred
        quantile_score_val = np.mean(
            np.where(residual >= 0, self.quantile_alpha * residual, 
                    (self.quantile_alpha - 1) * residual)
        )
        
        # Coverage probability (what percentage of actual values are below predictions)
        coverage = np.mean(y_true <= y_pred)
        
        # Coverage error (how far from target quantile)
        coverage_error = abs(coverage - self.quantile_alpha)
        
        return {
            "quantile_score": float(quantile_score_val),
            "coverage_probability": float(coverage),
            "coverage_error": float(coverage_error),
            "quantile_alpha": float(self.quantile_alpha)
        }