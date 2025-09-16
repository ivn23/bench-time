"""
Quantile XGBoost regression model implementation.

This module implements quantile regression using XGBoost with a custom objective function,
based on the approach from the quantile_xgboost_simple.ipynb notebook.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import xgboost as xgb

from .base import BaseModel, ModelTrainingError, ModelPredictionError


class XGBoostQuantileModel(BaseModel):
    """
    Quantile XGBoost regression model using custom objective function.
    
    This implementation uses xgb.train() with a custom pinball loss objective
    to perform quantile regression, following the approach from the notebook.
    """
    
    MODEL_TYPE = "xgboost_quantile"
    DESCRIPTION = "Quantile XGBoost regression model with custom objective"
    DEFAULT_HYPERPARAMETERS = {
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42
    }
    REQUIRES_QUANTILE = True
    
    def __init__(self, quantile_alphas: List[float] = None, **model_params):
        """
        Initialize XGBoost quantile model.
        
        Args:
            quantile_alphas: List of target quantile levels (e.g., [0.7] for 70% quantile)
            **model_params: XGBoost hyperparameters
        """
        super().__init__(**model_params)
        if quantile_alphas is None:
            quantile_alphas = [0.7]
        
        if not quantile_alphas or len(quantile_alphas) != 1:
            raise ValueError("XGBoost quantile model currently supports exactly one quantile level")
        
        self.quantile_alpha = quantile_alphas[0]  # Keep internal usage for now
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
