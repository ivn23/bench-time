"""
Quantile XGBoost regression model implementation.

This module implements quantile regression using XGBoost with a custom objective function,
based on the approach from the quantile_xgboost_simple.ipynb notebook.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import xgboost as xgb

from .base import BaseModel, ModelPredictionError


class XGBoostQuantileModel(BaseModel):
    """
    Quantile XGBoost regression model using custom objective function.
    
    This implementation uses xgb.train() with a custom pinball loss objective
    to perform quantile regression, following the approach from the notebook.
    """
    
    MODEL_TYPE = "xgboost_quantile"
    DESCRIPTION = "Quantile XGBoost regression model with custom objective"
    REQUIRES_QUANTILE = True
    
    def __init__(self, quantile_alphas: List[float] = None, **model_params):
        """
        Initialize XGBoost quantile model.

        Requires explicit configuration - no defaults are provided.

        Args:
            quantile_alphas: List of target quantile levels (e.g., [0.7] for 70% quantile).
                            Must contain exactly one value between 0 and 1.
            **model_params: XGBoost hyperparameters. Must include either 'n_estimators'
                          or 'num_boost_round' to specify number of boosting rounds.

        Raises:
            ValueError: If quantile_alphas is None, empty, contains != 1 value,
                       or if required parameters are missing.
        """
        super().__init__(**model_params)

        # Validate quantile_alphas is provided
        if quantile_alphas is None:
            raise ValueError(
                "quantile_alphas must be explicitly provided. "
                "Pass a list with one quantile level, e.g., quantile_alphas=[0.7]"
            )

        if not quantile_alphas or len(quantile_alphas) != 1:
            raise ValueError(
                "XGBoost quantile model currently supports exactly one quantile level. "
                f"Got {len(quantile_alphas) if quantile_alphas else 0} values."
            )

        self.quantile_alpha = quantile_alphas[0]  # Single quantile per model instance

        # Validate quantile_alpha is in valid range
        if not 0 < self.quantile_alpha < 1:
            raise ValueError(
                f"quantile_alpha must be between 0 and 1 (exclusive), got {self.quantile_alpha}"
            )

        # Handle n_estimators/num_boost_round extraction
        # Must be provided explicitly - no defaults
        # These control training iterations, not tree parameters, so remove from model_params
        if 'num_boost_round' in self.model_params:
            self.n_estimators = self.model_params.pop('num_boost_round')
        elif 'n_estimators' in self.model_params:
            self.n_estimators = self.model_params.pop('n_estimators')
        else:
            raise ValueError(
                "Either 'n_estimators' or 'num_boost_round' must be provided in model parameters. "
                "These specify the number of boosting rounds for training."
            )
    
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
            gradient = np.where(residual >= 0, -self.quantile_alpha, 1 - self.quantile_alpha)
            
            # Hessian approximation (constant for stability)
            hessian = np.ones_like(gradient) * 1 
            
            return gradient, hessian
        
        return objective
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the quantile XGBoost model using custom objective.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **training_kwargs: Additional training parameters
        """

        # Create DMatrix for training data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Use n_estimators stored during initialization as num_boost_round
        num_boost_round = self.n_estimators
        
        
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
            
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.

        Returns:
            Dictionary containing model type, parameters, and quantile-specific info
        """
        info = {
            "model_type": self.MODEL_TYPE,
            "parameters": self.model_params,
            "quantile_alpha": self.quantile_alpha,
            "is_trained": self.is_trained,
            "model_class": "XGBQuantileModel",
            "training_method": "xgb_train_custom_objective"
        }

        return info

    @staticmethod
    def get_search_space(trial, random_state: int) -> Dict[str, Any]:
        """
        Define hyperparameter search space for XGBoost quantile model.

        Args:
            trial: Optuna trial object for suggesting hyperparameters
            random_state: Random seed for reproducibility

        Returns:
            Dictionary of hyperparameters sampled from the search space
        """
        return {
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'seed': random_state
        }
