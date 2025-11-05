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
    Quantile XGBoost regression model using native reg:quantileerror objective.

    This implementation uses xgb.train() with XGBoost's native quantile regression
    objective, which provides proper handling of gradient asymmetry and stability.
    """
    
    MODEL_TYPE = "xgboost_quantile"
    DESCRIPTION = "Quantile XGBoost regression model with native reg:quantileerror objective"
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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **training_kwargs) -> None:
        """
        Train the quantile XGBoost model using native reg:quantileerror objective.

        Args:
            X_train: Training features
            y_train: Training targets
            **training_kwargs: Additional training parameters (e.g., compute_config)
        """

        # Create DMatrix for training data
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Use n_estimators stored during initialization as num_boost_round
        num_boost_round = self.n_estimators

        # Configure parameters with native quantile objective
        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': self.quantile_alpha,
            **self.model_params
        }

        # Train model with native quantile objective
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
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
            "training_method": "xgb_train_native_objective"
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
            # Learning rate: log scale, smaller is safer for stability on small data
            'eta': trial.suggest_float('eta', 0.01, 0.2, log=True),

            # Tree complexity: small range, shallow trees for few samples
            'max_depth': trial.suggest_int('max_depth', 2, 6),

            # Regularization on minimum child weight: controls overfitting on tiny leaves
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),

            # Sampling parameters: prevent overfitting, stable ranges
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

            # Split regularization: discourages small gain splits
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),

            # L1 / L2 regularization: log scale, large variation allowed
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),

            # Histogram binning — impacts performance and speed tradeoff
            'max_bin': trial.suggest_int('max_bin', 16, 256),

            # Grow policy and conditional leaves — adds flexibility
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            # Note: you can conditionally add max_leaves if grow_policy == 'lossguide'

            # n_estimators here corresponds to num_boost_round in native API
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),

            # Random seed for reproducibility
            'seed': random_state
        }
