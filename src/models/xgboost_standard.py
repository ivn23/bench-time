"""
Standard XGBoost regression model implementation.

This module implements the standard XGBoost regressor using the native XGBoost API
(xgb.train with DMatrix) for consistency with XGBoostQuantileModel.
"""

from typing import Any, Dict
import numpy as np
import xgboost as xgb

from .base import BaseModel, ModelPredictionError


class XGBoostStandardModel(BaseModel):
    """
    Standard XGBoost regression model using native XGBoost API.

    This implementation uses xgb.train() with DMatrix for consistency with
    XGBoostQuantileModel and to enable advanced XGBoost features.
    """

    MODEL_TYPE = "xgboost_standard"
    DESCRIPTION = "Standard XGBoost regression model with native API"

    def __init__(self, **model_params):
        """
        Initialize XGBoost standard model.

        Requires explicit configuration - no defaults are provided.

        Args:
            **model_params: XGBoost hyperparameters. Must include either 'n_estimators'
                          or 'num_boost_round' to specify number of boosting rounds.

        Raises:
            ValueError: If required parameters are missing.
        """
        super().__init__(**model_params)

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

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the XGBoost model using native API.

        Args:
            X_train: Training features
            y_train: Training targets
        """
        # Create DMatrix for training data
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Use n_estimators stored during initialization as num_boost_round
        num_boost_round = self.n_estimators

        # Train model with native API
        self.model = xgb.train(
            params=self.model_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False
        )

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions

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
            Dictionary containing model type, parameters, and other metadata
        """
        info = {
            "model_type": self.MODEL_TYPE,
            "parameters": self.model_params,
            "is_trained": self.is_trained,
            "model_class": "XGBBooster",
            "training_method": "xgb_train",
            "num_boost_round": self.n_estimators
        }

        return info
