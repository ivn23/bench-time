"""
Empirical Quantile Baseline Model Implementation.

This module provides a naive quantile baseline that computes a constant quantile
value from training data and returns this single value for all predictions,
completely ignoring input features. This serves as the simplest possible baseline
for quantile regression, demonstrating the performance floor when no feature
learning occurs.

The model answers: "Does learning from features provide value over simply
predicting the historical quantile?"
"""

import numpy as np
from typing import Dict, Any, List
import logging
from ..models.base import BaseModel, ModelPredictionError

logger = logging.getLogger(__name__)


class EmpiricalQuantileModel(BaseModel):
    """
    Naive empirical quantile baseline model.

    This model computes the empirical α-quantile from training targets and
    returns this constant value for all predictions, completely ignoring
    input features. It represents the simplest possible quantile model and
    provides a baseline for evaluating whether feature-based models add value.

    Training:
        q_α = np.quantile(y_train, α)

    Prediction:
        ŷ = q_α for all inputs (feature-agnostic)

    This is the performance floor for quantile models. Any reasonable feature-based
    model (XGBoost, Lightning, StatQuant) should outperform this baseline.
    """

    MODEL_TYPE = "empirical_quantile"
    DESCRIPTION = "Naive empirical quantile baseline (constant prediction)"
    REQUIRES_QUANTILE = True
    DEFAULT_HYPERPARAMETERS = {}  # No hyperparameters needed

    def __init__(self, quantile_alphas: List[float] = None, **model_params):
        """
        Initialize Empirical Quantile model.

        Args:
            quantile_alphas: List of target quantile levels (e.g., [0.7] for 70th percentile)
                           Must contain exactly one value.
            **model_params: Additional parameters (unused, for interface compatibility)
        """
        super().__init__(**model_params)

        if quantile_alphas is None:
            quantile_alphas = [0.5]

        if not quantile_alphas or len(quantile_alphas) != 1:
            raise ValueError("Empirical Quantile model currently supports exactly one quantile level")

        # Validate quantile_alpha
        if not (0 < quantile_alphas[0] < 1):
            raise ValueError(f"quantile_alphas values must be between 0 and 1, got {quantile_alphas[0]}")

        self.quantile_alpha = quantile_alphas[0]
        self.model_type = "empirical_quantile"
        self.quantile_value = None  # Computed during training

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **training_kwargs) -> None:
        """
        Train the Empirical Quantile model by computing the empirical quantile.

        This "training" simply computes the α-quantile from the training targets
        and stores it. Input features X_train are completely ignored.

        Args:
            X_train: Training features (ignored, but required for interface)
            y_train: Training targets
            **training_kwargs: Additional training parameters (unused)
        """
        # Flatten y_train if needed
        if len(y_train.shape) > 1:
            y_train = y_train.ravel()

        # Validate we have data
        if len(y_train) < 1:
            raise ValueError("y_train must contain at least one sample")

        # Compute empirical quantile - this is the entire "training" process
        self.quantile_value = float(np.quantile(y_train, self.quantile_alpha))

        # Store as the "model"
        self.model = self.quantile_value

        # Mark as trained
        self.is_trained = True

        logger.info(
            f"Empirical Quantile model (α={self.quantile_alpha}) training completed: "
            f"quantile value = {self.quantile_value:.6f} from {len(y_train)} samples"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make constant predictions using the trained empirical quantile value.

        Returns the same quantile value for all inputs, completely ignoring
        the feature values in X.

        Args:
            X: Features for prediction (ignored except for determining output shape)

        Returns:
            Array of constant predictions (all equal to quantile_value)

        Raises:
            ModelPredictionError: If model is not trained
        """
        if not self.is_trained or self.quantile_value is None:
            raise ModelPredictionError("Model must be trained before making predictions")

        # Determine number of predictions needed from input shape
        n_samples = X.shape[0] if len(X.shape) > 1 else len(X)

        # Return constant quantile value for all samples
        return np.full(n_samples, self.quantile_value, dtype=np.float64)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.

        Returns:
            Dictionary containing model type, parameters, and quantile-specific info
        """
        info = {
            "model_type": self.model_type,
            "parameters": {},  # No hyperparameters
            "quantile_alpha": self.quantile_alpha,
            "is_trained": self.is_trained,
            "model_class": "EmpiricalQuantileModel",
            "training_method": "empirical_quantile_numpy",
            "description": self.DESCRIPTION
        }

        # Add computed quantile value if trained
        if self.is_trained and self.quantile_value is not None:
            info.update({
                "quantile_value": self.quantile_value,
                "prediction_mode": "constant (feature-agnostic)"
            })

        return info

    def get_underlying_model(self) -> Any:
        """
        Get the underlying model object (the quantile value).

        Returns:
            The computed quantile value as a float
        """
        return self.quantile_value
