"""
StatQuant Model Implementation using statsmodels QuantReg.

This module implements a quantile regression model using statsmodels QuantReg
following the framework's BaseModel interface.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging
from ..models.base import BaseModel, ModelTrainingError, ModelPredictionError

logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
except ImportError:
    raise ImportError("statsmodels is required for StatQuantModel. Install with: pip install statsmodels")


class StatQuantModel(BaseModel):
    """
    Quantile regression model using statsmodels QuantReg.
    
    This implementation uses statsmodels' QuantReg class for quantile regression
    following the patterns established by XGBoost and Lightning quantile models.
    """
    
    MODEL_TYPE = "statquant"
    DESCRIPTION = "Statsmodels Quantile Regression for probabilistic forecasting"
    REQUIRES_QUANTILE = True
    
    def __init__(self, quantile_alphas: List[float] = None, **model_params):
        """
        Initialize StatQuant model.
        
        Args:
            quantile_alphas: List of target quantile levels (e.g., [0.7] for 70% quantile)
            **model_params: Additional parameters for QuantReg
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
                f"StatQuant model currently supports exactly one quantile level. "
                f"Got {len(quantile_alphas) if quantile_alphas else 0} values."
            )

        self.quantile_alpha = quantile_alphas[0]

        # Validate quantile_alpha is in valid range
        if not 0 < self.quantile_alpha < 1:
            raise ValueError(
                f"quantile_alpha must be between 0 and 1 (exclusive), got {self.quantile_alpha}"
            )
        self.model_type = "statquant"
        self.fitted_model = None  # Store the fitted QuantReg results
        
        # Set default parameters if not provided
        default_params = {
            'method': 'interior-point',
            'max_iter': 1000,
            'p_tol': 1e-6
        }
        
        # Update with provided parameters
        for key, value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = value
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **training_kwargs) -> None:
        """
        Train the StatQuant model using statsmodels QuantReg.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **training_kwargs: Additional training parameters
        """

        # Flatten y_train if needed
        if len(y_train.shape) > 1:
            y_train = y_train.ravel()
        
        # Data cleaning for statsmodels compatibility
        # Check for and handle inf/nan values
        X_clean = X_train.copy()
        y_clean = y_train.copy()
        
        # Replace inf values with finite values
        X_clean = np.where(np.isinf(X_clean), np.nan, X_clean)
        
        # Handle NaN/Inf values by removing rows with any missing data
        # This is necessary because statsmodels QuantReg is strict about data quality
        if np.isnan(X_clean).any() or np.isnan(y_clean).any():
            # Find rows with any NaN values
            valid_mask = ~(np.isnan(X_clean).any(axis=1) | np.isnan(y_clean))
            X_clean = X_clean[valid_mask]
            y_clean = y_clean[valid_mask]
            
            logger.info(f"Removed {np.sum(~valid_mask)} rows with missing values for StatQuant training")
        
        # Ensure we still have enough data after cleaning
        if len(X_clean) < 10:  # Minimum sample size check
            raise ValueError("Insufficient clean data for StatQuant training (< 10 samples after cleaning)")
        
        # Create QuantReg model - note that QuantReg expects (endog, exog)
        self.model = QuantReg(endog=y_clean, exog=X_clean)
        
        # Fit the model with specified quantile
        fit_params = {
            'q': self.quantile_alpha,
            **self.model_params,
            **training_kwargs
        }

        self.fitted_model = self.model.fit(**fit_params)
        
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
        if not self.is_trained or self.fitted_model is None:
            raise ModelPredictionError("Model must be trained before making predictions")
        
        return self.fitted_model.predict(X)
    
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
            "model_class": "StatQuantModel",
            "training_method": "statsmodels_quantreg",
            "description": self.DESCRIPTION
        }
        
        # Add fitted model information if available
        if self.is_trained and self.fitted_model is not None:
            info.update({
                "converged": getattr(self.fitted_model, 'converged', None),
                "n_iterations": getattr(self.fitted_model, 'n_iterations', None),
                "method": getattr(self.fitted_model, 'method', None),
                "quantile_level": self.quantile_alpha
            })
        
        return info