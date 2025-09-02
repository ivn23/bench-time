"""
StatQuant Model Implementation using statsmodels QuantReg.

This module implements a quantile regression model using statsmodels QuantReg
following the framework's BaseModel interface.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
from ..models.base import BaseModel, ModelTrainingError, ModelPredictionError

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
    DEFAULT_HYPERPARAMETERS = {
        "method": "interior-point",
        "max_iter": 1000,
        "p_tol": 1e-6,
        "random_state": 42
    }
    REQUIRES_QUANTILE = True
    
    def __init__(self, quantile_alpha: float = 0.7, **model_params):
        """
        Initialize StatQuant model.
        
        Args:
            quantile_alpha: Target quantile level (e.g., 0.7 for 70% quantile)
            **model_params: Additional parameters for QuantReg
        """
        super().__init__(**model_params)
        
        # Validate quantile_alpha
        if not (0 < quantile_alpha < 1):
            raise ValueError(f"quantile_alpha must be between 0 and 1, got {quantile_alpha}")
        
        self.quantile_alpha = quantile_alpha
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
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **training_kwargs) -> None:
        """
        Train the StatQuant model using statsmodels QuantReg.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, not used by statsmodels)
            y_val: Validation targets (optional, not used by statsmodels)
            **training_kwargs: Additional training parameters
        """
        try:
            # Flatten y_train if needed
            if len(y_train.shape) > 1:
                y_train = y_train.ravel()
            
            # Create QuantReg model - note that QuantReg expects (endog, exog)
            self.model = QuantReg(endog=y_train, exog=X_train)
            
            # Fit the model with specified quantile
            fit_params = {
                'q': self.quantile_alpha,
                **self.model_params,
                **training_kwargs
            }
            
            # Suppress iteration warnings if requested
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, 
                                      message="Maximum number of iterations")
                
                self.fitted_model = self.model.fit(**fit_params)
            
            self.is_trained = True
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train StatQuant model: {str(e)}")
    
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
        
        try:
            return self.fitted_model.predict(X)
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