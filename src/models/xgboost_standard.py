"""
Standard XGBoost regression model implementation.

This module implements the standard XGBoost regressor using the sklearn-style interface,
maintaining backward compatibility with the existing framework implementation.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from .base import BaseModel, ModelTrainingError, ModelPredictionError


class XGBoostStandardModel(BaseModel):
    """
    Standard XGBoost regression model using sklearn-style interface.
    
    This implementation maintains backward compatibility with the existing
    framework while conforming to the BaseModel interface.
    """
    
    def __init__(self, **model_params):
        """
        Initialize XGBoost standard model.
        
        Args:
            **model_params: XGBoost hyperparameters
        """
        super().__init__(**model_params)
        self.model_type = "xgboost"
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **training_kwargs) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used in standard implementation)
            y_val: Validation targets (not used in standard implementation) 
            **training_kwargs: Additional parameters (ignored)
        """
        try:
            # Create XGBoost regressor with parameters
            self.model = xgb.XGBRegressor(**self.model_params)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            self.is_trained = True
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train XGBoost standard model: {str(e)}")
            
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
            
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model type, parameters, and other metadata
        """
        info = {
            "model_type": self.model_type,
            "parameters": self.model_params,
            "is_trained": self.is_trained,
            "model_class": "XGBRegressor",
            "training_method": "sklearn_fit"
        }
        
        if self.is_trained and self.model is not None:
            info.update({
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "learning_rate": self.model.learning_rate
            })
            
        return info
        
    def get_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard regression evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of standard regression metrics
        """
        try:
            metrics = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred))
            }
            
            # Calculate additional metrics
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
                
            # Calculate accuracy within error bands
            abs_errors = np.abs(residuals)
            metrics.update({
                "within_1_unit": float(np.mean(abs_errors <= 1.0)),
                "within_2_units": float(np.mean(abs_errors <= 2.0)), 
                "within_5_units": float(np.mean(abs_errors <= 5.0))
            })
            
            return metrics
            
        except Exception as e:
            raise ValueError(f"Failed to calculate evaluation metrics: {str(e)}")