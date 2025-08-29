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
    
    MODEL_TYPE = "xgboost_standard"
    DESCRIPTION = "Standard XGBoost regression model"
    DEFAULT_HYPERPARAMETERS = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42
    }
    
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
