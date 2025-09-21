"""
Standard XGBoost regression model implementation.

This module implements the standard XGBoost regressor using the sklearn-style interface.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from .base import BaseModel, ModelTrainingError, ModelPredictionError


class XGBoostStandardModel(BaseModel):
    """
    Standard XGBoost regression model using sklearn-style interface.
    
    This implementation conforms to the BaseModel interface for consistent
    model handling throughout the framework.
    """
    
    MODEL_TYPE = "xgboost_standard"
    DESCRIPTION = "Standard XGBoost regression model"
    
    def __init__(self, **model_params):
        """
        Initialize XGBoost standard model.
        
        Args:
            **model_params: XGBoost hyperparameters
        """
        super().__init__(**model_params)
        self.model_type = "xgboost"
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        # Create XGBoost regressor with parameters
        self.model = xgb.XGBRegressor(**self.model_params)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
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
            
        return self.model.predict(X)
            
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
