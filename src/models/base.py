"""
Abstract base classes for model implementations in the M5 benchmarking framework.

This module defines the standard interface that all models must implement
to work with the framework's training, evaluation, and persistence systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    
    This class defines the standard interface that all models must implement
    to be compatible with the M5 benchmarking framework.
    """
    
    def __init__(self, **model_params):
        """
        Initialize the model with parameters.
        
        Args:
            **model_params: Model-specific parameters
        """
        self.model_params = model_params
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **training_kwargs) -> None:
        """
        Train the model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **training_kwargs: Additional training parameters
        
        Note:
            Models that require validation data should create internal train/validation
            splits from the provided training data to prevent data leakage.
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model is not trained
        """
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model type, parameters, and other metadata
        """
        pass

    def save_model(self, output_path: str, filename: str) -> None:
        """
        Save the trained model to the specified path.
        
        Args:
            output_path: Directory path where to save the model
            filename: Name of the file (without extension, .pkl will be added)
        
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        import pickle
        from pathlib import Path
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"{filename}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
    def is_model_trained(self) -> bool:
        """Check if model has been trained."""
        return self.is_trained
        
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model_params.copy()
        
    def set_model_params(self, **params) -> None:
        """Update model parameters."""
        self.model_params.update(params)
        
    def get_underlying_model(self) -> Any:
        """Get the underlying model object (for serialization)."""
        return self.model

        
    def get_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for model predictions.

        Args:
            y_true: True target values
            y_pred: Model predictions

        Returns:
            Dictionary containing evaluation metrics
        """
        from ..metrics import MetricsCalculator

        # Check if this is a quantile model
        quantile_alpha = getattr(self, 'quantile_alpha', None)

        return MetricsCalculator.calculate_all_metrics(
            y_true, y_pred, quantile_alpha=quantile_alpha
        )

    @staticmethod
    def get_search_space(trial, random_state: int) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for this model type.

        This method should be implemented by model classes that support hyperparameter
        tuning. It receives an Optuna trial object and should return a dictionary of
        hyperparameters using trial.suggest_* methods.

        Args:
            trial: Optuna trial object for suggesting hyperparameters
            random_state: Random seed for reproducibility

        Returns:
            Dictionary of hyperparameters sampled from the search space

        Raises:
            NotImplementedError: If the model does not support hyperparameter tuning

        Example:
            @staticmethod
            def get_search_space(trial, random_state: int) -> Dict[str, Any]:
                return {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'seed': random_state
                }
        """
        raise NotImplementedError(
            "Model does not support hyperparameter tuning. "
            "Implement get_search_space() method in the model class to enable tuning."
        )


class ModelTrainingError(Exception):
    """Exception raised when model training fails."""
    pass


class ModelPredictionError(Exception):
    """Exception raised when model prediction fails.""" 
    pass