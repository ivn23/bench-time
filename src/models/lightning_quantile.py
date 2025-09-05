"""
PyTorch Lightning quantile regression model implementation.

This module provides a PyTorch Lightning neural network model that performs
quantile regression using pinball loss, integrating with the M5 benchmarking
framework following the same patterns as other quantile models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from .base import BaseModel, ModelTrainingError, ModelPredictionError

logger = logging.getLogger(__name__)


class QuantileForecastingModel(L.LightningModule):
    """
    PyTorch Lightning module for quantile regression forecasting.
    
    Implements a 3-layer MLP architecture with dropout regularization
    and pinball loss for quantile regression.
    """
    
    def __init__(self, input_size: int, quantile_alpha: float = 0.7, 
                 hidden_size: int = 128, lr: float = 1e-3, dropout: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        self.quantile_alpha = quantile_alpha
        
        # Validate quantile_alpha
        if not 0 < quantile_alpha < 1:
            raise ValueError("quantile_alpha must be between 0 and 1")
        
        # layer so basteln, dass ich ddie size als HP machen kann
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        return self.model(x)
    
    def pinball_loss(self, y_pred, y_true):
        """
        Compute pinball (quantile) loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Pinball loss tensor
        """
        residual = y_true - y_pred
        loss = torch.maximum(
            self.quantile_alpha * residual,
            (self.quantile_alpha - 1) * residual
        )
        return torch.mean(loss)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.pinball_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.pinball_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LightningQuantileModel(BaseModel):
    """
    PyTorch Lightning quantile regression model for time series forecasting.
    
    This implementation integrates quantile regression using pinball loss
    into the M5 benchmarking framework following BaseModel interface patterns.
    """
    
    MODEL_TYPE = "lightning_quantile"
    DESCRIPTION = "PyTorch Lightning neural network with quantile regression"
    DEFAULT_HYPERPARAMETERS = {
        "hidden_size": 128,
        "lr": 1e-3,
        "dropout": 0.2,
        "max_epochs": 50,
        "batch_size": 64,
        "num_workers": 0,  # Conservative default for compatibility
        "random_state": 42
    }
    REQUIRES_QUANTILE = True
    
    def __init__(self, quantile_alpha: float = 0.7, **model_params):
        """
        Initialize Lightning quantile model.
        
        Args:
            quantile_alpha: Target quantile level (e.g., 0.7 for 70% quantile)
            **model_params: Lightning hyperparameters including:
                - hidden_size: Number of hidden units in first layer (default: 128)
                - lr: Learning rate (default: 1e-3)
                - dropout: Dropout probability (default: 0.2)
                - max_epochs: Maximum training epochs (default: 50)
                - batch_size: Training batch size (default: 64)
                - num_workers: DataLoader workers (default: 0)
                - random_state: Random seed (default: 42)
        """
        super().__init__(**model_params)
        self.quantile_alpha = quantile_alpha
        self.model_type = "lightning_quantile"
        self.lightning_model = None
        self.trainer = None
        self.input_size = None
        
        # Validate quantile_alpha
        if not 0 < quantile_alpha < 1:
            raise ValueError("quantile_alpha must be between 0 and 1")
        
        # Set random seeds for reproducibility
        if "random_state" in self.model_params:
            torch.manual_seed(self.model_params["random_state"])
            np.random.seed(self.model_params["random_state"])
        
    def _convert_to_tensors(self, X: np.ndarray, y: np.ndarray = None) -> tuple:
        """
        Convert numpy arrays to PyTorch tensors.
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            Tuple of converted tensors
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            if y_tensor.dim() > 1:
                y_tensor = y_tensor.squeeze()
            return X_tensor, y_tensor
        return X_tensor
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
        """
        Create PyTorch DataLoader from numpy arrays.
        
        Args:
            X: Features
            y: Targets
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader instance
        """
        X_tensor, y_tensor = self._convert_to_tensors(X, y)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Get batch size and num_workers from model params
        batch_size = self.model_params.get("batch_size", 64)
        num_workers = self.model_params.get("num_workers", 0)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=False  # Safer default
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **training_kwargs) -> None:
        """
        Train the Lightning quantile model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **training_kwargs: Additional training parameters
        """
        try:
            # Store input size for model architecture
            self.input_size = X_train.shape[1]
            
            # Create Lightning quantile model
            self.lightning_model = QuantileForecastingModel(
                input_size=self.input_size,
                quantile_alpha=self.quantile_alpha,
                hidden_size=self.model_params.get("hidden_size", 128),
                lr=self.model_params.get("lr", 1e-3),
                dropout=self.model_params.get("dropout", 0.2)
            )
            
            # Create data loaders
            train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
            val_loader = None
            if X_val is not None and y_val is not None:
                val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
            
            # Configure trainer
            max_epochs = self.model_params.get("max_epochs", 50)
            
            # Create trainer with minimal configuration for compatibility
            trainer_kwargs = {
                "max_epochs": max_epochs,
                "log_every_n_steps": max(10, len(train_loader) // 5),  # Adaptive logging
                "enable_checkpointing": False,  # Disable checkpointing for simplicity
                "logger": False,  # Disable logging for cleaner output
                "enable_progress_bar": False  # Disable progress bar for cleaner output
            }
            
            # Add any additional trainer kwargs from training_kwargs
            trainer_kwargs.update(training_kwargs.get("trainer_kwargs", {}))
            
            self.trainer = L.Trainer(**trainer_kwargs)
            
            # Train the model
            self.trainer.fit(self.lightning_model, train_loader, val_loader)
            
            # Set training flag
            self.is_trained = True
            self.model = self.lightning_model  # Store reference for consistency
            
            logger.info(f"Lightning quantile model training completed after {max_epochs} epochs")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train Lightning quantile model: {str(e)}")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make quantile predictions using the trained Lightning model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of quantile predictions
            
        Raises:
            ModelPredictionError: If model is not trained or prediction fails
        """
        if not self.is_trained or self.lightning_model is None:
            raise ModelPredictionError("Model must be trained before making predictions")
            
        try:
            # Convert to tensor
            X_tensor = self._convert_to_tensors(X)
            
            # Set model to evaluation mode and make predictions
            self.lightning_model.eval()
            with torch.no_grad():
                predictions = self.lightning_model(X_tensor).squeeze()
                
                # Convert back to numpy
                if predictions.dim() == 0:
                    # Single prediction
                    predictions = predictions.item()
                    return np.array([predictions])
                else:
                    return predictions.numpy()
                    
        except Exception as e:
            raise ModelPredictionError(f"Failed to make predictions: {str(e)}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Lightning quantile model information and metadata.
        
        Returns:
            Dictionary containing model type, parameters, and quantile-specific info
        """
        info = {
            "model_type": self.model_type,
            "parameters": self.model_params,
            "quantile_alpha": self.quantile_alpha,
            "is_trained": self.is_trained,
            "model_class": "QuantileForecastingModel",
            "training_method": "lightning_trainer_quantile"
        }
        
        if self.is_trained and self.lightning_model is not None:
            # Add model architecture information
            info.update({
                "input_size": self.input_size,
                "hidden_size": self.model_params.get("hidden_size", 128),
                "learning_rate": self.model_params.get("lr", 1e-3),
                "dropout": self.model_params.get("dropout", 0.2),
                "max_epochs": self.model_params.get("max_epochs", 50),
                "total_parameters": sum(p.numel() for p in self.lightning_model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.lightning_model.parameters() if p.requires_grad)
            })
            
        return info
    
    def get_underlying_model(self) -> Any:
        """Get the underlying Lightning quantile model object (for serialization)."""
        return self.lightning_model
    
    def save_state(self, path: Path) -> None:
        """
        Save model state for persistence.
        
        Args:
            path: Directory path to save model state
        """
        if not self.is_trained or self.lightning_model is None:
            raise ValueError("Cannot save state of untrained model")
            
        try:
            # Save Lightning model checkpoint
            checkpoint_path = path / "lightning_quantile_model.ckpt"
            self.trainer.save_checkpoint(str(checkpoint_path))
            
            # Save additional model info
            model_info_path = path / "model_info.pt"
            torch.save({
                "input_size": self.input_size,
                "quantile_alpha": self.quantile_alpha,
                "model_params": self.model_params,
                "is_trained": self.is_trained
            }, model_info_path)
            
        except Exception as e:
            logger.error(f"Failed to save Lightning quantile model state: {e}")
            raise
            
    def load_state(self, path: Path) -> None:
        """
        Load model state from persistence.
        
        Args:
            path: Directory path containing saved model state
        """
        try:
            # Load model info
            model_info_path = path / "model_info.pt"
            if model_info_path.exists():
                model_info = torch.load(model_info_path)
                self.input_size = model_info["input_size"]
                self.quantile_alpha = model_info["quantile_alpha"]
                self.model_params = model_info["model_params"]
                self.is_trained = model_info["is_trained"]
            
            # Load Lightning model
            checkpoint_path = path / "lightning_quantile_model.ckpt"
            if checkpoint_path.exists():
                self.lightning_model = QuantileForecastingModel.load_from_checkpoint(
                    str(checkpoint_path),
                    input_size=self.input_size,
                    quantile_alpha=self.quantile_alpha,
                    hidden_size=self.model_params.get("hidden_size", 128),
                    lr=self.model_params.get("lr", 1e-3),
                    dropout=self.model_params.get("dropout", 0.2)
                )
                self.model = self.lightning_model
                
        except Exception as e:
            logger.error(f"Failed to load Lightning quantile model state: {e}")
            raise