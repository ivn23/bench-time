"""
PyTorch Lightning standard model implementation.

This module provides a PyTorch Lightning neural network model that integrates
with the M5 benchmarking framework following the same patterns as XGBoost models.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from .base import BaseModel, ModelTrainingError, ModelPredictionError
from ..structures import ComputeConfig

logger = logging.getLogger(__name__)


class ForecastingModel(L.LightningModule):
    """
    PyTorch Lightning module for time series forecasting.
    
    Implements a 3-layer MLP architecture with dropout regularization.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, lr: float = 0.001, dropout: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        
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
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LightningStandardModel(BaseModel):
    """
    Standard PyTorch Lightning neural network model for time series forecasting.
    
    This implementation integrates the Lightning prototype neural network
    into the M5 benchmarking framework following BaseModel interface patterns.
    """
    
    MODEL_TYPE = "lightning_standard"
    DESCRIPTION = "PyTorch Lightning neural network for time series forecasting"
    DEFAULT_HYPERPARAMETERS = {
        "hidden_size": 128,
        "lr": 1e-3,
        "dropout": 0.2,
        "max_epochs": 50,
        "batch_size": 64,
        "num_workers": 0,  # Conservative default for compatibility
        "random_state": 42
    }
    
    def __init__(self, **model_params):
        """
        Initialize Lightning standard model.
        
        Args:
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
        self.model_type = "lightning"
        self.lightning_model = None
        self.trainer = None
        self.input_size = None
        
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

        # Get batch size from model params, num_workers from compute config
        batch_size = self.model_params.get("batch_size", 64)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.compute_config.dataloader_workers,
            persistent_workers=False  # Safer default
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, compute_config: ComputeConfig, **training_kwargs) -> None:
        """
        Train the Lightning model using all provided training data.

        Args:
            X_train: Training features
            y_train: Training targets
            compute_config: Compute resource configuration
            **training_kwargs: Additional training parameters

        Note:
            This method uses all provided training data for training without internal splits.
            The framework handles proper train/test separation at a higher level.
        """
        # Store compute config
        self.compute_config = compute_config

        # Store input size for model architecture
        self.input_size = X_train.shape[1]
        
        # Create Lightning model
        self.lightning_model = ForecastingModel(
            input_size=self.input_size,
            hidden_size=self.model_params.get("hidden_size", 128),
            lr=self.model_params.get("lr", 1e-3),
            dropout=self.model_params.get("dropout", 0.2)
        )
        
        # Create data loader using all training data
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        
        # Configure trainer
        max_epochs = self.model_params.get("max_epochs", 50)
        
        # Create trainer with minimal configuration for compatibility
        trainer_kwargs = {
            "max_epochs": max_epochs,
            "log_every_n_steps": max(10, len(train_loader) // 5),  # Adaptive logging
            "enable_checkpointing": False,  # Disable checkpointing for simplicity
            "logger": False,  # Disable logging for cleaner output
            "enable_progress_bar": False,  # Disable progress bar for cleaner output
            "accelerator": compute_config.accelerator,  # Use configured accelerator
            "strategy": "auto",  # Single-process strategy
            "enable_model_summary": False  # Reduce output verbosity
        }
        
        # Add any additional trainer kwargs from training_kwargs
        trainer_kwargs.update(training_kwargs.get("trainer_kwargs", {}))
        
        self.trainer = L.Trainer(**trainer_kwargs)
        
        # Train the model using all provided training data
        self.trainer.fit(self.lightning_model, train_loader)
        
        # Set training flag
        self.is_trained = True
        self.model = self.lightning_model  # Store reference for consistency
        
        logger.info(f"Lightning model training completed after {max_epochs} epochs using all {len(X_train)} training samples")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Lightning model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
            
        Raises:
            ModelPredictionError: If model is not trained or prediction fails
        """
        if not self.is_trained or self.lightning_model is None:
            raise ModelPredictionError("Model must be trained before making predictions")
            

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
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Lightning model information and metadata.
        
        Returns:
            Dictionary containing model type, parameters, and other metadata
        """
        info = {
            "model_type": self.model_type,
            "parameters": self.model_params,
            "is_trained": self.is_trained,
            "model_class": "ForecastingModel",
            "training_method": "lightning_trainer"
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
        """Get the underlying Lightning model object (for serialization)."""
        return self.lightning_model

    def save_model(self, output_path: str, filename: str) -> None:
        """
        Save Lightning model using checkpoint mechanism.
        
        Args:
            output_path: Directory path where to save the model
            filename: Name of the file (without extension)
        """
        if not self.is_trained or self.lightning_model is None:
            raise ValueError("Model must be trained before saving")
        
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Lightning checkpoint
        checkpoint_path = output_dir / f"{filename}.ckpt"
        self.trainer.save_checkpoint(str(checkpoint_path))
        
        # Save additional model metadata
        import torch
        metadata_path = output_dir / f"{filename}_metadata.pt"
        torch.save({
            "input_size": self.input_size,
            "model_params": self.model_params,
            "is_trained": self.is_trained,
            "model_type": self.model_type
        }, metadata_path)
    
    def save_state(self, path: Path) -> None:
        """
        Save model state for persistence.
        
        Args:
            path: Directory path to save model state
        """
        if not self.is_trained or self.lightning_model is None:
            raise ValueError("Cannot save state of untrained model")
            

        # Save Lightning model checkpoint
        checkpoint_path = path / "lightning_model.ckpt"
        self.trainer.save_checkpoint(str(checkpoint_path))
        
        # Save additional model info
        model_info_path = path / "model_info.pt"
        torch.save({
            "input_size": self.input_size,
            "model_params": self.model_params,
            "is_trained": self.is_trained
        }, model_info_path)
        

            
    def load_state(self, path: Path) -> None:
        """
        Load model state from persistence.
        
        Args:
            path: Directory path containing saved model state
        """

        # Load model info
        model_info_path = path / "model_info.pt"
        if model_info_path.exists():
            model_info = torch.load(model_info_path)
            self.input_size = model_info["input_size"]
            self.model_params = model_info["model_params"]
            self.is_trained = model_info["is_trained"]
        
        # Load Lightning model
        checkpoint_path = path / "lightning_model.ckpt"
        if checkpoint_path.exists():
            self.lightning_model = ForecastingModel.load_from_checkpoint(
                str(checkpoint_path),
                input_size=self.input_size,
                hidden_size=self.model_params.get("hidden_size", 128),
                lr=self.model_params.get("lr", 1e-3),
                dropout=self.model_params.get("dropout", 0.2)
            )
            self.model = self.lightning_model

    @staticmethod
    def get_search_space(trial, random_state: int) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Lightning standard model.

        Args:
            trial: Optuna trial object for suggesting hyperparameters
            random_state: Random seed for reproducibility

        Returns:
            Dictionary of hyperparameters sampled from the search space
        """
        return {
            'hidden_size': trial.suggest_int('hidden_size', 64, 512, step=64),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'max_epochs': trial.suggest_int('max_epochs', 10, 30),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'random_state': random_state
        }
                