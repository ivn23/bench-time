"""
Independent hyperparameter tuning module for the M5 Time Series Benchmarking Framework.

This module provides mode-based hyperparameter optimization completely separate from
the training workflow. Uses Optuna with cross-validation for robust parameter search.
"""

import time
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import optuna
from sklearn.model_selection import KFold


from .structures import ModelingDataset
from .model_types import model_registry

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Results from hyperparameter tuning optimization.

    Attributes:
        best_params: Dictionary of optimized hyperparameters
        best_score: Mean validation loss across CV folds
        n_trials: Number of Optuna trials executed
        n_folds: Number of cross-validation folds used
        model_type: Type of model that was tuned
        quantile_alpha: Quantile level for quantile models (None for standard models)
        n_skus_sampled: Number of SKUs sampled for tuning dataset
        optimization_time: Total time spent on optimization in seconds
    """
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    n_folds: int
    model_type: str
    quantile_alpha: Optional[float]
    n_skus_sampled: int
    optimization_time: float

    def get_summary(self) -> str:
        """Generate a formatted summary of tuning results."""
        summary = [
            "=" * 60,
            "HYPERPARAMETER TUNING RESULTS",
            "=" * 60,
            f"Model Type: {self.model_type}",
            f"Quantile Alpha: {self.quantile_alpha if self.quantile_alpha else 'N/A (standard model)'}",
            f"SKUs Sampled: {self.n_skus_sampled}",
            f"CV Folds: {self.n_folds}",
            f"Optuna Trials: {self.n_trials}",
            f"Best Validation Loss: {self.best_score:.6f}",
            f"Optimization Time: {self.optimization_time:.2f}s",
            "",
            "Best Hyperparameters:",
        ]
        for param, value in self.best_params.items():
            if isinstance(value, float):
                summary.append(f"  {param}: {value:.6f}")
            else:
                summary.append(f"  {param}: {value}")
        summary.append("=" * 60)
        return "\n".join(summary)


class HyperparameterTuner:
    """Independent hyperparameter optimization using Optuna and cross-validation.

    This class is completely decoupled from the training workflow and operates on
    ModelingDataset objects using only training data for CV-based optimization.
    """

    def __init__(self, random_state: int, n_jobs: int = 1,
                 dataloader_workers: int = 4, accelerator: str = 'cpu', devices: int = 1):
        """Initialize the tuner.

        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel Optuna trials (1=sequential, >1=parallel, -1=all cores)
            dataloader_workers: Number of DataLoader worker processes for parallel data loading
            accelerator: Device type for training ('cpu', 'cuda', 'mps')
            devices: Number of GPU devices to use for training (typically 1 for hyperparameter tuning)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.dataloader_workers = dataloader_workers
        self.accelerator = accelerator
        self.devices = devices
        logger.info(f"HyperparameterTuner initialized with random_state={random_state}")
        logger.info(f"Optuna parallelism: {self.n_jobs} jobs ({'sequential' if n_jobs == 1 else 'parallel'})")
        logger.info(f"Resource configuration: accelerator={accelerator}, devices={devices}, dataloader_workers={dataloader_workers}")

    def tune(self,
             dataset: ModelingDataset,
             model_type: str,
             quantile_alpha: Optional[float] = None,
             n_trials: int = 50,
             n_folds: int = 3) -> TuningResult:
        """Main tuning method using cross-validation and Optuna.

        Args:
            dataset: ModelingDataset containing training data
            model_type: Type of model to tune (e.g., 'xgboost_quantile')
            quantile_alpha: Quantile level for quantile models (None for standard)
            n_trials: Number of Optuna optimization trials
            n_folds: Number of cross-validation folds

        Returns:
            TuningResult with optimized hyperparameters and validation metrics
        """
        start_time = time.time()
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        logger.info(f"Configuration: {n_trials} trials, {n_folds} CV folds")

        # Prepare training data
        X_train, y_train = self._prepare_training_data(dataset)
        logger.info(f"Training data prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features")

        # Create CV splits
        cv_splits = self._create_cv_splits(X_train, y_train, n_folds)
        logger.info(f"Created {n_folds} cross-validation splits")

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        logger.info("Optuna study created with TPESampler")

        # Create objective function
        objective = self._create_objective(
            X_train, y_train, model_type, quantile_alpha, cv_splits
        )

        # Run optimization
        logger.info(f"Starting optimization: {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs, show_progress_bar=True)

        optimization_time = time.time() - start_time

        # Create result
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=n_trials,
            n_folds=n_folds,
            model_type=model_type,
            quantile_alpha=quantile_alpha,
            n_skus_sampled=len(dataset.sku_tuples),
            optimization_time=optimization_time
        )

        logger.info(f"Optimization complete! Best validation loss: {result.best_score:.6f}")
        logger.info(f"Best hyperparameters: {result.best_params}")

        return result

    def _prepare_training_data(self, dataset: ModelingDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract training features and target from ModelingDataset.

        Args:
            dataset: ModelingDataset with X_train and y_train

        Returns:
            Tuple of (X_train, y_train) as numpy arrays
        """
        # Convert Polars DataFrames to numpy
        X_train = dataset.X_train.to_numpy()
        y_train = dataset.y_train.to_numpy().flatten()

        print(f"Prepared training data with {X_train.shape[0]} samples and {X_train.shape[1]} features.")


        return X_train, y_train

    def _create_cv_splits(self, X_train: np.ndarray, y_train: np.ndarray,
                          n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create K-Fold cross-validation splits.

        Args:
            X_train: Training features
            y_train: Training targets
            n_folds: Number of CV folds

        Returns:
            List of (train_idx, val_idx) tuples for each fold
        """
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        cv_splits = list(kfold.split(X_train))

        logger.debug(f"CV splits created: {len(cv_splits)} folds")
        for i, (train_idx, val_idx) in enumerate(cv_splits):
            logger.debug(f"  Fold {i+1}: {len(train_idx)} train, {len(val_idx)} val samples")

        return cv_splits

    def _get_search_space(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Get hyperparameter search space from the model class.

        This method delegates to the model class's get_search_space() method,
        allowing each model to define its own search space.

        Args:
            trial: Optuna trial object
            model_type: Type of model (e.g., 'xgboost_quantile', 'lightning_standard')

        Returns:
            Dictionary of hyperparameters for the trial

        Raises:
            NotImplementedError: If the model class does not support hyperparameter tuning
        """
        try:
            # Get model class from registry
            model_class = model_registry.get_model_class(model_type)

            # Delegate to model class's search space method
            params = model_class.get_search_space(trial, self.random_state)

            return params

        except NotImplementedError as e:
            raise NotImplementedError(
                f"Model type '{model_type}' does not support hyperparameter tuning. "
                f"Implement get_search_space() method in the model class to enable tuning. "
                f"Original error: {str(e)}"
            )

    def _create_objective(self, X_train: np.ndarray, y_train: np.ndarray,
                          model_type: str, quantile_alpha: Optional[float],
                          cv_splits: List[Tuple[np.ndarray, np.ndarray]]):
        """Create Optuna objective function for optimization.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to optimize
            quantile_alpha: Quantile level for quantile models
            cv_splits: List of CV fold indices

        Returns:
            Objective function for Optuna to minimize
        """
        def objective(trial):
            try:
                # Get hyperparameters for this trial
                params = self._get_search_space(trial, model_type)

                # Add resource configuration parameters only for Lightning models
                # (XGBoost and other models don't use these PyTorch-specific parameters)
                if 'lightning' in model_type.lower():
                    params['num_workers'] = self.dataloader_workers
                    params['accelerator'] = self.accelerator
                    params['devices'] = self.devices

                logger.info(f"Trial {trial.number} started with params: {params}")

                # Perform cross-validation
                fold_scores = []

                for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                    logger.info(f"Trial {trial.number}, Fold {fold_idx + 1}/{len(cv_splits)} started")

                    try:
                        # Split data for this fold
                        X_fold_train = X_train[train_idx]
                        y_fold_train = y_train[train_idx]
                        X_fold_val = X_train[val_idx]
                        y_fold_val = y_train[val_idx]

                        # Get model class from registry
                        model_class = model_registry.get_model_class(model_type)

                        # Instantiate model with trial parameters
                        if model_registry.requires_quantile(model_type):
                            if quantile_alpha is None:
                                raise ValueError(
                                    f"quantile_alpha must be provided for model_type='{model_type}'"
                                )
                            model_instance = model_class(
                                quantile_alphas=[quantile_alpha],
                                **params
                            )
                        else:
                            model_instance = model_class(**params)

                        # Train model on fold
                        model_instance.train(X_fold_train, y_fold_train)

                        # Predict on validation fold
                        y_pred = model_instance.predict(X_fold_val)

                        # Calculate loss
                        loss = self._calculate_loss(y_fold_val, y_pred, quantile_alpha)
                        fold_scores.append(loss)

                        logger.info(f"Trial {trial.number}, Fold {fold_idx + 1}: loss={loss:.6f}")

                    except Exception as fold_error:
                        logger.error(f"Trial {trial.number}, Fold {fold_idx + 1} failed: {str(fold_error)}")
                        raise  # Re-raise to fail the trial

                # Return mean validation loss across folds
                mean_loss = np.mean(fold_scores)
                logger.info(f"Trial {trial.number} completed: mean_loss={mean_loss:.6f}")

                return mean_loss

            except Exception as trial_error:
                logger.error(f"Trial {trial.number} failed completely: {str(trial_error)}")
                raise  # Re-raise to let Optuna handle the failed trial

        return objective

    def _calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray,
                        quantile_alpha: Optional[float]) -> float:
        """Calculate validation loss (pinball loss for quantile, MSE for standard).

        Args:
            y_true: True target values
            y_pred: Predicted values
            quantile_alpha: Quantile level (None for standard regression)

        Returns:
            Loss value (lower is better)
        """
        if quantile_alpha is not None:
            # Quantile loss (pinball loss)
            residual = y_true - y_pred
            loss = np.where(
                residual >= 0,
                quantile_alpha * residual,
                (quantile_alpha - 1) * residual
            )
            return float(np.mean(loss))
        else:
            # MSE for standard regression
            return float(np.mean((y_true - y_pred) ** 2))
