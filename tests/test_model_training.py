"""
Unit tests for ModelTrainer component.
Tests core model training functionality and hyperparameter handling.
"""

import pytest
import polars as pl
import numpy as np
import xgboost as xgb

from src import ModelTrainer, ModelingStrategy, BenchmarkModel, TrainingConfig


class TestModelTrainer:
    """Test ModelTrainer component functionality."""

    def test_model_trainer_initialization(self, sample_training_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(sample_training_config)
        
        assert trainer.config == sample_training_config
        # TrainingConfig no longer has model_type directly - it's in model_configs
        assert isinstance(trainer.config.model_configs, dict)

    def test_train_model_combined_strategy(
        self, 
        sample_training_config, 
        prepared_model_data, 
        sample_sku_tuples
    ):
        """Test training a model with COMBINED strategy."""
        X, y, feature_cols = prepared_model_data
        
        # Split data for training
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        
        X_train = X.head(split_idx)
        y_train = y.head(split_idx) 
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        # Initialize trainer
        trainer = ModelTrainer(sample_training_config)
        
        # Train model
        model = trainer.train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            target_col="target",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=sample_sku_tuples,
            model_type="xgboost_standard"  # Added required parameter
        )
        
        # Validate returned model
        assert isinstance(model, BenchmarkModel)
        assert model.metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert model.metadata.sku_tuples == sample_sku_tuples
        assert model.metadata.model_type == "xgboost_standard"  # Updated expected value
        assert model.metadata.feature_columns == feature_cols
        assert model.metadata.target_column == "target"
        
        # Check model object - now wrapped in our extensible architecture
        assert model.model is not None
        from src.models import XGBoostStandardModel
        assert isinstance(model.model, XGBoostStandardModel)
        assert hasattr(model.model, 'predict')
        
        # Check data split information
        assert model.data_split.train_bdIDs is not None
        assert model.data_split.validation_bdIDs is not None
        assert len(model.data_split.train_bdIDs) > 0
        assert len(model.data_split.validation_bdIDs) > 0
        
        # Check performance metrics - now using centralized metrics
        metrics = model.metadata.performance_metrics
        expected_metrics = ["mse", "rmse", "mae", "r2", "mape", "max_error", "mean_error", "std_error",
                           "within_1_unit", "within_2_units", "within_5_units"]
        for metric in expected_metrics:
            assert metric in metrics
        
        # Metrics should be reasonable numbers
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert -1 <= metrics["r2"] <= 1  # R² can be negative for very poor models  # R² can be negative for very poor models

    def test_train_model_individual_strategy(
        self, 
        sample_training_config, 
        prepared_model_data, 
        single_sku_tuple
    ):
        """Test training a model with INDIVIDUAL strategy."""
        X, y, feature_cols = prepared_model_data
        
        # Use only data for the specific SKU (simulate filtering)
        # For testing purposes, we'll use all data but mark it as individual
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        # Initialize trainer
        trainer = ModelTrainer(sample_training_config)
        
        # Train model
        model = trainer.train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            target_col="target",
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            sku_tuples=single_sku_tuple,
            model_type="xgboost_standard"  # Add required model_type parameter
        )
        
        # Validate model
        assert isinstance(model, BenchmarkModel)
        assert model.metadata.modeling_strategy == ModelingStrategy.INDIVIDUAL
        assert model.metadata.sku_tuples == single_sku_tuple
        assert len(model.metadata.sku_tuples) == 1

    def test_hyperparameter_handling(self, prepared_model_data, single_sku_tuple):
        """Test that hyperparameters are properly handled."""
        X, y, feature_cols = prepared_model_data
        
        # Create custom training config with specific hyperparameters
        custom_config = TrainingConfig(random_state=42)
        custom_config.add_model_config(
            model_type="xgboost_standard",
            hyperparameters={
                "n_estimators": 20,
                "max_depth": 4,
                "learning_rate": 0.2,
                "subsample": 0.8
            }
        )
        
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        trainer = ModelTrainer(custom_config)
        
        model = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, single_sku_tuple,
            model_type="xgboost_standard"
        )
        
        # Check that hyperparameters were stored in metadata
        stored_params = model.metadata.hyperparameters
        assert stored_params["n_estimators"] == 20
        assert stored_params["max_depth"] == 4
        assert stored_params["learning_rate"] == 0.2
        assert stored_params["subsample"] == 0.8
        assert stored_params["random_state"] == custom_config.random_state

    def test_model_prediction(self, sample_training_config, prepared_model_data, single_sku_tuple):
        """Test that trained model can make predictions."""
        X, y, feature_cols = prepared_model_data
        
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        trainer = ModelTrainer(sample_training_config)
        
        model = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, single_sku_tuple,
            model_type="xgboost_standard"
        )
        
        # Test prediction on new data
        X_pred = X_test.select(feature_cols).to_numpy()
        predictions = model.model.predict(X_pred)
        
        # Validate predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert predictions.dtype in [np.float32, np.float64]
        
        # Predictions should be reasonable (non-negative for counts)
        assert all(pred >= 0 for pred in predictions)

    def test_different_model_types_error(self, sample_training_config, prepared_model_data, single_sku_tuple):
        """Test error handling for unsupported model types."""
        sample_training_config.model_type = "unsupported_model"
        
        X, y, feature_cols = prepared_model_data
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        trainer = ModelTrainer(sample_training_config)
        
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer.train_model(
                X_train, y_train, X_test, y_test,
                feature_cols, "target", ModelingStrategy.COMBINED, single_sku_tuple,
                model_type="unsupported_model"
            )

    def test_metrics_calculation(self, sample_training_config):
        """Test centralized metrics calculation functionality."""
        from src.metrics import MetricsCalculator
        
        # Create simple test data for metrics
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        # Check all expected metrics are present
        expected_metrics = ["mse", "rmse", "mae", "r2", "mape", "max_error", "mean_error", "std_error", 
                           "within_1_unit", "within_2_units", "within_5_units"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # Check that RMSE is sqrt of MSE
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10
        
        # Check MAPE calculation handles zero values
        y_true_with_zero = np.array([0, 1, 2, 3])
        y_pred_with_zero = np.array([0.1, 1.1, 1.9, 3.1])
        
        metrics_with_zero = MetricsCalculator.calculate_regression_metrics(y_true_with_zero, y_pred_with_zero)
        assert "mape" in metrics_with_zero
        
        # Test quantile metrics calculation
        quantile_metrics = MetricsCalculator.calculate_quantile_metrics(y_true, y_pred, 0.7)
        expected_quantile_metrics = ["quantile_score", "coverage_probability", "coverage_error", "quantile_alpha"]
        for metric in expected_quantile_metrics:
            assert metric in quantile_metrics
            assert isinstance(quantile_metrics[metric], (int, float))
        # MAPE should handle zero values appropriately

    def test_model_metadata_creation(self, sample_training_config, prepared_model_data, sample_sku_tuples):
        """Test that model metadata is created correctly."""
        X, y, feature_cols = prepared_model_data
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        trainer = ModelTrainer(sample_training_config)
        
        model = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, sample_sku_tuples,
            model_type="xgboost_standard"
        )
        
        metadata = model.metadata
        
        # Check all required metadata fields
        assert metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert metadata.sku_tuples == sample_sku_tuples
        assert metadata.model_type == "xgboost_standard"
        assert metadata.feature_columns == feature_cols
        assert metadata.target_column == "target"
        assert isinstance(metadata.hyperparameters, dict)
        assert isinstance(metadata.training_config, dict)
        assert isinstance(metadata.performance_metrics, dict)
        
        # Check model ID is generated
        assert metadata.model_id is not None
        assert len(metadata.model_id) > 0

    @pytest.mark.unit
    def test_train_model_with_params(self, sample_training_config):
        """Test internal _train_model_with_params method."""
        trainer = ModelTrainer(sample_training_config)
        
        # Create simple training data
        X_train = np.random.random((20, 5))
        y_train = np.random.random(20)
        
        hyperparameters = {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42
        }
        
        # Test XGBoost model creation
        model = trainer._train_model_with_params(
            X_train, y_train, hyperparameters, "xgboost_standard"
        )
        
        from src.models import XGBoostStandardModel
        assert isinstance(model, XGBoostStandardModel)
        assert hasattr(model, 'predict')
        
        # Test unsupported model type
        with pytest.raises(ValueError):
            trainer._train_model_with_params(
                X_train, y_train, hyperparameters, "unsupported"
            )

    def test_empty_data_handling(self, sample_training_config, sample_sku_tuples):
        """Test handling of empty training data."""
        trainer = ModelTrainer(sample_training_config)
        
        # Create minimal empty-like data that will still work
        X_empty = pl.DataFrame({"bdID": [1], "feature1": [0.0]})
        y_empty = pl.DataFrame({"bdID": [1], "target": [0]})
        
        # This should still work but might produce poor metrics
        try:
            model = trainer.train_model(
                X_empty, y_empty, X_empty, y_empty,
                ["feature1"], "target", ModelingStrategy.COMBINED, sample_sku_tuples,
                model_type="xgboost_standard"
            )
            # If it succeeds, check basic structure
            assert isinstance(model, BenchmarkModel)
        except Exception as e:
            # Some failure is acceptable with minimal data
            assert isinstance(e, (ValueError, IndexError))

    def test_training_reproducibility(self, sample_training_config, prepared_model_data, single_sku_tuple):
        """Test that training with same random seed produces consistent results."""
        X, y, feature_cols = prepared_model_data
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        # Train two models with same config
        trainer1 = ModelTrainer(sample_training_config)
        trainer2 = ModelTrainer(sample_training_config)
        
        model1 = trainer1.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, single_sku_tuple,
            model_type="xgboost_standard"
        )
        
        model2 = trainer2.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, single_sku_tuple,
            model_type="xgboost_standard"
        )
        
        # Results should be very similar due to random seed
        metrics1 = model1.metadata.performance_metrics
        metrics2 = model2.metadata.performance_metrics
        
        # Check that MSE is very close (should be identical with same random seed)
        assert abs(metrics1["mse"] - metrics2["mse"]) < 1e-6

    def test_lightning_model_training(self, prepared_model_data, sample_sku_tuples):
        """Test training Lightning neural network model."""
        X, y, feature_cols = prepared_model_data
        
        # Split data for training
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        # Configure Lightning-specific hyperparameters
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_standard",
            hyperparameters={
                "hidden_size": 64,  # Smaller for fast testing
                "lr": 1e-2,
                "dropout": 0.1,
                "max_epochs": 5,  # Very small for fast testing
                "batch_size": 32
            }
        )
        
        trainer = ModelTrainer(lightning_config)
        
        # Test Lightning model training
        model = trainer.train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            target_col="target",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=sample_sku_tuples[:2],
            model_type="lightning_standard"
        )
        
        # Test model object
        assert model is not None
        assert model.metadata.model_type == "lightning_standard"
        assert model.metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert len(model.metadata.sku_tuples) == 2
        
        # Test model hyperparameters
        expected_hyperparams = {
            "hidden_size": 64,
            "lr": 1e-2, 
            "dropout": 0.1,
            "max_epochs": 5,
            "batch_size": 32
        }
        for param, expected_value in expected_hyperparams.items():
            assert model.metadata.hyperparameters[param] == expected_value
        
        # Test performance metrics (Lightning should produce reasonable results)
        metrics = model.metadata.performance_metrics
        expected_metrics = ["mse", "rmse", "mae", "r2", "mape", "max_error", "mean_error", "std_error",
                           "within_1_unit", "within_2_units", "within_5_units"]
        for metric in expected_metrics:
            assert metric in metrics
        
        assert metrics['rmse'] > 0
        assert metrics['rmse'] < 1000  # Sanity check
        
        # Test model predictions capability
        from src.models import LightningStandardModel
        assert isinstance(model.model, LightningStandardModel)
        assert hasattr(model.model, 'predict')
        
        # Test actual predictions with feature columns
        X_test_features = X_test.select(feature_cols).to_numpy()
        predictions = model.model.predict(X_test_features)
        assert len(predictions) > 0
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_lightning_model_individual_strategy(self, prepared_model_data, sample_sku_tuples):
        """Test Lightning model with INDIVIDUAL strategy."""
        X, y, feature_cols = prepared_model_data
        
        # Split data for training
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_standard",
            hyperparameters={
                "hidden_size": 32,
                "lr": 1e-2,
                "max_epochs": 3,
                "batch_size": 16
            }
        )
        
        trainer = ModelTrainer(lightning_config)
        
        # Train individual Lightning model
        model = trainer.train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            target_col="target",
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            sku_tuples=[sample_sku_tuples[0]],  # Single SKU
            model_type="lightning_standard"
        )
        
        assert model.metadata.modeling_strategy == ModelingStrategy.INDIVIDUAL
        assert len(model.metadata.sku_tuples) == 1
        assert model.metadata.model_type == "lightning_standard"
        
        # Test Lightning model object
        from src.models import LightningStandardModel
        assert isinstance(model.model, LightningStandardModel)

    def test_lightning_vs_xgboost_consistency(self, prepared_model_data, sample_sku_tuples):
        """Test that Lightning and XGBoost models follow same interface patterns."""
        X, y, feature_cols = prepared_model_data
        
        # Split data for training
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        
        X_train = X.head(split_idx)
        y_train = y.head(split_idx)
        X_test = X.tail(n_samples - split_idx)
        y_test = y.tail(n_samples - split_idx)
        
        # XGBoost config
        xgb_config = TrainingConfig(random_state=42)
        xgb_config.add_model_config(
            model_type="xgboost_standard",
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )
        
        # Lightning config  
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_standard",
            hyperparameters={"hidden_size": 32, "max_epochs": 3}
        )
        
        # Train both models
        xgb_trainer = ModelTrainer(xgb_config)
        lightning_trainer = ModelTrainer(lightning_config)
        
        xgb_model = xgb_trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, sample_sku_tuples[:1],
            model_type="xgboost_standard"
        )
        
        lightning_model = lightning_trainer.train_model(
            X_train, y_train, X_test, y_test, 
            feature_cols, "target", ModelingStrategy.COMBINED, sample_sku_tuples[:1],
            model_type="lightning_standard"
        )
        
        # Test interface consistency
        xgb_metadata = xgb_model.metadata
        lightning_metadata = lightning_model.metadata
        
        # Both should have same required metadata fields
        required_fields = ['model_id', 'modeling_strategy', 'sku_tuples', 'model_type',
                          'hyperparameters', 'performance_metrics', 'feature_columns']
        
        for field in required_fields:
            assert hasattr(xgb_metadata, field)
            assert hasattr(lightning_metadata, field)
        
        # Both should have same metric keys (though values will differ)
        xgb_metric_keys = set(xgb_metadata.performance_metrics.keys())
        lightning_metric_keys = set(lightning_metadata.performance_metrics.keys()) 
        assert xgb_metric_keys == lightning_metric_keys
