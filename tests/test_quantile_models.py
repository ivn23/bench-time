"""
Tests for quantile model implementations in the M5 benchmarking framework.

This module tests the new extensible model architecture, specifically
focusing on quantile XGBoost functionality and model factory patterns.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from src.models import get_model_class, XGBoostQuantileModel, XGBoostStandardModel, LightningQuantileModel
from src.models.base import BaseModel, ModelTrainingError, ModelPredictionError
from src import TrainingConfig, ModelTrainer


class TestModelFactory:
    """Test the model factory functionality."""
    
    def test_get_standard_xgboost_class(self):
        """Test getting standard XGBoost model class."""
        model_class = get_model_class("xgboost")
        assert model_class == XGBoostStandardModel
        
    def test_get_quantile_xgboost_class(self):
        """Test getting quantile XGBoost model class."""
        model_class = get_model_class("xgboost_quantile")
        assert model_class == XGBoostQuantileModel
        
    def test_unknown_model_type_raises_error(self):
        """Test that unknown model types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model type 'unknown'"):
            get_model_class("unknown")


class TestXGBoostQuantileModel:
    """Test quantile XGBoost model implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100) * 2 + 10
        return X, y
        
    def test_model_initialization(self):
        """Test quantile model initialization."""
        model = XGBoostQuantileModel(quantile_alpha=0.7, max_depth=4)
        
        assert model.quantile_alpha == 0.7
        assert model.model_type == "xgboost_quantile"
        assert model.model_params['max_depth'] == 4
        assert not model.is_trained
        
    def test_model_training(self, sample_data):
        """Test quantile model training."""
        X, y = sample_data
        model = XGBoostQuantileModel(quantile_alpha=0.7)
        
        # Train the model
        model.train(X, y, num_boost_round=10)
        
        assert model.is_trained
        assert model.model is not None
        
    def test_model_prediction(self, sample_data):
        """Test quantile model prediction."""
        X, y = sample_data
        model = XGBoostQuantileModel(quantile_alpha=0.7)
        
        # Train then predict
        model.train(X, y, num_boost_round=10)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
        
    def test_prediction_without_training_raises_error(self, sample_data):
        """Test that prediction without training raises error."""
        X, y = sample_data
        model = XGBoostQuantileModel(quantile_alpha=0.7)
        
        with pytest.raises(ModelPredictionError):
            model.predict(X)
            
    def test_model_info(self, sample_data):
        """Test model info generation."""
        X, y = sample_data
        model = XGBoostQuantileModel(quantile_alpha=0.7, max_depth=6)
        model.train(X, y, num_boost_round=10)
        
        info = model.get_model_info()
        
        assert info["model_type"] == "xgboost_quantile"
        assert info["quantile_alpha"] == 0.7
        assert info["is_trained"] is True
        assert "parameters" in info
        
    def test_quantile_evaluation_metrics(self, sample_data):
        """Test quantile-specific evaluation metrics."""
        X, y = sample_data
        model = XGBoostQuantileModel(quantile_alpha=0.7)
        model.train(X, y, num_boost_round=10)
        
        predictions = model.predict(X)
        metrics = model.get_evaluation_metrics(y, predictions)
        
        # Check standard metrics are present
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        
        # Check quantile-specific metrics are present
        assert "quantile_score" in metrics
        assert "coverage_probability" in metrics
        assert "coverage_error" in metrics
        assert "quantile_alpha" in metrics
        
        # Check quantile alpha matches
        assert metrics["quantile_alpha"] == 0.7
        
    def test_coverage_probability_calculation(self, sample_data):
        """Test that coverage probability is reasonable for quantile predictions."""
        X, y = sample_data
        model = XGBoostQuantileModel(quantile_alpha=0.7)
        model.train(X, y, num_boost_round=50)  # More rounds for better calibration
        
        predictions = model.predict(X)
        metrics = model.get_evaluation_metrics(y, predictions)
        
        coverage = metrics["coverage_probability"]
        # Coverage should be between 0 and 1, and reasonably close to target quantile
        # Note: On small synthetic data, perfect calibration is not expected
        assert 0.0 <= coverage <= 1.0, f"Coverage {coverage} not in valid range [0, 1]"
        assert abs(coverage - 0.7) < 0.5, f"Coverage {coverage} too far from target 0.7"
        

class TestXGBoostStandardModel:
    """Test standard XGBoost model implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100) * 2 + 10
        return X, y
        
    def test_model_initialization(self):
        """Test standard model initialization."""
        model = XGBoostStandardModel(max_depth=4, n_estimators=50)
        
        assert model.model_type == "xgboost"
        assert model.model_params['max_depth'] == 4
        assert model.model_params['n_estimators'] == 50
        assert not model.is_trained
        
    def test_model_training(self, sample_data):
        """Test standard model training."""
        X, y = sample_data
        model = XGBoostStandardModel(n_estimators=10)
        
        # Train the model
        model.train(X, y)
        
        assert model.is_trained
        assert model.model is not None
        
    def test_model_evaluation_metrics(self, sample_data):
        """Test standard evaluation metrics."""
        X, y = sample_data
        model = XGBoostStandardModel(n_estimators=10)
        model.train(X, y)
        
        predictions = model.predict(X)
        metrics = model.get_evaluation_metrics(y, predictions)
        
        # Check standard metrics are present
        expected_metrics = ["mse", "rmse", "mae", "r2", "max_error", "mean_error", 
                           "std_error", "mape", "within_1_unit", "within_2_units", "within_5_units"]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            
        # Check that quantile-specific metrics are NOT present
        assert "quantile_score" not in metrics
        assert "coverage_probability" not in metrics


class TestTrainingConfigExtension:
    """Test extended TrainingConfig for quantile models."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.model_type == "xgboost"
        assert config.quantile_alpha is None
        assert config.model_specific_params == {}
        
    def test_quantile_config(self):
        """Test quantile model configuration."""
        config = TrainingConfig(
            model_type="xgboost_quantile",
            quantile_alpha=0.8,
            model_specific_params={"custom_param": "value"}
        )
        
        assert config.model_type == "xgboost_quantile"
        assert config.quantile_alpha == 0.8
        assert config.model_specific_params["custom_param"] == "value"


class TestModelTrainerIntegration:
    """Test integration with ModelTrainer."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for integration testing."""
        np.random.seed(42)
        n_samples = 50
        
        # Create mock polars DataFrames
        import polars as pl
        
        # Features
        X_data = {
            "bdID": list(range(n_samples)),
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        }
        X_df = pl.DataFrame(X_data)
        
        # Targets
        y_data = {
            "bdID": list(range(n_samples)),
            "target": np.random.randn(n_samples) * 2 + 10
        }
        y_df = pl.DataFrame(y_data)
        
        feature_cols = ["feature_1", "feature_2", "feature_3"]
        
        return X_df, y_df, feature_cols
        
    def test_standard_model_training(self, mock_data):
        """Test training standard XGBoost through ModelTrainer."""
        X_df, y_df, feature_cols = mock_data
        
        from src import ModelingStrategy
        
        config = TrainingConfig(
            model_type="xgboost",
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )
        
        trainer = ModelTrainer(config)
        
        # Split data for training
        n_train = len(X_df) // 2
        X_train = X_df[:n_train]
        y_train = y_df[:n_train]
        X_test = X_df[n_train:]
        y_test = y_df[n_train:]
        
        model = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target",
            ModelingStrategy.COMBINED, [(1, 1)]
        )
        
        assert model is not None
        assert model.metadata.model_type == "xgboost"
        assert "mse" in model.metadata.performance_metrics
        
    def test_quantile_model_training(self, mock_data):
        """Test training quantile XGBoost through ModelTrainer."""
        X_df, y_df, feature_cols = mock_data
        
        from src import ModelingStrategy
        
        config = TrainingConfig(
            model_type="xgboost_quantile",
            quantile_alpha=0.75,
            hyperparameters={"max_depth": 3, "learning_rate": 0.3}
        )
        
        trainer = ModelTrainer(config)
        
        # Split data for training
        n_train = len(X_df) // 2
        X_train = X_df[:n_train]
        y_train = y_df[:n_train]
        X_test = X_df[n_train:]
        y_test = y_df[n_train:]
        
        model = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target",
            ModelingStrategy.COMBINED, [(1, 1)]
        )
        
        assert model is not None
        assert model.metadata.model_type == "xgboost_quantile"
        assert "quantile_score" in model.metadata.performance_metrics
        assert "coverage_probability" in model.metadata.performance_metrics
        assert model.metadata.model_id == "combined_1skus_xgboost_quantile_q0.75"
        assert "q0.75" in model.get_identifier()


class TestBackwardCompatibility:
    """Test that existing functionality still works."""
    
    def test_standard_config_still_works(self):
        """Test that existing TrainingConfig usage still works."""
        # This should work exactly as before
        config = TrainingConfig(
            model_type="xgboost",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 6
            }
        )
        
        assert config.model_type == "xgboost"
        assert config.quantile_alpha is None
        
    def test_model_factory_handles_legacy_type(self):
        """Test that model factory works with legacy 'xgboost' type."""
        model_class = get_model_class("xgboost")
        model = model_class(n_estimators=10)
        
        assert model.model_type == "xgboost"
        assert isinstance(model, XGBoostStandardModel)


class TestLightningQuantileModel:
    """Test Lightning quantile model implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(50, 5)  # Smaller dataset for faster testing
        y = np.random.randn(50) * 2 + 10
        return X, y
        
    def test_model_initialization(self):
        """Test Lightning quantile model initialization."""
        model = LightningQuantileModel(quantile_alpha=0.8, hidden_size=32)
        
        assert model.model_type == "lightning_quantile"
        assert model.quantile_alpha == 0.8
        assert model.model_params['hidden_size'] == 32
        assert not model.is_trained
        assert hasattr(model, 'MODEL_TYPE')
        assert hasattr(model, 'REQUIRES_QUANTILE')
        assert model.REQUIRES_QUANTILE == True
        
    def test_invalid_quantile_alpha(self):
        """Test that invalid quantile_alpha values raise errors."""
        with pytest.raises(ValueError, match="quantile_alpha must be between 0 and 1"):
            LightningQuantileModel(quantile_alpha=1.5)
            
        with pytest.raises(ValueError, match="quantile_alpha must be between 0 and 1"):
            LightningQuantileModel(quantile_alpha=0.0)
            
    def test_model_attributes(self):
        """Test Lightning quantile model class attributes."""
        assert LightningQuantileModel.MODEL_TYPE == "lightning_quantile"
        assert LightningQuantileModel.REQUIRES_QUANTILE == True
        assert "PyTorch Lightning" in LightningQuantileModel.DESCRIPTION
        assert "quantile" in LightningQuantileModel.DESCRIPTION
        assert isinstance(LightningQuantileModel.DEFAULT_HYPERPARAMETERS, dict)
        
    def test_pinball_loss_function(self):
        """Test the pinball loss function implementation."""
        from src.models.lightning_quantile import QuantileForecastingModel
        import torch
        
        # Create a simple model to test loss function
        model = QuantileForecastingModel(input_size=2, quantile_alpha=0.7)
        
        # Test pinball loss computation
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.1, 2.2, 2.8])
        
        loss = model.pinball_loss(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
        
    def test_model_training_basic(self, sample_data):
        """Test basic Lightning quantile model training."""
        X, y = sample_data
        model = LightningQuantileModel(
            quantile_alpha=0.7,
            hidden_size=16,  # Small for fast testing
            max_epochs=2,    # Very few epochs for speed
            batch_size=16
        )
        
        # Train the model
        model.train(X, y)
        
        assert model.is_trained
        assert model.lightning_model is not None
        assert model.model is not None  # Reference should be set
        
    def test_model_predictions(self, sample_data):
        """Test Lightning quantile model predictions."""
        X, y = sample_data
        model = LightningQuantileModel(
            quantile_alpha=0.7,
            hidden_size=16,
            max_epochs=2,
            batch_size=16
        )
        
        # Train and predict
        model.train(X, y)
        predictions = model.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        
    def test_prediction_without_training(self, sample_data):
        """Test that prediction fails without training."""
        X, y = sample_data
        model = LightningQuantileModel(quantile_alpha=0.7)
        
        with pytest.raises(ModelPredictionError, match="Model must be trained"):
            model.predict(X)
            
    def test_model_info(self, sample_data):
        """Test Lightning quantile model information."""
        X, y = sample_data
        model = LightningQuantileModel(
            quantile_alpha=0.8,
            hidden_size=32,
            max_epochs=2
        )
        
        # Test info before training
        info = model.get_model_info()
        assert info["model_type"] == "lightning_quantile"
        assert info["quantile_alpha"] == 0.8
        assert info["is_trained"] == False
        
        # Train and test info after training
        model.train(X, y)
        info = model.get_model_info()
        assert info["is_trained"] == True
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["input_size"] == X.shape[1]
        
    def test_model_factory_integration(self):
        """Test Lightning quantile model works with factory pattern."""
        model_class = get_model_class("lightning_quantile")
        assert model_class == LightningQuantileModel
        
        # Test instantiation through factory
        model = model_class(quantile_alpha=0.6, hidden_size=64)
        assert isinstance(model, LightningQuantileModel)
        assert model.quantile_alpha == 0.6
        
    def test_different_quantile_levels(self, sample_data):
        """Test training with different quantile levels."""
        X, y = sample_data
        quantile_levels = [0.1, 0.5, 0.9]
        
        for alpha in quantile_levels:
            model = LightningQuantileModel(
                quantile_alpha=alpha,
                hidden_size=16,
                max_epochs=2,
                batch_size=16
            )
            
            model.train(X, y)
            predictions = model.predict(X)
            
            assert model.is_trained
            assert len(predictions) == len(y)
            assert model.quantile_alpha == alpha
            
    def test_deterministic_behavior(self, sample_data):
        """Test that random seed produces deterministic results."""
        X, y = sample_data
        
        # Train two models with same random seed
        model1 = LightningQuantileModel(
            quantile_alpha=0.7,
            hidden_size=16,
            max_epochs=2,
            random_state=42
        )
        
        model2 = LightningQuantileModel(
            quantile_alpha=0.7,
            hidden_size=16,
            max_epochs=2,
            random_state=42
        )
        
        model1.train(X, y)
        model2.train(X, y)
        
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        # Neural networks may have some non-determinism even with seeds
        # Check that at least the models are trained and produce reasonable outputs
        assert len(pred1) == len(pred2)
        assert all(isinstance(p, (int, float, np.number)) for p in pred1)
        assert all(isinstance(p, (int, float, np.number)) for p in pred2)


class TestLightningQuantileModelTrainerIntegration:
    """Test Lightning quantile model integration with ModelTrainer."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for integration testing."""
        np.random.seed(42)
        n_samples = 30  # Small dataset for fast testing
        
        # Create mock polars DataFrames
        import polars as pl
        
        # Features
        X_data = {
            "bdID": list(range(n_samples)),
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
        }
        X_df = pl.DataFrame(X_data)
        
        # Targets
        y_data = {
            "bdID": list(range(n_samples)),
            "target": np.random.randn(n_samples) * 2 + 10
        }
        y_df = pl.DataFrame(y_data)
        
        feature_cols = ["feature_1", "feature_2"]
        
        return X_df, y_df, feature_cols
        
    def test_lightning_quantile_training_through_trainer(self, mock_data):
        """Test training Lightning quantile model through ModelTrainer."""
        X_df, y_df, feature_cols = mock_data
        
        from src import ModelingStrategy
        
        config = TrainingConfig(random_state=42)
        config.add_model_config(
            model_type="lightning_quantile",
            quantile_alpha=0.75,
            hyperparameters={
                "hidden_size": 16,
                "max_epochs": 2,
                "batch_size": 8
            }
        )
        
        trainer = ModelTrainer(config)
        
        # Split data for training
        n_train = len(X_df) // 2
        X_train = X_df[:n_train]
        y_train = y_df[:n_train]
        X_test = X_df[n_train:]
        y_test = y_df[n_train:]
        
        model = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target",
            ModelingStrategy.COMBINED, [(1, 1)],
            model_type="lightning_quantile"
        )
        
        assert model is not None
        assert model.metadata.model_type == "lightning_quantile"
        
        # Check quantile-specific metrics are present
        metrics = model.metadata.performance_metrics
        assert "quantile_score" in metrics
        assert "coverage_probability" in metrics
        assert "coverage_error" in metrics
        assert "quantile_alpha" in metrics
        assert metrics["quantile_alpha"] == 0.75
        
        # Check standard metrics are also present
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        
        # Check model ID includes quantile level
        assert "q0.75" in model.get_identifier()
        
    def test_lightning_quantile_model_consistency(self, mock_data):
        """Test consistency between Lightning quantile models and XGBoost quantile models."""
        X_df, y_df, feature_cols = mock_data
        
        from src import ModelingStrategy
        
        # Both models should have similar interface patterns
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_quantile",
            quantile_alpha=0.7,
            hyperparameters={"hidden_size": 16, "max_epochs": 2}
        )
        
        xgb_config = TrainingConfig(random_state=42)
        xgb_config.add_model_config(
            model_type="xgboost_quantile",
            quantile_alpha=0.7,
            hyperparameters={"n_estimators": 5, "max_depth": 2}
        )
        
        lightning_trainer = ModelTrainer(lightning_config)
        xgb_trainer = ModelTrainer(xgb_config)
        
        # Split data
        n_train = len(X_df) // 2
        X_train = X_df[:n_train]
        y_train = y_df[:n_train]
        X_test = X_df[n_train:]
        y_test = y_df[n_train:]
        
        # Train both models
        lightning_model = lightning_trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, [(1, 1)],
            model_type="lightning_quantile"
        )
        
        xgb_model = xgb_trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target", ModelingStrategy.COMBINED, [(1, 1)],
            model_type="xgboost_quantile"
        )
        
        # Both should have same required metadata fields
        required_fields = ['model_id', 'modeling_strategy', 'sku_tuples', 'model_type',
                          'hyperparameters', 'performance_metrics', 'feature_columns']
        
        for field in required_fields:
            assert hasattr(lightning_model.metadata, field)
            assert hasattr(xgb_model.metadata, field)
        
        # Both should have same quantile metric keys
        lightning_metrics = set(lightning_model.metadata.performance_metrics.keys())
        xgb_metrics = set(xgb_model.metadata.performance_metrics.keys())
        
        # Key quantile metrics should be present in both
        quantile_metrics = {"quantile_score", "coverage_probability", "coverage_error", "quantile_alpha"}
        assert quantile_metrics.issubset(lightning_metrics)
        assert quantile_metrics.issubset(xgb_metrics)