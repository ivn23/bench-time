"""
Tests for quantile model implementations in the M5 benchmarking framework.

This module tests the new extensible model architecture, specifically
focusing on quantile XGBoost functionality and model factory patterns.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from src.models import get_model_class, XGBoostQuantileModel, XGBoostStandardModel
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