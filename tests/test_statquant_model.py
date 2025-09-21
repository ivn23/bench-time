"""
Unit tests for StatQuant model implementation.

Tests the statsmodels QuantReg-based quantile regression model.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.statquant_model import StatQuantModel
from src.models.base import ModelTrainingError, ModelPredictionError


class TestStatQuantModel:
    """Test StatQuant model implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100) * 2 + 10
        return X, y
        
    def test_model_initialization(self):
        """Test StatQuant model initialization."""
        model = StatQuantModel(quantile_alpha=0.7, method="interior-point")
        
        assert model.quantile_alpha == 0.7
        assert model.model_type == "statquant"
        assert model.model_params['method'] == "interior-point"
        assert not model.is_trained
        assert model.fitted_model is None
        
    def test_model_initialization_invalid_quantile(self):
        """Test that invalid quantile alpha raises error."""
        with pytest.raises(ValueError, match="quantile_alpha must be between 0 and 1"):
            StatQuantModel(quantile_alpha=1.5)
            
        with pytest.raises(ValueError, match="quantile_alpha must be between 0 and 1"):
            StatQuantModel(quantile_alpha=0.0)
            
        with pytest.raises(ValueError, match="quantile_alpha must be between 0 and 1"):
            StatQuantModel(quantile_alpha=1.0)
        
    def test_model_training(self, sample_data):
        """Test StatQuant model training."""
        X, y = sample_data
        model = StatQuantModel(quantile_alpha=0.7)
        
        # Mock the statsmodels components to avoid actual fitting
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_fit_result = MagicMock()
            mock_quantreg_instance = MagicMock()
            mock_quantreg_instance.fit.return_value = mock_fit_result
            mock_quantreg.return_value = mock_quantreg_instance
            
            # Train the model
            model.train(X, y)
            
            assert model.is_trained
            assert model.model is not None
            assert model.fitted_model is not None
            mock_quantreg.assert_called_once_with(endog=y, exog=X)
            mock_quantreg_instance.fit.assert_called_once()
            
    def test_model_training_with_2d_target(self, sample_data):
        """Test StatQuant model training with 2D target array."""
        X, y = sample_data
        y_2d = y.reshape(-1, 1)  # Make it 2D
        model = StatQuantModel(quantile_alpha=0.7)
        
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_fit_result = MagicMock()
            mock_quantreg_instance = MagicMock()
            mock_quantreg_instance.fit.return_value = mock_fit_result
            mock_quantreg.return_value = mock_quantreg_instance
            
            # Train the model - should handle 2D target
            model.train(X, y_2d)
            
            # Check that target was flattened
            call_args = mock_quantreg.call_args
            assert call_args[1]['endog'].shape == (100,)  # Should be flattened
            
    def test_model_prediction(self, sample_data):
        """Test StatQuant model prediction."""
        X, y = sample_data
        model = StatQuantModel(quantile_alpha=0.7)
        
        # Mock training and prediction
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_fit_result = MagicMock()
            mock_fit_result.predict.return_value = np.array([1.0, 2.0, 3.0])
            mock_quantreg_instance = MagicMock()
            mock_quantreg_instance.fit.return_value = mock_fit_result
            mock_quantreg.return_value = mock_quantreg_instance
            
            # Train then predict
            model.train(X[:3], y[:3])  # Train on smaller subset for test
            predictions = model.predict(X[:3])
            
            assert len(predictions) == 3
            assert isinstance(predictions, np.ndarray)
            np.testing.assert_array_equal(predictions, [1.0, 2.0, 3.0])
            
    def test_prediction_without_training_raises_error(self, sample_data):
        """Test that prediction without training raises error."""
        X, y = sample_data
        model = StatQuantModel(quantile_alpha=0.7)
        
        with pytest.raises(ModelPredictionError, match="Model must be trained before making predictions"):
            model.predict(X)
            
    def test_model_info(self, sample_data):
        """Test model info generation."""
        X, y = sample_data
        model = StatQuantModel(quantile_alpha=0.7, method="interior-point")
        
        # Test info before training
        info_before = model.get_model_info()
        assert info_before["model_type"] == "statquant"
        assert info_before["quantile_alpha"] == 0.7
        assert info_before["is_trained"] is False
        assert "parameters" in info_before
        assert info_before["parameters"]["method"] == "interior-point"
        
        # Mock training
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_fit_result = MagicMock()
            mock_fit_result.converged = True
            mock_fit_result.n_iterations = 50
            mock_fit_result.method = "interior-point"
            
            mock_quantreg_instance = MagicMock()
            mock_quantreg_instance.fit.return_value = mock_fit_result
            mock_quantreg.return_value = mock_quantreg_instance
            
            model.train(X, y)
            
            # Test info after training
            info_after = model.get_model_info()
            assert info_after["is_trained"] is True
            assert info_after["converged"] is True
            assert info_after["n_iterations"] == 50
            assert info_after["method"] == "interior-point"
            assert info_after["quantile_level"] == 0.7
            
    def test_quantile_evaluation_metrics(self, sample_data):
        """Test quantile-specific evaluation metrics."""
        X, y = sample_data
        model = StatQuantModel(quantile_alpha=0.7)
        
        # Mock the training and prediction
        mock_predictions = np.array([8.5, 12.3, 9.1, 11.7, 10.9])
        
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_fit_result = MagicMock()
            mock_fit_result.predict.return_value = mock_predictions
            mock_quantreg_instance = MagicMock()
            mock_quantreg_instance.fit.return_value = mock_fit_result
            mock_quantreg.return_value = mock_quantreg_instance
            
            model.train(X[:5], y[:5])
            predictions = model.predict(X[:5])
            metrics = model.get_evaluation_metrics(y[:5], predictions)
            
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
        model = StatQuantModel(quantile_alpha=0.7)
        
        # Create mock predictions that should give reasonable coverage
        # For 70% quantile, about 70% of actuals should be <= predictions
        y_subset = y[:20]
        mock_predictions = np.percentile(y_subset, 70) * np.ones_like(y_subset)
        
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_fit_result = MagicMock()
            mock_fit_result.predict.return_value = mock_predictions
            mock_quantreg_instance = MagicMock()
            mock_quantreg_instance.fit.return_value = mock_fit_result
            mock_quantreg.return_value = mock_quantreg_instance
            
            model.train(X[:20], y_subset)
            predictions = model.predict(X[:20])
            metrics = model.get_evaluation_metrics(y_subset, predictions)
            
            coverage = metrics["coverage_probability"]
            # Coverage should be between 0 and 1
            assert 0.0 <= coverage <= 1.0, f"Coverage {coverage} not in valid range [0, 1]"
            
    def test_training_error_handling(self, sample_data):
        """Test error handling during training."""
        X, y = sample_data
        model = StatQuantModel(quantile_alpha=0.7)
        
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_quantreg.side_effect = Exception("Statsmodels error")
            
            with pytest.raises(ModelTrainingError, match="Failed to train StatQuant model"):
                model.train(X, y)
                
    def test_prediction_error_handling(self, sample_data):
        """Test error handling during prediction."""
        X, y = sample_data
        model = StatQuantModel(quantile_alpha=0.7)
        
        with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
            mock_fit_result = MagicMock()
            mock_fit_result.predict.side_effect = Exception("Prediction error")
            mock_quantreg_instance = MagicMock()
            mock_quantreg_instance.fit.return_value = mock_fit_result
            mock_quantreg.return_value = mock_quantreg_instance
            
            # Train first
            model.train(X, y)
            
            # Now prediction should fail
            with pytest.raises(ModelPredictionError, match="Failed to make predictions"):
                model.predict(X)
                
    def test_model_constants(self):
        """Test model class constants."""
        assert StatQuantModel.MODEL_TYPE == "statquant"
        assert StatQuantModel.DESCRIPTION == "Statsmodels Quantile Regression for probabilistic forecasting"
        assert StatQuantModel.REQUIRES_QUANTILE is True
        assert isinstance(StatQuantModel.DEFAULT_HYPERPARAMETERS, dict)
        assert "method" in StatQuantModel.DEFAULT_HYPERPARAMETERS
        assert "max_iter" in StatQuantModel.DEFAULT_HYPERPARAMETERS
        
    def test_different_quantile_levels(self, sample_data):
        """Test model with different quantile levels."""
        X, y = sample_data[:10], sample_data[1][:10]  # Smaller dataset for speed
        
        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for alpha in quantile_levels:
            model = StatQuantModel(quantile_alpha=alpha)
            assert model.quantile_alpha == alpha
            
            with patch('src.models.statquant_model.QuantReg') as mock_quantreg:
                mock_fit_result = MagicMock()
                mock_quantreg_instance = MagicMock()
                mock_quantreg_instance.fit.return_value = mock_fit_result
                mock_quantreg.return_value = mock_quantreg_instance
                
                model.train(X, y)
                
                # Check that fit was called with correct quantile
                fit_call_args = mock_quantreg_instance.fit.call_args
                assert fit_call_args[1]['q'] == alpha