"""
Unit tests for ModelEvaluator component.
Tests model evaluation, metrics calculation, and comparison functionality.
"""

import pytest
import polars as pl
import numpy as np
from unittest.mock import Mock

from src import ModelEvaluator, BenchmarkModel, ModelMetadata, DataSplit, ModelingStrategy, ModelRegistry, DataLoader


class TestModelEvaluator:
    """Test ModelEvaluator component functionality."""

    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry for testing."""
        return Mock(spec=ModelRegistry)

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader for testing."""
        mock = Mock(spec=DataLoader)
        mock._is_loaded = True  # Add required attribute for evaluate_model
        # Add mock data frames that evaluate_model expects
        mock._features_df = Mock()
        mock._target_df = Mock()
        return mock

    @pytest.fixture
    def sample_trained_model(self, sample_training_config, single_sku_tuple):
        """Create a sample trained model for testing."""
        # Create a simple mock XGBoost model
        mock_xgb_model = Mock()
        mock_xgb_model.predict.return_value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        
        # Create metadata
        metadata = ModelMetadata(
            model_id="test_model_001",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=single_sku_tuple,
            model_type="xgboost",
            hyperparameters={"n_estimators": 10, "max_depth": 3},
            training_config=sample_training_config.__dict__,
            performance_metrics={"mse": 0.5, "rmse": 0.707, "mae": 0.4, "r2": 0.8},
            feature_columns=["month", "day_of_week", "price_0", "trend"],
            target_column="target"
        )
        
        # Create data split
        data_split = DataSplit(
            train_bdIDs=np.array([1001, 1002, 1003, 1004, 1005]),
            validation_bdIDs=np.array([1006, 1007, 1008, 1009, 1010])
        )
        
        return BenchmarkModel(
            metadata=metadata,
            model=mock_xgb_model,
            data_split=data_split
        )

    def test_evaluator_initialization(self, mock_data_loader, mock_model_registry):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        assert evaluator.data_loader == mock_data_loader
        assert evaluator.model_registry == mock_model_registry

    def test_evaluate_model_with_data(self, mock_data_loader, mock_model_registry, sample_trained_model):
        """Test evaluating a model with pre-engineered data."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Create sample X and y data for evaluation
        X_data = pl.DataFrame({
            "bdID": [1006, 1007, 1008, 1009, 1010],
            "month": [1, 1, 1, 2, 2],
            "day_of_week": [0, 1, 2, 3, 4],
            "price_0": [10.5, 11.2, 9.8, 10.0, 12.1],
            "trend": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        y_data = pl.DataFrame({
            "bdID": [1006, 1007, 1008, 1009, 1010],
            "target": [1, 2, 3, 4, 5]
        })
        
        # Evaluate model
        result = evaluator.evaluate_model_with_data(sample_trained_model, X_data, y_data)
        
        # Validate result structure
        assert isinstance(result, dict)
        assert "model_id" in result
        assert "modeling_strategy" in result
        assert "sku_tuples" in result
        assert "n_samples" in result
        assert "predictions" in result
        assert "actuals" in result
        assert "prediction_errors" in result
        assert "metrics" in result
        assert "data_split_name" in result
        
        # Check specific values
        assert result["model_id"] == sample_trained_model.get_identifier()
        assert result["modeling_strategy"] == "combined"
        assert result["n_samples"] == 5
        assert len(result["predictions"]) == 5
        assert len(result["actuals"]) == 5
        assert len(result["prediction_errors"]) == 5
        
        # Check metrics
        metrics = result["metrics"]
        required_metrics = ["mse", "rmse", "mae", "r2", "mape", "max_error", 
                          "mean_error", "std_error", "within_1_unit", 
                          "within_2_units", "within_5_units"]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.integer, np.floating))

    def test_comprehensive_metrics_calculation(self, mock_data_loader, mock_model_registry):
        """Test comprehensive metrics calculation."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Test with known values
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred)
        
        # Check all expected metrics are present
        expected_metrics = [
            "mse", "rmse", "mae", "r2", "mape", "max_error", 
            "mean_error", "std_error", "within_1_unit", 
            "within_2_units", "within_5_units"
        ]
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check some specific calculations
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert abs(metrics["mse"] - expected_mse) < 1e-10
        
        expected_rmse = np.sqrt(expected_mse)
        assert abs(metrics["rmse"] - expected_rmse) < 1e-10
        
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert abs(metrics["mae"] - expected_mae) < 1e-10
        
        # Check percentage within ranges
        abs_errors = np.abs(y_true - y_pred)
        expected_within_1 = np.mean(abs_errors <= 1.0) * 100
        assert abs(metrics["within_1_unit"] - expected_within_1) < 1e-10

    def test_metrics_with_zero_values(self, mock_data_loader, mock_model_registry):
        """Test metrics calculation with zero values in actuals."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Test MAPE with zero values
        y_true = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        y_pred = np.array([0.1, 1.1, 1.9, 0.2, 3.1])
        
        metrics = evaluator._calculate_comprehensive_metrics(y_true, y_pred)
        
        # MAPE should handle zero values appropriately
        assert "mape" in metrics
        # Should either be finite or inf, but not NaN
        assert not np.isnan(metrics["mape"])

    def test_compare_models(self, mock_data_loader, mock_model_registry):
        """Test comparing multiple models."""
        mock_model_registry.get_model.return_value = None  # Simulate model not found
        
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        model_ids = ["model_1", "model_2", "model_3"]
        result = evaluator.compare_models(model_ids)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "model_evaluations" in result
        assert "metrics_comparison" in result
        assert "rankings" in result
        
        # Since models weren't found, evaluations should be empty
        assert len(result["model_evaluations"]) == 0

    def test_create_metrics_comparison(self, mock_data_loader, mock_model_registry):
        """Test creating metrics comparison table."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Sample metrics from multiple models
        all_metrics = {
            "model_1": {"mse": 0.5, "rmse": 0.707, "mae": 0.4, "r2": 0.8},
            "model_2": {"mse": 0.6, "rmse": 0.775, "mae": 0.5, "r2": 0.75},
            "model_3": {"mse": 0.4, "rmse": 0.632, "mae": 0.35, "r2": 0.85}
        }
        
        comparison = evaluator._create_metrics_comparison(all_metrics)
        
        # Check structure
        assert isinstance(comparison, dict)
        
        # Check that all metrics are present
        expected_metrics = ["mse", "rmse", "mae", "r2"]
        for metric in expected_metrics:
            assert metric in comparison
            assert len(comparison[metric]) == 3  # Three models
            
        # Check specific values
        assert comparison["mse"]["model_1"] == 0.5
        assert comparison["r2"]["model_3"] == 0.85

    def test_rank_models(self, mock_data_loader, mock_model_registry):
        """Test model ranking functionality."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Sample metrics from multiple models
        all_metrics = {
            "model_1": {"mse": 0.5, "rmse": 0.707, "r2": 0.8},
            "model_2": {"mse": 0.6, "rmse": 0.775, "r2": 0.75},
            "model_3": {"mse": 0.4, "rmse": 0.632, "r2": 0.85}
        }
        
        rankings = evaluator._rank_models(all_metrics)
        
        # Check structure
        assert isinstance(rankings, dict)
        
        # Check MSE ranking (lower is better)
        mse_ranking = rankings["mse"]
        assert mse_ranking[0][0] == "model_3"  # Lowest MSE
        assert mse_ranking[1][0] == "model_1"
        assert mse_ranking[2][0] == "model_2"  # Highest MSE
        
        # Check R² ranking (higher is better)
        r2_ranking = rankings["r2"]
        assert r2_ranking[0][0] == "model_3"  # Highest R²
        assert r2_ranking[1][0] == "model_1"
        assert r2_ranking[2][0] == "model_2"  # Lowest R²

    def test_evaluate_by_modeling_strategy(self, mock_data_loader, mock_model_registry):
        """Test evaluating models by strategy."""
        # Mock the registry to return some model IDs
        mock_model_registry.list_models.return_value = ["model_1", "model_2"]
        
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Mock the compare_models method to avoid complex setup
        expected_result = {"model_evaluations": {}, "metrics_comparison": {}, "rankings": {}}
        evaluator.compare_models = Mock(return_value=expected_result)
        
        result = evaluator.evaluate_by_modeling_strategy(ModelingStrategy.COMBINED)
        
        # Should call compare_models internally
        mock_model_registry.list_models.assert_called_once_with(ModelingStrategy.COMBINED)
        evaluator.compare_models.assert_called_once_with(["model_1", "model_2"], None)
        
        # Check basic result structure
        assert isinstance(result, dict)
        assert result == expected_result

    def test_evaluate_by_modeling_strategy_no_models(self, mock_data_loader, mock_model_registry):
        """Test evaluating by strategy when no models exist."""
        # Mock empty model list
        mock_model_registry.list_models.return_value = []
        
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        result = evaluator.evaluate_by_modeling_strategy(ModelingStrategy.INDIVIDUAL)
        
        # Should return error message
        assert "error" in result
        assert "No models found" in result["error"]

    def test_generate_evaluation_report_single_model(self, mock_data_loader, mock_model_registry):
        """Test generating evaluation report for single model."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Sample evaluation result for single model
        evaluation_result = {
            "model_id": "test_model_001",
            "modeling_strategy": "combined",
            "sku_tuples": [(80558, 2)],
            "n_samples": 100,
            "metrics": {
                "mse": 0.5,
                "rmse": 0.707,
                "mae": 0.4,
                "r2": 0.8
            }
        }
        
        report = evaluator.generate_evaluation_report(evaluation_result)
        
        # Check report structure
        assert isinstance(report, str)
        assert "Model Evaluation Report" in report
        assert "test_model_001" in report
        assert "combined" in report
        assert "**MSE:** 0.5000" in report
        assert "**R2:** 0.8000" in report

    def test_generate_evaluation_report_comparison(self, mock_data_loader, mock_model_registry):
        """Test generating evaluation report for model comparison."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Sample comparison results
        comparison_results = {
            "model_evaluations": {
                "model_1": {"metrics": {"mse": 0.5, "r2": 0.8}},
                "model_2": {"metrics": {"mse": 0.6, "r2": 0.75}}
            },
            "metrics_comparison": {
                "mse": {"model_1": 0.5, "model_2": 0.6},
                "r2": {"model_1": 0.8, "model_2": 0.75}
            },
            "rankings": {
                "mse": [("model_1", 0.5), ("model_2", 0.6)],
                "r2": [("model_1", 0.8), ("model_2", 0.75)]
            }
        }
        
        report = evaluator.generate_evaluation_report(comparison_results)
        
        # Check report structure
        assert isinstance(report, str)
        assert "Model Comparison" in report
        assert "Performance Metrics" in report
        assert "Model Rankings" in report
        assert "MSE:" in report
        assert "R2:" in report

    def test_model_evaluation_error_handling(self, mock_data_loader, mock_model_registry, sample_trained_model):
        """Test error handling in model evaluation."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Test with empty data
        empty_X = pl.DataFrame({
            "bdID": [],
            "month": [],
            "day_of_week": [],
            "price_0": [],
            "trend": []
        })
        empty_y = pl.DataFrame({"bdID": [], "target": []})
        
        result = evaluator.evaluate_model_with_data(sample_trained_model, empty_X, empty_y)
        
        # Should handle empty data gracefully
        assert "error" in result
        assert "No test data available" in result["error"]

    def test_evaluation_with_mismatched_bdids(self, mock_data_loader, mock_model_registry, sample_trained_model):
        """Test evaluation when data bdIDs don't match model's validation bdIDs."""
        evaluator = ModelEvaluator(mock_data_loader, mock_model_registry)
        
        # Create data with different bdIDs than the model expects
        X_data = pl.DataFrame({
            "bdID": [2001, 2002, 2003],  # Different from model's validation bdIDs
            "month": [1, 1, 2],
            "day_of_week": [0, 1, 2],
            "price_0": [10.0, 11.0, 12.0],
            "trend": [0.1, 0.2, 0.3]
        })
        
        y_data = pl.DataFrame({
            "bdID": [2001, 2002, 2003],
            "target": [1, 2, 3]
        })
        
        result = evaluator.evaluate_model_with_data(sample_trained_model, X_data, y_data)
        
        # Should return error about no matching test data
        assert "error" in result
        assert "No test data available" in result["error"]