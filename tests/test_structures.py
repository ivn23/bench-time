"""
Unit tests for simplified data structures.

Tests the new simplified structures that replace the overengineered 11+ class system
with 4 essential classes focused on the user's actual needs.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from src.structures import (
    ModelConfig, TrainingResult, ExperimentResults, SplitInfo,
    ModelingStrategy, SkuTuple, SkuList, create_config,
    validate_sku_tuples, validate_modeling_strategy
)


class TestSimpleModelConfig:
    """Test SimpleModelConfig functionality."""

    def test_basic_config_creation(self):
        """Test creating basic model configuration."""
        config = SimpleModelConfig(
            model_type="xgboost_standard",
            hyperparameters={"n_estimators": 100, "max_depth": 6}
        )
        
        assert config.model_type == "xgboost_standard"
        assert config.hyperparameters["n_estimators"] == 100
        assert config.random_state == 42
        assert not config.is_quantile_model
        assert config.quantile_alphas is None

    def test_quantile_config_creation(self):
        """Test creating quantile model configuration."""
        config = SimpleModelConfig(
            model_type="xgboost_quantile",
            hyperparameters={"n_estimators": 50},
            quantile_alphas=[0.1, 0.5, 0.9]
        )
        
        assert config.model_type == "xgboost_quantile"
        assert config.quantile_alphas == [0.1, 0.5, 0.9]
        assert config.is_quantile_model

    def test_config_validation(self):
        """Test configuration validation."""
        # Empty model type should fail
        with pytest.raises(ValueError, match="model_type cannot be empty"):
            SimpleModelConfig(model_type="", hyperparameters={})
        
        # Non-dict hyperparameters should fail
        with pytest.raises(TypeError, match="hyperparameters must be a dictionary"):
            SimpleModelConfig(model_type="xgb", hyperparameters="invalid")
        
        # Invalid quantile alphas should fail
        with pytest.raises(ValueError, match="quantile_alphas must be between 0 and 1"):
            SimpleModelConfig(
                model_type="xgb", 
                hyperparameters={},
                quantile_alphas=[0.5, 1.5]  # 1.5 is invalid
            )

    def test_random_state_injection(self):
        """Test that random_state is automatically added to hyperparameters."""
        config = SimpleModelConfig(
            model_type="xgb",
            hyperparameters={"n_estimators": 100}
        )
        
        assert config.hyperparameters["random_state"] == 42
        
        # Should not override existing random_state
        config2 = SimpleModelConfig(
            model_type="xgb",
            hyperparameters={"n_estimators": 100, "random_state": 999}
        )
        
        assert config2.hyperparameters["random_state"] == 999

    def test_create_config_function(self):
        """Test convenience function for config creation."""
        config = create_config(
            model_type="xgboost_quantile",
            hyperparameters={"n_estimators": 200},
            quantile_alphas=[0.25, 0.75],
            random_state=123
        )
        
        assert config.model_type == "xgboost_quantile"
        assert config.hyperparameters["n_estimators"] == 200
        assert config.random_state == 123
        assert config.quantile_alphas == [0.25, 0.75]


class TestSplitInfo:
    """Test SplitInfo functionality."""

    def test_split_info_creation(self):
        """Test creating SplitInfo."""
        split_info = SplitInfo(
            train_bdIDs=np.array([1, 2, 3, 4]),
            validation_bdIDs=np.array([5, 6]),
            test_bdIDs=np.array([7, 8]),
            split_date="2020-03-01"
        )
        
        assert len(split_info.train_bdIDs) == 4
        assert len(split_info.validation_bdIDs) == 2
        assert len(split_info.test_bdIDs) == 2
        assert split_info.split_date == "2020-03-01"

    def test_split_info_validation(self):
        """Test SplitInfo validation."""
        # Empty train_bdIDs should fail
        with pytest.raises(ValueError, match="train_bdIDs cannot be empty"):
            SplitInfo(
                train_bdIDs=np.array([]),
                validation_bdIDs=np.array([1, 2])
            )
        
        # Empty validation_bdIDs should fail
        with pytest.raises(ValueError, match="validation_bdIDs cannot be empty"):
            SplitInfo(
                train_bdIDs=np.array([1, 2]),
                validation_bdIDs=np.array([])
            )

    def test_optional_fields(self):
        """Test optional fields in SplitInfo."""
        split_info = SplitInfo(
            train_bdIDs=np.array([1, 2]),
            validation_bdIDs=np.array([3, 4])
        )
        
        assert split_info.test_bdIDs is None
        assert split_info.split_date is None


class TestTrainingResult:
    """Test TrainingResult functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict.return_value = np.array([1.0, 2.0, 3.0])
        return model

    @pytest.fixture
    def sample_split_info(self):
        """Create sample split info for testing."""
        return SplitInfo(
            train_bdIDs=np.array([1, 2, 3]),
            validation_bdIDs=np.array([4, 5])
        )

    def test_training_result_creation(self, mock_model, sample_split_info):
        """Test creating TrainingResult."""
        result = TrainingResult(
            model=mock_model,
            model_type="xgboost_standard",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2), (80651, 5)],
            hyperparameters={"n_estimators": 100},
            feature_columns=["feature1", "feature2"],
            target_column="target",
            split_info=sample_split_info,
            training_loss=0.85,
            performance_metrics={"rmse": 12.34, "mae": 8.76}
        )
        
        assert result.model == mock_model
        assert result.model_type == "xgboost_standard"
        assert result.modeling_strategy == ModelingStrategy.COMBINED
        assert result.sku_tuples == [(80558, 2), (80651, 5)]
        assert result.training_loss == 0.85
        assert result.performance_metrics["rmse"] == 12.34
        assert result.has_test_metrics()
        assert not result.is_quantile_model()

    def test_quantile_training_result(self, mock_model, sample_split_info):
        """Test TrainingResult for quantile model."""
        result = TrainingResult(
            model=mock_model,
            model_type="xgboost_quantile",
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            sku_tuples=[(80558, 2)],
            hyperparameters={"n_estimators": 50},
            feature_columns=["feature1"],
            target_column="target",
            split_info=sample_split_info,
            quantile_level=0.5,
            performance_metrics={"coverage_probability": 0.89}
        )
        
        assert result.quantile_level == 0.5
        assert result.is_quantile_model()
        assert result.performance_metrics["coverage_probability"] == 0.89

    def test_model_id_generation(self, mock_model, sample_split_info):
        """Test automatic model ID generation."""
        result = TrainingResult(
            model=mock_model,
            model_type="xgb",
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            sku_tuples=[(80558, 2)],
            hyperparameters={},
            feature_columns=[],
            target_column="target",
            split_info=sample_split_info
        )
        
        model_id = result.get_identifier()
        assert "xgb" in model_id
        assert "individual" in model_id
        assert "80558x2" in model_id
        assert len(model_id) > 0

    def test_primary_sku(self, mock_model, sample_split_info):
        """Test primary SKU extraction."""
        result = TrainingResult(
            model=mock_model,
            model_type="xgb",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2), (80651, 5), (80700, 3)],
            hyperparameters={},
            feature_columns=[],
            target_column="target",
            split_info=sample_split_info
        )
        
        assert result.get_primary_sku() == (80558, 2)

    def test_validation(self, mock_model, sample_split_info):
        """Test TrainingResult validation."""
        # Empty SKU tuples should fail
        with pytest.raises(ValueError, match="sku_tuples cannot be empty"):
            TrainingResult(
                model=mock_model,
                model_type="xgb",
                modeling_strategy=ModelingStrategy.COMBINED,
                sku_tuples=[],
                hyperparameters={},
                feature_columns=[],
                target_column="target",
                split_info=sample_split_info
            )
        
        # Invalid SKU tuple should fail
        with pytest.raises(ValueError, match="Each SKU must be a 2-tuple"):
            TrainingResult(
                model=mock_model,
                model_type="xgb",
                modeling_strategy=ModelingStrategy.COMBINED,
                sku_tuples=[(80558,)],  # Missing store_id
                hyperparameters={},
                feature_columns=[],
                target_column="target",
                split_info=sample_split_info
            )

    def test_get_summary(self, mock_model, sample_split_info):
        """Test result summary generation."""
        result = TrainingResult(
            model=mock_model,
            model_type="xgb",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2), (80651, 5)],
            hyperparameters={"n_estimators": 100},
            feature_columns=["feature1"],
            target_column="target",
            split_info=sample_split_info,
            training_loss=0.75,
            performance_metrics={"rmse": 10.5}
        )
        
        summary = result.get_summary()
        assert summary["model_type"] == "xgb"
        assert summary["strategy"] == "combined"
        assert summary["num_skus"] == 2
        assert summary["skus"] == [(80558, 2), (80651, 5)]
        assert summary["training_loss"] == 0.75
        assert summary["test_metrics"]["rmse"] == 10.5
        assert summary["has_test_metrics"]


class TestExperimentResults:
    """Test ExperimentResults functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return SimpleModelConfig(
            model_type="xgboost_standard",
            hyperparameters={"n_estimators": 100}
        )

    @pytest.fixture
    def sample_training_results(self):
        """Create sample training results."""
        mock_model = Mock()
        split_info = SplitInfo(
            train_bdIDs=np.array([1, 2]),
            validation_bdIDs=np.array([3, 4])
        )
        
        result1 = TrainingResult(
            model=mock_model,
            model_type="xgb",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2)],
            hyperparameters={},
            feature_columns=[],
            target_column="target",
            split_info=split_info,
            performance_metrics={"rmse": 10.0, "mae": 7.5}
        )
        
        result2 = TrainingResult(
            model=mock_model,
            model_type="xgb",
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            sku_tuples=[(80651, 5)],
            hyperparameters={},
            feature_columns=[],
            target_column="target",
            split_info=split_info,
            quantile_level=0.5,
            performance_metrics={"rmse": 12.0, "coverage_probability": 0.89}
        )
        
        return [result1, result2]

    def test_experiment_results_creation(self, sample_config, sample_training_results):
        """Test creating ExperimentResults."""
        results = ExperimentResults(
            training_results=sample_training_results,
            experiment_name="test_experiment",
            config=sample_config
        )
        
        assert results.num_models == 2
        assert results.experiment_name == "test_experiment"
        assert results.config == sample_config
        assert len(results.model_identifiers) == 2

    def test_performance_summary(self, sample_config, sample_training_results):
        """Test performance summary calculation."""
        results = ExperimentResults(
            training_results=sample_training_results,
            experiment_name="test_experiment",
            config=sample_config
        )
        
        summary = results.get_performance_summary()
        assert summary["rmse_mean"] == 11.0  # (10.0 + 12.0) / 2
        assert summary["rmse_min"] == 10.0
        assert summary["rmse_max"] == 12.0
        assert summary["mae"] == 7.5  # Only one model has mae
        assert summary["coverage_probability"] == 0.89  # Only quantile model has this

    def test_filtering_methods(self, sample_config, sample_training_results):
        """Test result filtering methods."""
        results = ExperimentResults(
            training_results=sample_training_results,
            experiment_name="test_experiment", 
            config=sample_config
        )
        
        # Test strategy filtering
        combined_results = results.get_results_by_strategy(ModelingStrategy.COMBINED)
        assert len(combined_results) == 1
        assert combined_results[0].modeling_strategy == ModelingStrategy.COMBINED
        
        individual_results = results.get_results_by_strategy(ModelingStrategy.INDIVIDUAL)
        assert len(individual_results) == 1
        assert individual_results[0].modeling_strategy == ModelingStrategy.INDIVIDUAL
        
        # Test quantile filtering
        quantile_results = results.get_quantile_results()
        assert len(quantile_results) == 1
        assert quantile_results[0].is_quantile_model()

    def test_validation(self, sample_config):
        """Test ExperimentResults validation."""
        # Empty training results should fail
        with pytest.raises(ValueError, match="ExperimentResults must contain at least one training result"):
            ExperimentResults(
                training_results=[],
                experiment_name="test",
                config=sample_config
            )


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_sku_tuples(self):
        """Test SKU tuple validation."""
        # Valid SKU tuples should pass
        valid_skus = [(80558, 2), (80651, 5)]
        validate_sku_tuples(valid_skus)  # Should not raise
        
        # Empty list should fail
        with pytest.raises(ValueError, match="At least one SKU tuple must be provided"):
            validate_sku_tuples([])
        
        # Non-list should fail
        with pytest.raises(TypeError, match="sku_tuples must be a list"):
            validate_sku_tuples("invalid")
        
        # Invalid tuple should fail
        with pytest.raises(ValueError, match="SKU tuple .* must be a 2-tuple"):
            validate_sku_tuples([(80558,)])  # Missing store_id
        
        # Non-integer values should fail
        with pytest.raises(ValueError, match="SKU tuple .* must contain integers"):
            validate_sku_tuples([(80558, "2")])  # String store_id
        
        # Non-positive values should fail
        with pytest.raises(ValueError, match="SKU tuple .* must contain positive integers"):
            validate_sku_tuples([(80558, -2)])  # Negative store_id

    def test_validate_modeling_strategy(self):
        """Test modeling strategy validation."""
        valid_skus = [(80558, 2)]
        
        # Valid combinations should pass
        validate_modeling_strategy(ModelingStrategy.COMBINED, valid_skus)
        validate_modeling_strategy(ModelingStrategy.INDIVIDUAL, valid_skus)
        
        # Should not fail for empty SKUs (validation happens elsewhere)
        # This test just ensures the function doesn't crash on edge cases