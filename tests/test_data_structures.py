"""
Unit tests for data structures and configuration classes.
Tests enums, dataclasses, and configuration validation.
"""

import pytest
import json
import pickle
import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src import (
    ModelingStrategy, SkuTuple, SkuList, ModelMetadata, DataSplit, 
    BenchmarkModel, ModelRegistry, DataConfig
)


@pytest.fixture
def sample_model_metadata():
    """Create sample ModelMetadata for testing."""
    return ModelMetadata(
        model_id="test_model_003",
        modeling_strategy=ModelingStrategy.COMBINED,
        sku_tuples=[(80558, 2)],
        model_type="xgboost",
        hyperparameters={"n_estimators": 100},
        training_config={"seed": 42},
        performance_metrics={"mse": 0.8},
        feature_columns=["feature1", "feature2"],
        target_column="target"
    )


@pytest.fixture
def sample_data_split():
    """Create sample DataSplit for testing."""
    return DataSplit(
        train_bdIDs=np.array([1001, 1002]),
        validation_bdIDs=np.array([1003, 1004])
    )


class TestModelingStrategy:
    """Test ModelingStrategy enum."""

    def test_modeling_strategy_values(self):
        """Test ModelingStrategy enum values."""
        assert ModelingStrategy.COMBINED.value == "combined"
        assert ModelingStrategy.INDIVIDUAL.value == "individual"

    def test_modeling_strategy_iteration(self):
        """Test iterating over ModelingStrategy enum."""
        strategies = list(ModelingStrategy)
        assert len(strategies) == 2
        assert ModelingStrategy.COMBINED in strategies
        assert ModelingStrategy.INDIVIDUAL in strategies

    def test_modeling_strategy_comparison(self):
        """Test ModelingStrategy comparisons."""
        assert ModelingStrategy.COMBINED == ModelingStrategy.COMBINED
        assert ModelingStrategy.COMBINED != ModelingStrategy.INDIVIDUAL


class TestTypeAliases:
    """Test type aliases for SKU handling."""

    def test_sku_tuple_type(self):
        """Test SkuTuple type alias."""
        # This is more of a documentation test
        sku: SkuTuple = (80558, 2)
        assert isinstance(sku, tuple)
        assert len(sku) == 2
        assert isinstance(sku[0], int)
        assert isinstance(sku[1], int)

    def test_sku_list_type(self):
        """Test SkuList type alias."""
        sku_list: SkuList = [(80558, 2), (80651, 5)]
        assert isinstance(sku_list, list)
        assert len(sku_list) == 2
        for sku in sku_list:
            assert isinstance(sku, tuple)
            assert len(sku) == 2


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_model_metadata_creation(self):
        """Test creating ModelMetadata instance."""
        metadata = ModelMetadata(
            model_id="test_model_001",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2)],
            model_type="xgboost",
            hyperparameters={"n_estimators": 100, "max_depth": 6},
            training_config={"seed": 42, "validation_split": 0.2},
            performance_metrics={"mse": 0.5, "rmse": 0.707, "mae": 0.4},
            feature_columns=["feature1", "feature2", "feature3"],
            target_column="target"
        )
        
        assert metadata.model_id == "test_model_001"
        assert metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert metadata.sku_tuples == [(80558, 2)]
        assert metadata.model_type == "xgboost"
        assert metadata.hyperparameters["n_estimators"] == 100
        assert metadata.performance_metrics["mse"] == 0.5
        assert len(metadata.feature_columns) == 3

    def test_model_metadata_serialization(self):
        """Test ModelMetadata JSON serialization."""
        metadata = ModelMetadata(
            model_id="test_model_002",
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            sku_tuples=[(80558, 2), (80651, 5)],
            model_type="xgboost",
            hyperparameters={"n_estimators": 50},
            training_config={"seed": 42},
            performance_metrics={"mse": 1.2},
            feature_columns=["feature1"],
            target_column="target"
        )
        
        # Convert to dict for JSON serialization
        metadata_dict = {
            "model_id": metadata.model_id,
            "modeling_strategy": metadata.modeling_strategy.value,
            "sku_tuples": metadata.sku_tuples,
            "model_type": metadata.model_type,
            "hyperparameters": metadata.hyperparameters,
            "training_config": metadata.training_config,
            "performance_metrics": metadata.performance_metrics,
            "feature_columns": metadata.feature_columns,
            "target_column": metadata.target_column
        }
        
        # Should be JSON serializable
        json_str = json.dumps(metadata_dict)
        loaded_dict = json.loads(json_str)
        
        assert loaded_dict["model_id"] == "test_model_002"
        assert loaded_dict["modeling_strategy"] == "individual"
        assert loaded_dict["sku_tuples"] == [[80558, 2], [80651, 5]]


class TestDataSplit:
    """Test DataSplit dataclass."""

    def test_data_split_creation(self):
        """Test creating DataSplit instance."""
        train_bdids = np.array([1001, 1002, 1003, 1004])
        val_bdids = np.array([1005, 1006])
        test_bdids = np.array([1007, 1008])
        
        data_split = DataSplit(
            train_bdIDs=train_bdids,
            validation_bdIDs=val_bdids,
            test_bdIDs=test_bdids,
            split_date="2020-02-01"
        )
        
        assert np.array_equal(data_split.train_bdIDs, train_bdids)
        assert np.array_equal(data_split.validation_bdIDs, val_bdids)
        assert np.array_equal(data_split.test_bdIDs, test_bdids)
        assert data_split.split_date == "2020-02-01"

    def test_data_split_optional_fields(self):
        """Test DataSplit with optional fields."""
        train_bdids = np.array([1001, 1002])
        val_bdids = np.array([1003, 1004])
        
        # Create with minimal required fields
        data_split = DataSplit(
            train_bdIDs=train_bdids,
            validation_bdIDs=val_bdids
        )
        
        assert np.array_equal(data_split.train_bdIDs, train_bdids)
        assert np.array_equal(data_split.validation_bdIDs, val_bdids)
        assert data_split.test_bdIDs is None
        assert data_split.split_date is None

    def test_data_split_serialization(self):
        """Test DataSplit serialization compatibility."""
        train_bdids = np.array([1001, 1002])
        val_bdids = np.array([1003, 1004])
        
        data_split = DataSplit(
            train_bdIDs=train_bdids,
            validation_bdIDs=val_bdids,
            split_date="2020-01-15"
        )
        
        # Should be pickle-able (for model persistence)
        pickled = pickle.dumps(data_split)
        loaded_split = pickle.loads(pickled)
        
        assert np.array_equal(loaded_split.train_bdIDs, train_bdids)
        assert np.array_equal(loaded_split.validation_bdIDs, val_bdids)
        assert loaded_split.split_date == "2020-01-15"


class TestBenchmarkModel:
    """Test BenchmarkModel dataclass."""

    def test_benchmark_model_creation(self, sample_model_metadata, sample_data_split):
        """Test creating BenchmarkModel instance."""
        # Mock model object
        mock_model = {"type": "mock_xgboost_model"}
        
        benchmark_model = BenchmarkModel(
            metadata=sample_model_metadata,
            model=mock_model,
            data_split=sample_data_split
        )
        
        assert benchmark_model.metadata == sample_model_metadata
        assert benchmark_model.model == mock_model
        assert benchmark_model.data_split == sample_data_split

    def test_get_identifier(self, sample_model_metadata, sample_data_split):
        """Test BenchmarkModel identifier generation."""
        mock_model = {"type": "mock"}
        
        # Test with existing model_id - should return the model_id
        benchmark_model = BenchmarkModel(
            metadata=sample_model_metadata,
            model=mock_model,
            data_split=sample_data_split
        )
        
        identifier = benchmark_model.get_identifier()
        assert isinstance(identifier, str)
        assert len(identifier) > 0
        # Should return the model_id from metadata
        assert identifier == "test_model_003"
        
        # Test with empty model_id - should generate legacy identifier
        metadata_without_id = ModelMetadata(
            model_id="",  # Empty model_id
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2)],
            model_type="xgboost",
            hyperparameters={"n_estimators": 100},
            training_config={"seed": 42},
            performance_metrics={"mse": 0.8},
            feature_columns=["feature1", "feature2"],
            target_column="target"
        )
        
        benchmark_model_legacy = BenchmarkModel(
            metadata=metadata_without_id,
            model=mock_model,
            data_split=sample_data_split
        )
        
        legacy_identifier = benchmark_model_legacy.get_identifier()
        assert isinstance(legacy_identifier, str)
        assert len(legacy_identifier) > 0
        # Should be based on strategy, SKU tuples, and model type
        assert "combined" in legacy_identifier  # modeling strategy
        assert "80558x2" in legacy_identifier   # SKU tuple format
        assert "xgboost" in legacy_identifier   # model type

    def test_benchmark_model_with_different_strategies(self, sample_data_split):
        """Test BenchmarkModel with different modeling strategies."""
        mock_model = {"type": "mock"}
        
        # Test INDIVIDUAL strategy
        individual_metadata = ModelMetadata(
            model_id="individual_model",
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            sku_tuples=[(80558, 2)],
            model_type="xgboost",
            hyperparameters={},
            training_config={},
            performance_metrics={},
            feature_columns=[],
            target_column="target"
        )
        
        individual_model = BenchmarkModel(
            metadata=individual_metadata,
            model=mock_model,
            data_split=sample_data_split
        )
        
        assert individual_model.metadata.modeling_strategy == ModelingStrategy.INDIVIDUAL
        
        # Test COMBINED strategy
        combined_metadata = ModelMetadata(
            model_id="combined_model",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2), (80651, 5)],
            model_type="xgboost",
            hyperparameters={},
            training_config={},
            performance_metrics={},
            feature_columns=[],
            target_column="target"
        )
        
        combined_model = BenchmarkModel(
            metadata=combined_metadata,
            model=mock_model,
            data_split=sample_data_split
        )
        
        assert combined_model.metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert len(combined_model.metadata.sku_tuples) == 2


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_model_registry_initialization(self):
        """Test ModelRegistry initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "models"
            registry = ModelRegistry(storage_path)
            
            assert registry.storage_path == storage_path
            assert registry.storage_path.exists()

    def test_model_registry_default_path(self):
        """Test ModelRegistry with default storage path."""
        registry = ModelRegistry()
        assert registry.storage_path is not None
        assert isinstance(registry.storage_path, Path)

    def test_register_and_get_model(self, sample_model_metadata, sample_data_split):
        """Test registering and retrieving models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "models"
            registry = ModelRegistry(storage_path)
            
            mock_model = {"type": "mock"}
            benchmark_model = BenchmarkModel(
                metadata=sample_model_metadata,
                model=mock_model,
                data_split=sample_data_split
            )
            
            # Register model
            model_id = registry.register_model(benchmark_model)
            assert isinstance(model_id, str)
            assert len(model_id) > 0
            
            # Retrieve model
            retrieved_model = registry.get_model(model_id)
            assert retrieved_model is not None
            assert retrieved_model.metadata.model_id == benchmark_model.metadata.model_id

    def test_list_models(self, sample_data_split):
        """Test listing models in registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "models"
            registry = ModelRegistry(storage_path)
            
            # Initially empty
            models = registry.list_models()
            assert isinstance(models, list)
            assert len(models) == 0
            
            # Add a model
            metadata = ModelMetadata(
                model_id="list_test_model",
                modeling_strategy=ModelingStrategy.COMBINED,
                sku_tuples=[(80558, 2)],
                model_type="xgboost",
                hyperparameters={},
                training_config={},
                performance_metrics={},
                feature_columns=[],
                target_column="target"
            )
            
            benchmark_model = BenchmarkModel(
                metadata=metadata,
                model={},
                data_split=sample_data_split
            )
            
            model_id = registry.register_model(benchmark_model)
            
            # Should now have one model
            models = registry.list_models()
            assert len(models) == 1
            assert model_id in models

    def test_list_models_by_strategy(self, sample_data_split):
        """Test listing models filtered by strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "models"
            registry = ModelRegistry(storage_path)
            
            # Add models with different strategies
            combined_metadata = ModelMetadata(
                model_id="combined_test",
                modeling_strategy=ModelingStrategy.COMBINED,
                sku_tuples=[(80558, 2)],
                model_type="xgboost",
                hyperparameters={}, training_config={}, performance_metrics={},
                feature_columns=[], target_column="target"
            )
            
            individual_metadata = ModelMetadata(
                model_id="individual_test",
                modeling_strategy=ModelingStrategy.INDIVIDUAL,
                sku_tuples=[(80558, 2)],
                model_type="xgboost",
                hyperparameters={}, training_config={}, performance_metrics={},
                feature_columns=[], target_column="target"
            )
            
            combined_model = BenchmarkModel(combined_metadata, {}, sample_data_split)
            individual_model = BenchmarkModel(individual_metadata, {}, sample_data_split)
            
            combined_id = registry.register_model(combined_model)
            individual_id = registry.register_model(individual_model)
            
            # Test filtering by strategy
            combined_models = registry.list_models(ModelingStrategy.COMBINED)
            individual_models = registry.list_models(ModelingStrategy.INDIVIDUAL)
            
            assert len(combined_models) >= 1
            assert len(individual_models) >= 1
            assert combined_id in combined_models
            assert individual_id in individual_models

    def test_model_not_found(self):
        """Test retrieving non-existent model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "models"
            registry = ModelRegistry(storage_path)
            
            # Try to get non-existent model
            model = registry.get_model("non_existent_model")
            assert model is None


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_data_config_creation(self):
        """Test creating DataConfig instance."""
        config = DataConfig(
            features_path="/path/to/features.feather",
            target_path="/path/to/target.feather",
            mapping_path="/path/to/mapping.pkl",
            date_column="date",
            target_column="target",
            bdid_column="bdID",
            remove_not_for_sale=True,
            split_date="2020-02-01"
        )
        
        assert config.features_path == "/path/to/features.feather"
        assert config.target_path == "/path/to/target.feather"
        assert config.mapping_path == "/path/to/mapping.pkl"
        assert config.date_column == "date"
        assert config.target_column == "target"
        assert config.bdid_column == "bdID"
        assert config.remove_not_for_sale == True
        assert config.split_date == "2020-02-01"

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig(
            features_path="/path/to/features.feather",
            target_path="/path/to/target.feather",
            mapping_path="/path/to/mapping.pkl"
        )
        
        # Check defaults
        assert config.date_column == "date"
        assert config.target_column == "target"
        assert config.bdid_column == "bdID"
        assert config.remove_not_for_sale == True
        assert config.min_date is None
        assert config.max_date is None
        assert config.split_date is None

