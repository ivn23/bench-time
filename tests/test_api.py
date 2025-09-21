"""
Tests for API and imports.
Ensures all public components can be imported and instantiated correctly.
"""

import pytest
import numpy as np
import polars as pl


class TestAPIImports:
    """Test that all API components can be imported."""

    def test_import_modeling_strategy(self):
        """Test importing ModelingStrategy enum."""
        from src import ModelingStrategy
        
        assert ModelingStrategy.COMBINED is not None
        assert ModelingStrategy.INDIVIDUAL is not None
        assert hasattr(ModelingStrategy, 'COMBINED')
        assert hasattr(ModelingStrategy, 'INDIVIDUAL')

    def test_import_type_aliases(self):
        """Test importing type aliases."""
        from src import SkuTuple, SkuList
        
        # These are type aliases, so we can't instantiate them directly
        # But we can check they're importable
        assert SkuTuple is not None
        assert SkuList is not None

    def test_import_data_structures(self):
        """Test importing data structure classes."""
        from src import (
            DataConfig, ModelingDataset, ModelConfig, 
            SplitInfo, TrainingResult, ExperimentResults
        )
        
        assert DataConfig is not None
        assert ModelingDataset is not None
        assert ModelConfig is not None
        assert SplitInfo is not None
        assert TrainingResult is not None
        assert ExperimentResults is not None
        
        # Check they're actually classes
        assert callable(DataConfig)
        assert callable(ModelingDataset)
        assert callable(ModelConfig)
        assert callable(SplitInfo)
        assert callable(TrainingResult)
        assert callable(ExperimentResults)

    def test_import_factory_functions(self):
        """Test importing factory and validation functions."""
        from src import create_config, validate_sku_tuples, validate_modeling_strategy
        
        assert create_config is not None
        assert validate_sku_tuples is not None
        assert validate_modeling_strategy is not None
        assert callable(create_config)
        assert callable(validate_sku_tuples)
        assert callable(validate_modeling_strategy)

    def test_import_core_components(self):
        """Test importing core framework components."""
        from src import DataLoader, ModelTrainer, ModelEvaluator
        
        assert DataLoader is not None
        assert ModelTrainer is not None
        assert ModelEvaluator is not None
        
        assert callable(DataLoader)
        assert callable(ModelTrainer)
        assert callable(ModelEvaluator)

    def test_import_main_pipeline(self):
        """Test importing pipeline class."""
        from src import BenchmarkPipeline
        
        assert BenchmarkPipeline is not None
        assert callable(BenchmarkPipeline)

    def test_import_release_management(self):
        """Test importing release management."""
        from src import ReleaseManager
        
        assert ReleaseManager is not None
        assert callable(ReleaseManager)

    def test_import_all_at_once(self):
        """Test importing all components at once."""
        from src import (
            ModelingStrategy, SkuTuple, SkuList, DataConfig, ModelingDataset,
            ModelConfig, SplitInfo, TrainingResult, ExperimentResults,
            create_config, validate_sku_tuples, validate_modeling_strategy,
            DataLoader, ModelTrainer, ModelEvaluator, BenchmarkPipeline,
            ReleaseManager
        )
        
        # All should be importable without error
        components = [
            ModelingStrategy, SkuTuple, SkuList, DataConfig, ModelingDataset,
            ModelConfig, SplitInfo, TrainingResult, ExperimentResults,
            create_config, validate_sku_tuples, validate_modeling_strategy,
            DataLoader, ModelTrainer, ModelEvaluator, BenchmarkPipeline,
            ReleaseManager
        ]
        
        for component in components:
            assert component is not None

    def test_package_metadata(self):
        """Test package metadata is accessible."""
        import src
        
        assert hasattr(src, '__version__')
        assert hasattr(src, '__author__')
        assert hasattr(src, '__all__')
        
        # Check __all__ contains expected exports
        expected_exports = [
            'ModelingStrategy', 'SkuTuple', 'SkuList', 'DataConfig', 'ModelingDataset',
            'ModelConfig', 'SplitInfo', 'TrainingResult', 'ExperimentResults',
            'create_config', 'validate_sku_tuples', 'validate_modeling_strategy',
            'DataLoader', 'ModelTrainer', 'ModelEvaluator', 'BenchmarkPipeline',
            'ReleaseManager'
        ]
        
        for export in expected_exports:
            assert export in src.__all__


class TestBasicInstantiation:
    """Test basic instantiation of classes."""

    def test_modeling_strategy_usage(self):
        """Test ModelingStrategy can be used correctly."""
        from src import ModelingStrategy
        
        combined = ModelingStrategy.COMBINED
        individual = ModelingStrategy.INDIVIDUAL
        
        assert combined.value == "combined"
        assert individual.value == "individual"
        assert combined != individual

    def test_data_config_instantiation(self):
        """Test DataConfig can be instantiated with required parameters."""
        from src import DataConfig
        
        # Test DataConfig with minimal required parameters
        data_config = DataConfig(
            features_path="/path/to/features.feather",
            target_path="/path/to/target.feather", 
            mapping_path="/path/to/mapping.pkl"
        )
        assert data_config.features_path == "/path/to/features.feather"
        assert data_config.target_path == "/path/to/target.feather"
        assert data_config.mapping_path == "/path/to/mapping.pkl"
        assert data_config.date_column == "date"
        assert data_config.target_column == "target"
        assert data_config.bdid_column == "bdID"

    def test_model_config_factory(self):
        """Test ModelConfig can be created via factory."""
        from src import create_config
        
        # Test basic config creation
        config = create_config("xgboost_standard", {"n_estimators": 100})
        assert config.model_type == "xgboost_standard"
        assert config.hyperparameters["n_estimators"] == 100
        assert config.quantile_alphas is None
        assert not config.is_quantile_model
        
        # Test quantile config creation
        quantile_config = create_config(
            "xgboost_quantile", 
            {"n_estimators": 50}, 
            quantile_alphas=[0.1, 0.5, 0.9]
        )
        assert quantile_config.is_quantile_model
        assert quantile_config.quantile_alphas == [0.1, 0.5, 0.9]

    def test_split_info_instantiation(self):
        """Test SplitInfo can be instantiated."""
        from src import SplitInfo
        
        split_info = SplitInfo(
            train_bdIDs=np.array([1, 2, 3]),
            validation_bdIDs=np.array([4, 5]),
            split_date="2023-01-01"
        )
        assert len(split_info.train_bdIDs) == 3
        assert len(split_info.validation_bdIDs) == 2
        assert split_info.split_date == "2023-01-01"

    def test_validation_functions(self):
        """Test validation functions work correctly."""
        from src import validate_sku_tuples, validate_modeling_strategy, ModelingStrategy
        
        # Test SKU tuple validation
        valid_skus = [(80558, 2), (80651, 3)]
        validate_sku_tuples(valid_skus)  # Should not raise
        
        # Test modeling strategy validation
        validate_modeling_strategy(ModelingStrategy.COMBINED, valid_skus)  # Should not raise
        validate_modeling_strategy(ModelingStrategy.INDIVIDUAL, valid_skus)  # Should not raise

    def test_component_instantiation_with_configs(self):
        """Test core components can be instantiated with configs."""
        from src import DataLoader, ModelTrainer, DataConfig, create_config
        
        # Create test configs
        data_config = DataConfig(
            features_path="/fake/features.feather",
            target_path="/fake/target.feather",
            mapping_path="/fake/mapping.pkl"
        )
        
        model_config = create_config("xgboost_standard", {"n_estimators": 10})
        
        # Test instantiation (don't try to use them, just create them)
        data_loader = DataLoader(data_config)
        assert data_loader is not None
        assert hasattr(data_loader, 'load_data')
        assert hasattr(data_loader, 'prepare_modeling_dataset')
        
        model_trainer = ModelTrainer(model_config)
        assert model_trainer is not None
        assert hasattr(model_trainer, 'train_model')


class TestAPIConsistency:
    """Test API consistency and expected interfaces."""

    def test_enum_consistency(self):
        """Test enum values are consistent."""
        from src import ModelingStrategy
        
        # Test string representations
        assert str(ModelingStrategy.COMBINED) == "ModelingStrategy.COMBINED"
        assert ModelingStrategy.COMBINED.name == "COMBINED"
        assert ModelingStrategy.COMBINED.value == "combined"

    def test_main_classes_have_expected_methods(self):
        """Test main classes have expected public methods."""
        from src import BenchmarkPipeline, DataLoader, ModelTrainer, ModelEvaluator
        
        # BenchmarkPipeline expected methods
        expected_pipeline_methods = ['run_experiment']
        
        for method in expected_pipeline_methods:
            assert hasattr(BenchmarkPipeline, method)
        
        # DataLoader expected methods
        expected_loader_methods = [
            'load_data', 'prepare_modeling_dataset', 'create_temporal_split',
            'create_temporal_split_by_date'
        ]
        
        for method in expected_loader_methods:
            assert hasattr(DataLoader, method)
        
        # ModelTrainer expected methods
        expected_trainer_methods = ['train_model']
        
        for method in expected_trainer_methods:
            assert hasattr(ModelTrainer, method)
        
        # ModelEvaluator expected methods
        expected_evaluator_methods = ['evaluate_training_result', 'evaluate_multiple_results']
        
        for method in expected_evaluator_methods:
            assert hasattr(ModelEvaluator, method)

    def test_config_classes_have_expected_attributes(self):
        """Test configuration classes have expected attributes."""
        from src import DataConfig, create_config
        
        # DataConfig expected attributes
        data_config = DataConfig(
            features_path="test", target_path="test", mapping_path="test"
        )
        
        expected_data_attrs = [
            'features_path', 'target_path', 'mapping_path', 'date_column',
            'target_column', 'bdid_column', 'validation_split'
        ]
        
        for attr in expected_data_attrs:
            assert hasattr(data_config, attr)
        
        # ModelConfig expected attributes
        model_config = create_config("xgboost_standard", {})
        
        expected_model_attrs = [
            'model_type', 'hyperparameters', 'quantile_alphas', 'random_state'
        ]
        
        for attr in expected_model_attrs:
            assert hasattr(model_config, attr)

    def test_dataclass_structure_consistency(self):
        """Test dataclass structures are consistent."""
        from src import (
            DataConfig, ModelingDataset, ModelConfig, 
            SplitInfo, TrainingResult, ExperimentResults
        )
        import dataclasses
        
        # Check they are actually dataclasses
        assert dataclasses.is_dataclass(DataConfig)
        assert dataclasses.is_dataclass(ModelingDataset)
        assert dataclasses.is_dataclass(ModelConfig)
        assert dataclasses.is_dataclass(SplitInfo)
        assert dataclasses.is_dataclass(TrainingResult)
        assert dataclasses.is_dataclass(ExperimentResults)
        
        # Check expected fields exist
        data_config_fields = [f.name for f in dataclasses.fields(DataConfig)]
        expected_data_config_fields = [
            'features_path', 'target_path', 'mapping_path', 'date_column',
            'target_column', 'bdid_column', 'validation_split'
        ]
        
        for field in expected_data_config_fields:
            assert field in data_config_fields


class TestErrorHandling:
    """Test basic error handling in API."""

    def test_invalid_config_creation(self):
        """Test error handling for invalid configurations."""
        from src import DataConfig
        
        # DataConfig requires certain parameters
        with pytest.raises(TypeError):
            DataConfig()  # Missing required parameters
        
        # Should work with required parameters
        data_config = DataConfig(
            features_path="test",
            target_path="test", 
            mapping_path="test"
        )
        assert data_config is not None

    def test_invalid_sku_validation(self):
        """Test SKU validation with invalid inputs."""
        from src import validate_sku_tuples
        
        # Invalid SKU formats should raise ValueError
        with pytest.raises(ValueError):
            validate_sku_tuples([])  # Empty list
        
        with pytest.raises(ValueError):
            validate_sku_tuples([(1, 2, 3)])  # Too many elements
        
        with pytest.raises(ValueError):
            validate_sku_tuples([(1,)])  # Too few elements

    def test_invalid_enum_usage(self):
        """Test ModelingStrategy enum usage."""
        from src import ModelingStrategy
        
        # Valid usage
        strategy = ModelingStrategy.COMBINED
        assert strategy == ModelingStrategy.COMBINED
        
        # Test comparison with strings (should not be equal)
        assert strategy != "combined"  # Enum != string
        assert strategy.value == "combined"  # But value should match

    def test_import_errors(self):
        """Test that legacy imports fail appropriately."""
        # Test importing legacy classes that should no longer exist
        with pytest.raises(ImportError):
            from src import BenchmarkModel
        
        with pytest.raises(ImportError):
            from src import TrainingConfig
        
        with pytest.raises(ImportError):
            from src import ModelRegistry