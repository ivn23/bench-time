"""
Tests for public API and imports.
Ensures all public components can be imported and instantiated correctly.
"""

import pytest


class TestPublicAPIImports:
    """Test that all public API components can be imported."""

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
        from src import ModelMetadata, DataSplit, BenchmarkModel, ModelRegistry
        
        assert ModelMetadata is not None
        assert DataSplit is not None
        assert BenchmarkModel is not None
        assert ModelRegistry is not None
        
        # Check they're actually classes
        assert callable(ModelMetadata)
        assert callable(DataSplit)
        assert callable(BenchmarkModel)
        assert callable(ModelRegistry)

    def test_import_config_classes(self):
        """Test importing configuration classes."""
        from src import DataConfig, TrainingConfig
        
        assert DataConfig is not None
        assert TrainingConfig is not None
        assert callable(DataConfig)
        assert callable(TrainingConfig)

    def test_import_core_components(self):
        """Test importing core framework components."""
        from src import DataLoader, ModelTrainer, ModelEvaluator, VisualizationGenerator
        
        assert DataLoader is not None
        assert ModelTrainer is not None
        assert ModelEvaluator is not None
        assert VisualizationGenerator is not None
        
        assert callable(DataLoader)
        assert callable(ModelTrainer)
        assert callable(ModelEvaluator)
        assert callable(VisualizationGenerator)

    def test_import_main_pipeline(self):
        """Test importing main pipeline class."""
        from src import BenchmarkPipeline
        
        assert BenchmarkPipeline is not None
        assert callable(BenchmarkPipeline)

    def test_import_all_at_once(self):
        """Test importing all public components at once."""
        from src import (
            ModelingStrategy, SkuTuple, SkuList, ModelMetadata, DataSplit, 
            BenchmarkModel, ModelRegistry, DataConfig, TrainingConfig,
            DataLoader, ModelTrainer, ModelEvaluator, VisualizationGenerator,
            BenchmarkPipeline
        )
        
        # All should be importable without error
        components = [
            ModelingStrategy, SkuTuple, SkuList, ModelMetadata, DataSplit,
            BenchmarkModel, ModelRegistry, DataConfig, TrainingConfig,
            DataLoader, ModelTrainer, ModelEvaluator, VisualizationGenerator,
            BenchmarkPipeline
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
            'ModelingStrategy', 'SkuTuple', 'SkuList', 'ModelMetadata', 
            'DataSplit', 'BenchmarkModel', 'ModelRegistry', 'DataConfig', 
            'TrainingConfig', 'DataLoader', 'ModelTrainer', 'ModelEvaluator',
            'VisualizationGenerator', 'BenchmarkPipeline'
        ]
        
        for export in expected_exports:
            assert export in src.__all__


class TestBasicInstantiation:
    """Test basic instantiation of main classes."""

    def test_modeling_strategy_usage(self):
        """Test ModelingStrategy can be used correctly."""
        from src import ModelingStrategy
        
        combined = ModelingStrategy.COMBINED
        individual = ModelingStrategy.INDIVIDUAL
        
        assert combined.value == "combined"
        assert individual.value == "individual"
        assert combined != individual

    def test_config_instantiation(self):
        """Test configuration classes can be instantiated."""
        from src import DataConfig, TrainingConfig
        
        # Test DataConfig with minimal required parameters
        data_config = DataConfig(
            features_path="/path/to/features.feather",
            target_path="/path/to/target.feather", 
            mapping_path="/path/to/mapping.pkl"
        )
        assert data_config.features_path == "/path/to/features.feather"
        
        # Test TrainingConfig with defaults
        training_config = TrainingConfig()
        assert training_config.model_type == "xgboost"
        assert training_config.random_state == 42

    def test_model_registry_instantiation(self):
        """Test ModelRegistry can be instantiated."""
        from src import ModelRegistry
        
        # Should work with default path
        registry = ModelRegistry()
        assert registry is not None
        assert hasattr(registry, 'storage_path')
        assert hasattr(registry, 'register_model')
        assert hasattr(registry, 'get_model')
        assert hasattr(registry, 'list_models')

    def test_data_structures_instantiation(self):
        """Test data structure classes can be instantiated."""
        from src import ModelMetadata, DataSplit, ModelingStrategy
        import numpy as np
        
        # Test ModelMetadata
        metadata = ModelMetadata(
            model_id="test_id",
            modeling_strategy=ModelingStrategy.COMBINED,
            sku_tuples=[(80558, 2)],
            model_type="xgboost",
            hyperparameters={},
            training_config={},
            performance_metrics={},
            feature_columns=[],
            target_column="target"
        )
        assert metadata.model_id == "test_id"
        
        # Test DataSplit
        data_split = DataSplit(
            train_bdIDs=np.array([1, 2, 3]),
            validation_bdIDs=np.array([4, 5])
        )
        assert len(data_split.train_bdIDs) == 3

    def test_component_instantiation_with_configs(self, temp_output_dir):
        """Test core components can be instantiated with configs."""
        from src import DataLoader, ModelTrainer, BenchmarkPipeline, DataConfig, TrainingConfig
        
        # Create test configs
        data_config = DataConfig(
            features_path=str(temp_output_dir / "features.feather"),
            target_path=str(temp_output_dir / "target.feather"),
            mapping_path=str(temp_output_dir / "mapping.pkl")
        )
        
        training_config = TrainingConfig()
        
        # Test instantiation (don't try to use them, just create them)
        data_loader = DataLoader(data_config)
        assert data_loader is not None
        assert hasattr(data_loader, 'load_data')
        
        model_trainer = ModelTrainer(training_config)
        assert model_trainer is not None
        assert hasattr(model_trainer, 'train_model')
        
        pipeline = BenchmarkPipeline(data_config, training_config, temp_output_dir)
        assert pipeline is not None
        assert hasattr(pipeline, 'run_experiment')


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
        from src import BenchmarkPipeline, ModelRegistry, DataLoader, ModelTrainer
        
        # BenchmarkPipeline expected methods
        expected_pipeline_methods = [
            'load_and_prepare_data', 'run_experiment', 'evaluate_all_models',
            'save_evaluation_results', 'save_experiment_log'
        ]
        
        for method in expected_pipeline_methods:
            assert hasattr(BenchmarkPipeline, method)
        
        # ModelRegistry expected methods
        expected_registry_methods = [
            'register_model', 'get_model', 'list_models', 'save_model', 'load_model'
        ]
        
        for method in expected_registry_methods:
            assert hasattr(ModelRegistry, method)
        
        # DataLoader expected methods
        expected_loader_methods = [
            'load_data', 'get_data_for_tuples', 'prepare_features_for_modeling',
            'create_temporal_split_by_date'
        ]
        
        for method in expected_loader_methods:
            assert hasattr(DataLoader, method)
        
        # ModelTrainer expected methods
        expected_trainer_methods = ['train_model']
        
        for method in expected_trainer_methods:
            assert hasattr(ModelTrainer, method)

    def test_config_classes_have_expected_attributes(self):
        """Test configuration classes have expected attributes."""
        from src import DataConfig, TrainingConfig
        
        # DataConfig expected attributes
        data_config = DataConfig(
            features_path="test", target_path="test", mapping_path="test"
        )
        
        expected_data_attrs = [
            'features_path', 'target_path', 'mapping_path', 'date_column',
            'target_column', 'bdid_column', 'remove_not_for_sale'
        ]
        
        for attr in expected_data_attrs:
            assert hasattr(data_config, attr)
        
        # TrainingConfig expected attributes
        training_config = TrainingConfig()
        
        expected_training_attrs = [
            'validation_split', 'random_state', 'cv_folds', 'model_type',
            'hyperparameters', 'model_params'
        ]
        
        for attr in expected_training_attrs:
            assert hasattr(training_config, attr)

    def test_dataclass_structure_consistency(self):
        """Test dataclass structures are consistent."""
        from src import ModelMetadata, DataSplit, BenchmarkModel, DataConfig, TrainingConfig
        import dataclasses
        
        # Check they are actually dataclasses
        assert dataclasses.is_dataclass(ModelMetadata)
        assert dataclasses.is_dataclass(DataSplit)
        assert dataclasses.is_dataclass(BenchmarkModel)
        assert dataclasses.is_dataclass(DataConfig)
        assert dataclasses.is_dataclass(TrainingConfig)
        
        # Check expected fields exist
        metadata_fields = [f.name for f in dataclasses.fields(ModelMetadata)]
        expected_metadata_fields = [
            'model_id', 'modeling_strategy', 'sku_tuples', 'model_type',
            'hyperparameters', 'training_config', 'performance_metrics',
            'feature_columns', 'target_column'
        ]
        
        for field in expected_metadata_fields:
            assert field in metadata_fields


class TestErrorHandling:
    """Test basic error handling in API."""

    def test_invalid_config_creation(self):
        """Test error handling for invalid configurations."""
        from src import DataConfig, TrainingConfig
        
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
        """Test that non-existent imports fail appropriately."""
        # Test importing something that doesn't exist
        with pytest.raises(ImportError):
            from src import NonExistentClass
        
        # Test importing from wrong module
        with pytest.raises(ImportError):
            from src.nonexistent_module import SomeClass