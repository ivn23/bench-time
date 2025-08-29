"""
Test multi-quantile configuration and training functionality.

Tests the enhanced framework capabilities for training multiple quantile models
with a single configuration and quantile-aware hierarchical storage.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List

from src import (
    DataConfig, TrainingConfig, ModelingStrategy, BenchmarkPipeline,
    ModelTypeConfig, ModelMetadata, BenchmarkModel
)
from src.storage_utils import ModelStorageLocation, HierarchicalStorageManager
from src.model_types import model_registry


class TestMultiQuantileConfiguration:
    """Test multi-quantile configuration structures."""
    
    def test_model_type_config_quantile_alphas(self):
        """Test ModelTypeConfig with multiple quantile levels."""
        # Test multi-quantile configuration
        config = ModelTypeConfig(
            model_type="xgboost_quantile",
            quantile_alphas=[0.1, 0.5, 0.9],
            hyperparameters={"n_estimators": 100}
        )
        
        assert config.quantile_alphas == [0.1, 0.5, 0.9]
        assert config.quantile_alpha is None
        assert config.effective_quantile_alphas == [0.1, 0.5, 0.9]
        assert config.is_quantile_model is True
    
    def test_model_type_config_backward_compatibility(self):
        """Test backward compatibility with single quantile_alpha."""
        config = ModelTypeConfig(
            model_type="xgboost_quantile",
            quantile_alpha=0.7,
            hyperparameters={"n_estimators": 100}
        )
        
        assert config.quantile_alpha == 0.7
        assert config.quantile_alphas is None
        assert config.effective_quantile_alphas == [0.7]
        assert config.is_quantile_model is True
    
    def test_model_type_config_validation(self):
        """Test validation of quantile ranges."""
        # Test invalid quantile values
        with pytest.raises(ValueError, match="quantile_alpha must be between 0 and 1"):
            ModelTypeConfig(model_type="xgboost_quantile", quantile_alphas=[0.1, 1.5, 0.9])
        
        with pytest.raises(ValueError, match="quantile_alpha must be between 0 and 1"):
            ModelTypeConfig(model_type="xgboost_quantile", quantile_alpha=-0.1)
    
    def test_model_type_config_conflict_validation(self):
        """Test validation of conflicting quantile parameters."""
        with pytest.raises(ValueError, match="Cannot specify both quantile_alpha and quantile_alphas"):
            ModelTypeConfig(
                model_type="xgboost_quantile",
                quantile_alpha=0.5,
                quantile_alphas=[0.1, 0.9]
            )
    
    def test_training_config_multi_quantile(self):
        """Test TrainingConfig with multi-quantile model configuration."""
        training_config = TrainingConfig()
        
        # Add multi-quantile configuration
        training_config.add_model_config(
            model_type="xgboost_quantile",
            quantile_alphas=[0.1, 0.5, 0.9],
            hyperparameters={"n_estimators": 100}
        )
        
        model_config = training_config.get_model_config("xgboost_quantile")
        assert model_config.effective_quantile_alphas == [0.1, 0.5, 0.9]
        assert model_config.is_quantile_model is True
    
    def test_training_config_validation_error(self):
        """Test TrainingConfig validation with conflicting quantile parameters."""
        training_config = TrainingConfig()
        
        with pytest.raises(ValueError, match="Cannot specify both quantile_alpha and quantile_alphas"):
            training_config.add_model_config(
                model_type="xgboost_quantile",
                quantile_alpha=0.5,
                quantile_alphas=[0.1, 0.9]
            )


class TestQuantileAwareStorage:
    """Test quantile-aware hierarchical storage."""
    
    def test_model_storage_location_quantile(self):
        """Test ModelStorageLocation with quantile level."""
        location = ModelStorageLocation(
            store_id=1,
            product_id=2,
            model_type="xgboost_quantile",
            model_instance="default",
            quantile_level=0.7
        )
        
        assert location.quantile_level == 0.7
        components = location.to_path_components()
        assert len(components) == 5
        assert components == ("1", "2", "xgboost_quantile", "q0.7", "default")
    
    def test_model_storage_location_no_quantile(self):
        """Test ModelStorageLocation without quantile level."""
        location = ModelStorageLocation(
            store_id=1,
            product_id=2,
            model_type="xgboost_standard",
            model_instance="default"
        )
        
        assert location.quantile_level is None
        components = location.to_path_components()
        assert components == ("1", "2", "xgboost_standard", "standard", "default")
    
    def test_quantile_formatting(self):
        """Test quantile level formatting for directory names."""
        # Test various quantile levels
        test_cases = [
            (0.1, "q0.1"),
            (0.05, "q0.05"),
            (0.5, "q0.5"),
            (0.95, "q0.95"),
            (0.001, "q0.001")
        ]
        
        for quantile, expected in test_cases:
            location = ModelStorageLocation(1, 2, "test", quantile_level=quantile)
            components = location.to_path_components()
            assert components[3] == expected
    
    def test_from_sku_tuple_with_quantile(self):
        """Test creation from SKU tuple with quantile level."""
        sku_tuple = (123, 456)
        location = ModelStorageLocation.from_sku_tuple_with_quantile(
            sku_tuple, "xgboost_quantile", quantile_level=0.7
        )
        
        assert location.store_id == 456
        assert location.product_id == 123
        assert location.model_type == "xgboost_quantile"
        assert location.quantile_level == 0.7
    
    def test_quantile_validation(self):
        """Test quantile level validation in storage location."""
        with pytest.raises(ValueError, match="quantile_level must be between 0 and 1"):
            ModelStorageLocation(1, 2, "test", quantile_level=1.5)
        
        with pytest.raises(ValueError, match="quantile_level must be between 0 and 1"):
            ModelStorageLocation(1, 2, "test", quantile_level=-0.1)


class TestMultiQuantileTraining:
    """Test multi-quantile training workflows."""
    
    def test_model_trainer_returns_list(self, prepared_model_data, sample_feature_columns, tmp_path):
        """Test that ModelTrainer returns list of models for multi-quantile."""
        X, y, feature_cols = prepared_model_data
        
        # Create training config with multi-quantile
        training_config = TrainingConfig(random_state=42)
        training_config.add_model_config(
            model_type="xgboost_quantile",
            quantile_alphas=[0.1, 0.9],
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )
        
        from src.model_training import ModelTrainer
        
        # Create simple train/test split
        n_samples = len(X)
        train_size = int(0.8 * n_samples)
        train_bdids = X.select("bdID").to_numpy()[:train_size].flatten()
        test_bdids = X.select("bdID").to_numpy()[train_size:].flatten()
        
        X_train = X.filter(X["bdID"].is_in(train_bdids))
        y_train = y.filter(y["bdID"].is_in(train_bdids))
        X_test = X.filter(X["bdID"].is_in(test_bdids))
        y_test = y.filter(y["bdID"].is_in(test_bdids))
        
        # Train models
        trainer = ModelTrainer(training_config)
        models = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target",
            ModelingStrategy.INDIVIDUAL, [(80558, 2)], "xgboost_quantile"
        )
        
        # Check results
        assert isinstance(models, list)
        assert len(models) == 2  # Two quantile levels
        
        # Check quantile levels
        quantile_levels = [model.metadata.quantile_level for model in models]
        assert 0.1 in quantile_levels
        assert 0.9 in quantile_levels
        
        # Check model IDs contain quantile info
        for model in models:
            assert "q" in model.metadata.model_id
            if model.metadata.quantile_level == 0.1:
                assert "q0.1" in model.metadata.model_id
            elif model.metadata.quantile_level == 0.9:
                assert "q0.9" in model.metadata.model_id
    
    def test_single_quantile_backward_compatibility(self, prepared_model_data):
        """Test backward compatibility with single quantile configuration."""
        X, y, feature_cols = prepared_model_data
        
        # Create training config with single quantile (old way)
        training_config = TrainingConfig(random_state=42)
        training_config.add_model_config(
            model_type="xgboost_quantile",
            quantile_alpha=0.7,
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )
        
        from src.model_training import ModelTrainer
        
        # Create simple train/test split
        n_samples = len(X)
        train_size = int(0.8 * n_samples)
        train_bdids = X.select("bdID").to_numpy()[:train_size].flatten()
        test_bdids = X.select("bdID").to_numpy()[train_size:].flatten()
        
        X_train = X.filter(X["bdID"].is_in(train_bdids))
        y_train = y.filter(y["bdID"].is_in(train_bdids))
        X_test = X.filter(X["bdID"].is_in(test_bdids))
        y_test = y.filter(y["bdID"].is_in(test_bdids))
        
        # Train models
        trainer = ModelTrainer(training_config)
        models = trainer.train_model(
            X_train, y_train, X_test, y_test,
            feature_cols, "target",
            ModelingStrategy.INDIVIDUAL, [(80558, 2)], "xgboost_quantile"
        )
        
        # Check results
        assert isinstance(models, list)
        assert len(models) == 1  # Single quantile level
        assert models[0].metadata.quantile_level == 0.7
        assert "q0.7" in models[0].metadata.model_id


class TestMultiQuantileIntegration:
    """Test complete multi-quantile workflow integration."""
    
    def test_complete_multi_quantile_workflow(self, temp_data_dir, tmp_path):
        """Test complete multi-quantile workflow from config to storage."""
        
        # Create data configuration from temporary data
        data_config = DataConfig(
            features_path=str(temp_data_dir["features_path"]),
            target_path=str(temp_data_dir["target_path"]),
            mapping_path=str(temp_data_dir["mapping_path"]),
            validation_split=0.2
        )
        
        # Create training config with multi-quantile
        training_config = TrainingConfig(random_state=42)
        training_config.add_model_config(
            model_type="xgboost_quantile",
            quantile_alphas=[0.1, 0.5, 0.9],
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )
        
        # Initialize pipeline
        results_dir = tmp_path / "results"
        pipeline = BenchmarkPipeline(
            data_config, training_config, output_dir=results_dir
        )
        pipeline.load_and_prepare_data()
        
        # Define test SKUs
        sku_tuples = [(80558, 2), (80651, 5)]
        
        # Run experiment with INDIVIDUAL strategy
        models = pipeline.run_experiment(
            sku_tuples=sku_tuples,
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            experiment_name="multi_quantile_test"
        )
        
        # Validate results
        assert len(models) == 6  # 2 SKUs × 3 quantile levels
        
        # Check quantile levels are represented
        quantile_levels = set(model.metadata.quantile_level for model in models)
        assert quantile_levels == {0.1, 0.5, 0.9}
        
        # Check SKU coverage
        sku_coverage = set(model.metadata.sku_tuples[0] for model in models)
        assert sku_coverage == {(80558, 2), (80651, 5)}
        
        # Check model IDs
        for model in models:
            assert model.metadata.quantile_level is not None
            assert f"q{model.metadata.quantile_level}" in model.metadata.model_id
            assert "individual" in model.metadata.model_id
        
        # Check storage hierarchy
        for model in models:
            storage_location = model.get_storage_location()
            assert storage_location.quantile_level == model.metadata.quantile_level
            
            # Verify storage path structure
            components = storage_location.to_path_components()
            assert len(components) == 5
            assert components[3].startswith("q")  # Quantile component
        
        # Verify models are saved to correct directories
        models_dir = results_dir / "models"
        assert models_dir.exists()
        
        # Check directory structure includes quantile levels
        for model in models:
            location = model.get_storage_location()
            expected_path = pipeline.model_registry.storage_manager.create_model_path(location)
            assert expected_path.exists()
    
    def test_combined_strategy_multi_quantile(self, temp_data_dir, tmp_path):
        """Test multi-quantile with COMBINED modeling strategy."""
        
        # Create configs from temp data
        data_config = DataConfig(
            features_path=str(temp_data_dir["features_path"]),
            target_path=str(temp_data_dir["target_path"]),
            mapping_path=str(temp_data_dir["mapping_path"])
        )
        
        training_config = TrainingConfig(random_state=42)
        training_config.add_model_config(
            model_type="xgboost_quantile",
            quantile_alphas=[0.2, 0.8],
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )
        
        # Run pipeline
        pipeline = BenchmarkPipeline(data_config, training_config, tmp_path / "results")
        pipeline.load_and_prepare_data()
        
        sku_tuples = [(80558, 2), (80651, 5)]
        models = pipeline.run_experiment(
            sku_tuples=sku_tuples,
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="combined_multi_quantile"
        )
        
        # Validate results  
        assert len(models) == 2  # 2 quantile levels for combined strategy
        
        quantile_levels = {model.metadata.quantile_level for model in models}
        assert quantile_levels == {0.2, 0.8}
        
        # All models should cover both SKUs
        for model in models:
            assert len(model.metadata.sku_tuples) == 2
            assert set(model.metadata.sku_tuples) == {(80558, 2), (80651, 5)}
            assert "combined" in model.metadata.model_id
    
    def test_non_quantile_model_storage(self, temp_data_dir, tmp_path):
        """Test that non-quantile models get 'standard' quantile directory."""
        
        # Create config for standard (non-quantile) model from temp data
        data_config = DataConfig(
            features_path=str(temp_data_dir["features_path"]),
            target_path=str(temp_data_dir["target_path"]),
            mapping_path=str(temp_data_dir["mapping_path"])
        )
        
        training_config = TrainingConfig(random_state=42)
        training_config.add_model_config(
            model_type="xgboost_standard",
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )
        
        # Run pipeline
        pipeline = BenchmarkPipeline(data_config, training_config, tmp_path / "results")
        pipeline.load_and_prepare_data()
        
        models = pipeline.run_experiment(
            sku_tuples=[(80558, 2)],
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            experiment_name="standard_model_test"
        )
        
        # Validate results
        assert len(models) == 1
        model = models[0]
        assert model.metadata.quantile_level is None
        
        # Check storage location uses 'standard' for quantile directory
        storage_location = model.get_storage_location()
        assert storage_location.quantile_level is None
        
        components = storage_location.to_path_components()
        assert components[3] == "standard"  # Non-quantile models get 'standard' directory