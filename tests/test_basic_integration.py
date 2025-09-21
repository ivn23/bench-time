"""
Basic integration test to verify core pipeline functionality works.
Simplified test that focuses on essential workflow validation.
"""

import pytest
import tempfile
from pathlib import Path

from src import BenchmarkPipeline, ModelingStrategy
from tests.fixtures.sample_data import save_sample_data_to_temp


def test_basic_pipeline_workflow():
    """Test basic pipeline workflow with minimal setup."""
    # Create temporary directory and sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Generate and save sample data
        features_path, target_path, mapping_path = save_sample_data_to_temp(temp_path)
        
        # Create configurations
        from src import DataConfig, TrainingConfig
        
        data_config = DataConfig(
            features_path=str(features_path),
            target_path=str(target_path),
            mapping_path=str(mapping_path),
            remove_not_for_sale=False  # Disable filtering for test data
        )
        
        training_config = TrainingConfig()
        # Use simplified configuration for single model type
        training_config.set_model_config(
            model_type="xgboost_standard",
            hyperparameters={
                "n_estimators": 5,  # Minimal for fast testing
                "max_depth": 2,
                "random_state": 42
            }
        )
        
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=data_config,
            training_config=training_config,
            output_dir=temp_path / "results"
        )
        
        # Load data
        pipeline.load_and_prepare_data()
        
        # Test simple experiment
        sample_skus = [(80558, 2)]
        
        models = pipeline.run_experiment(
            sku_tuples=sample_skus,
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="basic_test"
        )
        
        # Validate results
        assert len(models) == 1
        model = models[0]
        
        assert model.metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert model.metadata.sku_tuples == sample_skus
        assert model.model is not None
        assert "mse" in model.metadata.performance_metrics
        
        print("✓ Basic pipeline workflow test passed!")


if __name__ == "__main__":
    test_basic_pipeline_workflow()