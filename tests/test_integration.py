"""
Integration tests for the M5 benchmarking framework.
Tests complete end-to-end workflows as users would actually use the framework.
"""

import pytest
import polars as pl
from pathlib import Path

from src import BenchmarkPipeline, ModelingStrategy


class TestPipelineIntegration:
    """Test complete pipeline workflows end-to-end."""

    @pytest.mark.integration
    def test_combined_strategy_workflow(
        self, 
        sample_data_config, 
        sample_training_config, 
        sample_sku_tuples,
        temp_output_dir
    ):
        """Test complete workflow with COMBINED modeling strategy."""
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            training_config=sample_training_config,
            output_dir=temp_output_dir
        )
        
        # Load data
        pipeline.load_and_prepare_data()
        
        # Run experiment with COMBINED strategy
        models = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples,
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="test_combined"
        )
        
        # Validate results
        assert len(models) == 1  # Combined strategy should create 1 model
        model = models[0]
        
        # Check model metadata
        assert model.metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert model.metadata.sku_tuples == sample_sku_tuples
        assert model.metadata.model_type == "xgboost"
        assert len(model.metadata.feature_columns) > 0
        assert model.metadata.target_column == "target"
        
        # Check model object exists
        assert model.model is not None
        assert hasattr(model.model, 'predict')
        
        # Check data split
        assert model.data_split.train_bdIDs is not None
        assert model.data_split.validation_bdIDs is not None
        assert len(model.data_split.train_bdIDs) > 0
        assert len(model.data_split.validation_bdIDs) > 0
        
        # Check performance metrics exist
        assert "mse" in model.metadata.performance_metrics
        assert "rmse" in model.metadata.performance_metrics
        assert "mae" in model.metadata.performance_metrics
        assert "r2" in model.metadata.performance_metrics
        
        # Check model registry
        model_id = model.get_identifier()
        retrieved_model = pipeline.model_registry.get_model(model_id)
        assert retrieved_model is not None
        assert retrieved_model.metadata.model_id == model.metadata.model_id

    @pytest.mark.integration
    def test_individual_strategy_workflow(
        self, 
        sample_data_config, 
        sample_training_config,
        single_sku_tuple,  # Use single SKU to keep test fast
        temp_output_dir
    ):
        """Test complete workflow with INDIVIDUAL modeling strategy."""
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            training_config=sample_training_config,
            output_dir=temp_output_dir
        )
        
        # Load data
        pipeline.load_and_prepare_data()
        
        # Run experiment with INDIVIDUAL strategy
        models = pipeline.run_experiment(
            sku_tuples=single_sku_tuple,
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            experiment_name="test_individual"
        )
        
        # Validate results
        assert len(models) == len(single_sku_tuple)  # One model per SKU
        model = models[0]
        
        # Check model metadata
        assert model.metadata.modeling_strategy == ModelingStrategy.INDIVIDUAL
        assert model.metadata.sku_tuples == single_sku_tuple
        assert model.metadata.model_type == "xgboost"
        
        # Check model registry
        all_model_ids = pipeline.model_registry.list_models()
        assert len(all_model_ids) >= len(models)

    @pytest.mark.integration
    def test_multiple_experiments_workflow(
        self,
        sample_data_config,
        sample_training_config, 
        sample_sku_tuples,
        temp_output_dir
    ):
        """Test running multiple experiments and evaluation."""
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            training_config=sample_training_config,
            output_dir=temp_output_dir
        )
        
        # Load data
        pipeline.load_and_prepare_data()
        
        # Run combined strategy experiment
        combined_models = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples[:2],  # Subset for performance
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="multi_test_combined"
        )
        
        # Run individual strategy experiment  
        individual_models = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples[:1],  # Single SKU for individual
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            experiment_name="multi_test_individual"
        )
        
        # Check models were created
        assert len(combined_models) == 1
        assert len(individual_models) == 1
        
        # Test evaluation
        evaluation_results = pipeline.evaluate_all_models()
        assert isinstance(evaluation_results, dict)
        
        # Should have results for both strategies
        if "combined" in evaluation_results:
            combined_results = evaluation_results["combined"]
            assert "model_evaluations" in combined_results
        
        if "individual" in evaluation_results:
            individual_results = evaluation_results["individual"]
            assert "model_evaluations" in individual_results
        
        # Test save_evaluation_results output
        pipeline.save_evaluation_results(evaluation_results)
        
        # Assert evaluation_results.json exists
        evaluation_dir = temp_output_dir / "evaluation_results"
        assert evaluation_dir.exists()
        assert (evaluation_dir / "evaluation_results.json").exists()
        
        # Check that the JSON file is valid
        import json
        with open(evaluation_dir / "evaluation_results.json") as f:
            saved_results = json.load(f)
        assert isinstance(saved_results, dict)
        
        # Check for strategy-specific reports
        if "combined" in evaluation_results:
            combined_report = evaluation_dir / "combined_evaluation_report.md"
            assert combined_report.exists()
            # Verify report contains expected content
            with open(combined_report) as f:
                report_content = f.read()
            assert "# Model Evaluation Report" in report_content
        
        if "individual" in evaluation_results:
            individual_report = evaluation_dir / "individual_evaluation_report.md"
            assert individual_report.exists()
            # Verify report contains expected content
            with open(individual_report) as f:
                report_content = f.read()
            assert "# Model Evaluation Report" in report_content

    @pytest.mark.integration 
    def test_split_date_workflow(
        self,
        sample_data_config_with_split_date,
        sample_training_config,
        single_sku_tuple,
        temp_output_dir
    ):
        """Test workflow with specific split date instead of percentage split."""
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config_with_split_date,
            training_config=sample_training_config,
            output_dir=temp_output_dir
        )
        
        # Load data
        pipeline.load_and_prepare_data()
        
        # Run experiment
        models = pipeline.run_experiment(
            sku_tuples=single_sku_tuple,
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="test_split_date"
        )
        
        # Validate split date was used
        model = models[0]
        assert model.data_split.split_date is not None
        # Split date should be the configured date
        assert "2020-02-01" in model.data_split.split_date

    @pytest.mark.integration
    def test_model_persistence_workflow(
        self,
        sample_data_config,
        sample_training_config,
        single_sku_tuple,
        temp_output_dir
    ):
        """Test model saving and loading workflow."""
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            training_config=sample_training_config,
            output_dir=temp_output_dir
        )
        
        # Load data and train model
        pipeline.load_and_prepare_data()
        models = pipeline.run_experiment(
            sku_tuples=single_sku_tuple,
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="test_persistence"
        )
        
        # Get model ID and verify it was saved
        model_id = models[0].get_identifier()
        model_dir = temp_output_dir / "models" / model_id
        assert model_dir.exists()
        
        # Check expected files exist
        expected_files = ["model.pkl", "metadata.json", "data_splits.json"]
        for filename in expected_files:
            assert (model_dir / filename).exists()
        
        # Verify data_splits.json content
        import json
        import numpy as np
        with open(model_dir / "data_splits.json") as f:
            splits_data = json.load(f)
        
        # Assert arrays match the model's data split
        original_model = models[0]
        assert np.array_equal(splits_data["train_bdIDs"], original_model.data_split.train_bdIDs)
        assert np.array_equal(splits_data["validation_bdIDs"], original_model.data_split.validation_bdIDs)
        
        # Assert split_date if it exists
        if original_model.data_split.split_date is not None:
            assert splits_data["split_date"] == original_model.data_split.split_date
        
        # Test loading the model
        loaded_model = pipeline.model_registry.load_model(model_id)
        assert loaded_model is not None
        assert loaded_model.metadata.model_id == models[0].metadata.model_id
        assert loaded_model.model is not None
        
        # Test that loaded model can make predictions
        assert hasattr(loaded_model.model, 'predict')

    @pytest.mark.integration
    def test_experiment_logging_workflow(
        self,
        sample_data_config,
        sample_training_config,
        single_sku_tuple,
        temp_output_dir
    ):
        """Test experiment logging functionality."""
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            training_config=sample_training_config,
            output_dir=temp_output_dir
        )
        
        # Load data and run experiment
        pipeline.load_and_prepare_data()
        models = pipeline.run_experiment(
            sku_tuples=single_sku_tuple,
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="test_logging"
        )
        
        # Check experiment log was created
        assert len(pipeline.experiment_log) > 0
        log_entry = pipeline.experiment_log[0]
        
        # Validate log entry structure
        expected_keys = [
            "experiment_name", "model_id", "modeling_strategy", 
            "sku_tuples", "n_samples", "n_features", "performance"
        ]
        for key in expected_keys:
            assert key in log_entry
        
        # Save experiment log
        pipeline.save_experiment_log()
        
        # Check log file was created
        log_file = temp_output_dir / "experiment_log.json"
        assert log_file.exists()
        
        # Verify log file is valid JSON
        import json
        with open(log_file) as f:
            loaded_log = json.load(f)
        assert isinstance(loaded_log, list)
        assert len(loaded_log) > 0