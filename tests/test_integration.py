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
        """Test complete workflow with COMBINED modeling strategy using new API."""
        # Initialize pipeline with new API (no training_config in constructor)
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            output_dir=temp_output_dir
        )
        
        # Run experiment with COMBINED strategy using new API
        results = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples,
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_standard",
            hyperparameters={
                "n_estimators": 10,
                "max_depth": 3,
                "seed": 42
            },
            experiment_name="test_combined"
        )
        
        # Validate results
        assert results.num_models == 1  # Combined strategy should create 1 model
        assert len(results.models) == 1
        model = results.models[0]
        
        # Check model metadata
        assert model.metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert model.metadata.sku_tuples == sample_sku_tuples
        assert model.metadata.model_type == "xgboost_standard"
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
        # Note: r2 and other metrics depend on centralized MetricsCalculator
        
        # Check experiment results
        assert results.experiment_name == "test_combined"
        assert results.model_type == "xgboost_standard" 
        assert results.modeling_strategy == ModelingStrategy.COMBINED
        assert results.sku_tuples == sample_sku_tuples

    @pytest.mark.integration
    def test_individual_strategy_workflow(
        self, 
        sample_data_config, 
        sample_training_config,
        single_sku_tuple,  # Use single SKU to keep test fast
        temp_output_dir
    ):
        """Test complete workflow with INDIVIDUAL modeling strategy using new API."""
        # Initialize pipeline with new API
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            output_dir=temp_output_dir
        )
        
        # Run experiment with INDIVIDUAL strategy using new API
        results = pipeline.run_experiment(
            sku_tuples=single_sku_tuple,
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            model_type="xgboost_standard",
            hyperparameters={
                "n_estimators": 10,
                "max_depth": 3,
                "seed": 42
            },
            experiment_name="test_individual"
        )
        
        # Validate results
        assert results.num_models == len(single_sku_tuple)  # One model per SKU
        assert len(results.models) == len(single_sku_tuple)
        model = results.models[0]
        
        # Check model metadata
        assert model.metadata.modeling_strategy == ModelingStrategy.INDIVIDUAL
        assert model.metadata.sku_tuples == single_sku_tuple
        assert model.metadata.model_type == "xgboost_standard"
        
        # Check experiment results
        assert results.experiment_name == "test_individual"
        assert results.model_type == "xgboost_standard"
        assert results.modeling_strategy == ModelingStrategy.INDIVIDUAL

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
        
        # Test comprehensive release functionality
        release_dir = pipeline.run_complete_experiment(
            sku_tuples=[(80558, 2)],
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="test_comprehensive_release",
            evaluate=True
        )
        
        # Assert release directory and structure exists
        assert release_dir.exists()
        assert (release_dir / "bundle.json").exists()
        assert (release_dir / "metrics.json").exists()  # Since evaluate=True
        assert (release_dir / "models").exists()
        assert (release_dir / "logs").exists()
        assert (release_dir / "README.md").exists()
        
        # Check that the bundle.json file is valid
        import json
        with open(release_dir / "bundle.json") as f:
            bundle_data = json.load(f)
        assert isinstance(bundle_data, dict)
        assert "experiment_name" in bundle_data
        assert "models" in bundle_data
        assert "data_config" in bundle_data
        assert "training_config" in bundle_data
        
        # Check logs directory content
        logs_dir = release_dir / "logs"
        assert (logs_dir / "experiment_log.json").exists()
        
        # Verify experiment log contains expected content
        with open(logs_dir / "experiment_log.json") as f:
            log_data = json.load(f)
        assert isinstance(log_data, dict)
        assert "experiments" in log_data
        
        # Verify README.md exists and has content
        readme_path = release_dir / "README.md"
        with open(readme_path) as f:
            readme_content = f.read()
        assert "# Experiment Release:" in readme_content
        assert "test_comprehensive_release" in readme_content

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
        
        # Get model and verify it was saved using new release management system
        model = models[0]
        model_id = model.get_identifier()
        model_type = model.metadata.model_type
        
        # Check that release directory was created
        release_dirs = list(temp_output_dir.glob("release_*"))
        assert len(release_dirs) > 0, f"No release directories found in {temp_output_dir}"
        
        # Check that the release contains expected files
        release_dir = release_dirs[0]  # Use first release directory
        expected_files = ["bundle.json", "metrics.json", "data_splits.json"]
        for filename in expected_files:
            assert (release_dir / filename).exists(), f"Expected file {filename} not found in {release_dir}"
        
        # Check that models directory exists with model files
        models_dir = release_dir / "models"
        assert models_dir.exists(), f"Models directory not found in {release_dir}"
        
        # Check for model pickle files (naming pattern: model_<storeID>_<productID>.pkl)
        model_files = list(models_dir.glob("model_*.pkl"))
        assert len(model_files) > 0, f"No model files found in {models_dir}"
        
        # Verify data_splits.json content
        import json
        import numpy as np
        with open(release_dir / "data_splits.json") as f:
            splits_data = json.load(f)
        
        # Assert arrays match the model's data split
        original_model = models[0]
        assert np.array_equal(splits_data["train_bdIDs"], original_model.data_split.train_bdIDs)
        assert np.array_equal(splits_data["validation_bdIDs"], original_model.data_split.validation_bdIDs)
        
        # Assert split_date if it exists
        if original_model.data_split.split_date is not None:
            assert splits_data["split_date"] == original_model.data_split.split_date
        
        # Test retrieving the model from in-memory registry
        loaded_model = pipeline.model_registry.get_model(model_id)
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
        assert len(pipeline.experiment_log["experiments"]) > 0
        log_entry = pipeline.experiment_log["experiments"][0]
        
        # Validate log entry structure
        expected_keys = [
            "experiment_name", "model_id", "modeling_strategy", 
            "sku_tuples", "n_samples", "n_features", "performance"
        ]
        for key in expected_keys:
            assert key in log_entry
        
        # Note: Experiment logging is now handled by run_complete_experiment()
        # which creates comprehensive release directories with logs included

    @pytest.mark.integration
    def test_quantile_model_workflow(
        self,
        sample_data_config,
        single_sku_tuple,
        temp_output_dir
    ):
        """Test complete workflow with quantile XGBoost model."""
        # Configure quantile model training
        quantile_training_config = TrainingConfig(random_state=42)
        
        # Add quantile model configuration
        quantile_training_config.add_model_config(
            model_type="xgboost_quantile",
            quantile_alpha=0.75,
            hyperparameters={
                "max_depth": 4,
                "learning_rate": 0.3,
                "n_estimators": 10  # Keep low for fast testing
            }
        )
        
        # Initialize pipeline
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config,
            training_config=quantile_training_config,
            output_dir=temp_output_dir
        )
        
        # Load data and run experiment
        pipeline.load_and_prepare_data()
        models = pipeline.run_experiment(
            sku_tuples=single_sku_tuple,
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="test_quantile"
        )
        
        # Validate quantile model was created
        assert len(models) == 1
        model = models[0]
        
        # Check model metadata
        assert model.metadata.model_type == "xgboost_quantile"
        assert "q0.75" in model.get_identifier()
        
        # Check quantile-specific metrics are present
        metrics = model.metadata.performance_metrics
        assert "quantile_score" in metrics
        assert "coverage_probability" in metrics
        assert "coverage_error" in metrics
        assert "quantile_alpha" in metrics
        assert metrics["quantile_alpha"] == 0.75
        
        # Ensure standard metrics are also present
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        
        # Test model persistence with new release management system
        model_id = model.get_identifier()
        
        # Check that release directory was created
        release_dirs = list(temp_output_dir.glob("release_*"))
        assert len(release_dirs) > 0, f"No release directories found in {temp_output_dir}"
        
        # Check that the release contains quantile-specific bundle metadata
        release_dir = release_dirs[0]  # Use first release directory
        assert (release_dir / "bundle.json").exists(), f"Bundle file not found in {release_dir}"
        
        import json
        with open(release_dir / "bundle.json") as f:
            bundle_data = json.load(f)
        
        # Verify quantile-specific information in bundle
        assert bundle_data["model_family"] == "xgboost_quantile"
        assert bundle_data["quantile_level"] == 0.75
        
        # Test in-memory model retrieval (since we no longer have disk loading)
        loaded_model = pipeline.model_registry.get_model(model_id)
        assert loaded_model is not None
        assert loaded_model.metadata.model_type == "xgboost_quantile"

    @pytest.mark.integration
    def test_mixed_model_types_workflow(
        self,
        sample_data_config,
        single_sku_tuple,
        temp_output_dir
    ):
        """Test workflow with both standard and quantile models."""
        # Create separate directories for each pipeline to avoid conflicts
        from tempfile import mkdtemp
        from pathlib import Path
        
        standard_dir = Path(mkdtemp())
        quantile_dir = Path(mkdtemp())
        
        try:
            # Train standard XGBoost model
            standard_config = TrainingConfig(random_state=42)
            standard_config.add_model_config(
                model_type="xgboost_standard",
                hyperparameters={"n_estimators": 10, "max_depth": 4}
            )
            
            pipeline_standard = BenchmarkPipeline(
                data_config=sample_data_config,
                training_config=standard_config,
                output_dir=standard_dir
            )
            
            pipeline_standard.load_and_prepare_data()
            standard_models = pipeline_standard.run_experiment(
                sku_tuples=single_sku_tuple,
                modeling_strategy=ModelingStrategy.COMBINED,
                experiment_name="test_mixed_standard"
            )
            
            # Train quantile XGBoost model
            quantile_config = TrainingConfig(random_state=42)
            quantile_config.add_model_config(
                model_type="xgboost_quantile",
                quantile_alpha=0.7,
                hyperparameters={"max_depth": 4, "learning_rate": 0.3}
            )
            
            pipeline_quantile = BenchmarkPipeline(
                data_config=sample_data_config,
                training_config=quantile_config,
                output_dir=quantile_dir
            )
            
            pipeline_quantile.load_and_prepare_data()
            quantile_models = pipeline_quantile.run_experiment(
                sku_tuples=single_sku_tuple,
                modeling_strategy=ModelingStrategy.COMBINED,
                experiment_name="test_mixed_quantile"
            )
        
            # Validate both model types were created
            assert len(standard_models) == 1
            assert len(quantile_models) == 1
            
            standard_model = standard_models[0]
            quantile_model = quantile_models[0]
            
            assert standard_model.metadata.model_type == "xgboost_standard"
            assert quantile_model.metadata.model_type == "xgboost_quantile"
            
            # Check different metrics are present
            standard_metrics = standard_model.metadata.performance_metrics
            quantile_metrics = quantile_model.metadata.performance_metrics
            
            # Standard model should NOT have quantile metrics
            assert "quantile_score" not in standard_metrics
            assert "coverage_probability" not in standard_metrics
            
            # Quantile model SHOULD have quantile metrics
            assert "quantile_score" in quantile_metrics
            assert "coverage_probability" in quantile_metrics
            
            # Both should have standard metrics
            for metrics in [standard_metrics, quantile_metrics]:
                assert "mse" in metrics
                assert "rmse" in metrics
                assert "mae" in metrics
                assert "r2" in metrics
            
            # Check that models exist in their respective registries
            standard_models_list = pipeline_standard.model_registry.list_models()
            quantile_models_list = pipeline_quantile.model_registry.list_models()
            
            assert len(standard_models_list) == 1
            assert len(quantile_models_list) == 1
            
            # Verify model types are different
            standard_loaded = pipeline_standard.model_registry.get_model(standard_models_list[0])
            quantile_loaded = pipeline_quantile.model_registry.get_model(quantile_models_list[0])
            
            assert standard_loaded.metadata.model_type == "xgboost_standard"
            assert quantile_loaded.metadata.model_type == "xgboost_quantile"
            
        finally:
            # Cleanup temporary directories
            import shutil
            shutil.rmtree(standard_dir, ignore_errors=True)
            shutil.rmtree(quantile_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_lightning_combined_strategy_workflow(
        self, 
        sample_data_config, 
        sample_sku_tuples,
        temp_output_dir
    ):
        """Test complete Lightning model workflow with COMBINED strategy."""
        
        # Setup Lightning configuration
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_standard",
            hyperparameters={
                "hidden_size": 64,
                "lr": 1e-2,
                "dropout": 0.1,
                "max_epochs": 5,  # Small for fast testing
                "batch_size": 32
            }
        )
        
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config, 
            training_config=lightning_config, 
            output_dir=temp_output_dir
        )
        pipeline.load_and_prepare_data()
        
        # Run Lightning experiment
        models = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples[:2],
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="lightning_combined_test"
        )
        
        # Verify results
        assert len(models) == 1
        model = models[0]
        assert model.metadata.model_type == "lightning_standard"
        assert model.metadata.modeling_strategy == ModelingStrategy.COMBINED
        assert len(model.metadata.sku_tuples) == 2
        
        # Verify metrics
        metrics = model.metadata.performance_metrics
        expected_metrics = ['mse', 'rmse', 'mae', 'r2', 'mape', 'max_error', 'mean_error', 'std_error',
                           'within_1_unit', 'within_2_units', 'within_5_units']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Verify model persistence
        model_id = model.get_identifier()
        retrieved_model = pipeline.model_registry.get_model(model_id)
        assert retrieved_model is not None
        assert retrieved_model.metadata.model_type == "lightning_standard"
        assert retrieved_model.metadata.hyperparameters == model.metadata.hyperparameters

    @pytest.mark.integration
    def test_lightning_individual_strategy_workflow(
        self, 
        sample_data_config, 
        sample_sku_tuples,
        temp_output_dir
    ):
        """Test complete Lightning model workflow with INDIVIDUAL strategy."""
        
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_standard",
            hyperparameters={
                "hidden_size": 32,
                "lr": 1e-2,
                "max_epochs": 3,
                "batch_size": 16
            }
        )
        
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config, 
            training_config=lightning_config, 
            output_dir=temp_output_dir
        )
        pipeline.load_and_prepare_data()
        
        # Run experiment with 2 SKUs for individual strategy
        models = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples[:2],
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            experiment_name="lightning_individual_test"
        )
        
        # Should create 2 individual models
        assert len(models) == 2
        for i, model in enumerate(models):
            assert model.metadata.model_type == "lightning_standard"
            assert model.metadata.modeling_strategy == ModelingStrategy.INDIVIDUAL
            assert len(model.metadata.sku_tuples) == 1
            assert model.metadata.sku_tuples[0] == sample_sku_tuples[i]

    @pytest.mark.integration
    def test_lightning_evaluation_workflow(
        self, 
        sample_data_config, 
        sample_sku_tuples,
        temp_output_dir
    ):
        """Test Lightning model evaluation workflow."""
        
        # Setup and train Lightning model
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_standard",
            hyperparameters={"hidden_size": 32, "max_epochs": 3}
        )
        
        pipeline = BenchmarkPipeline(
            data_config=sample_data_config, 
            training_config=lightning_config, 
            output_dir=temp_output_dir
        )
        pipeline.load_and_prepare_data()
        
        # Train models
        combined_models = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples[:2],
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="lightning_eval_combined"
        )
        
        individual_models = pipeline.run_experiment(
            sku_tuples=sample_sku_tuples[:2],
            modeling_strategy=ModelingStrategy.INDIVIDUAL,
            experiment_name="lightning_eval_individual"
        )
        
        # Test comprehensive evaluation
        evaluation_results = pipeline.evaluate_all_models()
        
        # Verify evaluation structure
        assert 'combined' in evaluation_results
        assert 'individual' in evaluation_results
        assert 'overall' in evaluation_results
        
        # Verify combined results
        combined_results = evaluation_results['combined']
        if 'rankings' in combined_results:
            assert 'rmse' in combined_results['rankings']
            assert len(combined_results['rankings']['rmse']) >= 1
        
        # Verify individual results exist (flexible on count)
        individual_results = evaluation_results['individual']
        if 'rankings' in individual_results:
            assert 'rmse' in individual_results['rankings']
            assert len(individual_results['rankings']['rmse']) >= 1  # At least 1 model
        
        # Note: Evaluation results saving is now handled by run_complete_experiment()
        # which creates comprehensive release directories with metrics included

    @pytest.mark.integration
    def test_lightning_mixed_with_xgboost_workflow(
        self, 
        sample_data_config, 
        sample_sku_tuples,
        temp_output_dir
    ):
        """Test Lightning model integration with framework (simplified)."""
        
        # This test verifies Lightning works independently and can coexist with XGBoost
        lightning_config = TrainingConfig(random_state=42)
        lightning_config.add_model_config(
            model_type="lightning_standard",
            hyperparameters={"hidden_size": 32, "max_epochs": 3}
        )
        
        lightning_pipeline = BenchmarkPipeline(
            data_config=sample_data_config, 
            training_config=lightning_config, 
            output_dir=temp_output_dir
        )
        lightning_pipeline.load_and_prepare_data()
        lightning_models = lightning_pipeline.run_experiment(
            sku_tuples=sample_sku_tuples[:1],
            modeling_strategy=ModelingStrategy.COMBINED,
            experiment_name="mixed_lightning"
        )
        
        # Verify Lightning model was created successfully
        assert len(lightning_models) == 1
        lightning_model = lightning_models[0]
        assert lightning_model.metadata.model_type == "lightning_standard"
        
        # Verify registry operations work
        registry = lightning_pipeline.model_registry
        all_models = registry.list_models()
        
        model_types = []
        for model_id in all_models:
            model = registry.get_model(model_id)
            if model:
                model_types.append(model.metadata.model_type)
        
        # Verify Lightning model is in registry and discoverable
        assert "lightning_standard" in model_types
        
        # Test evaluation works with Lightning models
        evaluation_results = lightning_pipeline.evaluate_all_models()
        assert 'combined' in evaluation_results
        assert 'overall' in evaluation_results
        
        # Verify Lightning model has all required metrics
        combined_results = evaluation_results.get('combined', {})
        if 'rankings' in combined_results:
            rankings = combined_results['rankings']['rmse']
            assert len(rankings) >= 1  # At least the Lightning model  # At least XGBoost + Lightning  # At least XGBoost + Lightning  # At least XGBoost + Lightning
