"""
Tests for hyperparameter tuning mode functionality.

Tests the independent hyperparameter tuning module and its integration
with the pipeline's mode-based architecture.
"""

import pytest
import numpy as np
from src import BenchmarkPipeline, DataConfig, ModelingStrategy
from src.hyperparameter_tuning import HyperparameterTuner, TuningResult
from src.data_loading import DataLoader
from src.structures import ExperimentResults


@pytest.fixture
def data_config():
    """Standard data configuration for testing."""
    return DataConfig(
        features_path="data/db_snapshot_offsite/train_data/processed/train_data_features.feather",
        target_path="data/db_snapshot_offsite/train_data/train_data/train_data_target.feather",
        mapping_path="data/feature_mapping_train.pkl",
        split_date="2016-01-01"
    )


@pytest.fixture
def test_skus():
    """Standard SKU tuples for testing."""
    return [(715, 7), (377, 1), (1912, 7)]


class TestTuningResult:
    """Tests for TuningResult dataclass."""

    def test_tuning_result_creation(self):
        """Test that TuningResult can be created with all fields."""
        result = TuningResult(
            best_params={'eta': 0.05, 'max_depth': 6},
            best_score=0.123,
            n_trials=10,
            n_folds=3,
            model_type='xgboost_quantile',
            quantile_alpha=0.7,
            n_skus_sampled=5,
            optimization_time=30.5
        )

        assert result.best_params == {'eta': 0.05, 'max_depth': 6}
        assert result.best_score == 0.123
        assert result.n_trials == 10
        assert result.model_type == 'xgboost_quantile'

    def test_tuning_result_summary(self):
        """Test that TuningResult generates a summary string."""
        result = TuningResult(
            best_params={'eta': 0.05},
            best_score=0.123,
            n_trials=10,
            n_folds=3,
            model_type='xgboost_quantile',
            quantile_alpha=0.7,
            n_skus_sampled=5,
            optimization_time=30.5
        )

        summary = result.get_summary()
        assert "HYPERPARAMETER TUNING RESULTS" in summary
        assert "Model Type: xgboost_quantile" in summary
        assert "Quantile Alpha: 0.7" in summary
        assert "eta: 0.05" in summary


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""

    def test_tuner_initialization(self):
        """Test tuner can be initialized."""
        tuner = HyperparameterTuner(random_state=42)
        assert tuner.random_state == 42

    def test_cv_splits_creation(self, data_config):
        """Test that CV splits are created correctly."""
        tuner = HyperparameterTuner(random_state=42)

        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)

        # Create 3-fold splits
        cv_splits = tuner._create_cv_splits(X_train, y_train, n_folds=3)

        assert len(cv_splits) == 3
        for train_idx, val_idx in cv_splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert len(train_idx) + len(val_idx) == 100

    def test_search_space_xgboost_quantile(self):
        """Test search space definition for XGBoost quantile."""
        import optuna
        tuner = HyperparameterTuner(random_state=42)

        study = optuna.create_study()
        trial = study.ask()

        params = tuner._get_search_space(trial, 'xgboost_quantile')

        # Check that all expected parameters are present
        expected_params = ['eta', 'max_depth', 'min_child_weight', 'subsample',
                          'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda',
                          'n_estimators', 'seed']
        for param in expected_params:
            assert param in params

    def test_search_space_xgboost_standard(self):
        """Test search space definition for XGBoost standard."""
        import optuna
        tuner = HyperparameterTuner(random_state=42)

        study = optuna.create_study()
        trial = study.ask()

        params = tuner._get_search_space(trial, 'xgboost_standard')

        # Check that seed is used (matching native XGBoost API)
        assert 'seed' in params
        assert params['seed'] == 42

    def test_calculate_loss_quantile(self):
        """Test quantile loss calculation."""
        tuner = HyperparameterTuner()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 2.0, 4.5, 4.0])

        loss = tuner._calculate_loss(y_true, y_pred, quantile_alpha=0.7)

        assert isinstance(loss, float)
        assert loss > 0  # Loss should be positive

    def test_calculate_loss_standard(self):
        """Test MSE loss calculation for standard regression."""
        tuner = HyperparameterTuner()

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 2.0, 4.5, 4.0])

        loss = tuner._calculate_loss(y_true, y_pred, quantile_alpha=None)

        # Should be MSE
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert np.isclose(loss, expected_mse)


class TestPipelineTuningMode:
    """Tests for pipeline integration with tuning mode."""

    def test_tuning_mode_basic(self, data_config, test_skus):
        """Test basic hyperparameter tuning mode."""
        pipeline = BenchmarkPipeline(data_config)

        result = pipeline.run_experiment(
            sku_tuples=test_skus,
            modeling_strategy=ModelingStrategy.COMBINED,  # Ignored in tuning mode
            model_type="xgboost_quantile",
            quantile_alphas=[0.7],
            mode="hp_tune",
            tune_on=3,
            tuning_config={'n_trials': 5, 'n_folds': 2},
            random_state=42
        )

        # Verify result type
        assert isinstance(result, TuningResult)

        # Verify result contents
        assert 'eta' in result.best_params
        assert 'max_depth' in result.best_params
        assert 'n_estimators' in result.best_params
        assert result.best_score > 0
        assert result.n_trials == 5
        assert result.n_folds == 2
        assert result.model_type == 'xgboost_quantile'
        assert result.quantile_alpha == 0.7

    def test_tuning_mode_samples_skus(self, data_config, test_skus):
        """Test that tuning mode samples the specified number of SKUs."""
        pipeline = BenchmarkPipeline(data_config)

        result = pipeline.run_experiment(
            sku_tuples=test_skus,
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            quantile_alphas=[0.7],
            mode="hp_tune",
            tune_on=2,  # Sample only 2 SKUs
            tuning_config={'n_trials': 3, 'n_folds': 2},
            random_state=42
        )

        assert isinstance(result, TuningResult)
        assert result.n_skus_sampled == 2

    def test_tuning_mode_multi_quantile_warning(self, data_config, test_skus):
        """Test that tuning with multiple quantiles uses only the first one."""
        pipeline = BenchmarkPipeline(data_config)

        result = pipeline.run_experiment(
            sku_tuples=test_skus,
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            quantile_alphas=[0.5, 0.7, 0.9],  # Multiple quantiles
            mode="hp_tune",
            tune_on=3,
            tuning_config={'n_trials': 3, 'n_folds': 2},
            random_state=42
        )

        # Should tune for first quantile only
        assert result.quantile_alpha == 0.5


class TestPipelineTrainingMode:
    """Tests for training mode backward compatibility."""

    def test_training_mode_unchanged(self, data_config, test_skus):
        """Test that existing training mode still works."""
        pipeline = BenchmarkPipeline(data_config)

        result = pipeline.run_experiment(
            sku_tuples=[test_skus[0]],
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            hyperparameters={'eta': 0.05, 'max_depth': 6, 'n_estimators': 50},
            quantile_alphas=[0.7],
            mode="train",
            evaluate_on_test=False,
            random_state=42
        )

        assert isinstance(result, ExperimentResults)
        assert result.num_models > 0

    def test_training_mode_default(self, data_config, test_skus):
        """Test that mode='train' is the default."""
        pipeline = BenchmarkPipeline(data_config)

        # Don't specify mode, should default to 'train'
        result = pipeline.run_experiment(
            sku_tuples=[test_skus[0]],
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            hyperparameters={'eta': 0.05, 'max_depth': 6, 'n_estimators': 50},
            quantile_alphas=[0.7],
            evaluate_on_test=False,
            random_state=42
        )

        assert isinstance(result, ExperimentResults)


class TestTuneThenTrainWorkflow:
    """Tests for complete tune-then-train workflow."""

    def test_tune_then_train(self, data_config, test_skus):
        """Test realistic workflow: tune hyperparameters then train."""
        pipeline = BenchmarkPipeline(data_config)

        # Step 1: Tune hyperparameters
        tuning_result = pipeline.run_experiment(
            sku_tuples=test_skus,
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            quantile_alphas=[0.7],
            mode="hp_tune",
            tune_on=3,
            tuning_config={'n_trials': 5, 'n_folds': 2},
            random_state=42
        )

        assert isinstance(tuning_result, TuningResult)

        # Step 2: Train with optimized parameters
        training_result = pipeline.run_experiment(
            sku_tuples=[test_skus[0]],
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            hyperparameters=tuning_result.best_params,
            quantile_alphas=[0.7],
            mode="train",
            evaluate_on_test=False,
            random_state=42
        )

        assert isinstance(training_result, ExperimentResults)
        assert training_result.num_models == 1


class TestModeValidation:
    """Tests for mode parameter validation."""

    def test_invalid_mode(self, data_config, test_skus):
        """Test that invalid mode raises ValueError."""
        pipeline = BenchmarkPipeline(data_config)

        with pytest.raises(ValueError, match="mode must be 'train' or 'hp_tune'"):
            pipeline.run_experiment(
                sku_tuples=test_skus,
                modeling_strategy=ModelingStrategy.COMBINED,
                model_type="xgboost_quantile",
                quantile_alphas=[0.7],
                mode="invalid_mode"
            )

    def test_hp_tune_without_tune_on(self, data_config, test_skus):
        """Test that hp_tune mode without tune_on raises ValueError."""
        pipeline = BenchmarkPipeline(data_config)

        with pytest.raises(ValueError, match="tune_on must be specified"):
            pipeline.run_experiment(
                sku_tuples=test_skus,
                modeling_strategy=ModelingStrategy.COMBINED,
                model_type="xgboost_quantile",
                quantile_alphas=[0.7],
                mode="hp_tune"
                # Missing tune_on parameter
            )

    def test_train_without_hyperparameters(self, data_config, test_skus):
        """Test that train mode without hyperparameters raises ValueError."""
        pipeline = BenchmarkPipeline(data_config)

        with pytest.raises(ValueError, match="hyperparameters must be provided"):
            pipeline.run_experiment(
                sku_tuples=test_skus,
                modeling_strategy=ModelingStrategy.COMBINED,
                model_type="xgboost_quantile",
                quantile_alphas=[0.7],
                mode="train"
                # Missing hyperparameters
            )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_tune_on_exceeds_available_skus(self, data_config, test_skus):
        """Test that tune_on larger than available SKUs uses all SKUs."""
        pipeline = BenchmarkPipeline(data_config)

        result = pipeline.run_experiment(
            sku_tuples=test_skus,  # Only 3 SKUs
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            quantile_alphas=[0.7],
            mode="hp_tune",
            tune_on=100,  # Request more than available
            tuning_config={'n_trials': 3, 'n_folds': 2},
            random_state=42
        )

        # Should use all available SKUs
        assert result.n_skus_sampled == 3

    def test_default_tuning_config(self, data_config, test_skus):
        """Test that default tuning config is applied when not provided."""
        pipeline = BenchmarkPipeline(data_config)

        result = pipeline.run_experiment(
            sku_tuples=test_skus,
            modeling_strategy=ModelingStrategy.COMBINED,
            model_type="xgboost_quantile",
            quantile_alphas=[0.7],
            mode="hp_tune",
            tune_on=3
            # No tuning_config provided, should use defaults
        )

        # Defaults should be n_trials=50, n_folds=3
        assert result.n_trials == 50
        assert result.n_folds == 3
