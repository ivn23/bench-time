"""
Pytest fixtures and configuration for M5 benchmarking framework tests.
Provides reusable test data, configurations, and utilities.
"""

import pytest
import polars as pl
import tempfile
from pathlib import Path
from typing import Tuple

from src import DataConfig, TrainingConfig, ModelingStrategy
from tests.fixtures.sample_data import (
    generate_sample_features_data,
    generate_sample_target_data, 
    save_sample_data_to_temp,
    create_sample_sku_tuples
)

@pytest.fixture(scope="session")
def sample_features_df():
    """Generate sample features DataFrame for testing."""
    return generate_sample_features_data(n_skus=5, n_days=50)

@pytest.fixture(scope="session")
def sample_target_df(sample_features_df):
    """Generate sample target DataFrame based on features."""
    return generate_sample_target_data(sample_features_df)

@pytest.fixture(scope="function")
def temp_data_dir():
    """Create temporary directory with sample data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        features_path, target_path, mapping_path = save_sample_data_to_temp(temp_path)
        
        yield {
            "dir": temp_path,
            "features_path": features_path,
            "target_path": target_path,
            "mapping_path": mapping_path
        }

@pytest.fixture(scope="function")
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_data_config(temp_data_dir):
    """Create DataConfig with sample data paths."""
    return DataConfig(
        features_path=str(temp_data_dir["features_path"]),
        target_path=str(temp_data_dir["target_path"]),
        mapping_path=str(temp_data_dir["mapping_path"]),
        date_column="date",
        target_column="target",
        bdid_column="bdID",
        remove_not_for_sale=True
    )

@pytest.fixture
def sample_data_config_with_split_date(temp_data_dir):
    """Create DataConfig with a specific split date."""
    return DataConfig(
        features_path=str(temp_data_dir["features_path"]),
        target_path=str(temp_data_dir["target_path"]),
        mapping_path=str(temp_data_dir["mapping_path"]),
        date_column="date",
        target_column="target",
        bdid_column="bdID",
        remove_not_for_sale=True,
        split_date="2020-02-01"
    )

@pytest.fixture
def sample_training_config():
    """Create TrainingConfig for testing."""
    return TrainingConfig(
        validation_split=0.2,
        random_state=42,
        model_type="xgboost",
        hyperparameters={
            "n_estimators": 10,  # Small for fast testing
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1  # Pin parallelism for reproducible tests
        }
    )

@pytest.fixture
def sample_sku_tuples():
    """Create sample SKU tuples for testing."""
    return create_sample_sku_tuples(3)

@pytest.fixture
def single_sku_tuple():
    """Create a single SKU tuple for testing."""
    return [(80558, 2)]

@pytest.fixture
def combined_strategy():
    """ModelingStrategy.COMBINED for testing."""
    return ModelingStrategy.COMBINED

@pytest.fixture 
def individual_strategy():
    """ModelingStrategy.INDIVIDUAL for testing."""
    return ModelingStrategy.INDIVIDUAL

@pytest.fixture
def sample_feature_columns():
    """Sample feature column names for testing."""
    return [
        "month", "day_of_week", "week_of_year", "quarter", "year", "is_weekend",
        "event_Christmas_0", "event_NewYear_0", "event_Halloween_0",
        "price_0", "trend"
    ]

@pytest.fixture
def minimal_features_df():
    """Generate minimal features DataFrame for unit testing."""
    return generate_sample_features_data(n_skus=2, n_days=20)

@pytest.fixture
def minimal_target_df(minimal_features_df):
    """Generate minimal target DataFrame for unit testing."""
    return generate_sample_target_data(minimal_features_df)

@pytest.fixture
def prepared_model_data(sample_features_df, sample_target_df):
    """Prepare X and y DataFrames ready for modeling."""
    # Simulate basic feature preparation
    feature_cols = [
        "month", "day_of_week", "week_of_year", "quarter", "year", "is_weekend",
        "event_Christmas_0", "event_NewYear_0", "event_Halloween_0", 
        "price_0", "trend"
    ]
    
    X = sample_features_df.select(["bdID"] + feature_cols)
    y = sample_target_df.select(["bdID", "target"])
    
    return X, y, feature_cols