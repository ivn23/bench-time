"""
Unit tests for DataLoader component.
Tests core data loading, filtering, and preparation functionality.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

from src import DataLoader, ModelingStrategy


class TestDataLoader:
    """Test DataLoader component functionality."""

    def test_data_loading(self, sample_data_config):
        """Test basic data loading functionality."""
        loader = DataLoader(sample_data_config)
        
        # Test loading data
        features_df, target_df, mapping = loader.load_data(lazy=False)
        
        # Validate data structure
        assert isinstance(features_df, pl.DataFrame)
        assert isinstance(target_df, pl.DataFrame)
        assert isinstance(mapping, dict)
        
        # Check required columns exist
        assert "bdID" in features_df.columns
        assert "date" in features_df.columns
        assert "productID" in features_df.columns
        assert "storeID" in features_df.columns
        
        assert "bdID" in target_df.columns
        assert "target" in target_df.columns
        
        # Check data has content
        assert len(features_df) > 0
        assert len(target_df) > 0
        assert len(mapping) > 0

    def test_lazy_loading(self, sample_data_config):
        """Test lazy loading functionality."""
        loader = DataLoader(sample_data_config)
        
        # Test lazy loading
        features_lf, target_lf, mapping = loader.load_data(lazy=True)
        
        # Should return LazyFrame for lazy=True
        assert isinstance(features_lf, pl.LazyFrame)
        assert isinstance(target_lf, pl.LazyFrame)
        
        # Can collect to get DataFrame
        features_df = features_lf.collect()
        target_df = target_lf.collect()
        
        assert isinstance(features_df, pl.DataFrame)
        assert isinstance(target_df, pl.DataFrame)
        assert len(features_df) > 0
        assert len(target_df) > 0

    def test_get_data_for_tuples_combined(self, sample_data_config, sample_sku_tuples):
        """Test getting data for SKU tuples with COMBINED strategy."""
        loader = DataLoader(sample_data_config)
        loader.load_data(lazy=False)
        
        # Get data for combined strategy
        features_df, target_df = loader.get_data_for_tuples(
            sample_sku_tuples, ModelingStrategy.COMBINED, collect=True
        )
        
        # Validate data structure
        assert isinstance(features_df, pl.DataFrame)
        assert isinstance(target_df, pl.DataFrame)
        assert len(features_df) > 0
        assert len(target_df) > 0
        
        # Check that data contains the requested SKUs
        unique_skus = features_df.select([
            pl.col("productID"), pl.col("storeID")
        ]).unique()
        
        # Should have data for requested SKU combinations
        assert len(unique_skus) <= len(sample_sku_tuples)

    def test_get_data_for_tuples_individual(self, sample_data_config, single_sku_tuple):
        """Test getting data for SKU tuples with INDIVIDUAL strategy."""
        loader = DataLoader(sample_data_config)
        loader.load_data(lazy=False)
        
        # Get data for individual strategy
        features_df, target_df = loader.get_data_for_tuples(
            single_sku_tuple, ModelingStrategy.INDIVIDUAL, collect=True
        )
        
        # Validate data structure
        assert isinstance(features_df, pl.DataFrame)
        assert isinstance(target_df, pl.DataFrame)
        assert len(features_df) > 0
        assert len(target_df) > 0
        
        # Check that data contains only the requested SKU
        unique_products = features_df.select("productID").unique().to_numpy().flatten()
        unique_stores = features_df.select("storeID").unique().to_numpy().flatten()
        
        expected_product, expected_store = single_sku_tuple[0]
        assert expected_product in unique_products
        assert expected_store in unique_stores

    def test_prepare_features_for_modeling(self, sample_data_config, minimal_features_df, minimal_target_df):
        """Test feature preparation for modeling."""
        loader = DataLoader(sample_data_config)
        
        # Prepare features for modeling
        X, y, feature_cols = loader.prepare_features_for_modeling(
            minimal_features_df, minimal_target_df, "target"
        )
        
        # Validate output structure
        assert isinstance(X, pl.DataFrame)
        assert isinstance(y, pl.DataFrame)
        assert isinstance(feature_cols, list)
        
        # Check that bdID is preserved
        assert "bdID" in X.columns
        assert "bdID" in y.columns
        
        # Check feature columns are valid
        assert len(feature_cols) > 0
        for col in feature_cols:
            assert col in X.columns
        
        # Check target column
        assert "target" in y.columns
        
        # Validate data consistency
        assert len(X) == len(y)
        assert len(X) > 0

    def test_get_feature_columns(self, sample_data_config, minimal_features_df):
        """Test getting feature column names."""
        loader = DataLoader(sample_data_config)
        
        feature_cols = loader.get_feature_columns(minimal_features_df)
        
        # Should return list of column names
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        
        # Should exclude non-feature columns
        excluded_cols = ["bdID", "date", "productID", "storeID", "skuID"]
        for col in excluded_cols:
            if col in minimal_features_df.columns:
                assert col not in feature_cols
        
        # Should include feature columns
        expected_features = ["month", "day_of_week", "price_0", "trend"]
        for col in expected_features:
            if col in minimal_features_df.columns:
                assert col in feature_cols

    def test_temporal_split_percentage(self, sample_data_config, minimal_features_df):
        """Test temporal split with percentage."""
        loader = DataLoader(sample_data_config)
        
        # Test temporal split
        train_bdids, test_bdids, split_info = loader.create_temporal_split(
            minimal_features_df, validation_split=0.3
        )
        
        # Validate output types
        assert isinstance(train_bdids, np.ndarray)
        assert isinstance(test_bdids, np.ndarray)
        
        # Check split ratios approximately correct
        total_samples = len(train_bdids) + len(test_bdids)
        test_ratio = len(test_bdids) / total_samples
        assert 0.2 < test_ratio < 0.4  # Should be around 0.3
        
        # Check no overlap between train and test
        assert len(set(train_bdids) & set(test_bdids)) == 0
        
        # Check all bdIDs are covered
        all_bdids = set(minimal_features_df.select("bdID").to_numpy().flatten())
        split_bdids = set(train_bdids) | set(test_bdids)
        assert split_bdids == all_bdids

    def test_temporal_split_by_date(self, sample_data_config, minimal_features_df):
        """Test temporal split by specific date."""
        loader = DataLoader(sample_data_config)
        
        # Use a date that should split the data
        split_date = "2020-01-15"
        
        train_bdids, test_bdids, returned_split_date = loader.create_temporal_split_by_date(
            minimal_features_df, split_date
        )
        
        # Validate output types
        assert isinstance(train_bdids, np.ndarray)
        assert isinstance(test_bdids, np.ndarray)
        assert returned_split_date is not None
        
        # Check that both splits have data
        assert len(train_bdids) > 0
        assert len(test_bdids) > 0
        
        # Check no overlap
        assert len(set(train_bdids) & set(test_bdids)) == 0
        
        # Validate temporal ordering by checking dates
        train_dates = minimal_features_df.filter(
            pl.col("bdID").is_in(train_bdids)
        ).select("date").to_series().max()
        
        test_dates = minimal_features_df.filter(
            pl.col("bdID").is_in(test_bdids)
        ).select("date").to_series().min()
        
        # Train data should be before test data (temporal split)
        assert train_dates <= test_dates

    def test_data_filtering(self, sample_data_config):
        """Test data filtering functionality."""
        loader = DataLoader(sample_data_config)
        features_df, target_df, _ = loader.load_data(lazy=False)
        
        # Test that we can filter by date range if config specifies it
        initial_count = len(features_df)
        assert initial_count > 0
        
        # Test filtering by SKU tuples works
        sample_tuples = [(80558, 2), (80651, 2)]
        filtered_features, filtered_target = loader.get_data_for_tuples(
            sample_tuples, ModelingStrategy.COMBINED, collect=True
        )
        
        # Should have filtered data
        assert len(filtered_features) > 0
        assert len(filtered_target) > 0
        
        # Check that filtering actually worked
        unique_combinations = filtered_features.select([
            pl.col("productID"), pl.col("storeID")
        ]).unique()
        
        # Should only have the requested combinations
        for row in unique_combinations.iter_rows(named=True):
            product_store = (row["productID"], row["storeID"])
            assert product_store in sample_tuples

    @pytest.mark.unit
    def test_error_handling(self, temp_output_dir):
        """Test error handling for invalid configurations."""
        from src import DataConfig
        
        # Test with non-existent files
        invalid_config = DataConfig(
            features_path=str(temp_output_dir / "nonexistent_features.feather"),
            target_path=str(temp_output_dir / "nonexistent_target.feather"), 
            mapping_path=str(temp_output_dir / "nonexistent_mapping.pkl"),
        )
        
        loader = DataLoader(invalid_config)
        
        # Should raise appropriate error when trying to load non-existent files
        with pytest.raises((FileNotFoundError, pl.exceptions.ComputeError, OSError)):
            loader.load_data()
        
    def test_empty_sku_tuples(self, sample_data_config):
        """Test handling of empty SKU tuple lists."""
        loader = DataLoader(sample_data_config)
        loader.load_data(lazy=False)
        
        # Test with empty tuple list - should handle gracefully
        with pytest.raises((ValueError, IndexError)):
            loader.get_data_for_tuples([], ModelingStrategy.COMBINED)