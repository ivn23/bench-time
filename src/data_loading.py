"""
Data loading and preprocessing module.
Memory-efficient data loading using Polars with filtering capabilities.
"""

import polars as pl
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from .data_structures import DataConfig, ModelingStrategy, ModelingDataset, SkuList

logger = logging.getLogger(__name__)


class DataLoader:
    """Memory-efficient data loader for M5 dataset using Polars."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._features_df = None
        self._target_df = None
        self._feature_mapping = None
        self._is_loaded = False
    
    def load_data(self, lazy: bool = True) -> Tuple[pl.DataFrame, pl.DataFrame, Dict]:
        """
        Load the M5 dataset from feather files.
        
        Args:
            lazy: If True, return LazyFrames for memory efficiency
            
        Returns:
            Tuple of (features_df, target_df, feature_mapping)
        """
        logger.info("Loading M5 dataset...")
        
        # Load feature mapping
        with open(self.config.mapping_path, 'rb') as f:
            self._feature_mapping = pickle.load(f)
        
        # Load data using Polars
        if lazy:
            self._features_df = pl.scan_ipc(self.config.features_path)
            self._target_df = pl.scan_ipc(self.config.target_path)
        else:
            self._features_df = pl.read_ipc(self.config.features_path)
            self._target_df = pl.read_ipc(self.config.target_path)
        
        # Apply basic filters
        if self.config.remove_not_for_sale:
            self._features_df = self._features_df.filter(pl.col("not_for_sale") != 1)
        
        if self.config.min_date:
            self._features_df = self._features_df.filter(
                pl.col(self.config.date_column) >= self.config.min_date
            )
            self._target_df = self._target_df.filter(
                pl.col(self.config.date_column) >= self.config.min_date
            )
        
        if self.config.max_date:
            self._features_df = self._features_df.filter(
                pl.col(self.config.date_column) <= self.config.max_date
            )
            self._target_df = self._target_df.filter(
                pl.col(self.config.date_column) <= self.config.max_date
            )
        
        self._is_loaded = True
        logger.info("Data loading completed")
        
        return self._features_df, self._target_df, self._feature_mapping
    
    
    def create_temporal_split(
        self, 
        df: pl.DataFrame, 
        validation_split: float = 0.2,
        date_column: str = "date"
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Create temporal train/validation split based on percentage.
        
        Args:
            df: DataFrame to split
            validation_split: Proportion of data for validation
            date_column: Date column name for temporal ordering
            
        Returns:
            Tuple of (train_bdIDs, validation_bdIDs, split_info)
        """
        # Get unique dates and sort them
        unique_dates = df.select(date_column).unique().sort(date_column)
        dates = unique_dates.to_series().to_list()
        
        # Calculate split index
        split_idx = int(len(dates) * (1 - validation_split))
        split_date = dates[split_idx] if split_idx < len(dates) else dates[-1]
        
        # Split data based on date
        train_data = df.filter(pl.col(date_column) < split_date)
        validation_data = df.filter(pl.col(date_column) >= split_date)
        
        train_bdids = train_data.select("bdID").to_numpy().flatten()
        validation_bdids = validation_data.select("bdID").to_numpy().flatten()
        
        return train_bdids, validation_bdids, str(split_date)
    
    def create_temporal_split_by_date(
        self, 
        df: pl.DataFrame, 
        split_date: str,
        date_column: str = "date"
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Create temporal train/validation split based on a specific date.
        
        Args:
            df: DataFrame to split
            split_date: Date to use for splitting (format: 'YYYY-MM-DD')
            date_column: Column containing dates
            
        Returns:
            Tuple of (train_bdIDs, validation_bdIDs, split_date)
        """
        # Sort by date to ensure chronological order
        df_sorted = df.sort(date_column)
        
        # Convert split_date string to date for comparison
        split_date_obj = pl.lit(split_date).str.strptime(pl.Date, "%Y-%m-%d")
        
        # Split based on date
        train_mask = pl.col(date_column) < split_date_obj
        val_mask = pl.col(date_column) >= split_date_obj
        
        train_bdids = df_sorted.filter(train_mask).select("bdID").to_numpy().flatten()
        val_bdids = df_sorted.filter(val_mask).select("bdID").to_numpy().flatten()
        
        logger.info(f"Created temporal split by date: {len(train_bdids)} train, {len(val_bdids)} validation")
        logger.info(f"Split date: {split_date}")
        
        return train_bdids, val_bdids, split_date

    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get list of feature columns, excluding metadata columns."""
        # Define metadata columns to exclude
        metadata_cols = {
            "frequency", "idx", "bdID", "base_date", "date", "dateID", 
            "skuID", "productID", "storeID", "companyID", "missing_value", 
            "not_for_sale", "target", "feature_0038", "target_lag_1"
        }
        
        # Get all columns except metadata
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        return feature_cols


    def prepare_modeling_dataset(
        self, 
        sku_tuples: SkuList, 
        modeling_strategy: ModelingStrategy
    ) -> ModelingDataset:
        """
        Complete data preparation pipeline - replaces all previous methods.
        Handles: data retrieval, filtering, feature prep, temporal splitting, train/test creation.
        
        Args:
            sku_tuples: List of (product_id, store_id) tuples
            modeling_strategy: COMBINED or INDIVIDUAL strategy
            
        Returns:
            ModelingDataset: Complete dataset ready for model training
        """
        logger.info(f"Preparing modeling dataset for {len(sku_tuples)} SKUs with {modeling_strategy.value} strategy")
        
        # Ensure data is loaded
        if not self._is_loaded:
            self.load_data(lazy=False)
        
        # Filter data for specified SKUs
        features_df, target_df = self._filter_sku_data(sku_tuples, modeling_strategy)
        
        # Prepare features for modeling
        X, y, feature_cols = self._prepare_features(features_df, target_df)
        
        # Apply temporal splitting with configuration
        X_train, y_train, X_test, y_test, split_info = self._apply_temporal_split(X, y)
        
        # Calculate dataset statistics
        dataset_stats = {
            "n_samples_total": len(X),
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(feature_cols),
            "n_skus": len(sku_tuples)
        }
        
        logger.info(f"Dataset prepared: {dataset_stats['n_samples_train']} train, {dataset_stats['n_samples_test']} test samples")
        
        return ModelingDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            target_col=self.config.target_column,
            split_info=split_info,
            dataset_stats=dataset_stats,
            sku_tuples=sku_tuples,
            modeling_strategy=modeling_strategy
        )
    
    def _filter_sku_data(self, sku_tuples: SkuList, strategy: ModelingStrategy) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Filter data for specified SKUs based on modeling strategy."""
        logger.info(f"Filtering data for {len(sku_tuples)} SKUs")
        
        # Create filter conditions for SKU tuples
        sku_conditions = []
        for product_id, store_id in sku_tuples:
            condition = (pl.col("productID") == product_id) & (pl.col("storeID") == store_id)
            sku_conditions.append(condition)
        
        # Combine conditions with OR
        if len(sku_conditions) == 1:
            sku_filter = sku_conditions[0]
        else:
            sku_filter = sku_conditions[0]
            for condition in sku_conditions[1:]:
                sku_filter = sku_filter | condition
        
        # Apply filtering
        features_filtered = self._features_df.filter(sku_filter)
        
        # Get corresponding target data by joining on bdID
        feature_bdids = features_filtered.select("bdID").unique()
        target_filtered = self._target_df.join(feature_bdids, on="bdID", how="inner")
        
        # Apply additional filters from config
        if self.config.remove_not_for_sale:
            # Assume 'not_for_sale' column exists - filter it out
            if "not_for_sale" in features_filtered.columns:
                features_filtered = features_filtered.filter(pl.col("not_for_sale") == False)
                # Re-filter targets
                feature_bdids = features_filtered.select("bdID").unique()
                target_filtered = self._target_df.join(feature_bdids, on="bdID", how="inner")
        
        # Apply date range filters
        if self.config.min_date or self.config.max_date:
            if self.config.date_column in features_filtered.columns:
                if self.config.min_date:
                    features_filtered = features_filtered.filter(pl.col(self.config.date_column) >= self.config.min_date)
                if self.config.max_date:
                    features_filtered = features_filtered.filter(pl.col(self.config.date_column) <= self.config.max_date)
                
                # Re-filter targets
                feature_bdids = features_filtered.select("bdID").unique()
                target_filtered = self._target_df.join(feature_bdids, on="bdID", how="inner")
        
        logger.info(f"Filtered to {len(features_filtered)} samples")
        return features_filtered, target_filtered
    
    def _prepare_features(self, features_df: pl.DataFrame, target_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        """Prepare features for modeling."""
        # Join features with targets to ensure alignment
        df_merged = features_df.join(target_df, on="bdID", how="inner")
        
        # Remove rows with null targets
        df_clean = df_merged.filter(pl.col(self.config.target_column).is_not_null())
        
        # Get feature columns (exclude metadata columns)
        exclude_cols = {self.config.bdid_column, self.config.target_column, 
                       self.config.date_column, "productID", "storeID"}
        if hasattr(df_clean, 'columns'):
            all_cols = set(df_clean.columns)
        else:
            all_cols = set(df_clean.schema.keys())
        
        feature_cols = sorted(list(all_cols - exclude_cols))
        
        # Prepare X and y with bdID for splitting
        date_col = self.config.date_column if self.config.date_column in df_clean.columns else None
        if date_col:
            X = df_clean.select(["bdID", date_col] + feature_cols)
        else:
            X = df_clean.select(["bdID"] + feature_cols)
        y = df_clean.select(["bdID", self.config.target_column])
        
        return X, y, feature_cols
    
    def _apply_temporal_split(self, X: pl.DataFrame, y: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
        """Apply temporal split using DataConfig settings."""
        # Choose split method based on configuration
        if self.config.split_date:
            train_bdids, test_bdids, split_date = self.create_temporal_split_by_date(X, self.config.split_date)
        else:
            train_bdids, test_bdids, split_date = self.create_temporal_split(X, self.config.validation_split)
        
        # Split data into train and test
        X_train = X.filter(pl.col("bdID").is_in(train_bdids))
        y_train = y.filter(pl.col("bdID").is_in(train_bdids))
        X_test = X.filter(pl.col("bdID").is_in(test_bdids))
        y_test = y.filter(pl.col("bdID").is_in(test_bdids))
        
        split_info = {
            "split_date": str(split_date),
            "train_bdids": train_bdids,
            "test_bdids": test_bdids,
            "split_method": "date_based" if self.config.split_date else "percentage_based",
            "validation_split": self.config.validation_split if not self.config.split_date else None
        }
        
        return X_train, y_train, X_test, y_test, split_info
