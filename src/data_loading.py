"""
Data loading and preprocessing module.
Memory-efficient data loading using Polars with filtering capabilities.
"""

import polars as pl
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .data_structures import DataConfig, ModelingStrategy

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
    
    def get_data_for_tuples(
        self, 
        sku_tuples: List[Tuple[int, int]],
        modeling_strategy: 'ModelingStrategy',
        collect: bool = True
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Filter data for specific SKU tuples (product_id, store_id pairs).
        
        Args:
            sku_tuples: List of (product_id, store_id) tuples defining SKUs
            modeling_strategy: How to handle multiple SKUs (combined vs individual)
            collect: Whether to collect LazyFrame to DataFrame
            
        Returns:
            Tuple of (filtered_features, filtered_target)
        """
        if not self._is_loaded:
            raise ValueError("Data must be loaded first. Call load_data().")
        
        if not sku_tuples:
            raise ValueError("At least one SKU tuple must be provided")
        
        logger.info(f"Filtering data for {len(sku_tuples)} SKU tuples using {modeling_strategy.value} strategy")
        
        features_query = self._features_df
        target_query = self._target_df
        
        # Create filter conditions for all requested SKUs
        sku_conditions = []
        for product_id, store_id in sku_tuples:
            condition = (pl.col("productID") == product_id) & (pl.col("storeID") == store_id)
            sku_conditions.append(condition)
        
        # Combine all conditions with OR
        combined_condition = sku_conditions[0]
        for condition in sku_conditions[1:]:
            combined_condition = combined_condition | condition
        
        # Apply filter
        features_query = features_query.filter(combined_condition)
        
        # Get matching bdIDs for target filtering
        if collect:
            # Check if we're working with LazyFrame or DataFrame
            if hasattr(features_query, 'collect'):
                # It's a LazyFrame
                bdids = features_query.select("bdID").collect().to_numpy().flatten()
            else:
                # It's already a DataFrame
                bdids = features_query.select("bdID").to_numpy().flatten()
            
            # Apply the same logic to target_query
            if hasattr(target_query, 'collect'):
                target_query = target_query.filter(pl.col("bdID").is_in(bdids))
            else:
                target_query = target_query.filter(pl.col("bdID").is_in(bdids))
        else:
            # For lazy evaluation, we need a different approach
            target_query = target_query.join(
                features_query.select("bdID"), 
                on="bdID", 
                how="inner"
            )
        
        # Collect if requested
        if collect:
            # Check if we need to collect or if already collected
            if hasattr(features_query, 'collect'):
                features_df = features_query.collect()
            else:
                features_df = features_query
                
            if hasattr(target_query, 'collect'):
                target_df = target_query.collect()
            else:
                target_df = target_query
                
            logger.info(f"Retrieved {len(features_df)} samples for SKU tuples")
            return features_df, target_df
        else:
            return features_query, target_query
    
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

    def prepare_features_for_modeling(
        self, 
        features_df: pl.DataFrame, 
        target_df: pl.DataFrame,
        target_col: str = "target"
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        """
        Prepare features and target data for modeling without feature engineering.
        
        Args:
            features_df: Features DataFrame
            target_df: Target DataFrame  
            target_col: Target column name
            
        Returns:
            Tuple of (X, y, feature_columns)
        """
        # Join features with target
        df = features_df.join(target_df.select(["bdID", target_col]), on="bdID", how="left")
        
        # Get feature column names
        feature_cols = self.get_feature_columns(df)
        
        # Drop rows with null values
        df_clean = df.drop_nulls()
        
        # Separate features and target - include date for temporal splitting
        date_col = "date" if "date" in df_clean.columns else None
        if date_col:
            X = df_clean.select(["bdID", "date"] + feature_cols)
        else:
            X = df_clean.select(["bdID"] + feature_cols)
        y = df_clean.select(["bdID", target_col])
        
        logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")
        
        return X, y, feature_cols
