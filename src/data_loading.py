"""
Data loading and preprocessing module.
Memory-efficient data loading using Polars with filtering capabilities.
Centralized data preparation for all model training and evaluation needs.
"""

import polars as pl
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import os

from .structures import DataConfig, ModelingStrategy, ModelingDataset, SkuList, TrainingResult

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
        Create temporal train/test split based on percentage.
        
        Args:
            df: DataFrame to split
            validation_split: Proportion of data for test set (despite the name, this creates test data)
            date_column: Date column name for temporal ordering
            
        Returns:
            Tuple of (train_bdIDs, test_bdIDs, split_info)
        """
        # Get unique dates and sort them
        unique_dates = df.select(date_column).unique().sort(date_column)
        dates = unique_dates.to_series().to_list()
        
        # Calculate split index
        split_idx = int(len(dates) * (1 - validation_split))
        split_date = dates[split_idx] if split_idx < len(dates) else dates[-1]
        
        # Split data based on date
        train_data = df.filter(pl.col(date_column) < split_date)
        test_data = df.filter(pl.col(date_column) >= split_date)
        
        train_bdids = train_data.select("bdID").to_numpy().flatten()
        test_bdids = test_data.select("bdID").to_numpy().flatten()
        
        return train_bdids, test_bdids, str(split_date)
    
    def create_temporal_split_by_date(
        self, 
        df: pl.DataFrame, 
        split_date: str,
        date_column: str = "date"
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Create temporal train/test split based on a specific date.
        
        Args:
            df: DataFrame to split
            split_date: Date to use for splitting (format: 'YYYY-MM-DD')
            date_column: Column containing dates
            
        Returns:
            Tuple of (train_bdIDs, test_bdIDs, split_date)
        """
        # Sort by date to ensure chronological order
        df_sorted = df.sort(date_column)
        
        # Convert split_date string to date for comparison
        split_date_obj = pl.lit(split_date).str.strptime(pl.Date, "%Y-%m-%d")
        
        # Split based on date
        train_mask = pl.col(date_column) < split_date_obj
        test_mask = pl.col(date_column) >= split_date_obj
        
        train_bdids = df_sorted.filter(train_mask).select("bdID").to_numpy().flatten()
        test_bdids = df_sorted.filter(test_mask).select("bdID").to_numpy().flatten()
        
        logger.info(f"Created temporal split by date: {len(train_bdids)} train, {len(test_bdids)} test")
        logger.info(f"Split date: {split_date}")
        
        return train_bdids, test_bdids, split_date

    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get list of feature columns, excluding metadata columns."""
        # Define metadata columns to exclude
        metadata_cols = {
            "frequency", "idx", "bdID", "base_date", "date", "dateID", 
            "skuID", "productID", "storeID", "companyID", "missing_value", 
            "not_for_sale", "target", "is_daily", "name", "name-2"
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
            sku_tuples=sku_tuples,
            modeling_strategy=modeling_strategy,
            train_bdids=split_info.get('train_bdids', np.array([])),
            test_bdids=split_info.get('test_bdids', np.array([])),
            split_date=split_info.get('split_date')
        )
    
    def _filter_sku_data(self, sku_tuples: SkuList, strategy: ModelingStrategy) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Filter data for specified SKUs using efficient Polars operations."""
        logger.info(f"Filtering data for {len(sku_tuples)} SKUs")
        
        # For single SKU (INDIVIDUAL strategy common case), use direct filtering
        if len(sku_tuples) == 1:
            product_id, store_id = sku_tuples[0]
            features_filtered = (self._features_df
                               .filter(pl.col('productID') == product_id)
                               .filter(pl.col('storeID') == store_id)
                               .drop_nulls()
                               .sort("date", "skuID"))
        else:
            # For multiple SKUs (COMBINED strategy), use is_in
            product_ids = [sku[0] for sku in sku_tuples]
            store_ids = [sku[1] for sku in sku_tuples]
            
            features_filtered = (self._features_df
                               .filter(pl.col('productID').is_in(product_ids))
                               .filter(pl.col('storeID').is_in(store_ids))
                               .drop_nulls()
                               .sort("date", "skuID"))
        
        # Get bdIDs and filter targets - direct operations on DataFrames
        feature_bdids = (features_filtered
                        .select('bdID')
                        .unique()
                        .to_numpy()
                        .flatten())
        
        target_filtered = (self._target_df
                          .filter(pl.col("bdID").is_in(feature_bdids))
                          .join(features_filtered.select("bdID", "skuID"), 
                               on="bdID", how="left")
                          .sort("date", "skuID"))
        
        logger.info(f"Filtered to {len(features_filtered)} samples")
        return features_filtered, target_filtered
    
    def _prepare_features(self, features_df: pl.DataFrame, target_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:

        
        
        

        feature_cols = self.get_feature_columns(features_df)

        X = features_df.select(feature_cols + ["bdID","date","skuID"])
        y = target_df.select([self.config.target_column, "bdID","date","skuID"])
        
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
        #sort by date and skuID
        X_train = X_train.sort("date","skuID")
        y_train = y_train.sort("date","skuID")
        X_test = X_test.sort("date","skuID")
        y_test = y_test.sort("date","skuID")
        #drop bdid and date columns from X and y
        X_train = X_train.drop(["bdID","date","skuID"])
        y_train = y_train.drop(["bdID","date","skuID"])
        X_test = X_test.drop(["bdID","date","skuID"])
        y_test = y_test.drop(["bdID","date","skuID"])

        return X_train, y_train, X_test, y_test, split_info
    
    # ===================================================================
    # CENTRALIZED DATA PREPARATION METHODS
    # All data format conversion and preparation logic consolidated here
    # ===================================================================
    
    @staticmethod
    def prepare_training_data(dataset: ModelingDataset, model_type: str) -> Tuple[Any, Any]:
        """
        Prepare training data in model-specific format.
        
        Args:
            dataset: ModelingDataset containing training data
            model_type: Model type determining format (xgboost needs pandas, others numpy)
            
        Returns:
            Tuple of (X_train, y_train) in appropriate format
        """
        if "xgboost" in model_type.lower():
            # XGBoost models need pandas DataFrames for feature names
            X_train = dataset.X_train.select(dataset.feature_cols).to_pandas()
            y_train = dataset.y_train.select(dataset.target_col).to_pandas()[dataset.target_col]
        else:
            # All other models use numpy arrays
            X_train = dataset.X_train.select(dataset.feature_cols).to_numpy()
            y_train = dataset.y_train.select(dataset.target_col).to_numpy().flatten()
        
        logger.debug(f"Prepared training data for {model_type}: X shape {X_train.shape}, y shape {y_train.shape}")
        return X_train, y_train
    
    @staticmethod
    def prepare_test_data(dataset: ModelingDataset, model_type: str) -> Tuple[Any, Any]:
        """
        Prepare test data in model-specific format.
        
        Args:
            dataset: ModelingDataset containing test data
            model_type: Model type determining format (xgboost needs pandas, others numpy)
            
        Returns:
            Tuple of (X_test, y_test) in appropriate format
        """
        if "xgboost" in model_type.lower():
            # XGBoost models need pandas DataFrames for feature names
            X_test = dataset.X_test.select(dataset.feature_cols).to_pandas()
            y_test = dataset.y_test.select(dataset.target_col).to_pandas()[dataset.target_col]
        else:
            # All other models use numpy arrays
            X_test = dataset.X_test.select(dataset.feature_cols).to_numpy()
            y_test = dataset.y_test.select(dataset.target_col).to_numpy().flatten()
        
        logger.debug(f"Prepared test data for {model_type}: X shape {X_test.shape}, y shape {y_test.shape}")
        return X_test, y_test
    
    @staticmethod
    def prepare_evaluation_data(X_test: pl.DataFrame, y_test: pl.DataFrame, 
                              training_result: TrainingResult) -> Tuple[Any, Any]:
        """
        Filter and format test data for model evaluation.
        
        Args:
            X_test: Test features DataFrame
            y_test: Test targets DataFrame
            training_result: TrainingResult containing model info and split details
            
        Returns:
            Tuple of (X_test_filtered, y_test_filtered) in model-appropriate format
        """
        # Use test bdIDs from training result split info
        bdids_to_use = training_result.split_info.test_bdIDs
        
        # Filter test data to match the bdIDs used during training
        test_features = X_test.filter(pl.col("bdID").is_in(bdids_to_use))
        test_targets = y_test.filter(pl.col("bdID").is_in(bdids_to_use))
        
        # Convert to model-specific format
        model_type = training_result.model_type
        if "xgboost" in model_type.lower():
            X_test_filtered = test_features.select(training_result.feature_columns).to_pandas()
        else:
            X_test_filtered = test_features.select(training_result.feature_columns).to_numpy()
        
        y_test_filtered = test_targets.select(training_result.target_column).to_numpy().flatten()
        
        logger.debug(f"Prepared evaluation data for {model_type}: {len(X_test_filtered)} samples")
        return X_test_filtered, y_test_filtered
    
    @staticmethod
    def convert_to_model_format(X: pl.DataFrame, y: pl.DataFrame, 
                              feature_cols: List[str], target_col: str, 
                              model_type: str) -> Tuple[Any, Any]:
        """
        Convert polars DataFrames to model-specific format.
        
        Args:
            X: Features DataFrame
            y: Targets DataFrame  
            feature_cols: List of feature column names
            target_col: Target column name
            model_type: Model type determining output format
            
        Returns:
            Tuple of (X_converted, y_converted) in appropriate format
        """
        if "xgboost" in model_type.lower():
            # XGBoost models need pandas DataFrames
            X_converted = X.select(feature_cols).to_pandas()
            y_converted = y.select(target_col).to_pandas()[target_col]
        else:
            # All other models use numpy arrays
            X_converted = X.select(feature_cols).to_numpy()
            y_converted = y.select(target_col).to_numpy().flatten()
        
        return X_converted, y_converted
