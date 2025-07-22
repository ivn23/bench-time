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

from .data_structures import DataConfig, GranularityLevel

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
    
    def get_data_for_granularity(
        self, 
        granularity: GranularityLevel,
        entity_ids: Dict[str, any],
        collect: bool = True
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Filter and aggregate data for specific granularity level.
        
        Args:
            granularity: SKU, PRODUCT, or STORE level
            entity_ids: Dictionary specifying which entities to include
            collect: Whether to collect LazyFrame to DataFrame
            
        Returns:
            Tuple of (filtered_features, filtered_target)
        """
        if not self._is_loaded:
            raise ValueError("Data must be loaded first. Call load_data().")
        
        features_query = self._features_df
        target_query = self._target_df
        
        # Apply granularity-specific filtering and aggregation
        if granularity == GranularityLevel.SKU:
            # SKU level: filter by specific SKU ID
            sku_id = entity_ids.get("skuID")
            if sku_id is None:
                raise ValueError("skuID must be specified for SKU granularity")
            
            features_query = features_query.filter(pl.col("skuID") == sku_id)
            
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
        
        elif granularity == GranularityLevel.PRODUCT:
            # Product level: filter by productID, create SKU dummy features
            product_id = entity_ids.get("productID")
            if product_id is None:
                raise ValueError("productID must be specified for PRODUCT granularity")
            
            features_query = features_query.filter(pl.col("productID") == product_id)
            
            # Create dummy features for each SKU within this product instead of aggregating
            # This preserves all individual observations while encoding SKU identity
            features_query = self._create_sku_dummies_for_product(features_query)
            
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
                target_query = target_query.join(
                    features_query.select("bdID"), 
                    on="bdID", 
                    how="inner"
                )
                
        elif granularity == GranularityLevel.STORE:
            # Store level: filter by storeID, aggregate across products
            store_id = entity_ids.get("storeID")
            if store_id is None:
                raise ValueError("storeID must be specified for STORE granularity")
            
            features_query = features_query.filter(pl.col("storeID") == store_id)
            
            # Aggregate features by date and store
            features_query = self._aggregate_by_store(features_query)
            
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
                
            return features_df, target_df
        else:
            return features_query, target_query
    
    def _aggregate_by_product(self, df_query: pl.LazyFrame) -> pl.LazyFrame:
        """Aggregate features at product level (across stores)."""
        # Define aggregation rules for different feature types
        sales_features = [col for col in df_query.columns if "feature_0038" in col or "sales" in col.lower()]
        price_features = [col for col in df_query.columns if "feature_0039" in col or "price" in col.lower()]
        
        # Aggregate by date and product
        agg_exprs = []
        
        # Sum sales-related features
        for col in sales_features:
            if col in df_query.columns:
                agg_exprs.append(pl.col(col).sum().alias(col))
        
        # Average price-related features
        for col in price_features:
            if col in df_query.columns:
                agg_exprs.append(pl.col(col).mean().alias(col))
        
        # Keep first value for categorical features (events, etc.)
        categorical_features = [col for col in df_query.columns 
                              if col.startswith("feature_") and col not in sales_features + price_features]
        for col in categorical_features:
            agg_exprs.append(pl.col(col).first().alias(col))
        
        # Keep key identifiers
        group_cols = ["date", "productID", "dateID"]
        keep_cols = ["frequency", "companyID", "bdID"]  # Include bdID and keep these as first values
        
        for col in keep_cols:
            if col in df_query.columns:
                agg_exprs.append(pl.col(col).first().alias(col))
        
        return df_query.group_by(group_cols).agg(agg_exprs)
    
    def _aggregate_by_store(self, df_query: pl.LazyFrame) -> pl.LazyFrame:
        """Aggregate features at store level (across products)."""
        # Similar to product aggregation but group by store instead
        sales_features = [col for col in df_query.columns if "feature_0038" in col or "sales" in col.lower()]
        price_features = [col for col in df_query.columns if "feature_0039" in col or "price" in col.lower()]
        
        agg_exprs = []
        
        # Sum sales-related features
        for col in sales_features:
            if col in df_query.columns:
                agg_exprs.append(pl.col(col).sum().alias(col))
        
        # Average price-related features
        for col in price_features:
            if col in df_query.columns:
                agg_exprs.append(pl.col(col).mean().alias(col))
        
        # Keep first value for categorical features
        categorical_features = [col for col in df_query.columns 
                              if col.startswith("feature_") and col not in sales_features + price_features]
        for col in categorical_features:
            agg_exprs.append(pl.col(col).first().alias(col))
        
        group_cols = ["date", "storeID", "dateID"]
        keep_cols = ["frequency", "companyID", "bdID"]  # Include bdID
        
        for col in keep_cols:
            if col in df_query.columns:
                agg_exprs.append(pl.col(col).first().alias(col))
        
        return df_query.group_by(group_cols).agg(agg_exprs)
    
    def _create_sku_dummies_for_product(self, df_query: pl.LazyFrame) -> pl.LazyFrame:
        """Create dummy variables for each SKU within a product instead of aggregating."""
        # Get unique SKUs for this product (need to collect to get the list)
        if hasattr(df_query, 'collect'):
            unique_skus = df_query.select("skuID").unique().collect().to_series().to_list()
        else:
            unique_skus = df_query.select("skuID").unique().to_series().to_list()
        
        logger.info(f"Creating dummy variables for {len(unique_skus)} SKUs in this product")
        
        # Create dummy variables for each SKU
        # Use Polars' when-then-otherwise for creating binary indicators
        dummy_expressions = []
        for sku_id in unique_skus:
            dummy_col_name = f"sku_{sku_id}_dummy"
            dummy_expr = pl.when(pl.col("skuID") == sku_id).then(1).otherwise(0).alias(dummy_col_name)
            dummy_expressions.append(dummy_expr)
        
        # Add all dummy columns to the dataframe
        df_with_dummies = df_query.with_columns(dummy_expressions)
        
        return df_with_dummies
    
    def get_unique_entities(self) -> Dict[str, List]:
        """Get lists of unique entities for each granularity level."""
        if not self._is_loaded:
            self.load_data()
        
        # Collect unique values
        df = self._features_df
        
        unique_entities = {
            "skuIDs": df.select("skuID").unique().to_series().to_list(),
            "productIDs": df.select("productID").unique().to_series().to_list(),
            "storeIDs": df.select("storeID").unique().to_series().to_list()
        }
        
        logger.info(f"Found {len(unique_entities['skuIDs'])} unique SKUs")
        logger.info(f"Found {len(unique_entities['productIDs'])} unique products")  
        logger.info(f"Found {len(unique_entities['storeIDs'])} unique stores")
        
        return unique_entities
    
    def create_temporal_split(
        self, 
        df: pl.DataFrame, 
        validation_split: float = 0.2,
        date_column: str = "date"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temporal train/validation split maintaining chronological order.
        
        Args:
            df: DataFrame to split
            validation_split: Fraction for validation set
            date_column: Column containing dates
            
        Returns:
            Tuple of (train_bdIDs, validation_bdIDs)
        """
        # Sort by date to ensure chronological order
        df_sorted = df.sort(date_column)
        
        # Calculate split point
        n_total = len(df_sorted)
        n_train = int(n_total * (1 - validation_split))
        
        # Get bdIDs for train and validation
        train_bdids = df_sorted[:n_train].select("bdID").to_numpy().flatten()
        val_bdids = df_sorted[n_train:].select("bdID").to_numpy().flatten()
        
        split_date = df_sorted[n_train].select(date_column).item()
        
        logger.info(f"Created temporal split: {len(train_bdids)} train, {len(val_bdids)} validation")
        logger.info(f"Split date: {split_date}")
        
        return train_bdids, val_bdids, split_date