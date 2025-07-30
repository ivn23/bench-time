"""
Feature engineering pipeline for different granularity levels.
Extends the existing Polars-based feature engineering from the notebook.
"""

import polars as pl
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

#from SDK import f_x , g_x 
from .data_structures import GranularityLevel

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline that works across different granularity levels."""
    
    def __init__(self, 
                 lag_features: List[int] = [1, 2, 3, 4, 5, 6, 7],
                 calendric_features: bool = True,
                 trend_features: bool = True):
        self.lag_features = lag_features
        self.calendric_features = calendric_features  
        self.trend_features = trend_features
    
    def create_features(self, 
                       features_df: pl.DataFrame, 
                       target_df: pl.DataFrame,
                       granularity: GranularityLevel,
                       entity_ids: Dict[str, any]) -> Tuple[pl.DataFrame, List[str]]:
        """
        Create full feature set for specified granularity level.
        
        Args:
            features_df: Base features DataFrame
            target_df: Target DataFrame  
            granularity: Level of aggregation
            entity_ids: Entity identifiers
            
        Returns:
            Tuple of (engineered_features_df, feature_column_names)
        """
        logger.info(f"Creating features for {granularity.value} level")
        
        # Join features with target to get target values for lag features
        df = features_df.join(target_df.select(["bdID", "target"]), on="bdID", how="left")
        
        # Create calendric features if enabled
        if self.calendric_features:
            df = self._create_calendric_features(df, "date")
        
        # Create lag features if enabled
        if self.lag_features:
            # Now works for all granularity levels including PRODUCT with SKU dummies
            df = self._add_lag_features(df, granularity, entity_ids)
        
        # Create trend features if enabled
        if self.trend_features:
            df = self._create_trend_features(df)
        
        # Get feature column names (exclude metadata columns)
        feature_cols = self._get_feature_columns(df)
        
        logger.info(f"Created {len(feature_cols)} features")
        
        return df, feature_cols
    
    def _create_calendric_features(self, df: pl.DataFrame, date_column: str) -> pl.DataFrame:
        """Create calendric features from date column."""
        logger.debug("Creating calendric features")
        
        # Ensure date column is properly typed
        df = df.with_columns(pl.col(date_column).cast(pl.Date).alias(date_column))
        
        # Create base calendric features
        df = df.with_columns([
            pl.col(date_column).dt.month().alias("month"),
            pl.col(date_column).dt.weekday().alias("day_of_week"), 
            pl.col(date_column).dt.strftime("%V").cast(pl.Int32).alias("week_of_year"),
            pl.col(date_column).dt.year().alias("year"),
            (pl.col(date_column).dt.weekday() >= 5).alias("is_weekend")
        ])
        
        # Create quarter feature
        df = df.with_columns(((pl.col("month") - 1) // 3 + 1).alias("quarter"))
        
        # Convert categorical features to dummy variables
        categorical_cols = ["day_of_week", "month", "quarter", "week_of_year", "year", "is_weekend"]
        
        for col in categorical_cols:
            if col in df.columns:
                df = df.to_dummies(columns=[col], separator="_")
        
        return df
    
    def _add_lag_features(self, 
                         df: pl.DataFrame, 
                         granularity: GranularityLevel,
                         entity_ids: Dict[str, any]) -> pl.DataFrame:
        """Add lag features based on granularity level."""
        logger.debug(f"Creating lag features for {granularity.value} level")
        
        # Define grouping columns based on granularity
        if granularity == GranularityLevel.SKU:
            group_cols = ["skuID", "frequency"]
            sort_cols = ["skuID", "frequency", "date"]
        elif granularity == GranularityLevel.PRODUCT:
            group_cols = ["productID", "frequency"] 
            sort_cols = ["productID", "frequency", "date"]
        elif granularity == GranularityLevel.STORE:
            group_cols = ["storeID", "frequency"]
            sort_cols = ["storeID", "frequency", "date"]
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
        
        # Sort data by time within groups
        df = df.sort(sort_cols)
        
        # Create lag features for sales (feature_0038) and target
        value_cols = ["feature_0038", "target"]
        
        for value_col in value_cols:
            if value_col in df.columns:
                lag_features = [
                    pl.col(value_col).shift(lag).over(group_cols).alias(f"{value_col}_lag_{lag}")
                    for lag in self.lag_features
                ]
                df = df.with_columns(lag_features)
        
        # Remove feature_0038 and lag_target_1 after creating lag features
        # feature_0038 is only needed for lag computation (it's the target variable)
        # lag_target_1 can be redundant with feature_0038_lag_1 (same values but 1-period shifted)
        columns_to_drop = ["feature_0038", "target_lag_1"]
        df = df.drop([col for col in columns_to_drop if col in df.columns])
        
        logger.debug(f"Dropped columns after lag computation: {[col for col in columns_to_drop if col in df.columns]}")
        
        return df
    
    def _create_trend_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create trend features."""
        logger.debug("Creating trend features")
        
        # Create linear trend based on date
        earliest_date = df.select("date").min().item()
        latest_date = df.select("date").max().item()
        
        # Create date range and trend mapping
        date_range = pl.date_range(earliest_date, latest_date, "1d", eager=True)
        trend_values = pl.int_range(1, len(date_range) + 1, eager=True)
        
        trend_df = pl.DataFrame({"date": date_range, "trend": trend_values})
        
        # Join trend feature
        df = df.join(trend_df, on="date", how="left")
        
        return df
    
    def _get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Get list of feature columns, excluding metadata columns."""
        # Define metadata columns to exclude
        metadata_cols = {
            "frequency", "idx", "bdID", "base_date", "date", "dateID", 
            "skuID", "productID", "storeID", "companyID", "missing_value", 
            "not_for_sale", "target", "feature_0038", "target_lag_1"  # Add dropped columns
        }
        
        # Get all columns except metadata
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        return feature_cols
    
    def prepare_model_data(self, 
                          df: pl.DataFrame, 
                          feature_cols: List[str],
                          target_col: str = "target") -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Prepare final X and y DataFrames for modeling.
        
        Args:
            df: Engineered features DataFrame
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) DataFrames
        """
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
        
        return X, y
    
    def get_feature_importance_mapping(self, 
                                     feature_cols: List[str]) -> Dict[str, str]:
        """Create mapping from feature names to human-readable descriptions."""
        feature_mapping = {}
        
        for col in feature_cols:
            if col.startswith("feature_"):
                # Map to original feature names if available
                if col == "feature_0038":
                    feature_mapping[col] = "sales"
                elif col == "feature_0039":
                    feature_mapping[col] = "price"
                else:
                    feature_mapping[col] = f"event_{col.split('_')[-1]}"
            elif "_lag_" in col:
                base_name = col.split("_lag_")[0]
                lag_num = col.split("_lag_")[1]
                if base_name == "feature_0038":
                    feature_mapping[col] = f"sales_lag_{lag_num}"
                elif base_name == "target":
                    feature_mapping[col] = f"target_lag_{lag_num}"
                else:
                    feature_mapping[col] = f"{base_name}_lag_{lag_num}"
            elif col in ["month", "day_of_week", "quarter", "week_of_year", "year", "is_weekend", "trend"]:
                feature_mapping[col] = col
            else:
                feature_mapping[col] = col
        
        return feature_mapping