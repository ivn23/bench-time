# Feature Engineering Documentation

This document explains the feature engineering pipeline in the M5 benchmarking framework and how to customize it for different use cases.

## Overview

The feature engineering is handled by the `FeatureEngineer` class in `src/feature_engineering.py`. It creates various types of features:

1. **Lag Features** - Historical values for prediction
2. **Calendric Features** - Time-based features (month, day of week, etc.)
3. **Trend Features** - Linear time trends
4. **Categorical Encoding** - One-hot encoding of categorical variables

## Key Design Decisions

### Column Dropping After Lag Feature Creation

**Important**: After creating lag features, certain columns are automatically dropped:

- `feature_0038`: This represents the raw sales values and is only used to compute lag features. Since it's essentially the target variable, it must be removed to prevent data leakage.
- `target_lag_1`: This is typically redundant with `feature_0038_lag_1` and can cause multicollinearity issues.

**Location**: This happens in the `_add_lag_features()` method in `src/feature_engineering.py` at lines 131-137.

```python
# Remove feature_0038 and lag_target_1 after creating lag features
# feature_0038 is only needed for lag computation (it's the target variable)
# lag_target_1 can be redundant with feature_0038_lag_1 (same values but 1-period shifted)
columns_to_drop = ["feature_0038", "target_lag_1"]
df = df.drop([col for col in columns_to_drop if col in df.columns])
```

## Feature Types

### 1. Lag Features

Lag features are created for both sales (`feature_0038`) and target values, then the original columns are dropped.

**Configuration**: 
- Set via `lag_features` parameter in `FeatureEngineer.__init__()`
- Default: `[1, 2, 3, 4, 5, 6, 7]` (1-7 day lags)

**Granularity Handling**:
- **SKU level**: Lags computed per `(skuID, frequency)` group
- **Product level**: Lags computed per `(productID, frequency)` group  
- **Store level**: Lags computed per `(storeID, frequency)` group

### 2. Calendric Features

Time-based features extracted from the date column:
- Month (1-12)
- Day of week (1-7)
- Week of year (1-53)
- Year
- Quarter (1-4)
- Is weekend (boolean)

All categorical features are automatically one-hot encoded.

### 3. Trend Features

Linear trend based on days since the earliest date in the dataset.

## Customization Guide

### Adding New Features

To add new feature types, modify the `create_features()` method:

```python
def create_features(self, features_df, target_df, granularity, entity_ids):
    # ... existing code ...
    
    # Add your custom features
    if self.custom_features:
        df = self._create_custom_features(df)
    
    # ... rest of method ...
```

### Modifying Column Dropping Behavior

To change which columns are dropped after lag computation:

1. **Location**: `src/feature_engineering.py`, `_add_lag_features()` method
2. **Modify**: The `columns_to_drop` list at line 134

```python
# Customize which columns to drop
columns_to_drop = ["feature_0038", "target_lag_1", "your_column_here"]
df = df.drop([col for col in columns_to_drop if col in df.columns])
```

3. **Update metadata exclusion**: Also update `_get_feature_columns()` method to exclude these from the feature list:

```python
metadata_cols = {
    "frequency", "idx", "bdID", "base_date", "date", "dateID", 
    "skuID", "productID", "storeID", "companyID", "missing_value", 
    "not_for_sale", "target", "feature_0038", "target_lag_1",
    "your_column_here"  # Add new columns to exclude
}
```

### Adding New Lag Sources

To create lags from additional columns:

1. Modify the `value_cols` list in `_add_lag_features()`:

```python
# Add new columns to create lags from
value_cols = ["feature_0038", "target", "feature_0039", "your_feature"]
```

2. Consider whether the source column should be dropped after lag creation.

### Changing Lag Periods

To modify which lag periods are created:

```python
# Initialize with different lags
feature_engineer = FeatureEngineer(
    lag_features=[1, 2, 3, 7, 14, 28],  # Daily, weekly, bi-weekly, monthly
    calendric_features=True,
    trend_features=True
)
```

## Granularity-Specific Considerations

### SKU Level
- Most straightforward - lags computed per individual SKU
- Full feature set available

### Product Level  
- **No aggregation**: Individual SKU observations are preserved within the product
- **SKU dummy features**: Binary indicator variables created for each SKU (e.g., `sku_12345_dummy`)
- **Lag features enabled**: Since observations are preserved, lag computation works normally per SKU
- **Enhanced modeling**: Model can learn SKU-specific patterns within the product

### Store Level
- Features aggregated across products before lag computation  
- Similar aggregation rules as product level
- Store-specific patterns captured

## Data Leakage Prevention

**Critical**: Always ensure that:

1. Target-related columns (`feature_0038`) are dropped after lag creation
2. Future information is not used (proper temporal ordering)
3. Lag computation respects grouping boundaries
4. Cross-validation maintains temporal order

## Testing Changes

After modifying feature engineering:

1. Run the test notebook to verify changes work end-to-end
2. Check that feature counts are as expected
3. Verify no data leakage by examining feature correlations
4. Test across different granularity levels

## Common Issues

### Memory Efficiency
- Use Polars lazy evaluation when possible
- Drop unnecessary columns early in the pipeline
- Consider feature selection for high-dimensional outputs

### Temporal Consistency
- Always sort by date before lag computation
- Ensure group-by operations maintain temporal order
- Validate split dates align with lag periods

### Missing Values
- Handle missing values before lag computation
- Consider forward-fill strategies for sparse time series
- Document assumptions about missing data treatment

## Example Usage

```python
from src.feature_engineering import FeatureEngineer
from src.data_structures import GranularityLevel

# Initialize with custom settings
feature_engineer = FeatureEngineer(
    lag_features=[1, 2, 3, 7],  # Shorter lag window
    calendric_features=True,
    trend_features=False  # Disable trend features
)

# Create features for SKU-level modeling
df_engineered, feature_cols = feature_engineer.create_features(
    features_df=features_df,
    target_df=target_df, 
    granularity=GranularityLevel.SKU,
    entity_ids={"skuID": 12345}
)

# Prepare for modeling (includes additional cleaning)
X, y = feature_engineer.prepare_model_data(df_engineered, feature_cols)
```

---

**Note**: This pipeline is designed to match the patterns from the original `load_data.ipynb` notebook while providing production-ready extensibility and error handling.