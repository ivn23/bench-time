"""
Mock data generation utilities for testing the M5 benchmarking framework.
Creates realistic but small datasets that match the expected schema.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import pickle
from pathlib import Path

# Set random seed for deterministic test data generation
np.random.seed(12345)

def generate_sample_features_data(
    n_skus: int = 5, 
    n_days: int = 100, 
    start_date: str = "2020-01-01"
) -> pl.DataFrame:
    """Generate sample features DataFrame matching M5 schema."""
    
    # Create date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start_dt + timedelta(days=i) for i in range(n_days)]
    
    # Sample SKU configurations
    sample_skus = [
        {"productID": 80558, "storeID": 2},
        {"productID": 80558, "storeID": 5}, 
        {"productID": 80651, "storeID": 2},
        {"productID": 80651, "storeID": 5},
        {"productID": 81234, "storeID": 3},
    ][:n_skus]
    
    data = []
    bdid_counter = 1000
    
    for sku in sample_skus:
        for i, date in enumerate(dates):
            # Create realistic but simple feature patterns
            row = {
                "bdID": bdid_counter,
                "date": date,
                "productID": sku["productID"],
                "storeID": sku["storeID"],
                "skuID": sku["productID"] * 1000 + sku["storeID"],  # Simple SKU ID generation
                
                # Calendar features
                "month": date.month,
                "day_of_week": date.weekday(),
                "week_of_year": date.isocalendar()[1],
                "quarter": (date.month - 1) // 3 + 1,
                "year": date.year,
                "is_weekend": 1 if date.weekday() >= 5 else 0,
                
                # Event features (simplified)
                "event_Christmas_0": 1 if date.month == 12 and date.day == 25 else 0,
                "event_NewYear_0": 1 if date.month == 1 and date.day == 1 else 0,
                "event_Halloween_0": 1 if date.month == 10 and date.day == 31 else 0,
                
                # Price and sales features
                "price_0": 10.0 + np.random.normal(0, 2),  # Base price around $10
                "sales_0": max(0, np.random.poisson(5) + (1 if date.weekday() >= 5 else 0)),  # Higher weekend sales
                
                # Additional numeric features
                "feature_0038": max(0, np.random.poisson(5)),  # Sales data feature
                "feature_0039": 10.0 + np.random.normal(0, 2),  # Price feature
                
                # Trend feature
                "trend": i / n_days,  # Simple linear trend
                
                # Add not_for_sale column (0 = for sale, 1 = not for sale)
                "not_for_sale": 0  # All items are for sale in test data
            }
            data.append(row)
            bdid_counter += 1
    
    return pl.DataFrame(data)

def generate_sample_target_data(
    features_df: pl.DataFrame, 
    target_col: str = "target"
) -> pl.DataFrame:
    """Generate target DataFrame based on features."""
    
    # Create targets with some relationship to sales and day patterns
    targets = []
    
    for row in features_df.iter_rows(named=True):
        # Base target influenced by sales and weekend patterns
        base_target = max(0, 
            row["sales_0"] + 
            np.random.normal(0, 1) + 
            (2 if row["is_weekend"] else 0) +
            (1 if row["event_Christmas_0"] else 0)
        )
        
        targets.append({
            "bdID": row["bdID"],
            target_col: int(base_target)
        })
    
    return pl.DataFrame(targets)

def generate_sample_mapping() -> Dict:
    """Generate sample feature mapping dictionary."""
    return {
        "feature_0000": {"feature_name": "event_Halloween_0", "company_name": "kaggle_m5"},
        "feature_0001": {"feature_name": "event_Christmas_0", "company_name": "kaggle_m5"},
        "feature_0002": {"feature_name": "event_NewYear_0", "company_name": "kaggle_m5"},
        "feature_0038": {"feature_name": "sales_0", "company_name": "kaggle_m5"},
        "feature_0039": {"feature_name": "price_0", "company_name": "kaggle_m5"},
    }

def save_sample_data_to_temp(temp_dir: Path) -> Tuple[Path, Path, Path]:
    """Generate and save sample data to temporary directory."""
    # Generate data
    features_df = generate_sample_features_data()
    target_df = generate_sample_target_data(features_df)
    mapping_dict = generate_sample_mapping()
    
    # Save to files
    features_path = temp_dir / "sample_features.feather"
    target_path = temp_dir / "sample_target.feather" 
    mapping_path = temp_dir / "sample_mapping.pkl"
    
    features_df.write_ipc(features_path)
    target_df.write_ipc(target_path)
    
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping_dict, f)
    
    return features_path, target_path, mapping_path

def create_sample_sku_tuples(n_tuples: int = 3) -> List[Tuple[int, int]]:
    """Create sample SKU tuples for testing."""
    sample_tuples = [
        (80558, 2),
        (80558, 5), 
        (80651, 2),
        (80651, 5),
        (81234, 3)
    ]
    return sample_tuples[:n_tuples]