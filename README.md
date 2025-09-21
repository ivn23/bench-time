# M5 Time Series Benchmarking Framework

A streamlined framework for time series forecasting on M5 competition data with dual modeling strategies, quantile regression support, and comprehensive model lifecycle management.

## Features

- **Tuple-Based Modeling**: Work with SKU tuples (product_id, store_id) representing store-product combinations
- **Modeling Strategies**: COMBINED (single model for all SKUs) vs INDIVIDUAL (separate models per SKU)
- **Multiple Model Types**: XGBoost, PyTorch Lightning neural networks, and statistical baselines
- **Quantile Regression**: Multi-quantile models for uncertainty quantification
- **Release Management**: Complete model lifecycle with automated experiment tracking

## Installation

```bash
git clone <repository-url>
cd Benchmarking
pip install -r requirements.txt
```

**Required Data Files** (place in `data/` directory):
- `train_data_features.feather` - M5 feature matrix in Arrow format
- `train_data_target.feather` - Target values aligned with features
- `feature_mapping_train.pkl` - Feature metadata dictionary

## Quick Start

### Basic Usage

```python
from src import BenchmarkPipeline, DataConfig, ModelingStrategy

# Configure data loading
data_config = DataConfig(
    features_path="data/train_data_features.feather",
    target_path="data/train_data_target.feather",
    mapping_path="data/feature_mapping_train.pkl"
)

# Initialize pipeline
pipeline = BenchmarkPipeline(data_config=data_config)

# Define SKU tuples: (product_id, store_id)
sku_tuples = [(80558, 2), (80651, 5)]

# Train COMBINED model (one model for all SKUs)
results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.COMBINED,
    model_type="xgboost_standard",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "random_state": 42
    },
    experiment_name="combined_experiment"
)

print(f"Trained {results.num_models} model(s)")
model = results.models[0]
print(f"RMSE: {model.metadata.performance_metrics['rmse']:.4f}")
```

### Individual SKU Modeling

```python
# Train separate models for each SKU
results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="xgboost_standard",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "random_state": 42
    },
    experiment_name="individual_experiment"
)

print(f"Trained {results.num_models} individual models")
for model in results.models:
    sku = model.metadata.sku_tuples[0]
    rmse = model.metadata.performance_metrics['rmse']
    print(f"SKU {sku[0]}x{sku[1]}: RMSE {rmse:.4f}")
```

### Quantile Regression

```python
# Train quantile models for uncertainty quantification
results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.COMBINED,
    model_type="xgboost_quantile",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "quantile_alpha": 0.1,  # 10th percentile
        "random_state": 42
    },
    experiment_name="quantile_experiment"
)

# Check quantile-specific metrics
model = results.models[0]
metrics = model.metadata.performance_metrics
print(f"Coverage Probability: {metrics['coverage_probability']:.4f}")
print(f"Quantile Score: {metrics['quantile_score']:.4f}")
```

### Multi-Quantile Models

```python
# Train multiple quantile levels in one experiment
results = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="xgboost_quantile",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "random_state": 42
    },
    quantile_alphas=[0.1, 0.5, 0.9],  # Multiple quantile levels
    experiment_name="multi_quantile_experiment"
)

print(f"Trained {results.num_models} models across quantile levels")
```

## Available Model Types

- **xgboost_standard**: Standard XGBoost regression
- **xgboost_quantile**: Quantile XGBoost with coverage analysis
- **lightning_standard**: PyTorch Lightning neural networks
- **lightning_quantile**: Neural network quantile regression
- **statquant**: Fast statistical quantile baseline

Discover all available models:
```python
from src.model_types import model_registry
print(model_registry.list_available_types())
```

## Data Configuration Options

```python
data_config = DataConfig(
    features_path="data/train_data_features.feather",
    target_path="data/train_data_target.feather",
    mapping_path="data/feature_mapping_train.pkl",
    split_date="2020-03-01",  # Optional: specific split date
    validation_split=0.2,     # Optional: percentage split
    remove_not_for_sale=True  # Optional: filter out non-sale items
)
```

## Release Management

```python
from src import ReleaseManager

# Create experiment release
release_manager = ReleaseManager()
release_dir = release_manager.create_complete_release(
    experiment_results=results,
    base_output_dir=Path("releases")
)

print(f"Release created at: {release_dir}")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Quick integration test
pytest tests/test_basic_integration.py -v

# Multi-quantile functionality
pytest tests/test_multi_quantile.py -v
```

## Data Requirements

**Expected Schema:**
- **Features**: bdID, productID, storeID, date, feature_* columns
- **Targets**: bdID, target columns
- **Mapping**: Dictionary with feature metadata

**File Formats:**
- Features/Targets: Arrow/Feather format (.feather)
- Mapping: Pickle format (.pkl)

## Output Structure

Results are organized in release directories:
```
releases/experiment_name/
├── bundle.json      # Experiment metadata
├── metrics.json     # Performance metrics
└── model_params/    # Saved model files
```

## Dependencies

**Core Requirements:**
- polars>=0.19.0 (DataFrame operations)
- numpy>=1.21.0 (Numerical computations)
- scikit-learn>=1.0.0 (ML utilities)
- xgboost>=1.7.0 (Gradient boosting)
- torch>=1.9.0 (PyTorch models)
- pytorch-lightning>=1.5.0 (Training framework)

Install all dependencies:
```bash
pip install -r requirements.txt
```