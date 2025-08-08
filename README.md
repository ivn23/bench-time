# M5 Benchmarking Framework

A production-ready framework for multi-granularity time series forecasting benchmarking using the M5 competition dataset. Provides memory-efficient data processing, XGBoost training with configurable hyperparameters, and comprehensive evaluation across SKU, Product, and Store levels.

## What This Framework Does

**Multi-Granularity Forecasting**: Train and evaluate time series forecasting models at three hierarchical levels:
- **SKU Level**: Individual products at specific stores
- **Product Level**: Same product aggregated across stores  
- **Store Level**: All products within specific stores

**Complete ML Pipeline**: End-to-end machine learning workflow including data loading, feature engineering, model training, hyperparameter optimization, evaluation, and model persistence.

**Capabilities**: Experiment tracking, model registry, evaluation metrics, and automated reporting.

## How It Works

### Architecture
```
src/
├── data_structures.py      # Core data structures and configuration
├── data_loading.py         # Polars-based data loading with lazy evaluation
├── feature_engineering.py  # Multi-granularity feature engineering  
├── model_training.py       # XGBoost training with configurable hyperparameters
├── evaluation.py           # Comprehensive evaluation and visualization
├── benchmark_pipeline.py   # Main orchestration pipeline
└── __init__.py            # Package initialization
```

### Core Components

**Data Management**: Uses Polars LazyFrames for memory-efficient loading and processing. Supports dynamic filtering by granularity without data duplication. Maintains proper temporal ordering for time series integrity.

**Feature Engineering**: Creates lag features (1-28 days), calendric features (month, weekday, etc.), and trend features. Automatically handles granularity-specific grouping and removes target-related columns to prevent data leakage.

**Model Training**: XGBoost models with configurable hyperparameters. Supports proper time series cross-validation and is extensible to other algorithms.

**Evaluation**: Multiple metrics (RMSE, MAE, R², MAPE), model comparison, visualization generation, and automated report creation.

**Model Registry**: Complete model persistence with metadata, hyperparameters, performance metrics, and data split information for reproducibility.

## How to Use

### Installation

```bash
git clone <repository-url>
cd Benchmarking
pip install -r requirements.txt
```

**Required Data Files** (place in `data/` directory):
- `train_data_features.feather` - M5 feature matrix
- `train_data_target.feather` - Target values  
- `feature_mapping_train.pkl` - Feature metadata

### Basic Usage

```python
from src import BenchmarkPipeline, DataConfig, TrainingConfig, GranularityLevel
from pathlib import Path

# Configure data and training
data_config = DataConfig(
    features_path="data/train_data_features.feather",
    target_path="data/train_data_target.feather",
    mapping_path="data/feature_mapping_train.pkl",
    lag_features=[1, 2, 3, 7, 14],
    calendric_features=True,
    trend_features=True
)

training_config = TrainingConfig(
    validation_split=0.2,
    n_trials=50,
    model_type="xgboost"
)

# Initialize and run pipeline
pipeline = BenchmarkPipeline(data_config, training_config)
pipeline.load_and_prepare_data()

# Train single model
model = pipeline.run_single_model_experiment(
    granularity=GranularityLevel.SKU,
    entity_ids={"skuID": 278993},
    experiment_name="first_model"
)

# Evaluate model
evaluation_results = pipeline.evaluate_all_models()
print(f"Model RMSE: {evaluation_results['overall']['models'][0]['metrics']['rmse']:.4f}")
```

### Multi-Granularity Training

```python
# Train across multiple granularity levels
sample_entities = {
    "skuIDs": [278993, 279024, 279854],
    "productIDs": [80558, 80651, 80702],
    "storeIDs": [2, 5, 8]
}

all_models = pipeline.run_full_benchmark_suite(
    sample_entities=sample_entities,
    max_models_per_granularity=10
)

# Compare performance
for granularity, models in all_models.items():
    avg_rmse = np.mean([m.metadata.performance_metrics.get('rmse', 0) for m in models])
    print(f"{granularity.value}: {len(models)} models, avg RMSE: {avg_rmse:.4f}")
```

### Model Registry

```python
from src import ModelRegistry

registry = ModelRegistry(Path("my_models"))

# Save and load models
model_id = registry.register_model(trained_model)
registry.save_model(model_id)
loaded_model = registry.load_model(model_id)

# Query models
sku_models = registry.list_models(GranularityLevel.SKU)
```

## Dataset Requirements

**M5 Competition Format**:
- `train_data_features.feather`: 59M+ rows with M5 features in Arrow format
- `train_data_target.feather`: Aligned target values
- `feature_mapping_train.pkl`: Feature metadata

**Key Columns**: `bdID` (temporal), `skuID`/`productID`/`storeID` (entities), `date` (temporal), `feature_*` (predictors), `target` (forecast target)


## Dependencies

**Core**: polars>=0.19.0, numpy, pandas, scikit-learn, xgboost, optuna  
**Optional**: lets-plot (visualization), mlflow (experiment tracking), pyarrow (Arrow format)

```bash
pip install -r requirements.txt
```

## Output Structure

```
benchmark_results/
├── models/                    # Trained model storage
│   └── sku_278993_xgboost/   # Individual model directories
│       ├── model.pkl         # Serialized model
│       ├── metadata.json     # Model metadata
│       └── data_splits.json  # Train/validation splits
├── evaluation_results/        # Evaluation outputs
│   ├── evaluation_results.json
│   └── *_evaluation_report.md
└── experiment_log.json       # Complete experiment audit trail
```
