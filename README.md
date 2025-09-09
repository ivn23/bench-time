# M5 Time Series Benchmarking Framework

A production-ready, tuple-based framework for time series forecasting on M5 competition data. Features dual modeling strategies (COMBINED and INDIVIDUAL), centralized metrics calculation, dynamic model discovery, multi-quantile regression support, PyTorch Lightning neural networks, statistical baselines, and comprehensive model lifecycle management with complete experiment tracking.

## Key Features

**Tuple-Based Modeling**: Work with SKU tuples (product_id, store_id pairs) representing individual store-product combinations. Choose between COMBINED strategy (one model for all SKUs) or INDIVIDUAL strategy (separate models per SKU).

**Centralized Metrics System**: Unified MetricsCalculator provides consistent evaluation across all models with comprehensive metrics including MSE, RMSE, MAE, R², MAPE, accuracy bands, and quantile-specific metrics.

**Dynamic Model Discovery**: Plugin architecture with ModelTypeRegistry automatically discovers and registers available model types, supporting XGBoost (standard/quantile), PyTorch Lightning neural networks, and statistical baselines.

**Multi-Quantile Regression Support**: Native support for quantile models with multiple quantile levels, coverage probability analysis, and specialized neural network implementations for uncertainty quantification and risk assessment.

**Fixed Hyperparameter Training**: Direct hyperparameter specification eliminates optimization complexity, enabling rapid experimentation cycles with predetermined parameters.

**Hierarchical Model Storage**: Organized model persistence with complete metadata tracking, data splits preservation, and intelligent directory naming for easy model management.

**Temporal Data Handling**: Supports both percentage-based and date-based temporal splitting strategies for proper time series validation with full reproducibility.

**Memory-Efficient Processing**: Polars LazyFrame integration provides efficient handling of large M5 datasets with lazy evaluation support and smart caching.

## Architecture Overview

```
src/
├── data_structures.py      # Core data models: ModelingStrategy enum, SKU tuples, configurations
├── data_loading.py          # Polars-based data loading with tuple filtering
├── model_training.py        # Model training coordination with factory pattern
├── evaluation.py            # Comprehensive evaluation and visualization
├── benchmark_pipeline.py    # End-to-end orchestration and workflow management
├── metrics.py              # Centralized metrics calculation system
├── model_types.py          # Dynamic model discovery and registration
├── storage_utils.py        # Hierarchical model storage utilities
└── models/
    ├── base.py               # Abstract base model interface
    ├── xgboost_standard.py   # Standard XGBoost implementation
    ├── xgboost_quantile.py   # Quantile XGBoost implementation
    ├── lightning_quantile.py # PyTorch Lightning quantile models
    └── statquant_model.py    # Statistical quantile baseline
```

**Core Components:**

- **ModelingStrategy**: COMBINED (unified model) vs INDIVIDUAL (per-SKU models)
- **DataLoader**: Efficient tuple-based data filtering and temporal splitting
- **ModelTrainer**: Coordinated training with dynamic model type support
- **ModelEvaluator**: Multi-metric evaluation with centralized metrics calculation
- **MetricsCalculator**: Unified metrics computation with quantile support
- **ModelTypeRegistry**: Dynamic model discovery with plugin architecture
- **BenchmarkPipeline**: Complete workflow orchestration with experiment tracking
- **ModelRegistry**: Hierarchical model storage and lifecycle management

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
from src import BenchmarkPipeline, DataConfig, TrainingConfig, ModelingStrategy
from pathlib import Path

# Configure data loading
data_config = DataConfig(
    features_path="data/train_data_features.feather",
    target_path="data/train_data_target.feather", 
    mapping_path="data/feature_mapping_train.pkl"
)

# Configure training with fixed hyperparameters
training_config = TrainingConfig(
    validation_split=0.2,
    random_state=42
)

# Add model-specific configuration
training_config.add_model_config(
    model_type="xgboost_standard",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "random_state": 42
    }
)

# Initialize pipeline
pipeline = BenchmarkPipeline(data_config, training_config)
pipeline.load_and_prepare_data()

# Define SKU tuples: (product_id, store_id)
sku_tuples = [
    (80558, 2),  # Product 80558 at Store 2
    (80558, 5),  # Product 80558 at Store 5  
    (80651, 2)   # Product 80651 at Store 2
]

# Train COMBINED model (one model for all SKUs)
combined_models = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.COMBINED,
    experiment_name="combined_3skus"
)

print(f"Trained model: {combined_models[0].get_identifier()}")
print(f"Test RMSE: {combined_models[0].metadata.performance_metrics['rmse']:.4f}")
```

### Individual SKU Modeling

```python
# Train separate models for each SKU
individual_models = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    experiment_name="individual_3skus"
)

print(f"Trained {len(individual_models)} individual models:")
for model in individual_models:
    sku = model.metadata.sku_tuples[0]
    rmse = model.metadata.performance_metrics['rmse']
    print(f"  SKU {sku[0]}x{sku[1]}: RMSE {rmse:.4f}")
```

### Model Evaluation and Comparison

```python
# Evaluate all trained models
evaluation_results = pipeline.evaluate_all_models()

# Compare models within each strategy
for strategy, results in evaluation_results.items():
    if strategy != "overall" and "error" not in results:
        print(f"\n{strategy.upper()} Strategy Results:")
        if "rankings" in results:
            rmse_ranking = results["rankings"]["rmse"][:3]  # Top 3 models
            for i, (model_id, rmse) in enumerate(rmse_ranking, 1):
                print(f"  {i}. {model_id}: RMSE {rmse:.4f}")

# Save evaluation results and experiment log
pipeline.save_evaluation_results(evaluation_results)
pipeline.save_experiment_log()
```

### Date-Based Temporal Splitting

```python
# Use specific date for train/validation split
data_config_with_date = DataConfig(
    features_path="data/train_data_features.feather",
    target_path="data/train_data_target.feather",
    mapping_path="data/feature_mapping_train.pkl",
    split_date="2020-03-01"  # Fixed split date
)

pipeline_dated = BenchmarkPipeline(data_config_with_date, training_config)
pipeline_dated.load_and_prepare_data()

# Models will use deterministic date-based splits
models = pipeline_dated.run_experiment(sku_tuples, ModelingStrategy.COMBINED)
```

### Quantile Regression for Uncertainty Quantification

```python
# Configure quantile model for risk assessment
training_config = TrainingConfig(
    validation_split=0.2,
    random_state=42
)

# Add quantile model configuration  
training_config.add_model_config(
    model_type="xgboost_quantile",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "quantile_alpha": 0.1,  # 10th percentile prediction
        "random_state": 42
    }
)

pipeline = BenchmarkPipeline(data_config, training_config)
pipeline.load_and_prepare_data()

# Train quantile models for prediction intervals
quantile_models = pipeline.run_experiment(
    sku_tuples=sku_tuples,
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    experiment_name="quantile_uncertainty"
)

# Examine quantile-specific metrics
for model in quantile_models:
    metrics = model.metadata.performance_metrics
    coverage = metrics.get('coverage_probability', 'N/A')
    print(f"SKU {model.metadata.sku_tuples[0]}: Coverage Probability = {coverage}")
```

## Advanced Usage

### Custom Hyperparameter Configuration

```python
# Advanced XGBoost configuration with multiple model types
advanced_training_config = TrainingConfig(
    validation_split=0.2,
    random_state=42
)

# Configure standard XGBoost with advanced parameters
advanced_training_config.add_model_config(
    model_type="xgboost_standard",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1  # Use all CPU cores
    }
)

# Also configure quantile model for uncertainty analysis
advanced_training_config.add_model_config(
    model_type="xgboost_quantile", 
    hyperparameters={
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
        "quantile_alpha": 0.05,  # 5th percentile for conservative estimates
        "random_state": 42,
        "n_jobs": -1
    }
)
```

### Model Type Discovery and Registry Operations

```python
from src import ModelRegistry
from src.model_types import model_registry

# Discover available model types
available_types = model_registry.list_available_types()
print(f"Available model types: {available_types}")

# Get information about specific model type
model_info = model_registry.get_model_info("xgboost_quantile")
print(f"Description: {model_info.description}")
print(f"Requires quantile: {model_info.requires_quantile}")
print(f"Default params: {model_info.default_hyperparameters}")

# Work directly with model registry
registry = ModelRegistry(Path("my_models"))

# List all models
all_models = registry.list_models()
combined_models = registry.list_models(ModelingStrategy.COMBINED)

# Load specific model
model = registry.load_model("combined_3tuples_12345_xgboost_standard")
print(f"Model covers {len(model.metadata.sku_tuples)} SKU tuples")

# Access model performance
metrics = model.metadata.performance_metrics
print(f"R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")

# Check for quantile-specific metrics
if 'coverage_probability' in metrics:
    print(f"Coverage Probability: {metrics['coverage_probability']:.4f}")
```

### Direct Component Usage

```python
from src import DataLoader, ModelTrainer, ModelEvaluator

# Use components individually for custom workflows
data_loader = DataLoader(data_config)
features_df, target_df, mapping = data_loader.load_data()

# Filter data for specific SKUs
sku_features, sku_target = data_loader.get_data_for_tuples(
    [(80558, 2), (80558, 5)], 
    ModelingStrategy.COMBINED
)

# Prepare data for modeling
X, y, feature_cols = data_loader.prepare_features_for_modeling(sku_features, sku_target)

# Create temporal split
train_bdids, val_bdids, split_date = data_loader.create_temporal_split(X, 0.2)

# Train model directly
trainer = ModelTrainer(training_config)
# ... continue with training workflow
```

## Data Requirements

**M5 Competition Format**:
- **Features**: Arrow/Feather format with bdID, productID, storeID, date, feature columns
- **Targets**: Arrow/Feather format with bdID and target columns  
- **Mapping**: Pickle dictionary with feature metadata

**Expected Columns**:
- `bdID`: Unique identifier linking features to targets
- `productID`, `storeID`: Entity identifiers forming SKU tuples
- `date`: Date column for temporal splitting
- `target`: Forecasting target variable
- Feature columns: Numerical features for model training

**Filtering Options**:
- `not_for_sale`: Automatic exclusion of non-sale items
- Date range filtering with `min_date` and `max_date`
- SKU tuple-based filtering for specific product-store combinations

## Output Structure

```
benchmark_results/
├── models/                          # Trained model storage
│   ├── combined_80558x2_80558x5_xgboost/
│   │   ├── model.pkl                # Serialized XGBoost model
│   │   ├── metadata.json            # Complete model metadata
│   │   └── data_splits.json         # Train/validation split info
│   └── individual_80558x2_xgboost/
│       ├── model.pkl
│       ├── metadata.json
│       └── data_splits.json
├── evaluation_results/              # Evaluation outputs  
│   ├── evaluation_results.json      # Complete evaluation data
│   ├── combined_evaluation_report.md
│   └── individual_evaluation_report.md
└── experiment_log.json             # Complete experiment audit trail
```

## Evaluation Metrics

The framework uses a centralized **MetricsCalculator** to ensure consistent evaluation across all models, with automatic support for quantile-specific metrics.

**Core Regression Metrics**:
- **MSE/RMSE**: Mean (squared) error for magnitude assessment
- **MAE**: Mean absolute error for robust evaluation  
- **R²**: Coefficient of determination for explained variance
- **MAPE**: Mean absolute percentage error for relative accuracy

**Error Distribution Analysis**:
- **Max Error**: Worst-case prediction error
- **Mean Error**: Bias measurement (positive/negative)
- **Std Error**: Error consistency assessment

**Accuracy Band Analysis**:
- **Within 1 Unit**: Percentage of predictions within ±1 unit
- **Within 2 Units**: Percentage of predictions within ±2 units  
- **Within 5 Units**: Percentage of predictions within ±5 units

**Quantile-Specific Metrics** (for quantile models):
- **Coverage Probability**: Actual coverage vs. theoretical quantile level
- **Quantile Loss**: Specialized loss function for quantile regression evaluation

All metrics are calculated through the centralized system, ensuring consistency across training, evaluation, and comparison workflows.

## Testing

The framework includes comprehensive tests with 80/20 integration focus:

```bash
# Run complete test suite
pytest tests/ -v

# Run integration tests only
pytest tests/test_integration.py -v

# Run component tests
pytest tests/test_data_loading.py tests/test_model_training.py -v

# Quick functionality check
pytest tests/test_basic_integration.py -v
```

**Test Features**:
- Realistic mock data generation (5 SKUs × 50 days)
- Both COMBINED and INDIVIDUAL strategy coverage
- Temporal directory isolation for test independence
- Fast execution (< 30 seconds total runtime)

## Dependencies

**Core Requirements**:
```
polars>=0.19.0          # Efficient DataFrame operations
numpy>=1.21.0           # Numerical computations  
scikit-learn>=1.0.0     # ML metrics and utilities
xgboost>=1.7.0          # Gradient boosting models
torch>=1.9.0            # PyTorch for neural network models
pytorch-lightning>=1.5.0 # Lightning framework for training
```

**Optional Dependencies**:
```
lets-plot>=4.0.0        # Visualization (graceful degradation if missing)
pandas>=1.3.0           # DataFrame compatibility for visualization
```

**Development/Testing**:
```
pytest>=7.0.0           # Test framework
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Framework Evolution

This framework represents a significant evolution from complex granularity-based modeling to streamlined tuple-based approaches:

**Key Changes**:
- **Simplified Architecture**: Tuple-based SKU identification replaces hierarchical granularities
- **Strategy Clarity**: COMBINED vs INDIVIDUAL modeling strategies address practical decisions
- **Performance Focus**: Fixed hyperparameters enable rapid experimentation cycles
- **Real-world Alignment**: (product_id, store_id) tuples map directly to business entities

**Design Philosophy**:
- **Simplicity**: Intuitive interfaces over complex abstractions
- **Speed**: Fast training cycles for iterative experimentation  
- **Reliability**: Comprehensive testing with production-ready error handling
- **Extensibility**: Clear patterns for adding new models and strategies

## Contributing

When extending the framework:

1. **New Model Types**: Create new model class inheriting from `BaseModel` with required attributes (`MODEL_TYPE`, `DESCRIPTION`, `DEFAULT_HYPERPARAMETERS`). The ModelTypeRegistry will automatically discover and register it.

2. **New Metrics**: Extend `MetricsCalculator.calculate_all_metrics()` to add new evaluation metrics that will be consistently applied across all models.

3. **New Modeling Strategies**: Extend `ModelingStrategy` enum and implement corresponding logic in `BenchmarkPipeline`.

4. **Storage Extensions**: Extend `storage_utils.py` functions to support new storage backends or metadata formats.

5. **Testing**: Prioritize integration tests that cover complete workflows, following the existing 80/20 testing approach.

The plugin architecture makes most extensions seamless - new model types are automatically discovered, metrics are centrally calculated, and storage follows established patterns.

Refer to `context/comprehensive_code_analysis.md` for detailed architectural insights and testing documentation for guidelines.