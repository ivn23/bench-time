# CLAUDE.md - Quick Reference for M5 Benchmarking Framework

## 1. Overview

This is a tuple-based time series forecasting framework. The framework emphasizes SKU tuple modeling (product_id, store_id pairs) with COMBINED (single model for all SKUs) and INDIVIDUAL (separate models per SKU) strategies. Features centralized metrics, dynamic model discovery, quantile regression, hierarchical storage, and release management.

## 2. Key Concepts

- **SKU Tuples**: `(product_id, store_id)` pairs as fundamental modeling units
- **Modeling Strategies**: COMBINED (one model for all SKUs) vs INDIVIDUAL (model per SKU)
- **Data Pipeline**: Polars-based loading with lazy evaluation and temporal splitting
- **Model Training**: Factory pattern with multi-quantile support and centralized metrics
- **Evaluation**: Comprehensive metrics calculation with quantile-specific analysis
- **Release Management**: Production deployment with versioning and monitoring

## 3. Common Tasks

### 3.1 Training Models
- Use `BenchmarkPipeline` to orchestrate complete training workflows
- Configure with `DataConfig` (data paths, filtering) and `TrainingConfig` (models, hyperparameters)
- Support for multiple model types: XGBoost, PyTorch Lightning, statistical baselines
- See: [example_usage.py](example_usage.py) and [test_pipeline.ipynb](test_pipeline.ipynb)

### 3.2 Evaluating Models
- `ModelEvaluator` provides comprehensive evaluation with centralized metrics
- Compare models across strategies with automatic ranking
- Generate markdown reports with performance summaries
- See: [test_integration.py](tests/test_integration.py)

### 3.3 Multi-Quantile Training
- Configure `quantile_alphas=[0.1, 0.5, 0.9]` for uncertainty quantification
- Automatic coverage probability analysis for quantile models
- Support for XGBoost and PyTorch Lightning quantile implementations
- See: [test_multi_quantile.py](tests/test_multi_quantile.py)

### 3.4 Creating Releases
- `ComprehensiveReleaseManager` creates complete experiment packages
- Includes models, configurations, metrics, and auto-generated documentation
- Hierarchical storage with intelligent naming conventions
- See: [comprehensive_manager.py](src/release_management/comprehensive_manager.py)

## 4. Best Practices

- **Use tuple-based SKU specification** for clear business entity mapping
- **Leverage lazy evaluation** with Polars for memory efficiency on large datasets
- **Configure deterministic splits** using `split_date` for reproducible experiments
- **Utilize centralized metrics** to ensure consistent evaluation across all models
- **Implement comprehensive testing** focusing on integration over unit tests
- **Follow hierarchical storage** patterns for organized model persistence
- **No backwards compatability** future code changes should focus on clean architecture and not legacy functionalities.
 
## 5. Directory Structure

```
src/
├── benchmark_pipeline.py        # Main orchestration
├── data_loading.py              # Polars-based data processing
├── model_training.py            # Factory pattern training
├── evaluation.py                # Comprehensive evaluation
├── metrics.py                   # Centralized metrics calculation
├── data_structures.py           # Core data models and configurations
├── model_types.py               # Dynamic model discovery
├── storage_utils.py             # Hierarchical model storage
├── models/                      # Model implementations
│   ├── base.py                  # Abstract base interface
│   ├── xgboost_*.py             # XGBoost implementations
│   ├── lightning_*.py           # PyTorch Lightning models
│   └── statquant_model.py       # Statistical baseline
└── release_management/          # Production deployment system
```

## 6. Configuration Templates

### Basic Pipeline Setup
```python
from src import BenchmarkPipeline, DataConfig, TrainingConfig, ModelingStrategy

# Data configuration
data_config = DataConfig(
    features_path="data/train_data_features.feather",
    target_path="data/train_data_target.feather", 
    mapping_path="data/feature_mapping_train.pkl",
    split_date="2020-03-01"  # Deterministic temporal split
)

# Training configuration
training_config = TrainingConfig(random_state=42)
training_config.set_model_config("xgboost_standard", n_estimators=100, max_depth=6)

# Pipeline execution
pipeline = BenchmarkPipeline(data_config, training_config)
pipeline.load_and_prepare_data()
models = pipeline.run_experiment([(80558, 2), (80651, 2)], ModelingStrategy.COMBINED)
```

### Multi-Quantile Configuration
```python
training_config.set_model_config(
    "xgboost_quantile", 
    n_estimators=100,
    quantile_alphas=[0.1, 0.5, 0.9]  # Multiple quantile levels
)
```

## 7. Available Model Types

- **xgboost_standard**: Standard XGBoost regression
- **xgboost_quantile**: Quantile XGBoost with coverage analysis
- **lightning_standard**: PyTorch Lightning neural networks
- **lightning_quantile**: Neural network quantile regression
- **statquant_model**: Fast numpy-based baseline

Use `model_registry.list_available_types()` to discover all available models.

## 8. Data Requirements

### Expected Schema
```
Features: bdID (int64), productID (int64), storeID (int64), date (date), feature_* (float64)
Targets: bdID (int64), target (float64)
Mapping: Dictionary with feature metadata
```

### File Formats
- **Features**: `train_data_features.feather` (Arrow/Feather format)
- **Targets**: `train_data_target.feather` (Arrow/Feather format) 
- **Mapping**: `feature_mapping_train.pkl` (Pickle format)

## 9. Testing Framework

### Test Distribution (80/20 Integration Approach)
- **70% Integration**: End-to-end workflow validation in [test_integration.py](tests/test_integration.py)
- **25% Component**: Critical functionality in [test_*_loading.py](tests/)
- **5% Validation**: API consistency checks in [test_api.py](tests/test_api.py)

### Running Tests
```bash
pytest tests/ -v                    # Full suite
pytest tests/test_integration.py    # Integration only
pytest tests/test_multi_quantile.py # Quantile functionality
```

## 10. Troubleshooting

### Common Issues
- **Import Errors**: Ensure `PYTHONPATH` includes project root
- **GPU Issues**: Check PyTorch Lightning and CUDA availability
- **Memory Issues**: Use `lazy=True` in data loading for large datasets
- **Model Discovery**: Verify model classes inherit from `BaseModel` with proper attributes

### Quick Validation

You are already in the Benchmarking directory , so there is NO NEED TO DO:

```bash
cd /Users/ivn/Documents/PhD/Transformer\ Research/Code/Benchmarking 
```

Just do a quick check befoore you run the command above if to see if you really need to do it:

```bash
ls ..
```


```bash
# Test environment setup (using UV)
uv run python -c "from src import *; print('All imports successful')"

# Check available models
uv run python -c "from src.model_types import model_registry; print(model_registry.list_available_types())"

# Validate data files
uv run python -c "import polars as pl; print(pl.scan_ipc('data/train_data_features.feather').shape)"
```

**Note**: All commands use `uv run` to execute within the project's virtual environment (`.venv`). If you need to activate the environment manually:
```bash
source .venv/bin/activate
```

## 11. Parallel Training

The framework supports automatic parallel training for INDIVIDUAL modeling strategy to dramatically speed up large-scale experiments.

### How It Works

- **Automatic Detection**: XGBoost/CPU models automatically parallelize across all cores; Lightning/GPU models remain sequential
- **Process-Based**: Uses `ProcessPoolExecutor` for robust multi-process training
- **Thread Safety**: Each XGBoost model trains with `nthread=1` to avoid thread contention
- **Progress Tracking**: Real-time progress bars show training status across all workers

### Usage Example

```python
# Parallel training for 7,000 SKUs with 5 quantiles = 35,000 models
results = pipeline.run_experiment(
    sku_tuples=large_sku_list,  # 7,000 SKUs
    modeling_strategy=ModelingStrategy.INDIVIDUAL,
    model_type="xgboost_quantile",
    quantile_alphas=[0.1, 0.3, 0.5, 0.7, 0.9],
    hyperparameters={'n_estimators': 100, 'max_depth': 6, 'nthread': -1},
    n_workers=None  # Auto-detect: will use cpu_count-1 workers
)
# Training 35,000 models in parallel takes ~10 minutes vs 24+ hours sequential
```

### Manual Worker Control

```python
# Override automatic detection
results = pipeline.run_experiment(
    ...
    n_workers=10  # Force 10 parallel workers
)

# Force sequential for debugging
results = pipeline.run_experiment(
    ...
    n_workers=1  # Disable parallelization
)
```

### Performance Expectations

- **Small Models** (sample size ≤1.5k, XGBoost nthread=1): Near-linear speedup with CPU count
- **7,000 SKUs × 5 quantiles = 35,000 models**:
  - Sequential: ~24 hours
  - Parallel (190 workers): ~10 minutes
  - Speedup: ~144x
- **Memory**: Each worker loads only its assigned dataset (minimal overhead)

### Model Type Behavior

| Model Type | Default Behavior | Reason |
|-----------|------------------|---------|
| `xgboost_quantile` | Parallel (N-1 workers) | CPU-bound, benefits from parallelization |
| `xgboost_standard` | Parallel (N-1 workers) | CPU-bound, benefits from parallelization |
| `lightning_quantile` | Sequential (1 worker) | GPU-exclusive, cannot parallelize |
| `lightning_standard` | Sequential (1 worker) | GPU-exclusive, cannot parallelize |
| `statquant` | Parallel (N-1 workers) | CPU-bound, lightweight |

### Best Practices

- **INDIVIDUAL strategy**: Parallel training is highly effective
- **COMBINED strategy**: Single model training (parallelization not applicable)
- **Set nthread=1**: For XGBoost in hyperparameters (automatically enforced in parallel mode)
- **Memory management**: Monitor RAM usage with large SKU counts (35k SKUs ≈ 50GB dataset memory)

## 12. Resource Configuration and Parallelism

The framework supports three independent types of parallelism that serve different purposes. Understanding these is crucial for optimal performance configuration.

### Three Types of Parallelism

#### 1. Optuna Trial Parallelism (`n_jobs`)
**What it controls**: Number of hyperparameter trials running simultaneously

**Configuration**: Set in `tuning_config['n_jobs']`

**When to use**:
- `n_jobs=1`: Sequential trials (use for GPU models - each trial needs exclusive GPU access)
- `n_jobs=-1`: All CPU cores (use for CPU models like XGBoost)
- `n_jobs=N`: N parallel trials (requires N GPUs with proper CUDA_VISIBLE_DEVICES setup)

**Example**: With `n_jobs=4`, Optuna runs 4 different hyperparameter combinations simultaneously

#### 2. DataLoader Worker Parallelism (`dataloader_workers`)
**What it controls**: Number of processes loading/preprocessing data in parallel while GPU trains

**Configuration**: Set in `tuning_config['dataloader_workers']`

**When to use**:
- `dataloader_workers=0`: Single-threaded (slow, but simple for debugging)
- `dataloader_workers=4-8`: Good default for GPU training
- `dataloader_workers=8-12`: Optimal for large datasets (>400k samples) on GPU

**Example**: With `dataloader_workers=8`, 8 processes load batches in parallel while GPU computes gradients

#### 3. GPU Device Parallelism (`devices`)
**What it controls**: Number of GPUs used for training a single model via Distributed Data Parallel (DDP)

**Configuration**: Set in `tuning_config['devices']`

**How it works**:
- Model is replicated on each GPU
- Training data is split across GPUs (each GPU sees different batches)
- Gradients are synchronized after each backward pass

**When to use**:
- `devices=1`: Single GPU per trial (**recommended for hyperparameter tuning**)
- `devices=N`: Use N GPUs for one large model (not typical for hyperparameter tuning)

**Example**: With `devices=4`, one model trains on 4 GPUs simultaneously using DDP

### Resource Configuration via tuning_config

All resource parameters are now passed through the `tuning_config` dictionary:

```python
tuning_config = {
    # Hyperparameter search configuration
    'n_trials': 100,           # Number of hyperparameter combinations to try
    'n_folds': 5,              # Cross-validation folds per trial

    # Resource configuration (NEW)
    'n_jobs': 1,               # Optuna trial parallelism
    'dataloader_workers': 8,   # DataLoader worker processes
    'accelerator': 'cuda',     # Device type: 'cuda', 'cpu', or 'mps'
    'devices': 1               # Number of GPUs per trial
}

pipeline.run_experiment(
    sku_tuples=sku_list,
    modeling_strategy=ModelingStrategy.COMBINED,
    model_type="lightning_quantile",
    mode="hp_tune",
    tuning_config=tuning_config
)
```

### Common Configuration Scenarios

#### Single GPU Hyperparameter Tuning (Typical)
```python
tuning_config = {
    'n_trials': 100,
    'n_folds': 5,
    'n_jobs': 1,               # Sequential trials
    'dataloader_workers': 8,   # Parallel data loading
    'accelerator': 'cuda',
    'devices': 1               # One GPU per trial
}
```

**What happens**: Trials run one at a time, each using 1 GPU with 8 data loading workers

#### CPU-Only Training
```python
tuning_config = {
    'n_trials': 100,
    'n_folds': 5,
    'n_jobs': -1,              # Parallel trials on all CPU cores
    'dataloader_workers': 4,   # Lower for CPU
    'accelerator': 'cpu',
    'devices': 1
}
```

**What happens**: Multiple trials run in parallel across CPU cores

#### Multi-GPU Hyperparameter Tuning (Advanced)
Run multiple scripts in parallel, each on a different GPU:
```bash
# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 python run_hpt_mlp.py &

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 python run_hpt_mlp.py &
```

With shared Optuna storage (SQLite/PostgreSQL), each GPU runs independent trials

### Auto-Detection

The `run_hpt_mlp.py` script automatically detects available GPUs and configures optimal defaults:

```python
if torch.cuda.is_available():
    accelerator = 'cuda'
    devices = 1
    dataloader_workers = 8
else:
    accelerator = 'cpu'
    devices = 1
    dataloader_workers = 4
```

You can override these by explicitly setting values in `tuning_config`

### For Your Use Case: Tuning One Global Model Across 100 SKUs

**Scenario**: Combined model across all SKUs, hyperparameter tuning on GPU

**Configuration**:
```python
tuning_config = {
    'n_trials': 100,           # Try 100 hyperparameter combinations
    'n_folds': 5,              # 5-fold CV per trial
    'n_jobs': 1,               # Sequential trials (GPU needs exclusive access)
    'dataloader_workers': 8,   # 8 workers load data while GPU trains
    'accelerator': 'cuda',     # Use GPU
    'devices': 1               # Each trial uses 1 GPU
}
```

**What happens**:
1. Optuna tries 100 hyperparameter combinations **sequentially** (one at a time)
2. Each trial trains a Lightning model on your combined dataset (all 100 SKUs together)
3. Model training uses **1 GPU** with **8 DataLoader workers** for fast data loading
4. No multi-GPU parallelization within a trial (`devices=1`)

**Why these settings**:
- `n_jobs=1`: GPU training is fast; parallel trials would cause GPU memory contention
- `dataloader_workers=8`: Keeps GPU fed with data (prevents GPU idling)
- `devices=1`: Hyperparameter tuning benefits from many trials, not parallel model training

## 13. Performance Tips

- **Use lazy evaluation** with `load_data(lazy=True)` for memory efficiency
- **Process SKU batches** for large-scale experiments
- **Configure fewer estimators** during development (n_estimators=50)
- **Leverage parallel training** for INDIVIDUAL strategy with XGBoost models (automatic)
- **Enable GPU training** for Lightning models by setting `accelerator='cuda'` in `tuning_config`
- **Optimize DataLoader workers** based on dataset size (8-12 for large datasets on GPU)

## 14. Further Reading

- **Architecture Details**: See Single Responsibility Principle analysis in conversation logs
- **Model Implementation**: Review [src/models/](src/models/) for specific implementations
- **Release Management**: Study [src/release_management/](src/release_management/) for deployment patterns
- **Testing Examples**: Examine [tests/](tests/) for comprehensive usage patterns
- **Data Processing**: Review [src/data_loading.py](src/data_loading.py) for advanced filtering options
