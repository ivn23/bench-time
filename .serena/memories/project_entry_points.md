# Project Entry Points and Usage

## Main Entry Point
**File**: `src/benchmark_pipeline.py`
- Contains `main()` function with example usage
- Executable with `if __name__ == "__main__":` guard
- **Run**: `python src/benchmark_pipeline.py`

## Package Import Structure
**File**: `src/__init__.py` 
- All major classes and functions exported via `__all__`
- Clean import interface for external usage

## Key Classes and Their Usage

### BenchmarkPipeline (Main Orchestrator)
```python
from src import BenchmarkPipeline, DataConfig, TrainingConfig

# Initialize pipeline
pipeline = BenchmarkPipeline(data_config, training_config)
pipeline.load_and_prepare_data()

# Run experiments
model = pipeline.run_single_model_experiment(...)
all_models = pipeline.run_full_benchmark_suite(...)
```

### Individual Components
```python
from src import DataLoader, FeatureEngineer, ModelTrainer, ModelEvaluator

# Use components separately for custom workflows
data_loader = DataLoader(data_config)
feature_engineer = FeatureEngineer(lag_features=[1,2,3,7])
trainer = ModelTrainer(training_config)
evaluator = ModelEvaluator(data_loader, model_registry)
```

### Model Registry
```python
from src import ModelRegistry, BenchmarkModel

# Manage trained models
registry = ModelRegistry(Path("my_models"))
model_id = registry.register_model(trained_model)
loaded_model = registry.load_model(model_id)
```

## Configuration Setup
```python
from src import DataConfig, TrainingConfig, GranularityLevel

# Configure data sources
data_config = DataConfig(
    features_path="data/train_data_features.feather",
    target_path="data/train_data_target.feather", 
    mapping_path="data/feature_mapping_train.pkl"
)

# Configure training parameters
training_config = TrainingConfig(
    validation_split=0.2,
    n_trials=50,
    model_type="xgboost"
)
```

## Example Workflows

### Single Model Training
```python
model = pipeline.run_single_model_experiment(
    granularity=GranularityLevel.SKU,
    entity_ids={"skuID": 278993},
    experiment_name="demo_model"
)
```

### Multi-Granularity Benchmarking
```python
all_models = pipeline.run_full_benchmark_suite(
    sample_entities={"skuIDs": [278993, 279024]},
    max_models_per_granularity=5
)
```

### Model Evaluation
```python
evaluation_results = pipeline.evaluate_all_models()
pipeline.save_evaluation_results(evaluation_results)
```

## Output Locations
- **Models**: `benchmark_results/models/`
- **Evaluations**: `benchmark_results/evaluation_results/` 
- **Logs**: `benchmark_results/experiment_log.json`