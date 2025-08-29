# Plan: Centralized Evaluation Metrics Calculation

## Metadata
adw_id: `evaluation_refactor`
prompt: `I want you to refactor the evaluation metric calculation. right now there are several palces where metrics are calculated. I want to change that and want to have a centralized calculation of evaluation metrics instead of having it all over the place. Take your previous answer into account but in metadata remove the "validation" where metrics from post-training should be stored. This should not have any back compatability at all. I want a consistent and central way to calculate the metrics. Additionally with the centralized approach the metrics should not be calculated in three places but at a specific point in the pipeline.`
task_type: refactor
complexity: complex

## Task Description
Refactor the M5 Time Series Benchmarking Framework to use centralized evaluation metric calculation instead of the current distributed approach where metrics are calculated in multiple locations (model_training.py, individual model classes, and evaluation.py). Eliminate redundant metric calculations and establish a single point of truth for metrics computation in the pipeline.

## Objective
Create a unified, consistent evaluation metrics system that:
- Centralizes all metric calculations in a single module
- Eliminates code duplication across model classes and evaluation components
- Establishes metrics calculation at one specific point in the training pipeline
- Removes validation-specific metadata storage concepts
- Maintains comprehensive metrics (MSE, RMSE, MAE, R², MAPE, error analysis, accuracy bands)
- Supports both standard and quantile-specific metrics

## Problem Statement
The current framework has metric calculation scattered across three different locations:
1. `ModelTrainer._calculate_metrics()` - Basic metrics during training
2. Model classes (`XGBoostStandardModel.get_evaluation_metrics()`, `XGBoostQuantileModel.get_evaluation_metrics()`) - Comprehensive model-specific metrics
3. `ModelEvaluator._calculate_comprehensive_metrics()` - Post-hoc evaluation metrics

This leads to code duplication, inconsistent metric implementations, and maintenance challenges. The framework needs a single source of truth for metrics calculation.

## Solution Approach
Create a centralized metrics calculation system with the following architecture:
- New `src/metrics.py` module containing all metric calculation logic
- Single calculation point in `ModelTrainer.train_model()` after prediction generation
- Remove metric calculation methods from model classes and evaluator
- Simplified metadata structure without validation-specific storage
- Support for both standard regression and quantile-specific metrics

## Relevant Files
Use these files to complete the task:

- `src/model_training.py` - Main integration point for centralized metrics, remove existing _calculate_metrics()
- `src/models/xgboost_standard.py` - Remove get_evaluation_metrics() method, focus on model training/prediction only
- `src/models/xgboost_quantile.py` - Remove get_evaluation_metrics() method, retain quantile-specific functionality
- `src/models/base.py` - Remove get_evaluation_metrics() interface from base model class
- `src/evaluation.py` - Remove _calculate_comprehensive_metrics(), refactor to use centralized metrics
- `src/data_structures.py` - Review and simplify ModelMetadata structure
- `tests/test_model_training.py` - Update tests for new metrics integration
- `tests/test_evaluation.py` - Update tests for simplified evaluation module
- `tests/test_integration.py` - Update integration tests for new metrics flow

### New Files
- `src/metrics.py` - Centralized metrics calculation module with MetricsCalculator class

## Implementation Phases

### Phase 1: Foundation
Create the centralized metrics module and establish the new calculation interface.

### Phase 2: Core Implementation  
Integrate centralized metrics into ModelTrainer and remove redundant metric calculations from model classes and evaluator.

### Phase 3: Integration & Polish
Update evaluation module to use centralized metrics, clean up metadata structure, and ensure all tests pass with new architecture.

## Step by Step Tasks

### 1. Create Centralized Metrics Module
- Create `src/metrics.py` with `MetricsCalculator` class
- Implement `calculate_regression_metrics()` static method with comprehensive standard metrics
- Implement `calculate_quantile_metrics()` static method for quantile-specific metrics  
- Include all current metrics: MSE, RMSE, MAE, R², MAPE, max_error, mean_error, std_error, accuracy bands
- Add proper error handling and zero-division protection

### 2. Update ModelTrainer Integration
- Modify `ModelTrainer.train_model()` to use centralized metrics calculation
- Remove existing `_calculate_metrics()` method
- Replace model-specific metric calls with centralized calculation
- Ensure metrics are calculated once after prediction generation
- Handle both standard and quantile model types in single calculation point

### 3. Remove Model-Specific Metric Methods
- Remove `get_evaluation_metrics()` method from `XGBoostStandardModel`
- Remove `get_evaluation_metrics()` method from `XGBoostQuantileModel`
- Remove `get_evaluation_metrics()` interface from `BaseModel`
- Update model classes to focus only on training and prediction functionality
- Retain quantile-specific parameters and logic in quantile model for metrics calculation

### 4. Refactor ModelEvaluator
- Remove `_calculate_comprehensive_metrics()` method from `ModelEvaluator`
- Update `evaluate_model()` and `evaluate_model_with_data()` to use centralized metrics
- Simplify evaluator to focus on model loading, data preparation, and results formatting
- Maintain model comparison and ranking functionality using centralized metrics

### 5. Simplify Metadata Structure
- Review `ModelMetadata` structure in `data_structures.py`
- Remove validation-specific metadata fields if present
- Ensure `performance_metrics` field stores metrics from single calculation point
- Maintain backward compatibility for model loading (if any stored models exist)

### 6. Update Tests and Validation
- Update `test_model_training.py` to test new centralized metrics integration
- Update `test_evaluation.py` for simplified evaluator functionality
- Update `test_integration.py` to verify end-to-end metrics flow
- Ensure all metric values remain consistent with previous calculations
- Validate that quantile models receive appropriate quantile-specific metrics

## Testing Strategy
- Unit tests for `MetricsCalculator` class with various edge cases (zero targets, negative values, etc.)
- Integration tests verifying metrics are calculated once and propagated correctly
- Comparison tests ensuring metric values remain consistent with previous implementation
- Model loading/saving tests to verify metadata compatibility
- Performance tests to ensure centralized calculation doesn't impact training speed

## Acceptance Criteria
- All metric calculations occur in single centralized module (`src/metrics.py`)
- Metrics are calculated exactly once per model in `ModelTrainer.train_model()`
- No metric calculation code exists in model classes (xgboost_standard.py, xgboost_quantile.py)
- No metric calculation code exists in `ModelEvaluator`
- All existing metrics are preserved with identical calculation logic
- Quantile models receive quantile-specific metrics in addition to standard metrics
- All integration tests pass with new architecture
- Model metadata contains performance metrics from single calculation point
- No validation-specific metadata storage concepts remain

## Validation Commands
Execute these commands to validate the task is complete:

- `cd "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking" && PYTHONPATH=. pytest tests/test_model_training.py -v` - Test model training with centralized metrics
- `cd "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking" && PYTHONPATH=. pytest tests/test_evaluation.py -v` - Test simplified evaluation module
- `cd "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking" && PYTHONPATH=. pytest tests/test_integration.py -v` - Test end-to-end integration
- `cd "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking" && PYTHONPATH=. python -c "from src.metrics import MetricsCalculator; print('Centralized metrics module imported successfully')"` - Verify new metrics module
- `cd "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking" && grep -r "get_evaluation_metrics" src/models/ || echo "No metric methods found in model classes"` - Verify removal of model-specific metrics
- `cd "/Users/ivn/Documents/PhD/Transformer Research/Code/Benchmarking" && jupyter nbconvert --to notebook --execute src/test_pipeline.ipynb --output test_pipeline_executed.ipynb` - Ensure notebook still works with refactored metrics

## Notes
- Maintain identical metric calculation logic to ensure consistency with existing results
- Focus on single calculation point in training pipeline rather than distributed calculations
- Quantile-specific metrics (quantile_score, coverage_probability, etc.) should be added to standard metrics for quantile models
- Error handling should be robust for edge cases (zero targets, infinite values, etc.)
- The refactor should not impact model persistence or loading functionality