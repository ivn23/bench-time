# Plan: Remove Feature Engineering from Framework

## Metadata
adw_id: `remove_feature_eng`
prompt: `I want you to completely remove feature engineering from this framework. Everything else should stay the same`
task_type: refactor
complexity: complex

## Task Description
Completely remove all feature engineering functionality from the M5 Time Series Benchmarking Framework while preserving all other functionality. The framework currently uses a FeatureEngineer class that handles both feature engineering (lag features, calendric features, trend features) and basic data preparation (X/y splitting, null handling). The goal is to eliminate feature engineering entirely while maintaining the core benchmarking pipeline functionality.

## Objective
Remove the FeatureEngineer class and all feature engineering capabilities from the framework, while preserving the benchmarking pipeline, model training, evaluation, and data loading functionality. The framework should work directly with precomputed features without any dynamic feature creation.

## Problem Statement
The current framework includes comprehensive feature engineering capabilities that the user wants removed. The FeatureEngineer class is tightly integrated into the BenchmarkPipeline and handles both feature engineering and essential data preparation tasks. Removing it requires careful refactoring to preserve core functionality while eliminating all feature creation logic.

## Solution Approach
1. Move essential data preparation functionality (X/y splitting, null handling) from FeatureEngineer to DataLoader
2. Update BenchmarkPipeline to work directly with raw features without feature engineering
3. Remove FeatureEngineer class and all related imports
4. Remove feature engineering parameters from DataConfig
5. Update public API exports
6. Preserve all other framework functionality (training, evaluation, model registry)

## Relevant Files
Use these files to complete the task:

- `src/feature_engineering.py` - Contains FeatureEngineer class to be completely removed
- `src/benchmark_pipeline.py` - Uses FeatureEngineer, needs refactoring to work without it
- `src/data_loading.py` - Needs new data preparation methods moved from FeatureEngineer
- `src/data_structures.py` - Contains feature engineering parameters in DataConfig to remove
- `src/__init__.py` - Exports FeatureEngineer, needs update
- `README.md` - May contain feature engineering documentation to update

## Implementation Phases
### Phase 1: Foundation
Extract essential data preparation functionality from FeatureEngineer and move to DataLoader

### Phase 2: Core Implementation
Remove FeatureEngineer class and update BenchmarkPipeline to work without feature engineering

### Phase 3: Integration & Polish
Clean up imports, exports, configurations, and documentation

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Extract Data Preparation Methods
- Move `prepare_model_data()` method from FeatureEngineer to DataLoader class
- Move `_get_feature_columns()` helper method to DataLoader class
- Ensure these methods work independently without feature engineering logic

### 2. Update DataLoader Interface
- Add new `prepare_features_for_modeling()` method that combines raw feature preparation
- Add logic to handle feature column identification without engineering
- Ensure method returns same format as current FeatureEngineer.prepare_model_data()

### 3. Remove FeatureEngineer Class
- Delete entire `src/feature_engineering.py` file
- Remove FeatureEngineer import from `src/__init__.py`
- Remove FeatureEngineer from __all__ exports in `src/__init__.py`

### 4. Update BenchmarkPipeline
- Remove FeatureEngineer import
- Remove feature_engineer initialization in __init__()
- Replace feature_engineer.create_features() call with direct data processing
- Replace feature_engineer.prepare_model_data() call with data_loader method
- Update run_single_model_experiment() method accordingly

### 5. Clean Up DataConfig
- Remove feature_engineering parameter
- Remove feature_engineering_methods parameter  
- Remove lag_features parameter
- Remove calendric_features parameter
- Remove trend_features parameter
- Keep only essential data loading parameters

### 6. Update BenchmarkPipeline Constructor
- Remove FeatureEngineer instantiation
- Remove feature engineering parameter passing to FeatureEngineer
- Simplify initialization to only use DataLoader, ModelTrainer, etc.

### 7. Update Example Configuration
- Remove feature engineering parameters from create_default_configs() function
- Ensure example still works without feature engineering

### 8. Validate All Pipeline Methods
- Ensure run_single_model_experiment() works without feature engineering
- Ensure run_multi_entity_experiment() works without feature engineering  
- Ensure run_full_benchmark_suite() works without feature engineering
- Ensure all evaluation methods continue to work

## Testing Strategy
- Test that BenchmarkPipeline can initialize without errors
- Test that single model experiments run successfully using raw features
- Test that multi-entity experiments work correctly
- Test that model training, evaluation, and registry functionality remains intact
- Test that the framework works with precomputed features only
- Verify no feature engineering is applied to the data

## Acceptance Criteria
- FeatureEngineer class and feature_engineering.py file are completely removed
- BenchmarkPipeline works without any feature engineering functionality
- All model training, evaluation, and registry functionality remains intact
- DataConfig contains no feature engineering parameters
- Framework processes raw features directly without modification
- All existing pipeline methods (run_single_model_experiment, etc.) work correctly
- Public API (__init__.py exports) updated to remove FeatureEngineer
- No imports or references to feature engineering remain in codebase

## Validation Commands
Execute these commands to validate the task is complete:

- `python -c "from src import BenchmarkPipeline, DataConfig, TrainingConfig; print('Imports work')"` - Test basic imports
- `python -c "import src; print('FeatureEngineer' not in dir(src))"` - Verify FeatureEngineer not exported  
- `find src -name "*.py" -exec grep -l "FeatureEngineer\|feature_engineer" {} \;` - Should return no files
- `find src -name "*.py" -exec grep -l "feature_engineering\|lag_features\|calendric_features\|trend_features" {} \;` - Should return no files
- `python -m py_compile src/*.py` - Test all Python files compile
- `ls src/feature_engineering.py 2>/dev/null || echo "File successfully removed"` - Verify file deletion

## Notes
The framework is designed to work with precomputed M5 features, so removing dynamic feature engineering should not affect core functionality. The key challenge is preserving the data preparation aspects (X/y splitting, null handling, feature column identification) that were embedded in FeatureEngineer. Care must be taken to ensure the same data format is maintained for downstream model training and evaluation components.