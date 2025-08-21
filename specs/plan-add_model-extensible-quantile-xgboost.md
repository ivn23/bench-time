# Plan: Extensible Model Architecture with Quantile XGBoost

## Metadata
adw_id: `add_model`
prompt: `What i no want to do is the possibility to add another model to the framework, not just only xgboost but also the quantile version from the notebook to be able to train it with the framework. Write a plan on what would be the eaziest and quickest way to add a model without loosing the generality of the framework. I should then just be able to pass a quantile to the configs if a quantile model was choosen and the pipeline would train it. I want to be able to add other models in the future, for example a model implemented in pytorch lightning. It is important that you work with the approach from the notebook. Dont forget to appropriately figure out how to fully incorporate this into the framework such that i also would be able to use other parts of the framework with the new models, such as the model registry`
task_type: feature
complexity: complex

## Task Description
Extend the M5 benchmarking framework to support multiple model types beyond the current XGBoost implementation. The primary goal is to add quantile XGBoost regression using the custom objective function approach from the working notebook, while creating an extensible architecture that can accommodate future model types (e.g., PyTorch Lightning). The solution must maintain full framework integration including model registry, evaluation, and pipeline orchestration, while preserving backward compatibility and generality.

## Objective
Create an extensible model architecture that allows seamless integration of new model types through configuration, starting with quantile XGBoost regression. Users should be able to specify model type and model-specific parameters (like quantile level) in training configs, and the entire framework ecosystem (training, evaluation, persistence, comparison) should work transparently with all model types.

## Problem Statement
The current framework is tightly coupled to standard XGBoost regression through XGBRegressor, limiting its extensibility for different model types and specialized regression variants like quantile regression. The quantile XGBoost implementation requires a fundamentally different training approach (xgb.train with custom objective) that doesn't fit the current sklearn-style interface. Additionally, different model types may require specialized evaluation metrics and persistence strategies that aren't currently supported.

## Solution Approach
Implement a model factory pattern with abstract base classes that define standard interfaces for training, prediction, and evaluation while allowing model-specific implementations. Extend the configuration system to support model-specific parameters and create specialized training, evaluation, and persistence logic for each model type. The quantile XGBoost implementation will serve as the first demonstration of this extensible architecture, using the proven custom objective approach from the notebook.

## Relevant Files
Use these files to complete the task:

- `src/model_training.py` - Core training logic that needs abstract model interface
- `src/data_structures.py` - TrainingConfig class requiring extension for model-specific parameters  
- `src/evaluation.py` - Evaluation system needing quantile-specific metrics
- `notebooks/quantile_xgboost_simple.ipynb` - Reference implementation for quantile approach
- `tests/test_model_training.py` - Tests requiring updates for new model types
- `tests/test_integration.py` - Integration tests for end-to-end workflows

### New Files
- `src/models/` - New directory for model implementations
- `src/models/base.py` - Abstract base classes for model interface
- `src/models/xgboost_standard.py` - Standard XGBoost implementation
- `src/models/xgboost_quantile.py` - Quantile XGBoost implementation using notebook approach
- `src/models/__init__.py` - Model factory and registry
- `tests/test_quantile_models.py` - Quantile-specific tests

## Implementation Phases

### Phase 1: Foundation
Create abstract model interfaces and refactor existing XGBoost implementation to fit the new architecture. This establishes the foundation for extensibility without breaking current functionality.

### Phase 2: Core Implementation  
Implement quantile XGBoost using the notebook approach, extend configuration system for model-specific parameters, and add quantile-specific evaluation metrics.

### Phase 3: Integration & Polish
Ensure full framework integration with model registry, pipeline orchestration, and comprehensive testing. Validate that all existing functionality works with both model types and document the extension pattern for future model additions.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Create Abstract Model Architecture
- Create `src/models/` directory and `__init__.py`
- Implement `src/models/base.py` with abstract `BaseModel` class defining standard interface
- Define abstract methods: `train()`, `predict()`, `get_model_info()`, `get_evaluation_metrics()`
- Create model factory function in `src/models/__init__.py`

### 2. Refactor Existing XGBoost Implementation
- Create `src/models/xgboost_standard.py` implementing `BaseModel` interface
- Move current XGBoost training logic from `ModelTrainer._train_model_with_params`
- Ensure backward compatibility with existing hyperparameter structure
- Update import in `src/model_training.py` to use model factory

### 3. Extend Configuration System
- Modify `TrainingConfig` in `src/data_structures.py` to support model-specific parameters
- Add `quantile_alpha` parameter (default None for non-quantile models)
- Add `model_specific_params` dict field for extensibility
- Maintain backward compatibility with existing configurations

### 4. Implement Quantile XGBoost Model
- Create `src/models/xgboost_quantile.py` using notebook approach
- Implement `quantile_objective()` function from notebook
- Use `xgb.train()` with custom objective instead of XGBRegressor
- Handle DMatrix creation and parameter conversion
- Implement model-specific prediction and evaluation methods

### 5. Extend Evaluation System
- Add quantile-specific metrics to evaluation framework
- Implement `quantile_score()` and `coverage_probability()` functions from notebook
- Extend `ModelEvaluator._calculate_comprehensive_metrics()` to use model-specific metrics
- Ensure quantile metrics are included in reports and comparisons

### 6. Update Model Training Pipeline
- Modify `ModelTrainer._train_model_with_params()` to use model factory
- Pass model-specific parameters from config to model implementations
- Ensure consistent metadata creation for all model types
- Update logging to reflect model type and specific parameters

### 7. Ensure Model Registry Compatibility
- Verify `ModelRegistry` works with all model types for save/load operations
- Test serialization compatibility (pickle) for both XGBoost variants
- Ensure metadata correctly identifies and distinguishes model types
- Validate model loading and prediction after persistence

### 8. Create Comprehensive Tests
- Create `tests/test_quantile_models.py` with quantile-specific test cases
- Test quantile parameter passing through configuration
- Verify quantile-specific evaluation metrics are calculated correctly
- Test model registry operations with quantile models
- Update existing tests to work with model factory pattern

### 9. Update Integration Tests
- Extend `tests/test_integration.py` to test both model types
- Verify end-to-end pipeline works with quantile models
- Test mixed experiments with both standard and quantile models
- Validate evaluation and comparison across different model types

### 10. Documentation and Validation
- Add model extension documentation for future implementations
- Create usage examples showing how to configure quantile models
- Validate all existing functionality remains unchanged

## Testing Strategy
Implement comprehensive testing covering the new extensible architecture:
- **Unit Tests**: Test individual model implementations, configuration parsing, and evaluation metrics
- **Integration Tests**: Test end-to-end pipelines with both model types, mixed experiments, and model registry operations
- **Compatibility Tests**: Ensure backward compatibility with existing configurations and saved models
- **Extension Tests**: Validate that the pattern can accommodate future model types through mock implementations

## Acceptance Criteria
- [ ] Framework supports both standard XGBoost and quantile XGBoost models through configuration
- [ ] Users can specify `model_type="xgboost_quantile"` and `quantile_alpha=0.7` in TrainingConfig
- [ ] Quantile models use the exact approach from the working notebook (custom objective with xgb.train)
- [ ] All evaluation metrics work correctly for both model types, with quantile-specific metrics for quantile models
- [ ] Model registry fully supports save/load operations for both model types
- [ ] Existing functionality and backward compatibility maintained (all current tests pass)
- [ ] BenchmarkPipeline works seamlessly with both model types
- [ ] Model comparison and evaluation reports include appropriate metrics for each model type
- [ ] Architecture is extensible for future model types with clear patterns and documentation
- [ ] Comprehensive test coverage for all new functionality

## Validation Commands
Execute these commands to validate the task is complete:

- `python -m py_compile src/models/*.py` - Verify all new model files compile
- `python -c "from src.models import get_model_class; print('Model factory works')"` - Test model factory import
- `PYTHONPATH=. pytest tests/test_quantile_models.py -v` - Run quantile-specific tests
- `PYTHONPATH=. pytest tests/test_model_training.py -v` - Verify refactored training works
- `PYTHONPATH=. pytest tests/test_integration.py -v` - Test end-to-end integration
- `PYTHONPATH=. pytest tests/ -v` - Run complete test suite for regression testing
- `python -c "from src import TrainingConfig; c=TrainingConfig(model_type='xgboost_quantile', quantile_alpha=0.7); print(f'Config: {c.model_type}, α={c.quantile_alpha}')"` - Test configuration
- `python -c "import pickle; from src.models import get_model_class; print('Serialization test passed')"` - Test model serialization imports

## Notes
- The quantile XGBoost implementation must use the exact approach from `notebooks/quantile_xgboost_simple.ipynb` with `xgb.train()` and custom objective function
- Maintain backward compatibility by defaulting to existing behavior when no model type is specified
- Consider future PyTorch Lightning integration by designing interfaces that can accommodate different training paradigms (epoch-based vs tree-based)
- Quantile-specific evaluation metrics (quantile_score, coverage_probability) should only be calculated for quantile models
- Model factory pattern should be easily extensible - adding a new model type should require minimal changes to existing code
- Ensure all model types work with the tuple-based modeling strategies (COMBINED/INDIVIDUAL) and SKU-based workflows