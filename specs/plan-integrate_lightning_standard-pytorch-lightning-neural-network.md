# Plan: Integrate PyTorch Lightning Neural Network Model

## Metadata
adw_id: `integrate_lightning_standard`
prompt: `integrate_lightning_standard. I want to Integrate the lightning prototype model from the lightning_prototype.ipynb notebook in the notebooks/models/ directory into my framework. It should be usable like the xgboost model in the framework, where you have to provide split information, model specific hyperparameters etc. I want a solution that does not change the overall functionality but makes use of already existing modules int he framework.`
task_type: feature
complexity: medium

## Task Description
Integrate the PyTorch Lightning neural network model from the `lightning_prototype.ipynb` notebook into the existing M5 benchmarking framework. The integration should follow the same patterns as the existing XGBoost models, making the Lightning model available through the framework's plugin architecture with full support for hyperparameter configuration, training with split information, and evaluation through the centralized metrics system.

## Objective
Create a new `LightningStandardModel` class that:
- Inherits from `BaseModel` and implements the required abstract methods
- Wraps the PyTorch Lightning `ForecastingModel` from the prototype
- Integrates seamlessly with the existing framework's model discovery, training, and evaluation systems
- Supports configurable hyperparameters (hidden_size, learning_rate, dropout, max_epochs, batch_size)
- Works with both COMBINED and INDIVIDUAL modeling strategies
- Provides consistent interface with existing XGBoost models

## Problem Statement
The framework currently only supports XGBoost-based models (standard and quantile). To expand the modeling capabilities and provide neural network alternatives, the PyTorch Lightning prototype needs to be integrated following the established plugin architecture patterns. The integration must maintain the framework's design principles of plugin-based model discovery, centralized metrics calculation, and consistent model lifecycle management.

## Solution Approach
Create a new model implementation that:
1. Wraps the existing Lightning neural network architecture in a BaseModel-compliant interface
2. Handles the PyTorch-specific data conversion (numpy to torch tensors)
3. Manages Lightning trainer configuration and training lifecycle
4. Provides proper model serialization/deserialization for the framework's storage system
5. Integrates with the centralized metrics calculation system
6. Follows the same patterns as XGBoost models for consistency

## Relevant Files
Use these files to complete the task:

- `notebooks/model_prototypes/lightning_prototype.ipynb` - Source Lightning implementation to adapt
- `src/models/base.py` - BaseModel abstract class defining the interface requirements
- `src/models/xgboost_standard.py` - Reference implementation pattern to follow
- `src/model_types.py` - Model type registry for automatic discovery
- `src/model_training.py` - ModelTrainer class that will use the new model
- `src/metrics.py` - Centralized metrics calculation that the model will integrate with
- `tests/test_model_training.py` - Test patterns to follow for the new model

### New Files
- `src/models/lightning_standard.py` - New Lightning neural network model implementation

## Implementation Phases

### Phase 1: Foundation
- Analyze the Lightning prototype architecture and hyperparameters
- Create the basic LightningStandardModel class structure
- Implement the BaseModel interface methods (train, predict, get_model_info)
- Define default hyperparameters and model type constants

### Phase 2: Core Implementation
- Implement the PyTorch Lightning model wrapper and data handling
- Add tensor conversion utilities for numpy <-> torch compatibility
- Implement training logic with proper validation handling
- Add model serialization/deserialization support for framework storage
- Integrate with the centralized metrics system

### Phase 3: Integration & Polish
- Test integration with the model type registry and discovery system
- Validate compatibility with both COMBINED and INDIVIDUAL strategies
- Add comprehensive error handling and logging
- Create integration tests following existing test patterns
- Update documentation if needed

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Analyze Lightning Prototype and Plan Integration
- Study the lightning_prototype.ipynb implementation details
- Identify key components: ForecastingModel class, hyperparameters, training logic
- Map prototype hyperparameters to framework configuration structure
- Plan data conversion strategy between numpy arrays and PyTorch tensors

### 2. Create Lightning Standard Model Foundation
- Create `src/models/lightning_standard.py` file
- Implement basic `LightningStandardModel` class inheriting from `BaseModel`
- Define class constants: `MODEL_TYPE`, `DESCRIPTION`, `DEFAULT_HYPERPARAMETERS`
- Add necessary imports for PyTorch, Lightning, and framework dependencies
- Implement `__init__` method with hyperparameter handling

### 3. Implement Core PyTorch Lightning Integration
- Create internal `ForecastingModel` Lightning module within the model class
- Implement neural network architecture matching the prototype (3-layer MLP)
- Add data conversion utilities (numpy to torch tensors and vice versa)
- Implement the `train` method with proper Lightning trainer configuration
- Handle validation data setup and DataLoader creation

### 4. Implement Prediction and Model Info Methods
- Implement the `predict` method with tensor conversion and model evaluation mode
- Add proper error handling for untrained model predictions
- Implement `get_model_info` method returning model metadata and parameters
- Add model serialization support for framework storage system

### 5. Add Error Handling and Validation
- Import and use `ModelTrainingError` and `ModelPredictionError` exceptions
- Add comprehensive error handling for training failures and prediction errors
- Validate input data shapes and types in all methods
- Add logging for training progress and debugging

### 6. Test Model Discovery and Integration
- Verify the model is automatically discovered by the ModelTypeRegistry
- Test model creation through the factory pattern
- Validate integration with ModelTrainer's `_train_model_with_params` method
- Test with sample data to ensure end-to-end functionality

### 7. Create Integration Tests
- Add Lightning model tests to existing test files
- Test model training with both small and realistic datasets
- Verify metrics calculation integration through centralized system
- Test model persistence and loading functionality
- Validate compatibility with COMBINED and INDIVIDUAL strategies

### 8. Validate Complete Integration
- Run complete test suite to ensure no regressions
- Test Lightning model through BenchmarkPipeline end-to-end workflow
- Verify model shows up in model type discovery
- Confirm metrics are calculated correctly through centralized system

## Testing Strategy
- Unit tests for Lightning model class methods (train, predict, get_model_info)
- Integration tests with ModelTrainer and model factory system  
- End-to-end tests through BenchmarkPipeline with Lightning models
- Performance tests comparing Lightning vs XGBoost on same data
- Error handling tests for invalid inputs and training failures
- Serialization/deserialization tests for model persistence

## Acceptance Criteria
- [x] LightningStandardModel class created and inherits from BaseModel
- [x] Model is automatically discovered by ModelTypeRegistry
- [x] Model can be created and trained through existing ModelTrainer interface
- [x] Model produces predictions comparable to prototype (within 5% RMSE)
- [x] Model integrates with centralized metrics calculation
- [x] Model works with both COMBINED and INDIVIDUAL strategies
- [x] Model can be saved and loaded through framework storage system
- [x] All existing tests continue to pass
- [x] New integration tests added and passing
- [x] Model supports configurable hyperparameters (hidden_size, lr, dropout, max_epochs)

## Validation Commands
Execute these commands to validate the task is complete:

- `python -c "from src.model_types import model_registry; print(model_registry.list_available_types())"` - Verify Lightning model is discovered
- `python -c "from src.model_types import model_registry; info = model_registry.get_model_info('lightning_standard'); print(f'Model: {info.name}, Description: {info.description}')"` - Check model info
- `pytest tests/test_model_training.py::test_lightning_model_training -v` - Test Lightning model training
- `pytest tests/test_integration.py -v` - Ensure integration tests pass with new model
- `python -c "from src.models.lightning_standard import LightningStandardModel; print('Lightning model imports successfully')"` - Test direct import
- `pytest tests/ -v` - Run complete test suite to ensure no regressions

## Notes
- Requires PyTorch and Lightning dependencies - add with `uv add torch lightning`
- Lightning models require different data handling (tensors vs numpy arrays)
- Model persistence will need to handle PyTorch state_dict serialization
- Training may require different batch sizes and epochs compared to XGBoost
- Consider GPU availability but ensure CPU fallback works correctly
- Lightning models may have different performance characteristics - monitor memory usage