# Plan: Testing Strategy for M5 Time Series Benchmarking Framework

## Metadata
adw_id: `testing_integration`
prompt: `I want have tests for my framework. Can you please come up with an apropriate testing strategy for this framework. Use context7 if you need documentation. I am expecting something that is enough to test the core functionalities of my framework. What i want to avoid is an overengineered approach. This said i want a method that does 80% of the work for 20% effort.`
task_type: feature
complexity: medium

## Task Description
Create a comprehensive but focused testing strategy for the M5 Time Series Benchmarking Framework that tests core functionality without overengineering. The approach should follow the 80/20 principle - achieving 80% of testing value with 20% of the effort by focusing on integration tests, critical component testing, and essential validation.

## Objective
Implement a pytest-based testing suite that:
- Tests core pipeline functionality end-to-end
- Validates critical component behavior (DataLoader, ModelTrainer, ModelEvaluator)
- Ensures data structures work correctly
- Provides confidence in the framework's reliability
- Is maintainable and not overly complex

## Problem Statement
The M5 benchmarking framework currently has no automated testing, making it difficult to:
- Ensure code changes don't break existing functionality
- Validate that the pipeline works correctly with different configurations
- Catch regressions early in development
- Provide confidence when refactoring or adding new features

## Solution Approach
Focus on **integration testing** as the primary strategy, supplemented by targeted unit tests for critical components. This approach tests the framework as users would actually use it, providing maximum confidence with minimal test maintenance overhead.

Key principles:
1. **Integration-first**: Test complete workflows rather than isolated units
2. **Mock external dependencies**: Use sample data instead of requiring real M5 dataset
3. **Fixture-based setup**: Reusable test data and configurations
4. **Essential validation only**: Focus on core functionality, not edge cases

## Relevant Files
Use these files to complete the task:

- `src/benchmark_pipeline.py` - Main pipeline orchestration (primary integration test target)
- `src/data_loading.py` - DataLoader class (core component testing)
- `src/model_training.py` - ModelTrainer class (core component testing)  
- `src/evaluation.py` - ModelEvaluator class (core component testing)
- `src/data_structures.py` - Data classes and enums (validation testing)
- `src/__init__.py` - Public API exports (import testing)

### New Files
- `tests/conftest.py` - Pytest fixtures and configuration
- `tests/test_integration.py` - End-to-end pipeline tests
- `tests/test_data_loading.py` - DataLoader component tests
- `tests/test_model_training.py` - ModelTrainer component tests
- `tests/test_evaluation.py` - ModelEvaluator component tests
- `tests/test_data_structures.py` - Data structure validation tests
- `tests/fixtures/sample_data.py` - Mock data generation utilities
- `pytest.ini` - Pytest configuration

## Implementation Phases
### Phase 1: Foundation
Set up testing infrastructure with pytest configuration, create mock data generation utilities, and establish test fixtures.

### Phase 2: Core Implementation
Implement integration tests for main pipeline workflows and unit tests for critical components.

### Phase 3: Integration & Polish
Add validation tests for data structures, ensure good test coverage of essential paths, and create documentation.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Setup Testing Infrastructure
- Install pytest and create pytest.ini configuration
- Create tests/ directory structure
- Set up basic conftest.py with common fixtures

### 2. Create Mock Data Generation Utilities
- Create fixtures/sample_data.py for generating test data
- Implement mock feature and target DataFrames
- Create sample configurations for testing

### 3. Implement Core Fixtures in conftest.py
- Create data_config and training_config fixtures
- Create sample_features and sample_target fixtures using Polars
- Create temporary directory fixture for model storage

### 4. Integration Tests (Primary Focus)
- Test complete pipeline workflows in test_integration.py
- Test COMBINED modeling strategy end-to-end
- Test INDIVIDUAL modeling strategy end-to-end
- Test model registry save/load functionality

### 5. DataLoader Component Tests
- Test data loading with mock data
- Test get_data_for_tuples with different strategies
- Test temporal split functionality
- Test feature preparation for modeling

### 6. ModelTrainer Component Tests
- Test model training with mock data
- Test hyperparameter handling
- Test model metadata creation
- Test different modeling strategies

### 7. ModelEvaluator Component Tests
- Test model evaluation with pre-engineered data
- Test metrics calculation
- Test model comparison functionality

### 8. Data Structure Validation Tests
- Test ModelingStrategy enum
- Test dataclass creation and serialization
- Test ModelRegistry functionality
- Test configuration validation

### 9. Import and API Tests
- Test that all public API components can be imported
- Test basic instantiation of main classes

### 10. Documentation and Final Validation
- Add pytest commands to validation section
- Document test structure and usage
- Run complete test suite validation

## Testing Strategy
### Primary Approach: Integration Testing (70% of effort)
- **End-to-end pipeline tests**: Test complete workflows from data loading through model training to evaluation
- **Realistic scenarios**: Use tuple-based SKU selection as users would
- **Both strategies**: Test COMBINED and INDIVIDUAL modeling approaches
- **Model persistence**: Test save/load functionality

### Secondary Approach: Targeted Unit Testing (25% of effort)  
- **Critical components only**: DataLoader, ModelTrainer, ModelEvaluator core methods
- **Data structure validation**: Ensure dataclasses and enums work correctly
- **Configuration testing**: Validate configs handle expected inputs

### Minimal Approach: Edge Case Testing (5% of effort)
- **Import validation**: Ensure public API works
- **Basic error handling**: Test obvious failure modes
- **Configuration validation**: Test invalid config detection

### Mock Data Strategy
Instead of requiring the full M5 dataset:
- Generate small Polars DataFrames with realistic structure
- Create 3-5 sample SKUs with ~100 time periods each
- Include all necessary columns (bdID, date, target, features)
- Use pytest fixtures for reusable test data

### Fixture Organization
- **conftest.py**: Common fixtures (configs, mock data, temp directories)
- **sample_data.py**: Mock data generation utilities
- **Scoped fixtures**: Use appropriate scopes (function, module, session) for efficiency

## Acceptance Criteria
- [ ] Complete integration test covering COMBINED strategy workflow
- [ ] Complete integration test covering INDIVIDUAL strategy workflow
- [ ] Unit tests for DataLoader core methods (get_data_for_tuples, temporal splits)
- [ ] Unit tests for ModelTrainer core functionality
- [ ] Unit tests for ModelEvaluator with mock model and data
- [ ] Validation tests for key data structures (ModelMetadata, DataSplit, etc.)
- [ ] Tests run successfully with `pytest tests/` command
- [ ] All tests use mock data, no external dataset dependencies
- [ ] Test coverage focuses on core functionality, not edge cases
- [ ] Tests complete in under 30 seconds
- [ ] Clear documentation on running tests

## Validation Commands
Execute these commands to validate the task is complete:

- `pytest tests/ -v` - Run all tests with verbose output
- `pytest tests/test_integration.py -v` - Run integration tests specifically  
- `pytest tests/ --tb=short` - Run tests with concise error reporting
- `pytest tests/ -x` - Run tests, stop on first failure
- `python -c "import src; print('Import successful')"` - Verify imports work

## Notes
### Dependencies
- Add pytest to requirements: `pytest>=7.0.0`
- Consider pytest-mock for advanced mocking if needed: `pytest-mock>=3.10.0`

### Key Testing Principles
1. **Integration over isolation**: Test workflows, not individual methods
2. **Mock external dependencies**: No real datasets, network calls, or file I/O beyond temp directories
3. **Fast execution**: All tests should complete quickly to encourage frequent running
4. **Maintainable**: Tests should be simple to understand and modify
5. **Focused scope**: Test what matters most to users

### Framework-Specific Considerations
- Use Polars for mock DataFrames to match production code
- Test both ModelingStrategy.COMBINED and INDIVIDUAL workflows
- Validate tuple-based SKU selection works correctly
- Ensure model registry and persistence work with temporary directories
- Test that XGBoost integration works without needing to validate XGBoost itself

### Avoid Overengineering
- Don't test third-party libraries (XGBoost, Polars, sklearn)
- Don't test complex error scenarios unless they're critical
- Don't test visualization functionality in detail
- Don't create comprehensive performance tests
- Don't test every possible configuration combination