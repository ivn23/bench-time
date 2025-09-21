# M5 Benchmarking Framework - Test Suite

This test suite provides comprehensive coverage for the M5 Time Series Benchmarking Framework using an 80/20 approach - focusing on integration testing and critical component validation rather than exhaustive edge case testing.

## Test Structure

### Core Test Files

- **`test_api.py`** - Public API and import validation tests
- **`test_basic_integration.py`** - Simplified integration test for core workflow
- **`test_integration.py`** - Comprehensive end-to-end pipeline tests
- **`test_data_loading.py`** - DataLoader component tests
- **`test_model_training.py`** - ModelTrainer component tests
- **`test_evaluation.py`** - ModelEvaluator component tests
- **`test_data_structures.py`** - Data structure and configuration validation tests

### Test Infrastructure

- **`conftest.py`** - Pytest fixtures and shared configuration
- **`fixtures/sample_data.py`** - Mock data generation utilities
- **`pytest.ini`** - Pytest configuration

## Running Tests

### Prerequisites

```bash
pip install pytest
```

### Run All Tests

```bash
# Run complete test suite
pytest tests/ -v

# Run with short error messages
pytest tests/ --tb=short

# Stop on first failure
pytest tests/ -x
```

### Run Specific Test Categories

```bash
# Integration tests only
pytest tests/test_integration.py -v

# API tests only
pytest tests/test_api.py -v

# Component tests only
pytest tests/test_data_loading.py tests/test_model_training.py tests/test_evaluation.py -v

# Basic functionality test
pytest tests/test_basic_integration.py -v
```

### Run with Markers

```bash
# Run integration tests
pytest -m integration -v

# Run unit tests
pytest -m unit -v
```

## Test Strategy

### Primary Focus: Integration Testing (70%)

The test suite prioritizes integration testing to validate complete workflows:

1. **End-to-end pipeline tests** - Test complete workflows from data loading through model training to evaluation
2. **Both modeling strategies** - COMBINED (one model for all SKUs) and INDIVIDUAL (model per SKU)
3. **Model persistence** - Save/load functionality with temporary directories
4. **Configuration handling** - Both percentage-based and date-based splits

### Secondary Focus: Component Testing (25%)

Critical component functionality:

1. **DataLoader** - Data loading, filtering, temporal splits, feature preparation
2. **ModelTrainer** - Model training, hyperparameter handling, metrics calculation
3. **ModelEvaluator** - Model evaluation, metrics calculation, comparison functionality
4. **Data Structures** - Dataclass creation, serialization, enum validation

### Minimal Focus: Edge Cases (5%)

Basic validation only:
1. **API imports** - Ensure all public components are importable
2. **Configuration validation** - Test invalid configurations are caught
3. **Error handling** - Test obvious failure modes

## Mock Data Strategy

Tests use generated mock data instead of requiring the full M5 dataset:

- **Small scale**: 5 SKUs, 50 time periods
- **Realistic structure**: Matches expected M5 schema
- **Fast execution**: All tests complete in under 30 seconds
- **No external dependencies**: Self-contained test data

### Sample Data Features

```python
# Generated in tests/fixtures/sample_data.py
- 5 SKUs across different product/store combinations
- 50 time periods (2020-01-01 to ~2020-02-19)
- Calendar features (month, day_of_week, etc.)
- Price and sales patterns
- Event indicators (Christmas, New Year, Halloween)
- Simple linear trends
```

## Test Coverage

### Integration Tests (`test_integration.py`)

- ✅ COMBINED strategy end-to-end workflow
- ✅ INDIVIDUAL strategy end-to-end workflow  
- ✅ Multiple experiments in single session
- ✅ Date-based vs percentage-based splits
- ✅ Model persistence and loading
- ✅ Experiment logging

### Component Tests

**DataLoader** (`test_data_loading.py`):
- ✅ Data loading (lazy and eager)
- ✅ SKU tuple filtering for both strategies
- ✅ Feature preparation for modeling
- ✅ Temporal splitting (percentage and date-based)
- ✅ Data validation and error handling

**ModelTrainer** (`test_model_training.py`):
- ✅ Model training for both strategies
- ✅ Hyperparameter handling and storage
- ✅ Model prediction capabilities
- ✅ Metrics calculation
- ✅ Reproducibility with random seeds
- ✅ Error handling for unsupported models

**ModelEvaluator** (`test_evaluation.py`):
- ✅ Model evaluation with pre-engineered data
- ✅ Comprehensive metrics calculation
- ✅ Model comparison functionality
- ✅ Report generation
- ✅ Error handling for edge cases

### Validation Tests

**Data Structures** (`test_data_structures.py`):
- ✅ ModelingStrategy enum
- ✅ ModelMetadata, DataSplit, BenchmarkModel dataclasses
- ✅ ModelRegistry functionality
- ✅ Configuration classes (DataConfig, TrainingConfig)
- ✅ Serialization compatibility

**API Tests** (`test_api.py`):
- ✅ All public imports work
- ✅ Basic instantiation
- ✅ Expected method signatures
- ✅ Configuration consistency
- ✅ Error handling

## Key Design Decisions

### 1. Integration-First Approach
Tests focus on complete workflows rather than isolated units, providing confidence that the framework works as users expect.

### 2. Mock Data Generation
Generated small, realistic datasets eliminate external dependencies while maintaining schema compatibility.

### 3. Fast Execution
All tests complete quickly to encourage frequent running during development.

### 4. Minimal Hyperparameters
Tests use small model configurations (5 estimators, depth 2) for speed while maintaining functionality validation.

### 5. Temporary Directories
All file I/O uses temporary directories, ensuring tests don't interfere with each other or leave artifacts.

## Common Issues and Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root directory
cd /path/to/Benchmarking

# Set PYTHONPATH if needed
PYTHONPATH=. pytest tests/
```

### Missing Dependencies
```bash
# Install required packages
pip install pytest polars numpy xgboost scikit-learn
```

### Test Failures
1. **Data structure issues**: Check that mock data matches expected schema
2. **Method name mismatches**: Verify method names match between tests and implementation
3. **Configuration issues**: Ensure test configurations are valid

### Performance Issues
Tests are designed to be fast. If tests are slow:
1. Check that `n_estimators` is small in test configurations
2. Verify mock data size is reasonable (default: 5 SKUs × 50 days)
3. Ensure temporary directories are being cleaned up

## Adding New Tests

### For New Components
1. Create new test file: `test_new_component.py`
2. Add fixtures to `conftest.py` if shared
3. Follow integration-first approach
4. Use mock data from `fixtures/sample_data.py`

### For New Features
1. Add integration test in `test_integration.py` first
2. Add component tests if needed
3. Update API tests if public interface changes
4. Document any new fixtures or utilities

## Continuous Integration

These tests are designed to run in CI environments:

- No external dependencies beyond Python packages
- Fast execution (< 30 seconds total)
- Clear pass/fail indicators
- Comprehensive error messages
- Temporary file cleanup

Example CI command:
```bash
pytest tests/ --tb=short -v
```