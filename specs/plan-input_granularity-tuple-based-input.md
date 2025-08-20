# Plan: Replace Granularity-Based Input with Tuple-Based Input

## Metadata
adw_id: `input_granularity`
prompt: `I want you now to change the input of the pipeline. I dont care anymore about granularity. what i would have is a list of tuples. the tuples is a pair of (product_id,store_id). this list can contain from 1 to several thousands touples. additionally there should be a flag where I can set if this should be one model for all the tuples(a tuple defines an sku)  or a model for every tuple(one model per sku).`
task_type: refactor
complexity: complex

## Task Description
Replace the current granularity-based pipeline input system (SKU, PRODUCT, STORE levels) with a tuple-based system where:
- Input is a list of (product_id, store_id) tuples 
- Each tuple defines an SKU (product at specific store)
- List can contain 1 to several thousand tuples
- Add a modeling strategy flag that determines:
  - `combined`: One model trained on all tuples together
  - `individual`: One model per tuple (individual SKU models)

## Objective
Completely refactor the pipeline input interface to eliminate granularity concepts and provide a simpler, more direct way to specify which SKUs to model, with flexible modeling strategies for handling multiple SKUs.

## Problem Statement
The current granularity-based system (SKU/PRODUCT/STORE) adds complexity and limits flexibility in specifying exactly which SKUs should be included in modeling. Users need a more direct way to specify SKU combinations and control whether they want combined or individual models.

## Solution Approach
1. Replace GranularityLevel enum with ModelingStrategy enum
2. Update all pipeline interfaces to accept List[Tuple[int, int]] instead of granularity + entity_ids
3. Modify DataLoader to filter data based on tuple lists
4. Update model training to support both combined and individual strategies
5. Adjust metadata and registry to track tuple lists instead of granularity
6. Maintain backward compatibility where possible for evaluation and visualization

## Relevant Files
Use these files to complete the task:

- `src/data_structures.py` - Contains GranularityLevel enum and ModelMetadata to be updated
- `src/data_loading.py` - Contains get_data_for_granularity method to be replaced
- `src/benchmark_pipeline.py` - Main pipeline methods need interface changes
- `src/model_training.py` - Model training methods need to support new strategies
- `src/evaluation.py` - Evaluation methods may need updates for new metadata
- `src/__init__.py` - Public API exports need updates

## Implementation Phases
### Phase 1: Foundation
Replace core data structures and enums with tuple-based equivalents

### Phase 2: Core Implementation
Update DataLoader and BenchmarkPipeline to work with tuple lists and modeling strategies

### Phase 3: Integration & Polish
Update model training, evaluation, and ensure all components work together

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Update Core Data Structures
- Replace GranularityLevel enum with ModelingStrategy enum (`combined`, `individual`)
- Update ModelMetadata to store tuple lists instead of granularity
- Create helper types for tuple lists and modeling strategies
- Update BenchmarkModel identifier generation for new structure

### 2. Create New DataLoader Methods
- Replace get_data_for_granularity with get_data_for_tuples method
- Add support for filtering data by list of (product_id, store_id) tuples
- Handle both combined and individual data preparation strategies
- Remove granularity-specific aggregation methods that are no longer needed

### 3. Update BenchmarkPipeline Interface
- Replace run_single_model_experiment method with new tuple-based interface
- Add run_experiment method that accepts tuple lists and modeling strategy
- Update run_multi_entity_experiment to work with tuple lists
- Modify run_full_benchmark_suite to use new interface or remove if not applicable

### 4. Update ModelTrainer Interface
- Update train_model method to accept ModelingStrategy instead of GranularityLevel
- Ensure model metadata stores tuple information correctly
- Update model ID generation to reflect tuple-based approach

### 5. Update ModelRegistry and Evaluation
- Update ModelRegistry to filter by modeling strategy instead of granularity
- Modify evaluation methods to work with new metadata structure
- Update report generation for tuple-based models

### 6. Update Public API and Examples
- Update __init__.py exports to remove GranularityLevel, add ModelingStrategy
- Update example configuration and main function to use new interface
- Clean up any remaining granularity references

### 7. Add Comprehensive Validation
- Test combined modeling strategy with multiple tuples
- Test individual modeling strategy with multiple tuples  
- Test edge cases (single tuple, large tuple lists)
- Validate model metadata and registry functionality

## Testing Strategy
- Unit tests for tuple list validation and processing
- Integration tests for both combined and individual modeling strategies
- Performance tests with large tuple lists (hundreds to thousands)
- Edge case testing with single tuples and empty lists
- Backward compatibility tests for existing model registry functionality

## Acceptance Criteria
- Pipeline accepts List[Tuple[int, int]] as input instead of granularity + entity_ids
- ModelingStrategy enum replaces GranularityLevel enum
- Combined strategy trains one model on all specified SKUs
- Individual strategy trains separate models for each SKU
- Model metadata correctly stores tuple lists and modeling strategy
- All existing pipeline functionality (training, evaluation, registry) works with new structure
- Public API updated to reflect new interface
- No references to granularity remain in codebase
- Performance acceptable for large tuple lists (1000+ tuples)

## Validation Commands
Execute these commands to validate the task is complete:

- `python -c "from src import ModelingStrategy; print(list(ModelingStrategy))"` - Test new enum exists
- `python -c "import src; print('GranularityLevel' not in dir(src))"` - Verify GranularityLevel removed
- `find src -name "*.py" -exec grep -l "GranularityLevel\|granularity" {} \;` - Should return minimal files
- `python -c "from src import BenchmarkPipeline; import inspect; sig = inspect.signature(BenchmarkPipeline().run_experiment if hasattr(BenchmarkPipeline(), 'run_experiment') else lambda x: None); print('New interface exists')"` - Test new interface
- `python -m py_compile src/*.py` - Test all files compile

## Notes
This is a significant architectural change that affects the core interface of the framework. Care must be taken to:
- Ensure model metadata correctly captures the new tuple-based information
- Maintain performance with large tuple lists by using efficient Polars operations
- Consider memory usage when processing thousands of tuples simultaneously
- Provide clear examples of both modeling strategies
- Consider adding validation for tuple list inputs (valid product_id/store_id combinations)

The new interface should be more intuitive: users specify exactly which SKUs they want (as product_id, store_id pairs) and choose whether to model them together or separately.