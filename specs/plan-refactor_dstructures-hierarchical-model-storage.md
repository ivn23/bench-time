# Plan: Hierarchical Model Storage and Configuration Refactoring

## Metadata
adw_id: `refactor_dstructures`
prompt: `I want you to plan the restructuring of the data_structures script and the data saving/loading. for this take a look at the data_structures.py script in the src/ directory. I want the config classes to be adjusted for the fact that now it is possible to train different models. Additionally I want the model saving be restructured in the following way: In the models/directory i want to have a dictory for every store id, that should again contain a directory for for every product id and in that directory one directory for every model type that is supported by this framework. The right directory should be figured out by the store x product combination that is passed to the config files when initializing the pipeline. It is important that you only plan the steps and do not implement them`
task_type: refactor
complexity: complex

## Task Description
Refactor the data structures and model storage system to implement a hierarchical directory structure for models organized by store_id/product_id/model_type, while updating configuration classes to better support multiple model types. The new structure should organize models as `models/{store_id}/{product_id}/{model_type}/` instead of the current flat `models/{model_id}/` structure.

## Objective
Create a hierarchical model storage system that organizes models by business entities (store, product, model type) while updating configuration classes to support the extensible model architecture, making model management more intuitive and scalable.

## Problem Statement
The current model storage system uses a flat directory structure with complex model IDs, making it difficult to:
1. Locate models for specific store/product combinations
2. Compare different model types for the same SKU
3. Manage models at scale across many stores and products
4. Support the growing variety of model types in the framework

The configuration classes also need updates to better support the multi-model architecture that has been developed.

## Solution Approach
Implement a hierarchical storage system organized by business entities:
- **Store Level**: Top-level directories for each store_id
- **Product Level**: Subdirectories for each product_id within stores  
- **Model Type Level**: Further subdirectories for each model type (xgboost_standard, xgboost_quantile, etc.)
- **Enhanced Configs**: Update configuration classes to better support model selection and parameterization

## Relevant Files
- **src/data_structures.py** - Core data structures that need refactoring (ModelRegistry, configs)
- **src/benchmark_pipeline.py** - Pipeline that needs to use new directory structure
- **src/models/__init__.py** - Model type registration and discovery
- **src/models/base.py** - Base model interface
- **src/models/xgboost_standard.py** - Existing model implementations
- **src/models/xgboost_quantile.py** - Existing model implementations
- **tests/test_data_structures.py** - Tests that need updating for new structure
- **tests/test_integration.py** - Integration tests for new model storage

### New Files
- **src/model_types.py** - Registry and factory for supported model types
- **src/storage_utils.py** - Utilities for hierarchical path management
- **context/model_storage_migration.md** - Documentation for migration from old to new structure

## Implementation Phases

### Phase 1: Foundation
- Define new hierarchical storage structure and path utilities
- Create model type registry system for dynamic model discovery
- Update configuration classes to support multi-model scenarios
- Design backward compatibility strategy

### Phase 2: Core Implementation  
- Refactor ModelRegistry to use hierarchical paths
- Update ModelMetadata to include explicit store_id/product_id fields
- Modify model saving/loading methods for new directory structure
- Implement model type factory pattern

### Phase 3: Integration & Polish
- Update BenchmarkPipeline to use new storage system
- Create migration utilities for existing models
- Update all tests for new structure
- Add comprehensive documentation

## Step by Step Tasks

### 1. Design New Storage Architecture
- Define hierarchical path structure: `{base}/models/{store_id}/{product_id}/{model_type}/{model_instance}/`
- Create `StoragePath` utility class for path management
- Design model instance naming within model type directories
- Plan backward compatibility approach for existing models

### 2. Create Model Type Registry System
- Create `src/model_types.py` with `ModelTypeRegistry` class
- Implement dynamic model type discovery from `src/models/` directory
- Define standard interface for model type metadata (name, class, default configs)
- Add factory method for creating models by type string

### 3. Refactor Configuration Classes
- Update `TrainingConfig` to support model-specific parameter dictionaries
- Add `ModelSelectionConfig` class for specifying which models to train
- Enhance `DataConfig` to include default store/product filtering options
- Add validation methods for configuration consistency

### 4. Update Core Data Structures
- Add `store_id` and `product_id` fields to `ModelMetadata`
- Create `ModelStorageLocation` dataclass for path management
- Update `SkuTuple` type hints and utilities for better type safety
- Modify `BenchmarkModel.get_identifier()` for new naming convention

### 5. Refactor ModelRegistry Class
- Update `__init__()` to accept hierarchical storage configuration
- Refactor `save_model()` to create appropriate directory hierarchy
- Modify `load_model()` to navigate hierarchical paths
- Add methods: `list_models_by_store()`, `list_models_by_product()`, `list_model_types()`
- Implement `find_models()` with filtering by store/product/model_type

### 6. Create Storage Utilities
- Implement `create_model_path()` for generating hierarchical paths
- Add `ensure_model_directory()` for directory creation with proper permissions
- Create `parse_model_path()` for extracting store/product/model info from paths
- Implement path validation and sanitization utilities

### 7. Update BenchmarkPipeline Integration
- Modify pipeline to extract store_id/product_id from sku_tuples
- Update experiment naming to reflect hierarchical organization
- Ensure model registration uses new path structure
- Add validation for store/product consistency in sku_tuples

### 8. Implement Backward Compatibility
- Create `ModelMigration` class for converting old model storage to new format
- Add detection logic for old vs new storage formats
- Implement automatic migration on first load of old models
- Provide migration validation and rollback capabilities

### 9. Update Model Loading/Saving Logic
- Modify model file structure within model type directories
- Update metadata.json to include hierarchical location information
- Ensure data_splits.json handling works with new structure
- Add model versioning support within model type directories

### 10. Comprehensive Testing Updates
- Update `test_data_structures.py` for new ModelRegistry behavior
- Modify integration tests to validate hierarchical storage
- Add tests for migration functionality
- Create performance tests for model discovery at scale

### 11. Documentation and Migration Guide
- Update README.md with new storage structure documentation
- Create migration guide in `context/model_storage_migration.md`
- Update CLAUDE.md with new storage examples
- Add troubleshooting guide for common migration issues

## Testing Strategy
- **Unit Tests**: Test each component (ModelRegistry, storage utils, configs) in isolation
- **Integration Tests**: Test complete pipeline with new storage structure
- **Migration Tests**: Verify old models can be migrated to new structure without data loss
- **Performance Tests**: Ensure hierarchical storage doesn't significantly impact performance
- **Edge Case Tests**: Handle edge cases like special characters in store/product IDs
- **Backward Compatibility Tests**: Ensure old code can still load migrated models

## Acceptance Criteria
- [ ] Models are stored in hierarchical structure: `models/{store_id}/{product_id}/{model_type}/`
- [ ] Configuration classes support multiple model types with appropriate parameter isolation
- [ ] Existing models can be migrated to new structure without data loss
- [ ] Model discovery and loading works efficiently with hierarchical paths
- [ ] All existing tests pass with updated storage structure
- [ ] Pipeline can train multiple model types for same SKU and store appropriately
- [ ] Model metadata includes explicit store_id and product_id fields
- [ ] Directory creation handles edge cases (special characters, long IDs)
- [ ] Performance is maintained or improved compared to current system

## Validation Commands
- `python -m pytest tests/test_data_structures.py -v` - Verify core data structure changes
- `python -m pytest tests/test_integration.py -v` - Test complete pipeline functionality
- `python -c "from src.model_types import ModelTypeRegistry; print(ModelTypeRegistry.list_available_types())"` - Verify model type discovery
- `python -c "from src.data_structures import ModelRegistry; r = ModelRegistry(); print(r.list_models_by_store('1234'))"` - Test hierarchical queries
- `python -c "from src import BenchmarkPipeline, DataConfig, TrainingConfig; print('Import successful')"` - Verify API compatibility
- `python src/benchmark_pipeline.py` - Run pipeline end-to-end with new structure

## Notes
- **Backward Compatibility**: Critical to maintain ability to load existing models
- **Performance**: Hierarchical structure should not significantly impact model loading speed
- **Scalability**: Design should handle thousands of store/product combinations efficiently  
- **Model Versioning**: Consider future need for model versioning within model type directories
- **Concurrency**: Ensure thread-safe model saving/loading in hierarchical structure
- **Storage**: Directory structure should be platform-independent (Windows/Unix)
- **Migration Strategy**: Provide clear migration path and validation for existing deployments