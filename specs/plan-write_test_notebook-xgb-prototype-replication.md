# Plan: Rewrite test_pipeline notebook to replicate xgb_prototype functionality

## Metadata
adw_id: `write_test_notebook`
prompt: `write test_notebook. I want you to rewrite the test_pipeline notebook in the way that it does exactl what xgb_prototpe is doing but instead would pass allt he necessary hyperparameters, data pathts etc. to the pipeline and get the results to evaluate. try to keep it as short as possible with no exessive print statements and no intensive comments. For this check what you the newly changed pipeline and only then start.`
task_type: refactor
complexity: medium

## Task Description
Rewrite the test_pipeline notebook to replicate the exact functionality of xgb_prototype.ipynb using the current extensible pipeline architecture. The new notebook should use the pipeline to achieve the same results as the prototype (filtering for productID 80558, using optimized hyperparameters, and evaluating performance) but with minimal code and output.

## Objective
Create a concise notebook that uses BenchmarkPipeline to replicate xgb_prototype's workflow: loading M5 data, training XGBoost on productID 80558 with optimized hyperparameters, and evaluating performance metrics (MSE, R²).

## Problem Statement
The current test_pipeline notebook is verbose and doesn't match the specific workflow of xgb_prototype.ipynb. The xgb_prototype uses manual data processing, Optuna optimization, and focuses on a single product (80558), while the test_pipeline uses the extensible pipeline architecture with different configuration patterns.

## Solution Approach
Leverage the current BenchmarkPipeline architecture to replicate xgb_prototype's exact behavior by:
1. Using the extensible model configuration system with optimized hyperparameters from the prototype
2. Configuring data loading to match prototype's paths and product filtering
3. Using COMBINED strategy to mirror the prototype's single-model approach
4. Implementing equivalent evaluation and minimal essential output

## Relevant Files
- `src/test_pipeline.ipynb` - Target notebook to rewrite
- `notebooks/model_prototypes/xgb_prototype.ipynb` - Reference implementation to replicate
- `src/benchmark_pipeline.py` - Pipeline class to use
- `src/data_structures.py` - Configuration classes (TrainingConfig, DataConfig)
- `src/model_types.py` - Extensible model architecture

### New Files
None - modifying existing notebook only.

## Implementation Phases
### Phase 1: Analysis & Configuration
Extract key parameters and workflow from xgb_prototype and map them to current pipeline architecture.

### Phase 2: Core Implementation  
Rewrite notebook using pipeline architecture with equivalent functionality.

### Phase 3: Validation & Optimization
Ensure results match prototype and minimize code/output for conciseness.

## Step by Step Tasks

### 1. Extract xgb_prototype Parameters
- Extract optimized hyperparameters from xgb_prototype (n_estimators: 78, max_depth: 3, etc.)
- Identify data paths and filtering criteria (productID == 80558)
- Note evaluation metrics used (MSE, R²)

### 2. Configure Pipeline Architecture
- Set up DataConfig with correct paths matching xgb_prototype
- Create TrainingConfig using extensible architecture with extracted hyperparameters
- Use add_model_config() to specify xgboost_standard with optimized parameters

### 3. Implement Core Workflow
- Load data using pipeline.load_and_prepare_data()
- Define SKU tuples for productID 80558 (multiple store combinations)
- Run experiment using ModelingStrategy.COMBINED to mirror prototype's approach
- Extract performance metrics from trained model

### 4. Add Minimal Evaluation Output
- Display final MSE and R² metrics matching prototype output format
- Remove excessive print statements and comments
- Keep only essential information for validation

### 5. Validate Results
- Compare output metrics with xgb_prototype results
- Ensure workflow efficiency and code conciseness
- Test notebook execution from start to finish

## Testing Strategy
- Execute rewritten notebook and compare MSE/R² results with xgb_prototype
- Verify that pipeline correctly handles productID 80558 filtering
- Confirm hyperparameters are properly applied through extensible architecture
- Validate data loading paths and configuration

## Acceptance Criteria
- Notebook replicates xgb_prototype's core functionality using current pipeline
- Uses extracted optimized hyperparameters: n_estimators=78, max_depth=3, learning_rate=0.0636, etc.
- Filters data for productID == 80558 equivalent to prototype
- Produces comparable MSE and R² metrics to original prototype
- Code is concise with minimal print statements and comments
- Uses extensible model architecture (TrainingConfig.add_model_config())
- Executes successfully without errors

## Validation Commands
Execute these commands to validate the task is complete:

- `cd /Users/ivn/Documents/PhD/Transformer\ Research/Code/Benchmarking && jupyter nbconvert --to notebook --execute src/test_pipeline.ipynb --output test_pipeline_executed.ipynb` - Execute notebook to ensure it runs without errors
- Visual comparison of final MSE/R² metrics between rewritten notebook and original xgb_prototype
- Code review to ensure conciseness and removal of excessive output

## Notes
- The xgb_prototype uses hyperparameters optimized via Optuna: {'n_estimators': 78, 'max_depth': 3, 'learning_rate': 0.06356393066232492, 'subsample': 0.8136751464901273, 'colsample_bytree': 0.820105725620293, 'reg_alpha': 4.979378780027597, 'reg_lambda': 6.663822635873432}
- Original prototype focuses on productID 80558 with multiple SKUs (different stores)
- Current pipeline uses tuple-based approach: (product_id, store_id) pairs
- Must use COMBINED strategy to replicate single-model approach from prototype
- Data paths in prototype: '../data/processed/train_data_features.feather' (note: processed subdirectory)