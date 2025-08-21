# Plan: Working Notebook Development

## Metadata
adw_id: `notebook_test`
prompt: `read the laod_data notebook in the notebooks/ directory. Based on what is done there create a new notebook working_notebook and do the following steps: 1)adjust the filtering and preprocessing the data in the way it is done in the framework, that means by store and product id instead of sku. Process the data like it is done in the cells before training and add a date that splits the training and test data instead of a percent split. skip hyperparameter tuning and pass a predefined parameter set to the model. do this for three productID/storeID combinations. all should be trained with the same hyperparameter set. Set seeds where needed to enforece deterministic behaviour. Key is to get output from the models that then can be tested on the same data using the pipeline. Pay attention to the vizualizations, they should be done in lets-plot, like in the load_data notebook. Use context7 to get documentation on packages like lets-plot or sklearn`
task_type: feature
complexity: complex

## Task Description
Create a new Jupyter notebook called "working_notebook.ipynb" that adapts the existing "laod_data.ipynb" notebook to align with the benchmarking framework's approach. The notebook will train XGBoost models on three productID/storeID combinations using tuple-based filtering, date-based train/test splitting, predefined hyperparameters, and deterministic seeds. All visualizations will use lets-plot, and the output will be compatible for testing with the framework pipeline.

## Objective
Develop a standalone notebook that demonstrates the framework's data processing approach while maintaining compatibility with the pipeline for validation. The notebook will serve as a reference implementation showing how to process data by productID/storeID tuples, use date-based splitting, and generate model outputs that can be compared with the framework's results.

## Problem Statement
The existing "laod_data.ipynb" notebook uses SKU-level filtering and percentage-based train/test splits, which doesn't align with the framework's updated approach of tuple-based filtering (productID, storeID) and date-based splitting. Additionally, the notebook performs hyperparameter tuning, while the framework now uses predefined hyperparameters for consistency.

## Solution Approach
Create a new notebook that:
1. Implements tuple-based data filtering for three specific productID/storeID combinations
2. Uses date-based temporal splitting instead of percentage-based splitting  
3. Applies the same predefined hyperparameters across all models for consistency
4. Maintains deterministic behavior through proper seed setting
5. Uses lets-plot for all visualizations to match the existing notebook style
6. Generates model outputs compatible with framework pipeline testing

## Relevant Files
- `notebooks/laod_data.ipynb` - Source notebook to adapt and improve
- `src/data_structures.py` - TrainingConfig with predefined hyperparameters
- `src/data_loading.py` - DataLoader with tuple-based filtering methods
- `context7 documentation` - lets-plot and scikit-learn documentation for implementation

### New Files
- `notebooks/working_notebook.ipynb` - New notebook implementing framework-aligned approach

## Implementation Phases
### Phase 1: Foundation
- Set up notebook structure and imports
- Implement data loading using framework patterns
- Define the three productID/storeID combinations for testing

### Phase 2: Core Implementation
- Implement tuple-based data filtering
- Create date-based train/test splitting
- Apply feature engineering with framework patterns
- Train models with predefined hyperparameters

### Phase 3: Integration & Polish
- Add comprehensive visualizations using lets-plot
- Validate outputs for framework compatibility
- Add documentation and explanatory text

## Step by Step Tasks

### 1. Setup Notebook Infrastructure
- Create new notebook `notebooks/working_notebook.ipynb`
- Add imports for polars, xgboost, lets-plot, sklearn, numpy, pandas
- Initialize lets-plot with `LetsPlot.setup_html()`
- Set random seeds for reproducibility (numpy, random, xgboost)

### 2. Define Experimental Configuration
- Define three productID/storeID combinations for testing
- Set predefined XGBoost hyperparameters matching framework defaults
- Define date split point for temporal train/test division
- Set random state seeds for deterministic behavior

### 3. Implement Data Loading and Filtering
- Load features, targets, and mapping data using polars
- Implement tuple-based filtering for each productID/storeID combination
- Filter out not-for-sale items and apply date ranges
- Validate data shapes and consistency across combinations

### 4. Create Feature Engineering Pipeline  
- Add calendric features (month, day_of_week, quarter, etc.)
- Create lag features (1-7 days) with proper grouping
- Add trend features based on date progression
- Encode categorical variables using one-hot encoding
- Remove redundant columns to prevent data leakage

### 5. Implement Date-Based Train/Test Splitting
- Define split date for temporal division
- Create train/test sets based on date threshold
- Ensure chronological ordering in time series data
- Validate split proportions and temporal integrity

### 6. Train Models with Predefined Hyperparameters
- Configure XGBoost with framework's default hyperparameters
- Train separate models for each productID/storeID combination
- Ensure deterministic training with random state seeds
- Store model objects and predictions for each combination

### 7. Generate Model Predictions and Evaluation
- Create predictions for test sets
- Calculate evaluation metrics (MSE, RMSE, MAE, R²)
- Store predictions in format compatible with framework pipeline
- Document model performance for each combination

### 8. Create Comprehensive Visualizations
- Implement prediction vs actual plots using lets-plot
- Create aggregated forecast plots (mean and sum by date)
- Generate individual SKU-level prediction plots
- Add error distribution boxplots by SKU
- Include optimization progress visualization (static, since no tuning)

### 9. Add Pipeline Compatibility Features
- Structure output data to match framework expectations
- Create metadata for model comparison with pipeline
- Add export functionality for pipeline testing
- Document data formats and structure

### 10. Documentation and Validation
- Add markdown cells explaining each step
- Include performance summaries and insights
- Validate deterministic behavior across runs
- Test output compatibility with framework pipeline

## Testing Strategy
- **Deterministic Testing**: Run notebook multiple times to ensure identical outputs
- **Data Validation**: Verify data filtering produces expected sample counts
- **Model Consistency**: Confirm all three models use identical hyperparameters
- **Pipeline Compatibility**: Test that outputs can be loaded and used by framework
- **Visualization Testing**: Ensure all lets-plot visualizations render correctly

## Acceptance Criteria
- [x] Notebook successfully processes three productID/storeID combinations
- [x] Date-based splitting replaces percentage-based splitting
- [x] Predefined hyperparameters used consistently across all models
- [x] All random seeds set for deterministic behavior
- [x] lets-plot used for all visualizations matching original notebook style
- [x] Model outputs structured for framework pipeline compatibility
- [x] Feature engineering follows framework patterns (calendric, lag, trend features)
- [x] Performance metrics calculated and displayed for each model
- [x] Individual SKU and aggregated visualizations included
- [x] Notebook runs end-to-end without errors

## Validation Commands
Execute these commands to validate the task is complete:

- `jupyter nbconvert --to notebook --execute notebooks/working_notebook.ipynb --output working_notebook_test.ipynb` - Test notebook execution
- `python -c "import polars as pl; print('Polars version:', pl.__version__)"` - Verify polars installation
- `python -c "from lets_plot import *; print('lets-plot imported successfully')"` - Verify lets-plot installation
- `python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"` - Verify XGBoost installation

## Notes
**Key Dependencies**: 
- polars>=0.20.0 for efficient data processing
- lets-plot for Grammar of Graphics visualizations  
- xgboost for gradient boosting models
- scikit-learn for evaluation metrics and utilities

**Framework Alignment**: The notebook will mirror the framework's approach while remaining standalone for educational and validation purposes. Special attention to tuple-based filtering and date-based splitting to match the updated framework architecture.

**Deterministic Behavior**: All random operations will be seeded to ensure reproducible results across runs, enabling reliable comparison with framework outputs.

**Data Compatibility**: Output format will be designed to enable direct comparison with framework pipeline results, facilitating validation of the framework's implementation.