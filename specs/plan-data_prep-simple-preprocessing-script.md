# Plan: Simple Data Preprocessing Script

## Metadata
adw_id: `data_prep`
prompt: `based on the laod_data notebook in the notebooks directory. create a python script that loads the data, prepares it like in the notebook and then saves it in the data/ directory under the directory processed_data/. I should be able to run this script from terminal and then it would create processed datasets in the mentioned directories. Keep it as simple as possible.`
task_type: chore
complexity: simple

## Task Description
Create a standalone Python script that replicates the data loading and preprocessing steps from the `laod_data.ipynb` notebook. The script should load raw M5 data files, apply feature engineering transformations, and save the processed datasets to a new `data/processed_data/` directory for easy reuse. The script must be executable from the terminal with minimal dependencies.

## Objective
Create a simple, terminal-executable Python script that automates the data preprocessing pipeline from the notebook, enabling quick generation of processed M5 datasets without requiring Jupyter notebook execution.

## Relevant Files
Use these files to complete the task:

- `notebooks/laod_data.ipynb` - Source notebook containing data preprocessing logic to replicate
- `data/train_data_features.feather` - Raw M5 feature matrix input
- `data/train_data_target.feather` - Raw target values input  
- `data/feature_mapping_train.pkl` - Feature metadata dictionary input
- `src/data_loading.py` - Reference for existing data loading patterns (optional)

### New Files
- `scripts/prepare_data.py` - Main preprocessing script to be created
- `data/processed_data/` - New directory for processed datasets

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Create Scripts Directory Structure
- Create `scripts/` directory if it doesn't exist
- Create `data/processed_data/` directory structure

### 2. Extract Core Functions from Notebook
- Extract `create_calendric_features()` function from notebook
- Extract `add_lag_features()` function from notebook
- Simplify functions to remove unused parameters and focus on core functionality

### 3. Create Main Preprocessing Script
- Create `scripts/prepare_data.py` with proper imports
- Implement data loading logic (features, target, mapping)
- Apply feature engineering pipeline in sequence:
  - Calendric features creation
  - Categorical variable dummy encoding
  - Lag features (1-7 days) for sales data
  - Filter not-for-sale items
  - Create trend feature based on date range

### 4. Add Data Saving Logic
- Save processed features to `data/processed_data/processed_features.feather`
- Save processed target to `data/processed_data/processed_target.feather`
- Save feature mapping to `data/processed_data/processed_feature_mapping.pkl`
- Add progress logging for user feedback

### 5. Add Command Line Interface
- Add argument parsing for optional parameters (input/output paths)
- Add help text and usage information
- Make script executable with `python scripts/prepare_data.py`

### 6. Create Documentation and Validation
- Add docstring documentation to script
- Test script execution from terminal
- Verify output files are created correctly
- Validate data shape and basic statistics match notebook results

## Acceptance Criteria
- [ ] Script runs successfully from terminal with `python scripts/prepare_data.py`
- [ ] Creates `data/processed_data/` directory with processed datasets
- [ ] Processed features include all transformations from notebook:
  - Calendric features (month, day_of_week, week_of_year, quarter, year, is_weekend)
  - Dummy variables for categorical features
  - Lag features (feature_0038_lag_1 through feature_0038_lag_7)
  - Filtered not-for-sale items
  - Trend feature based on date progression
- [ ] Output files are in correct format (feather for dataframes, pickle for mapping)
- [ ] Script includes progress logging and completion confirmation
- [ ] No external dependencies beyond those already in requirements.txt
- [ ] Processed data shape and basic statistics match notebook preprocessing results

## Validation Commands
Execute these commands to validate the task is complete:

- `python -m py_compile scripts/prepare_data.py` - Verify script compiles without errors
- `python scripts/prepare_data.py --help` - Test help functionality
- `python scripts/prepare_data.py` - Execute full preprocessing pipeline
- `ls -la data/processed_data/` - Verify output files are created
- `python -c "import polars as pl; df = pl.read_ipc('data/processed_data/processed_features.feather'); print(f'Shape: {df.shape}'); print(f'Columns: {len(df.columns)}')"` - Validate processed features
- `python -c "import polars as pl; df = pl.read_ipc('data/processed_data/processed_target.feather'); print(f'Target shape: {df.shape}')"` - Validate processed target

## Notes
- Keep the script simple and focused on core preprocessing steps from the notebook
- Use Polars for consistency with existing framework
- Avoid complex command line arguments - prioritize simplicity
- Include basic error handling for missing input files
- The script should complete in reasonable time for the full M5 dataset