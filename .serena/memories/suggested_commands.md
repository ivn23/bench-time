# Suggested Commands for Development

## Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup project directory structure
mkdir -p data benchmark_results
```

## Running the Framework
```bash
# Run main pipeline example
python src/benchmark_pipeline.py

# Run pipeline from project root as module
python -m src.benchmark_pipeline
```

## Development Commands
**Note**: No specific linting, formatting, or testing configuration files found in the project. The following are suggested based on Python best practices:

### Testing (if implemented)
```bash
# Run tests (if test files exist)
python -m pytest tests/
python -m unittest discover -s tests/
```

### Code Quality (suggested)
```bash
# Linting (if configured)
flake8 src/
pylint src/

# Formatting (if configured)
black src/
isort src/

# Type checking (if configured)
mypy src/
```

## Data Management
```bash
# Verify data directory structure
ls -la data/
# Should contain:
# - train_data_features.feather
# - train_data_target.feather  
# - feature_mapping_train.pkl
```

## System Utilities (macOS/Darwin)
```bash
# File operations
ls -la              # List files with details
find . -name "*.py" # Find Python files
grep -r "pattern"   # Search for patterns (prefer ripgrep if available)

# Git operations
git status          # Check repository status
git add .           # Stage changes
git commit -m "msg" # Commit changes
git log --oneline   # View commit history
```

## Project-Specific Operations
```bash
# Check output directory
ls -la benchmark_results/models/

# Monitor log files
tail -f logs/*.log

# Clean temporary files
rm -rf __pycache__/ *.pyc test_registry/ test_models/
```

## Environment Management
```bash
# Python environment (if using conda/venv)
python --version    # Check Python version
pip list           # List installed packages
pip freeze         # Export current environment
```