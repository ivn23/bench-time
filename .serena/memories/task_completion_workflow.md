# Task Completion Workflow

## When a Development Task is Completed

### Code Quality Checks
**Note**: No specific linting/testing configuration found in project. These are general recommendations:

1. **Manual Code Review**
   - Check for type consistency and proper imports
   - Verify docstring documentation for new functions/classes
   - Ensure proper error handling and logging
   - Follow established naming conventions (snake_case, PascalCase)

2. **Testing (if applicable)**
   ```bash
   # Run any existing tests
   python -m pytest tests/ # if tests directory exists
   python -m unittest discover # if using unittest
   ```

3. **Code Formatting (suggested)**
   ```bash
   # Format code (if tools configured)
   black src/
   isort src/
   ```

### Data Validation
1. **Check Data Dependencies**
   ```bash
   # Ensure required data files exist
   ls -la data/train_data_features.feather
   ls -la data/train_data_target.feather  
   ls -la data/feature_mapping_train.pkl
   ```

2. **Validate Output Structure**
   ```bash
   # Check output directories
   ls -la benchmark_results/models/
   ls -la benchmark_results/evaluation_results/
   ```

### Integration Testing
1. **Run Main Pipeline**
   ```bash
   # Test complete pipeline functionality
   python src/benchmark_pipeline.py
   ```

2. **Module Import Testing**
   ```python
   # Verify clean imports work
   from src import BenchmarkPipeline, DataConfig, TrainingConfig
   ```

### Documentation Updates
1. **Update Comments**: Ensure code changes are reflected in docstrings
2. **Update Type Hints**: Verify all new functions have proper type annotations  
3. **Log Changes**: Add appropriate logging statements for new functionality

### Version Control
1. **Stage Changes**
   ```bash
   git add .
   ```

2. **Commit Changes**
   ```bash
   git commit -m "feat: [description of changes]"
   # or
   git commit -m "fix: [description of bug fix]" 
   # or
   git commit -m "refactor: [description of refactoring]"
   ```

### Cleanup
1. **Remove Temporary Files**
   ```bash
   # Clean up test artifacts
   rm -rf __pycache__/ *.pyc
   rm -rf test_registry/ test_models/ pipeline_demo_results/
   ```

2. **Check Git Status**
   ```bash
   git status  # Ensure no untracked important files
   ```

## Before Pushing Changes
1. **Final Integration Test**: Run complete pipeline end-to-end
2. **Memory Usage Check**: Monitor for memory leaks with large datasets
3. **Performance Validation**: Ensure no significant performance regressions
4. **Documentation Accuracy**: Verify README and CLAUDE.md still accurate

## Post-Completion
1. **Update Experiment Log**: Document any significant changes to model behavior
2. **Backup Models**: Ensure important trained models are preserved
3. **Monitor Results**: Check that evaluation metrics are reasonable