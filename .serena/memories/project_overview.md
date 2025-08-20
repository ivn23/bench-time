# Project Overview: M5 Benchmarking Framework

## Purpose
A comprehensive benchmarking framework for time series forecasting models using the M5 competition dataset. The framework provides end-to-end capabilities for multi-granularity forecasting (SKU, Product, Store levels), feature engineering, model training with hyperparameter optimization, evaluation, and comparison.

## Tech Stack
- **Core Data Processing**: Polars (lazy evaluation for memory efficiency), NumPy, Pandas
- **Machine Learning**: XGBoost, scikit-learn
- **Hyperparameter Optimization**: Optuna  
- **Visualization**: lets-plot
- **Experiment Tracking**: MLflow
- **Database**: SQLAlchemy, PostgreSQL (psycopg2-binary)
- **Serialization**: joblib, pyarrow
- **Language**: Python 3.x

## Project Structure
```
src/
├── __init__.py               # Package initialization with exports
├── data_structures.py        # Core data classes and enums
├── data_loading.py           # Memory-efficient data loading with Polars
├── feature_engineering.py    # Multi-granularity feature creation
├── model_training.py         # Training with Optuna hyperparameter optimization
├── evaluation.py             # Comprehensive model evaluation and comparison
└── benchmark_pipeline.py     # Main orchestration pipeline

data/                         # Ignored by git - M5 dataset files
├── train_data_features.feather
├── train_data_target.feather
└── feature_mapping_train.pkl

benchmark_results/            # Output directory structure
├── models/                   # Trained model storage
├── evaluation_results/       # Evaluation reports
└── experiment_log.json       # Complete experiment tracking
```

## Key Features
- Multi-granularity modeling (SKU, Product, Store, Global levels)
- Memory-efficient processing with Polars LazyFrames
- Automated feature engineering with lag, calendric, and trend features
- Hyperparameter optimization using Optuna with time series cross-validation
- Complete model registry with metadata and reproducibility tracking
- Comprehensive evaluation with multiple metrics and ranking
- Extensible architecture for new algorithms and granularity levels