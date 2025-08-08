"""
Example usage of the M5 benchmarking framework.
Demonstrates how to use the framework for different scenarios.
"""

import logging
from pathlib import Path
import polars as pl

from src import (
    DataConfig, TrainingConfig, GranularityLevel,
    BenchmarkPipeline, DataLoader, FeatureEngineer,
    ModelTrainer, ModelRegistry, ModelEvaluator
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_single_sku_model():
    """Example: Train a single model for one SKU."""
    logger.info("=== Example: Single SKU Model ===")
    
    # Configure data and training
    data_config = DataConfig(
        features_path="data/train_data_features.feather",
        target_path="data/train_data_target.feather", 
        mapping_path="data/feature_mapping_train.pkl",
        lag_features=[1, 2, 3, 7],  # Reduced for faster training
        calendric_features=True,
        trend_features=True
    )
    
    training_config = TrainingConfig(
        validation_split=0.2,
        model_type="xgboost",
        hyperparameters={
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
    )
    
    # Initialize pipeline
    pipeline = BenchmarkPipeline(data_config, training_config, Path("results_single_sku"))
    
    # Load data
    pipeline.load_and_prepare_data()
    
    # Train model for specific SKU
    model = pipeline.run_single_model_experiment(
        granularity=GranularityLevel.SKU,
        entity_ids={"skuID": 278993},  # Use your SKU ID
        experiment_name="demo_sku_278993"
    )
    
    # Evaluate model
    evaluation_results = pipeline.evaluate_all_models()
    
    # Save results
    pipeline.save_experiment_log()
    
    logger.info("Single SKU model example completed")
    return model


def example_multi_granularity_comparison():
    """Example: Compare models across different granularity levels."""
    logger.info("=== Example: Multi-Granularity Comparison ===")
    
    # Configure for faster execution
    data_config = DataConfig(
        features_path="data/train_data_features.feather",
        target_path="data/train_data_target.feather",
        mapping_path="data/feature_mapping_train.pkl",
        lag_features=[1, 3, 7],  # Reduced feature set
        calendric_features=True,
        trend_features=False  # Disable for speed
    )
    
    training_config = TrainingConfig(
        validation_split=0.2,
        model_type="xgboost",
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.2,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0
        }
    )
    
    # Initialize pipeline
    pipeline = BenchmarkPipeline(data_config, training_config, Path("results_multi_granularity"))
    pipeline.load_and_prepare_data()
    
    # Get sample entities
    unique_entities = pipeline.data_loader.get_unique_entities()
    
    # Train models at different granularity levels (limited samples)
    sample_entities = {
        "skuIDs": unique_entities["skuIDs"][:2],      # 2 SKUs
        "productIDs": unique_entities["productIDs"][:2],  # 2 Products  
        "storeIDs": unique_entities["storeIDs"][:1]       # 1 Store
    }
    
    # Run limited benchmark suite
    all_models = pipeline.run_full_benchmark_suite(
        sample_entities=sample_entities,
        max_models_per_granularity=2
    )
    
    # Comprehensive evaluation
    evaluation_results = pipeline.evaluate_all_models()
    
    # Generate comparison report
    if "overall" in evaluation_results:
        report = pipeline.evaluator.generate_evaluation_report(
            evaluation_results["overall"],
            Path("results_multi_granularity/granularity_comparison_report.md")
        )
        print("\n" + "="*50)
        print("GRANULARITY COMPARISON REPORT")
        print("="*50)
        print(report)
    
    pipeline.save_experiment_log()
    
    logger.info("Multi-granularity comparison completed")
    return all_models, evaluation_results


def example_custom_feature_engineering():
    """Example: Custom feature engineering workflow."""
    logger.info("=== Example: Custom Feature Engineering ===")
    
    # Configure data loading only
    data_config = DataConfig(
        features_path="data/train_data_features.feather",
        target_path="data/train_data_target.feather",
        mapping_path="data/feature_mapping_train.pkl"
    )
    
    # Load data
    data_loader = DataLoader(data_config)
    features_df, target_df, mapping = data_loader.load_data(lazy=False)
    
    # Get data for a specific SKU
    sku_features, sku_target = data_loader.get_data_for_granularity(
        GranularityLevel.SKU,
        {"skuID": 278993},
        collect=True
    )
    
    logger.info(f"Raw data shape: {sku_features.shape}")
    
    # Custom feature engineering
    feature_engineer = FeatureEngineer(
        lag_features=[1, 2, 3, 4, 5, 6, 7, 14, 21],  # Include longer lags
        calendric_features=True,
        trend_features=True
    )
    
    # Engineer features
    engineered_df, feature_cols = feature_engineer.create_features(
        sku_features, sku_target, GranularityLevel.SKU, {"skuID": 278993}
    )
    
    logger.info(f"Engineered data shape: {engineered_df.shape}")
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # Show feature importance mapping
    feature_mapping = feature_engineer.get_feature_importance_mapping(feature_cols[:10])
    logger.info("Sample feature mapping:")
    for orig, mapped in list(feature_mapping.items())[:10]:
        logger.info(f"  {orig} -> {mapped}")
    
    # Prepare final model data
    X, y = feature_engineer.prepare_model_data(engineered_df, feature_cols)
    logger.info(f"Final model data: X{X.shape}, y{y.shape}")
    
    logger.info("Custom feature engineering example completed")
    return X, y, feature_cols


def example_model_storage_and_retrieval():
    """Example: Model storage and retrieval workflow."""
    logger.info("=== Example: Model Storage and Retrieval ===")
    
    # Train a simple model first
    data_config = DataConfig(
        features_path="data/train_data_features.feather",
        target_path="data/train_data_target.feather",
        mapping_path="data/feature_mapping_train.pkl",
        lag_features=[1, 2, 3],
        calendric_features=True
    )
    
    training_config = TrainingConfig(
        validation_split=0.2,
        model_type="xgboost",
        hyperparameters={
            'n_estimators': 80,
            'max_depth': 4,
            'learning_rate': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0
        }
    )
    
    # Initialize components
    pipeline = BenchmarkPipeline(data_config, training_config, Path("results_storage"))
    pipeline.load_and_prepare_data()
    
    # Train and save a model
    model = pipeline.run_single_model_experiment(
        granularity=GranularityLevel.SKU,
        entity_ids={"skuID": 278993},
        experiment_name="storage_demo"
    )
    
    model_id = model.get_identifier()
    logger.info(f"Trained and saved model: {model_id}")
    
    # Demonstrate model registry operations
    registry = pipeline.model_registry
    
    # List all models
    all_models = registry.list_models()
    logger.info(f"All models in registry: {all_models}")
    
    # List models by granularity
    sku_models = registry.list_models(GranularityLevel.SKU)
    logger.info(f"SKU-level models: {sku_models}")
    
    # Load model from disk
    loaded_model = registry.load_model(model_id)
    logger.info(f"Loaded model: {loaded_model.metadata.model_id}")
    logger.info(f"Model performance: {loaded_model.metadata.performance_metrics}")
    
    # Access model components
    logger.info(f"Model type: {loaded_model.metadata.model_type}")
    logger.info(f"Feature columns: {len(loaded_model.metadata.feature_columns)}")
    logger.info(f"Training samples: {len(loaded_model.data_split.train_bdIDs)}")
    logger.info(f"Validation samples: {len(loaded_model.data_split.validation_bdIDs)}")
    
    logger.info("Model storage and retrieval example completed")
    return loaded_model


def example_evaluation_and_visualization():
    """Example: Advanced evaluation and visualization."""
    logger.info("=== Example: Evaluation and Visualization ===")
    
    # First train a model to evaluate
    model = example_single_sku_model()  # Reuse previous example
    
    # Initialize evaluation components
    data_config = DataConfig(
        features_path="data/train_data_features.feather",
        target_path="data/train_data_target.feather",
        mapping_path="data/feature_mapping_train.pkl"
    )
    
    data_loader = DataLoader(data_config)
    data_loader.load_data(lazy=False)
    
    registry = ModelRegistry(Path("results_single_sku/models"))
    evaluator = ModelEvaluator(data_loader, registry)
    
    # Load the model
    model_id = model.get_identifier()
    loaded_model = registry.load_model(model_id)
    
    # Detailed evaluation
    eval_result = evaluator.evaluate_model(loaded_model)
    
    # Print detailed metrics
    logger.info("Detailed Evaluation Results:")
    for metric, value in eval_result["metrics"].items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Feature importance analysis
    if eval_result.get("feature_importance"):
        logger.info("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(eval_result["feature_importance"].items())[:10], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    # Generate visualization if lets-plot is available
    from src.evaluation import VisualizationGenerator
    viz_gen = VisualizationGenerator()
    
    if viz_gen.lets_plot_available:
        # Create prediction plot
        pred_plot = viz_gen.create_prediction_plot(eval_result)
        if pred_plot:
            logger.info("Prediction plot created (use .show() to display)")
        
        # Create error distribution plot  
        error_plot = viz_gen.create_error_distribution_plot(eval_result)
        if error_plot:
            logger.info("Error distribution plot created")
    else:
        logger.info("lets-plot not available for visualization")
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        eval_result,
        Path("results_single_sku/detailed_evaluation_report.md")
    )
    
    print("\n" + "="*50)
    print("DETAILED EVALUATION REPORT")
    print("="*50)
    print(report)
    
    logger.info("Evaluation and visualization example completed")
    return eval_result


if __name__ == "__main__":
    """
    Run examples based on what data you have available.
    Uncomment the examples you want to run.
    """
    
    print("M5 Benchmarking Framework - Example Usage")
    print("=" * 50)
    
    try:
        # Example 1: Single SKU model
        example_single_sku_model()
        
        # Example 2: Multi-granularity comparison
        # all_models, evaluation_results = example_multi_granularity_comparison()
        
        # Example 3: Custom feature engineering
        # X, y, feature_cols = example_custom_feature_engineering()
        
        # Example 4: Model storage and retrieval
        # loaded_model = example_model_storage_and_retrieval()
        
        # Example 5: Evaluation and visualization  
        # eval_result = example_evaluation_and_visualization()
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        logger.error("Make sure you have the M5 data files in the 'data/' directory")
        logger.error("Expected files:")
        logger.error("  - data/train_data_features.feather")
        logger.error("  - data/train_data_target.feather") 
        logger.error("  - data/feature_mapping_train.pkl")
    
    print("\nExample execution completed!")
    print("Check the 'results_*' directories for outputs.")