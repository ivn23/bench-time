"""
Comprehensive Release Manager for consolidating complete experiment outputs.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from ..data_structures import ExperimentResults, BenchmarkModel
from ..storage_utils import HierarchicalStorageManager
from .factory import ReleaseManagerFactory

logger = logging.getLogger(__name__)


class ComprehensiveReleaseManager:
    """
    Comprehensive release manager that consolidates all experiment outputs
    into complete, self-contained release packages.
    
    This manager acts as a post-training consolidator that gathers:
    - Trained models
    - Configuration files
    - Evaluation results (optional)
    - Experiment logs
    - Metadata
    
    And creates structured release directories with all artifacts.
    """
    
    def __init__(self):
        self.storage_manager = None
    
    def create_complete_release(
        self, 
        experiment_results: ExperimentResults, 
        base_output_dir: Path
    ) -> Path:
        """
        Create a complete release package from experiment results.
        
        Args:
            experiment_results: Complete experiment results container
            base_output_dir: Base directory for release creation
            
        Returns:
            Path to created release directory
        """
        # Create timestamped release directory
        timestamp_str = experiment_results.timestamp.strftime("%Y%m%d_%H%M%S")
        release_name = f"release_{experiment_results.experiment_name}_{timestamp_str}"
        release_dir = base_output_dir / release_name
        
        logger.info(f"Creating comprehensive release: {release_name}")
        
        # Ensure release directory exists
        release_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage manager for this release
        self.storage_manager = HierarchicalStorageManager(release_dir)
        
        # Create release components
        self._create_bundle_json(experiment_results, release_dir)
        self._create_models_directory(experiment_results.models, release_dir)
        self._create_metrics_json(experiment_results.evaluation_results, release_dir)
        self._create_logs_directory(experiment_results.experiment_log, release_dir)
        self._create_readme(experiment_results, release_dir)
        
        logger.info(f"Release created successfully at: {release_dir}")
        return release_dir
    
    def _create_bundle_json(self, experiment_results: ExperimentResults, release_dir: Path):
        """Create bundle.json with merged configs and metadata."""
        logger.debug("Creating bundle.json")
        
        # Extract configurations
        data_config = experiment_results.configurations.get('data_config')
        training_config = experiment_results.configurations.get('training_config')
        
        # Build bundle data
        bundle_data = {
            # Experiment metadata
            'experiment_name': experiment_results.experiment_name,
            'timestamp': experiment_results.timestamp.isoformat(),
            'model_count': len(experiment_results.models),
            
            # Configurations (merged)
            'data_config': self._serialize_config(data_config),
            'training_config': self._serialize_config(training_config),
            
            # Model metadata
            'models': [self._serialize_model_metadata(model) for model in experiment_results.models],
            
            # Release metadata
            'release_version': '1.0',
            'framework_version': 'M5-Benchmarking-v1.0'
        }
        
        # Write bundle.json
        bundle_path = release_dir / "bundle.json"
        with open(bundle_path, 'w') as f:
            json.dump(bundle_data, f, indent=2, default=str)
        
        logger.debug(f"Bundle created: {bundle_path}")
    
    def _create_models_directory(self, models: list[BenchmarkModel], release_dir: Path):
        """Create models directory with all trained models."""
        logger.debug("Creating models directory")
        
        models_dir = release_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model in models:
            # Get appropriate release manager for this model type
            release_manager = ReleaseManagerFactory.get_manager_for_model(model)
            
            # Use existing model-specific persistence logic
            release_manager.save_model_object(models_dir, model)
        
        logger.debug(f"Models saved to: {models_dir}")
    
    def _create_metrics_json(self, evaluation_results: Optional[Dict[str, Any]], release_dir: Path):
        """Create optional metrics.json if evaluation results exist."""
        if evaluation_results is None:
            logger.debug("No evaluation results - skipping metrics.json")
            return
        
        logger.debug("Creating metrics.json")
        
        # Extract metrics from evaluation results
        metrics_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_metrics': {},
            'per_model_metrics': {},
            'strategy_metrics': {}
        }
        
        # Process evaluation results structure
        for key, value in evaluation_results.items():
            if key == 'overall':
                # Overall comparison metrics
                if isinstance(value, dict) and 'summary' in value:
                    metrics_data['overall_metrics'] = value['summary']
            elif key in ['combined', 'individual']:
                # Strategy-specific results
                if isinstance(value, dict):
                    metrics_data['strategy_metrics'][key] = value
            elif isinstance(value, dict) and 'performance_metrics' in value:
                # Individual model metrics
                metrics_data['per_model_metrics'][key] = value['performance_metrics']
        
        # Write metrics.json
        metrics_path = release_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.debug(f"Metrics created: {metrics_path}")
    
    def _create_logs_directory(self, experiment_log: Dict[str, Any], release_dir: Path):
        """Create logs directory with experiment log."""
        logger.debug("Creating logs directory")
        
        logs_dir = release_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Write experiment_log.json
        log_path = logs_dir / "experiment_log.json"
        with open(log_path, 'w') as f:
            json.dump(experiment_log, f, indent=2, default=str)
        
        logger.debug(f"Experiment log saved: {log_path}")
    
    def _create_readme(self, experiment_results: ExperimentResults, release_dir: Path):
        """Create auto-generated README.md with experiment summary."""
        logger.debug("Creating README.md")
        
        readme_content = self._generate_readme_content(experiment_results)
        
        readme_path = release_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.debug(f"README created: {readme_path}")
    
    def _generate_readme_content(self, experiment_results: ExperimentResults) -> str:
        """Generate README.md content."""
        models = experiment_results.models
        has_evaluation = experiment_results.evaluation_results is not None
        
        content = f"""# Experiment Release: {experiment_results.experiment_name}

**Release Date:** {experiment_results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Model Count:** {len(models)}
**Evaluation Included:** {'Yes' if has_evaluation else 'No'}

## Overview

This release contains a complete experiment package with all trained models, configurations, and associated metadata.

## Contents

- `bundle.json` - Complete experiment configuration and metadata
- `models/` - All trained model files
- `logs/` - Experiment logs and execution details
{f"- `metrics.json` - Evaluation results and performance metrics" if has_evaluation else ""}
- `README.md` - This summary file

## Models Included

"""
        
        # Add model information
        for i, model in enumerate(models, 1):
            strategy = model.metadata.modeling_strategy.value
            model_type = model.metadata.model_type
            sku_count = len(model.metadata.sku_tuples)
            
            content += f"{i}. **{model.get_identifier()}**\n"
            content += f"   - Strategy: {strategy}\n"
            content += f"   - Type: {model_type}\n"
            content += f"   - SKUs: {sku_count}\n"
            
            # Add performance metrics if available
            if model.metadata.performance_metrics:
                rmse = model.metadata.performance_metrics.get('rmse', 'N/A')
                mae = model.metadata.performance_metrics.get('mae', 'N/A')
                content += f"   - RMSE: {rmse}\n"
                content += f"   - MAE: {mae}\n"
            content += "\n"
        
        content += """## Usage

This release package contains everything needed to reproduce or deploy the experiment results. 
Load the models using the framework's model loading utilities and refer to the bundle.json 
for complete configuration details.

---
*Generated automatically by M5 Time Series Benchmarking Framework*
"""
        
        return content
    
    def _serialize_config(self, config) -> Dict[str, Any]:
        """Serialize configuration object to dictionary."""
        if config is None:
            return {}
        
        if hasattr(config, '__dict__'):
            # Handle dataclass objects
            result = {}
            for key, value in config.__dict__.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    result[key] = value
                else:
                    result[key] = str(value)
            return result
        elif isinstance(config, dict):
            return config
        else:
            return str(config)
    
    def _serialize_model_metadata(self, model: BenchmarkModel) -> Dict[str, Any]:
        """Serialize model metadata for bundle.json."""
        metadata = model.metadata
        
        return {
            'model_id': metadata.model_id,
            'model_type': metadata.model_type,
            'modeling_strategy': metadata.modeling_strategy.value,
            'sku_tuples': metadata.sku_tuples,
            'store_id': metadata.store_id,
            'product_id': metadata.product_id,
            'hyperparameters': metadata.hyperparameters,
            'performance_metrics': metadata.performance_metrics,
            'feature_columns': metadata.feature_columns,
            'target_column': metadata.target_column,
            'training_date_range': metadata.training_date_range,
            'model_instance': metadata.model_instance,
            'quantile_level': metadata.quantile_level
        }