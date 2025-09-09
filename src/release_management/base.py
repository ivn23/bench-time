"""
Abstract base classes for the benchmarking release management system.

This module defines the core interfaces that all model-specific release managers
must provide for release creation functionality in the M5 benchmarking framework.
"""

import pickle
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

from .utils import save_json, ensure_directory

if t.TYPE_CHECKING:
    from ..data_structures import BenchmarkModel


# init_functions for subclasses schreiben
# validate model parameter schreiben (methode)=



class BaseReleaseManager(ABC):
    """
    Abstract base class for model-specific release managers.

    Each model type (xgboost_standard, xgboost_quantile, etc.) must implement this
    interface to integrate with the benchmarking release management system.
    """

    @property
    @abstractmethod
    def family_name(self) -> str:
        """Return the unique identifier for this model family."""
        ...

    @abstractmethod
    def create_bundle_metadata(self, benchmark_model: "BenchmarkModel") -> dict:
        """
        Create bundle metadata from BenchmarkModel.

        Args:
            benchmark_model: BenchmarkModel to extract metadata from

        Returns:
            Dictionary containing all model-relevant details for bundle.json
        """
        ...

    def export_release(
        self,
        version: str,
        benchmark_model: "BenchmarkModel",
        output_dir: Path,
    ) -> Path:
        """
        Export release with standardized structure for benchmarking framework.

        Creates the release directory structure:
        release_<version>/
        ├── bundle.json (model parameters, name, and all relevant details)
        ├── metrics.json (optional model performance metrics)
        ├── data_splits.json (train/validation split information)
        └── models/
            └── model_<storeID_productID>.pkl (pickled model object)

        Args:
            version: Version identifier (from BenchmarkPipeline output_dir)
            benchmark_model: BenchmarkModel object to create release from
            output_dir: Base directory for release

        Returns:
            Path to created release directory
        """
        # Create release directory
        release_dir = output_dir / f"release_{version}"
        ensure_directory(release_dir)
        
        # Create bundle metadata
        bundle_metadata = self.create_bundle_metadata(benchmark_model)
        
        # Save bundle.json
        bundle_file = release_dir / "bundle.json"
        save_json(bundle_metadata, bundle_file)
        
        # Save metrics.json if performance metrics exist
        if benchmark_model.metadata.performance_metrics:
            metrics_file = release_dir / "metrics.json"
            save_json(benchmark_model.metadata.performance_metrics, metrics_file)
        
        # Save data_splits.json
        data_splits = {
            "train_bdIDs": benchmark_model.data_split.train_bdIDs.tolist(),
            "validation_bdIDs": benchmark_model.data_split.validation_bdIDs.tolist(),
            "test_bdIDs": (
                benchmark_model.data_split.test_bdIDs.tolist() 
                if benchmark_model.data_split.test_bdIDs is not None 
                else None
            ),
            "split_date": benchmark_model.data_split.split_date
        }
        splits_file = release_dir / "data_splits.json"
        save_json(data_splits, splits_file)
        
        # Create models directory
        models_dir = release_dir / "models"
        ensure_directory(models_dir)
        
        # Save model object
        self.save_model_object(models_dir, benchmark_model)
        
        return release_dir
    
    def save_model_object(self, models_dir: Path, benchmark_model: "BenchmarkModel") -> None:
        """
        Save the actual model object to the models directory.

        Args:
            models_dir: Directory to save model files to
            benchmark_model: BenchmarkModel containing the model to save
        """
        # Create model filename based on SKU information
        if benchmark_model.metadata.sku_tuples:
            # For individual models, use first SKU tuple
            sku = benchmark_model.metadata.sku_tuples[0]
            model_filename = f"model_{sku[1]}_{sku[0]}.pkl"  # store_id, product_id
        else:
            # Fallback for combined models or models without SKU info
            model_filename = "model.pkl"
        
        # Save model using pickle serialization
        model_path = models_dir / model_filename
        with open(model_path, "wb") as f:
            pickle.dump(benchmark_model.model, f)
    
    def check_bundle(self, bundle: dict) -> None:
        """
        Validate bundle dictionary structure.

        Ensures required fields are present in the bundle metadata.

        Args:
            bundle: Bundle metadata dictionary

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["model_id", "model_type", "modeling_strategy"]
        for field in required_fields:
            if field not in bundle:
                raise ValueError(f"Bundle missing required field: {field}")