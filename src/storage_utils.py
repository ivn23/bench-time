"""
Storage Utilities for Hierarchical Model Management

This module provides utilities for managing hierarchical model storage paths
organized by store_id/product_id/model_type structure.
"""

import re
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from .data_structures import SkuTuple


@dataclass
class ModelStorageLocation:
    """Represents a hierarchical storage location for a model."""
    store_id: int
    product_id: int
    model_type: str
    model_instance: str = "default"
    quantile_level: Optional[float] = None  # New quantile level support
    
    def __post_init__(self):
        """Validate quantile level after initialization."""
        if self.quantile_level is not None and not (0 < self.quantile_level < 1):
            raise ValueError(f"quantile_level must be between 0 and 1, got {self.quantile_level}")
    
    def to_path_components(self) -> Tuple[str, str, str, str, str]:
        """Convert to path components (store_id, product_id, model_type, quantile_level, instance)."""
        quantile_component = self._format_quantile_component()
        return (
            self._sanitize_path_component(str(self.store_id)),
            self._sanitize_path_component(str(self.product_id)),
            self._sanitize_path_component(self.model_type),
            self._sanitize_path_component(quantile_component),
            self._sanitize_path_component(self.model_instance)
        )
    
    def _format_quantile_component(self) -> str:
        """Format quantile level for use in directory names."""
        if self.quantile_level is not None:
            return f"q{self.quantile_level:.3f}".rstrip('0').rstrip('.')
        return "standard"  # Default for non-quantile models
    
    @staticmethod
    def _sanitize_path_component(component: str) -> str:
        """Sanitize a path component to be filesystem-safe."""
        # Remove or replace invalid characters for cross-platform compatibility
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', component)
        # Ensure it's not empty or just dots
        if not sanitized or sanitized.strip('.'):
            sanitized = sanitized or "default"
        return sanitized[:255]  # Limit length
    
    @classmethod
    def from_sku_tuple(cls, sku_tuple: SkuTuple, model_type: str, 
                      model_instance: str = "default") -> 'ModelStorageLocation':
        """Create storage location from SKU tuple (backward compatibility)."""
        product_id, store_id = sku_tuple
        return cls(
            store_id=store_id,
            product_id=product_id,
            model_type=model_type,
            model_instance=model_instance
        )
    
    @classmethod
    def from_sku_tuple_with_quantile(cls, sku_tuple: SkuTuple, model_type: str,
                                   quantile_level: Optional[float] = None,
                                   model_instance: str = "default") -> 'ModelStorageLocation':
        """Create storage location from SKU tuple with quantile level."""
        product_id, store_id = sku_tuple
        return cls(
            store_id=store_id,
            product_id=product_id,
            model_type=model_type,
            model_instance=model_instance,
            quantile_level=quantile_level
        )


class HierarchicalStorageManager:
    """Manages hierarchical model storage paths and operations."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
    
    def create_model_path(self, location: ModelStorageLocation) -> Path:
        """Generate full hierarchical path for a model storage location."""
        store_dir, product_dir, model_type_dir, quantile_dir, instance_dir = location.to_path_components()
        
        return self.models_path / store_dir / product_dir / model_type_dir / quantile_dir / instance_dir
    
    def ensure_model_directory(self, location: ModelStorageLocation) -> Path:
        """Create directory hierarchy for model storage and return the path."""
        model_path = self.create_model_path(location)
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path
    
    def parse_model_path(self, model_path: Path) -> Optional[ModelStorageLocation]:
        """Extract storage location information from a model path."""
        try:
            # Expect path like: base/models/store_id/product_id/model_type/quantile_level/instance
            relative_path = model_path.relative_to(self.models_path)
            parts = relative_path.parts
            
            if len(parts) < 5:
                return None
                
            store_id = int(parts[0])
            product_id = int(parts[1])
            model_type = parts[2]
            quantile_str = parts[3]
            model_instance = parts[4] if len(parts) > 4 else "default"
            
            # Parse quantile level
            quantile_level = None
            if quantile_str.startswith('q') and quantile_str != "qstandard":
                try:
                    quantile_level = float(quantile_str[1:])
                except ValueError:
                    pass
            
            return ModelStorageLocation(
                store_id=store_id,
                product_id=product_id,
                model_type=model_type,
                model_instance=model_instance,
                quantile_level=quantile_level
            )
        except (ValueError, IndexError):
            return None
    
    def list_stores(self) -> List[int]:
        """List all store IDs that have models."""
        stores = []
        if self.models_path.exists():
            for store_path in self.models_path.iterdir():
                if store_path.is_dir():
                    try:
                        stores.append(int(store_path.name))
                    except ValueError:
                        continue
        return sorted(stores)
    
    def list_products_for_store(self, store_id: int) -> List[int]:
        """List all product IDs for a specific store."""
        products = []
        store_path = self.models_path / str(store_id)
        if store_path.exists():
            for product_path in store_path.iterdir():
                if product_path.is_dir():
                    try:
                        products.append(int(product_path.name))
                    except ValueError:
                        continue
        return sorted(products)
    
    def list_model_types_for_sku(self, store_id: int, product_id: int) -> List[str]:
        """List all model types available for a specific SKU."""
        model_types = []
        sku_path = self.models_path / str(store_id) / str(product_id)
        if sku_path.exists():
            for model_type_path in sku_path.iterdir():
                if model_type_path.is_dir():
                    model_types.append(model_type_path.name)
        return sorted(model_types)
    
    def list_model_instances(self, store_id: int, product_id: int, 
                           model_type: str) -> List[str]:
        """List all model instances for a specific SKU and model type."""
        instances = []
        type_path = self.models_path / str(store_id) / str(product_id) / model_type
        if type_path.exists():
            for instance_path in type_path.iterdir():
                if instance_path.is_dir():
                    instances.append(instance_path.name)
        return sorted(instances)
    
    def find_models(self, store_id: Optional[int] = None,
                   product_id: Optional[int] = None,
                   model_type: Optional[str] = None) -> List[ModelStorageLocation]:
        """Find models matching the specified criteria."""
        locations = []
        
        # Start from appropriate level based on filters
        if store_id is not None:
            stores_to_search = [store_id]
        else:
            stores_to_search = self.list_stores()
        
        for sid in stores_to_search:
            if product_id is not None:
                products_to_search = [product_id] if product_id in self.list_products_for_store(sid) else []
            else:
                products_to_search = self.list_products_for_store(sid)
            
            for pid in products_to_search:
                if model_type is not None:
                    types_to_search = [model_type] if model_type in self.list_model_types_for_sku(sid, pid) else []
                else:
                    types_to_search = self.list_model_types_for_sku(sid, pid)
                
                for mtype in types_to_search:
                    instances = self.list_model_instances(sid, pid, mtype)
                    for instance in instances:
                        locations.append(ModelStorageLocation(
                            store_id=sid,
                            product_id=pid,
                            model_type=mtype,
                            model_instance=instance
                        ))
        
        return locations
    
    def get_model_directory_size(self, location: ModelStorageLocation) -> int:
        """Get the total size of files in a model directory in bytes."""
        model_path = self.create_model_path(location)
        if not model_path.exists():
            return 0
        
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def cleanup_empty_directories(self):
        """Remove empty directories in the hierarchy."""
        if not self.models_path.exists():
            return
        
        # Remove empty directories bottom-up
        for store_path in self.models_path.iterdir():
            if not store_path.is_dir():
                continue
                
            for product_path in store_path.iterdir():
                if not product_path.is_dir():
                    continue
                    
                for model_type_path in product_path.iterdir():
                    if not model_type_path.is_dir():
                        continue
                    
                    # Remove empty instance directories
                    for instance_path in model_type_path.iterdir():
                        if instance_path.is_dir() and not any(instance_path.iterdir()):
                            instance_path.rmdir()
                    
                    # Remove empty model type directories
                    if not any(model_type_path.iterdir()):
                        model_type_path.rmdir()
                
                # Remove empty product directories
                if not any(product_path.iterdir()):
                    product_path.rmdir()
            
            # Remove empty store directories
            if not any(store_path.iterdir()):
                store_path.rmdir()