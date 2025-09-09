"""
Release Management Module for M5 Benchmarking Framework.

This module provides a release management system that completely replaces
the ModelRegistry.save_model() functionality with a plugin architecture
based on BaseReleaseManager interface.

Key Components:
- BaseReleaseManager: Abstract interface for model-specific release managers
- ReleaseManagerFactory: Factory for selecting appropriate release managers
- Model-specific managers: XGBoostStandardManager, XGBoostQuantileManager, etc.
"""

from .base import BaseReleaseManager
from .factory import ReleaseManagerFactory

# Import managers to trigger registration
from . import managers

__all__ = [
    "BaseReleaseManager",
    "ReleaseManagerFactory",
]