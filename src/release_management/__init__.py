"""
Release Management Module for M5 Benchmarking Framework.

This module provides a clean release management system where models handle
their own persistence through the save_model() method.

Key Components:
- ReleaseManager: Orchestrates complete experiment releases
"""

from .release_manager import ReleaseManager

__all__ = [
    "ReleaseManager",
]