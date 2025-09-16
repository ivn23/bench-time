"""
Release Management Module for M5 Benchmarking Framework.

This module provides a clean release management system where models handle
their own persistence through the save_model() method.

Key Components:
- ComprehensiveReleaseManager: Orchestrates complete experiment releases
"""

from .comprehensive_manager import ComprehensiveReleaseManager

__all__ = [
    "ComprehensiveReleaseManager",
]