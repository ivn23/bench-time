"""
Model-specific Release Managers.

This module contains implementations of BaseReleaseManager for specific
model types in the benchmarking framework.
"""

from .xgboost_standard_manager import XGBoostStandardReleaseManager
from .xgboost_quantile_manager import XGBoostQuantileReleaseManager
from .lightning_standard_manager import LightningStandardReleaseManager
from .lightning_quantile_manager import LightningQuantileReleaseManager
from .statquant_manager import StatQuantReleaseManager

# Import factory to register managers
from ..factory import ReleaseManagerFactory

# Register all release managers
ReleaseManagerFactory.register_manager("xgboost_standard", XGBoostStandardReleaseManager)
ReleaseManagerFactory.register_manager("xgboost_quantile", XGBoostQuantileReleaseManager)
ReleaseManagerFactory.register_manager("lightning_standard", LightningStandardReleaseManager)
ReleaseManagerFactory.register_manager("lightning_quantile", LightningQuantileReleaseManager)
ReleaseManagerFactory.register_manager("statquant", StatQuantReleaseManager)

__all__ = [
    "XGBoostStandardReleaseManager",
    "XGBoostQuantileReleaseManager",
    "LightningStandardReleaseManager",
    "LightningQuantileReleaseManager",
    "StatQuantReleaseManager",
]