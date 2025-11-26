# src/graph/__init__.py
"""IPAG Graph construction and feature extraction modules."""

from .ipag_builder import IPAGBuilder
from .features import BuildNodeFeatures

try:
    from .features_accelerated import BuildNodeFeaturesAccelerated
    ACCELERATED_AVAILABLE = True
except ImportError:
    ACCELERATED_AVAILABLE = False
    BuildNodeFeaturesAccelerated = None

__all__ = [
    'IPAGBuilder',
    'BuildNodeFeatures',
    'BuildNodeFeaturesAccelerated',
    'ACCELERATED_AVAILABLE'
]