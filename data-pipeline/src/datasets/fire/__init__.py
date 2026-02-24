"""
Boston Pulse - Fire Dataset Package

This module exposes the Fire dataset components so they can be imported as:

    from src.datasets.fire import FireIngester
    from src.datasets.fire import FirePreprocessor
    from src.datasets.fire import FireFeatureBuilder
"""

from .ingest import FireIngester
from .preprocess import FirePreprocessor
from .features import FireFeatureBuilder

__all__ = [
    "FireIngester",
    "FirePreprocessor",
    "FireFeatureBuilder",
]