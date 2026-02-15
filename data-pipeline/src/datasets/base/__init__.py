"""
Boston Pulse - Base Classes for Datasets

Abstract base classes that all dataset implementations inherit from.
These provide a consistent interface for:
- Data ingestion (BaseIngester)
- Data preprocessing (BasePreprocessor)
- Feature building (BaseFeatureBuilder)

Usage:
    from src.datasets.base import BaseIngester, BasePreprocessor, BaseFeatureBuilder

    class CrimeIngester(BaseIngester):
        def fetch_data(self, since: Optional[datetime]) -> pd.DataFrame:
            ...
"""

# Base classes will be implemented in Phase 2
# from src.datasets.base.ingester import BaseIngester
# from src.datasets.base.preprocessor import BasePreprocessor
# from src.datasets.base.feature_builder import BaseFeatureBuilder

# __all__ = ["BaseIngester", "BasePreprocessor", "BaseFeatureBuilder"]
