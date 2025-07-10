# File: src/metrics/metric_types.py

from pathlib import Path
from typing import Callable, Literal, TypedDict, Union, Dict, List
from abc import ABC, abstractmethod


class MetricResult(TypedDict):
    name: str
    value: object
    units: str | None
    success: bool
    error: str | None


# Function-style metric extractor (used for plugin systems)
MetricExtractor = Callable[[Path], List[MetricResult]]


# Base class for OO-style metric extractors (e.g. PydocstyleExtractor, BanditExtractor)
class MetricExtractorBase(ABC):
    """
    Abstract base class for class-based metric extractors.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Extract metrics for the file.

        Returns:
            dict[str, int | float]: Metric name-value pairs.
        """
        pass


# Optional: Plugin schema for ML registration / cataloguing
MetricPlugin = dict[str, object]
"""
MetricPlugin = TypedDict("MetricPlugin", {
    "name": str,
    "type": Literal["static_analysis", "complexity", "security"],
    "extractor": MetricExtractor,
    "domain": Literal["code", "security"],
    "language": str,
    "source": str,
    "version": str,
    "format": Literal["metrics"],
    "tool": str,
    "scope": Literal["file", "project"],
    "outputs": list[str],
})
"""
