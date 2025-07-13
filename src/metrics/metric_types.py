from pathlib import Path
from typing import Callable, Literal, TypedDict, Union, Dict, List, Protocol
from abc import ABC, abstractmethod


class MetricResult(TypedDict):
    """
    Represents a single metric extraction result.

    Fields:
        name (str): Name of the metric.
        value (object): Extracted value (int, float, str, etc.).
        units (str | None): Units of the value (e.g. 'lines', '%', 'errors').
        success (bool): Whether the metric was successfully extracted.
        error (str | None): Error message if extraction failed.
    """
    name: str
    value: object
    units: Union[str, None]
    success: bool
    error: Union[str, None]


# Callable signature for simple gatherer-style metric extractors
MetricExtractor = Callable[[Path], List[MetricResult]]


class MetricExtractorBase(ABC):
    """
    Abstract base class for class-based metric extractors.

    Implementations must define an `extract()` method returning
    a dictionary of metric name → scalar value (int or float).
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Perform metric extraction.

        Returns:
            dict[str, int | float]: Dictionary of extracted metrics.
        """
        pass


# Optional plugin metadata type for registry, ML, or API export
MetricPlugin = Dict[str, object]
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
