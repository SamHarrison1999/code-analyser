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

    Subclasses must implement the extract() method, returning
    a dictionary of metric name to scalar values.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def extract(self) -> Dict[str, Union[int, float]]:
        """
        Perform metric extraction.

        Returns:
            dict[str, int | float]: Extracted metrics.
        """
        ...


class SonarStyleExtractorBase(ABC):
    """
    Specialised extractor interface for Sonar-style tools
    that consume a raw context and a file path.
    """

    def __init__(self, file_path: str, context: dict):
        self.file_path = file_path
        self.context = context

    @abstractmethod
    def extract(self) -> Dict[str, float]:
        """
        Extract metrics from a raw data context.

        Returns:
            dict[str, float]: Metric values from the Sonar context.
        """
        ...


# Optional plugin metadata type for registry, ML, or API export
MetricPlugin = Dict[str, object]
"""
MetricPlugin = TypedDict("MetricPlugin", {
    "name": str,
    "type": Literal["static_analysis", "complexity", "security", "coverage"],
    "extractor": MetricExtractor,
    "domain": Literal["code", "security", "coverage"],
    "language": str,
    "source": str,
    "version": str,
    "format": Literal["metrics"],
    "tool": str,
    "scope": Literal["file", "project"],
    "outputs": list[str],
})
"""
