from pathlib import Path
from typing import Callable, Literal, TypedDict


class MetricResult(TypedDict):
    name: str
    value: object
    units: str | None
    success: bool
    error: str | None


# A MetricExtractor is a callable that takes a Path and returns a list of metric result dicts
MetricExtractor = Callable[[Path], list[MetricResult]]


# Core definition for any registered plugin
MetricPlugin = dict[str, object]  # should match the following structure:
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
