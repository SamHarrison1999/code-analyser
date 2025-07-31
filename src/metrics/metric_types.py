"""
metric_types.py — Unified typing for static + AI-enhanced metric extractors.

Includes:
- Plugin base classes (MetricExtractorBase, SonarStyleExtractorBase)
- Standardised output bundles (MetricResult, AIAnnotationOverlay)
- AI-specific logging for token logits and RL reward shaping
"""

from pathlib import Path
from typing import Callable, Literal, TypedDict, Union, Dict, List, Tuple, Any
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

    Subclasses must implement extract() and may optionally
    implement confidence and metadata-aware variants.
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
        raise NotImplementedError("Subclasses must implement extract()")

    def extract_with_confidence(self) -> Dict[str, Tuple[Union[int, float], float]]:
        """
        Optional: Return both metric value and confidence for each key.

        Returns:
            Dict[str, Tuple[Union[int, float], float]]: Each metric mapped to (value, confidence).
        """
        raise NotImplementedError("Subclasses may implement extract_with_confidence()")

    def extract_with_metadata(self) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """
        Optional: Return extended metadata per metric for GUI/ML export.

        Example:
            {
                "cyclomatic_complexity": {
                    "value": 7.0,
                    "confidence": 0.88,
                    "severity": "medium",
                    "source": "radon"
                }
            }

        Returns:
            Dict[str, Dict[str, Union[str, float, int]]]: Rich metadata per metric.
        """
        raise NotImplementedError("Subclasses may implement extract_with_metadata()")


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
        raise NotImplementedError("Subclasses must implement extract()")

    def extract_with_confidence(self) -> Dict[str, Tuple[Union[int, float], float]]:
        """
        Optional: Return both metric value and confidence for each key.

        Returns:
            Dict[str, Tuple[Union[int, float], float]]: Each metric mapped to (value, confidence).
        """
        raise NotImplementedError("Subclasses may implement extract_with_confidence()")

    def extract_with_metadata(self) -> Dict[str, Dict[str, Union[str, float, int]]]:
        """
        Optional: Return extended metadata per metric for GUI/ML export.

        Example:
            {
                "coverage": {
                    "value": 100.0,
                    "confidence": 1,
                    "severity": "low",
                    "source": "sonar"
                }
            }

        Returns:
            Dict[str, Dict[str, Union[str, float, int]]]: Rich metadata per metric.
        """
        raise NotImplementedError("Subclasses may implement extract_with_metadata()")


# Optional plugin registry metadata
MetricPlugin = Dict[str, object]


# ✅ For ML pipelines and dataset export
class TokenExplanationMap(TypedDict):
    line: int
    token: str
    confidence: float
    severity: str


class HuggingFaceDatasetSample(TypedDict):
    file_path: str
    content: str
    labels: List[str]
    annotations: List[TokenExplanationMap]


class ONNXExportBundle(TypedDict):
    model_path: str
    tokenizer_path: str
    input_example: Any
    output_example: Any


class AIAnnotationOverlay(TypedDict, total=False):
    line: int
    token_span: Tuple[int, int]
    label: str
    confidence: float
    severity: Literal["low", "medium", "high"]
    scope: str
    explanation: str


class AISummaryBundle(TypedDict):
    file: str
    avg_confidence: float
    high_risk_count: int
    total_annotations: int
    timestamp: str


class TokenLogits(TypedDict):
    line: int
    token: str
    index: int
    logits: List[float]
    predicted_label: str
    confidence: float


class RLRewardLog(TypedDict):
    file: str
    episode: int
    reward: float
    timestamp: str
    model_version: str
    source: Literal["actor", "critic"]
    explanation: str
