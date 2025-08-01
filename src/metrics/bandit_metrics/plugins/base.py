# File: code_analyser/src/metrics/bandit_metrics/plugins/base.py

import abc
from typing import Dict


class BanditMetricPlugin(abc.ABC):
    """
    Abstract base class for Bandit metric plugins.

    Each Bandit plugin must implement:
    - name(): Return a unique string identifier for the metric
    - extract(data): Compute the metric from Bandit's parsed JSON output

    Optional:
    - plugin_name: Canonical plugin ID for registration
    - plugin_tags: Tags for filtering and UI grouping
    - confidence_score(): Signal strength of the computed metric
    - severity_level(): Optional severity classification
    """

    # ✅ Optional metadata for dynamic discovery
    plugin_name: str = ""
    plugin_tags: list[str] = []

    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the unique name for this Bandit metric.

        Returns:
            str: Unique metric name (used as dictionary key and CSV column).
        """
        raise NotImplementedError("Bandit plugin must implement the name() method.")

    @abc.abstractmethod
    def extract(self, data: Dict) -> int:
        """
        Compute the metric value from Bandit's JSON analysis results.

        Args:
            data (dict): Parsed Bandit output using `-f json`.

        Returns:
            int: Computed value for this metric.
        """
        raise NotImplementedError("Bandit plugin must implement the extract() method.")

    def confidence_score(self, data: Dict) -> float:
        """
        Optional: Estimate confidence score (0.0–1.0) in the extracted metric.

        Returns:
            float: Confidence score (default = 1.0)
        """
        return 1.0

    def severity_level(self, data: Dict) -> str:
        """
        Optional: Return severity level ('low', 'medium', 'high') based on extracted metric.

        Returns:
            str: Severity category (default = 'low')
        """
        return "low"
