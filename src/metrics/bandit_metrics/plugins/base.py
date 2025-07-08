# File: src/metrics/bandit_metrics/plugins/base.py

import abc
from typing import Protocol


class BanditMetricPlugin(abc.ABC):
    """
    Abstract base class for Bandit metric plugins.

    All Bandit plugins must:
    - Provide a unique metric name via `name()`
    - Implement `extract(data)` to compute the metric from Bandit's JSON
    """

    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: The unique name of the metric (used as dictionary key and CSV column).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, data: dict) -> int:
        """
        Compute the metric value from Bandit's parsed JSON results.

        Args:
            data (dict): Parsed Bandit output (from -f json)

        Returns:
            int: The value for this specific metric.
        """
        raise NotImplementedError
