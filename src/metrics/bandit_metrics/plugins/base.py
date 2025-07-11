import abc
from typing import Dict


class BanditMetricPlugin(abc.ABC):
    """
    Abstract base class for Bandit metric plugins.

    Each Bandit plugin must implement:
    - name(): Return a unique string identifier for the metric
    - extract(data): Compute the metric from Bandit's parsed JSON output
    """

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
