"""
Base class for Lizard metric plugins.

Each plugin inspects preprocessed Lizard metric entries and extracts one metric.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict


class LizardMetricPlugin(ABC):
    """
    Abstract base class for Lizard metric plugins.

    Plugins operate on parsed Lizard metric data (typically a list of dictionaries)
    and extract one scalar metric each (int or float) from that structure.

    All subclasses must implement:
    - name(): a unique string identifier for the metric
    - extract(): a computation from parsed Lizard metric data
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the metric.

        Returns:
            str: A globally unique name used in metric output dictionaries.
        """
        # ✅ Best Practice: Unique metric keys ensure safe integration into CSV/ML/GUI pipelines
        raise NotImplementedError("LizardMetricPlugin must implement the name() method.")

    @abstractmethod
    def extract(self, lizard_metrics: List[Dict[str, Any]], file_path: str) -> Any:
        """
        Compute and return a scalar metric from parsed Lizard data.

        Args:
            lizard_metrics (List[Dict[str, Any]]): Parsed entries for a single file from Lizard.
            file_path (str): Full path to the Python file being analysed.

        Returns:
            Any: The computed metric (typically an int or float).
        """
        # ⚠️ SAST Risk: Plugins must validate numerical types to avoid type errors during aggregation
        # 🧠 ML Signal: Plugins standardise feature vector structure across files
        raise NotImplementedError("LizardMetricPlugin must implement the extract() method.")
