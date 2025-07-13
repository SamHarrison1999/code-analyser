import logging
from typing import Union
from vulture import Vulture
from metrics.metric_types import MetricExtractorBase


class VultureExtractor(MetricExtractorBase):
    """
    Extracts unused code metrics using the Vulture static analysis tool.
    """

    def extract(self) -> dict[str, Union[int, float]]:
        """
        Analyses the file with Vulture and returns unused code metrics.

        Returns:
            dict[str, int]: Metrics including:
                - unused_functions
                - unused_classes
                - unused_variables
                - unused_imports
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                code = f.read()

            v = Vulture()
            v.scan(code)
            unused_items = v.get_unused_code()

            result = {
                "unused_functions": sum(1 for item in unused_items if item.typ == "function"),
                "unused_classes": sum(1 for item in unused_items if item.typ == "class"),
                "unused_variables": sum(1 for item in unused_items if item.typ == "variable"),
                "unused_imports": sum(1 for item in unused_items if item.typ == "import"),
            }

            logging.info(f"[VultureExtractor] Metrics for {self.file_path}:\n{result}")
            return result

        except Exception as e:
            logging.error(f"[VultureExtractor] Failed to analyse {self.file_path}: {type(e).__name__}: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> dict[str, int]:
        """
        Returns default zero values for all metrics on failure.

        Returns:
            dict[str, int]: All-zero fallback metrics.
        """
        return {
            "unused_functions": 0,
            "unused_classes": 0,
            "unused_variables": 0,
            "unused_imports": 0,
        }

    def extract_items(self) -> list:
        """
        Returns raw Vulture unused item list for plugin-based analysis.

        Returns:
            list: List of Vulture unused code items.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                code = f.read()

            v = Vulture()
            v.scan(code)
            return v.get_unused_code()

        except Exception as e:
            logging.error(f"[VultureExtractor] Failed to extract items: {type(e).__name__}: {e}")
            return []


def run_vulture(file_path: str) -> dict[str, int]:
    """
    Entry point to extract standard Vulture metrics via standalone call.

    Args:
        file_path (str): Path to Python file.

    Returns:
        dict[str, int]: Vulture metrics using default logic.
    """
    return VultureExtractor(file_path).extract()


__all__ = ["VultureExtractor", "run_vulture"]
