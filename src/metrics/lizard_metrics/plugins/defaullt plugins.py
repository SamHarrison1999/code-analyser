"""
Default Lizard metric plugins.

Each plugin extracts one metric from the pre-parsed Lizard output.
"""

from metrics.lizard_metrics.plugins.base import LizardPlugin


class AverageFunctionLengthPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "average_function_length"

    def extract(self, lizard_metrics: list[dict], file_path: str) -> float:
        """
        Calculate the average function length across all entries.

        Args:
            lizard_metrics (list[dict]): Lizard metric entries.
            file_path (str): Path to the analysed file.

        Returns:
            float: Average function length.
        """
        lengths = [m["value"] for m in lizard_metrics if m["name"] == "average_function_length"]
        return round(sum(lengths) / len(lengths), 2) if lengths else 0.0


class AverageCyclomaticComplexityPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "average_function_complexity"

    def extract(self, lizard_metrics: list[dict], file_path: str) -> float:
        """
        Calculate the average cyclomatic complexity.

        Args:
            lizard_metrics (list[dict]): Lizard metric entries.
            file_path (str): Path to the analysed file.

        Returns:
            float: Average cyclomatic complexity.
        """
        values = [m["value"] for m in lizard_metrics if m["name"] == "average_function_complexity"]
        return round(sum(values) / len(values), 2) if values else 0.0


class MaintainabilityIndexPlugin(LizardPlugin):
    @classmethod
    def name(cls) -> str:
        return "maintainability_index"

    def extract(self, lizard_metrics: list[dict], file_path: str) -> float:
        """
        Extract the maintainability index.

        Args:
            lizard_metrics (list[dict]): Lizard metric entries.
            file_path (str): Path to the analysed file.

        Returns:
            float: Maintainability index if found, else 0.0
        """
        for m in lizard_metrics:
            if m["name"] == "maintainability_index":
                return float(m["value"])
        return 0.0


def load_plugins() -> list[LizardPlugin]:
    """
    Loads all default Lizard metric plugins.

    Returns:
        list[LizardPlugin]: List of plugin instances.
    """
    return [
        AverageFunctionLengthPlugin(),
        AverageCyclomaticComplexityPlugin(),
        MaintainabilityIndexPlugin(),
    ]


# ✅ Exported list of plugin instances for unified access
plugins = load_plugins()
