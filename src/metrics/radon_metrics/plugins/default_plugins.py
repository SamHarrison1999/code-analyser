"""
Default Radon metric plugins.

Each plugin extracts one metric from Radon JSON output.
"""

from metrics.radon_metrics.plugins.base import RadonMetricPlugin


class LogicalLinesPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "number_of_logical_lines"

    def extract(self, radon_data: dict) -> int:
        return int(radon_data.get("number_of_logical_lines", 0))


class BlankLinesPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "number_of_blank_lines"

    def extract(self, radon_data: dict) -> int:
        return int(radon_data.get("number_of_blank_lines", 0))


class DocstringLinesPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "number_of_doc_strings"

    def extract(self, radon_data: dict) -> int:
        return int(radon_data.get("number_of_doc_strings", 0))


class AverageHalsteadVolumePlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "average_halstead_volume"

    def extract(self, radon_data: dict) -> float:
        return float(radon_data.get("average_halstead_volume", 0.0))


class AverageHalsteadDifficultyPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "average_halstead_difficulty"

    def extract(self, radon_data: dict) -> float:
        return float(radon_data.get("average_halstead_difficulty", 0.0))


class AverageHalsteadEffortPlugin(RadonMetricPlugin):
    def name(self) -> str:
        return "average_halstead_effort"

    def extract(self, radon_data: dict) -> float:
        return float(radon_data.get("average_halstead_effort", 0.0))


DEFAULT_PLUGINS = [
    LogicalLinesPlugin,
    BlankLinesPlugin,
    DocstringLinesPlugin,
    AverageHalsteadVolumePlugin,
    AverageHalsteadDifficultyPlugin,
    AverageHalsteadEffortPlugin,
]


def load_plugins() -> list[RadonMetricPlugin]:
    """
    Instantiate and return all default Radon metric plugins.

    Returns:
        list[RadonMetricPlugin]: List of plugin instances.
    """
    return [plugin() for plugin in DEFAULT_PLUGINS]
