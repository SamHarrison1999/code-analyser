from typing import Literal

from ...metric_types import MetricPlugin
from ..extractor import get_lizard_extractor

LizardPlugin: MetricPlugin = {
    "name": "lizard",
    "type": "static_analysis",
    "extractor": get_lizard_extractor(),
    "domain": "code",
    "language": "python",
    "source": "lizard",
    "version": "1.17.10",  # replace with your actual installed version if dynamic versioning isn't used
    "format": "metrics",
    "tool": "lizard",
    "scope": "file",
    "outputs": [
        "average_function_complexity",
        "average_token_count",
        "average_parameter_count",
        "average_function_length",
        "number_of_functions",
        "maintainability_index",
    ],
}
