from pathlib import Path
from typing import Callable

from .gather import gather


def get_lizard_extractor() -> Callable[[Path], list[dict[str, object]]]:
    """
    Returns a callable that extracts Lizard metrics for a given file path.

    This function conforms to the expected extractor interface used in the overall metrics framework.

    Returns:
        Callable[[Path], list[dict[str, object]]]: Function that takes a Path and returns Lizard metrics.
    """
    return gather
