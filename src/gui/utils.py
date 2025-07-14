import logging
from collections.abc import Mapping
from typing import Dict, Any, Union

# ✅ Structured logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def merge_nested_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge nested per-tool metric dictionaries into a single dot-notated flat dictionary.

    Examples:
        {"flake8": {"unused_imports": 2}} → {"flake8.unused_imports": 2}
        {"sqale_rating": 1.0} → {"sqale_rating": 1.0}

    Args:
        metrics_dict (Dict[str, Any]): Raw combined metrics from one file.

    Returns:
        Dict[str, Any]: Flattened dictionary with tool-prefixed keys.
    """
    merged = {}

    for section, subdict in metrics_dict.items():
        if isinstance(subdict, Mapping):
            for key, value in subdict.items():
                flat_key = f"{section}.{key}"
                merged[flat_key] = value
                logger.debug(f"[merge_nested_metrics] Merged: {flat_key} = {value}")
        else:
            merged[section] = subdict
            logger.debug(f"[merge_nested_metrics] Added top-level: {section} = {subdict}")

    return merged


def flatten_metrics(
    d: dict[str, Any],
    prefix: str = "",
    numeric_only: bool = False
) -> dict[str, Union[str, int, float]]:
    """
    Recursively flatten deeply nested dictionaries into a single-level dictionary.

    Supports:
        - {"flake8": {"unused_imports": 2}} → {"flake8.unused_imports": 2}
        - {"foo.bar": {"x": 1}} → {"foo.bar.x": 1}

    Args:
        d (dict): Input nested dictionary to flatten.
        prefix (str): Prefix for recursive keys.
        numeric_only (bool): If True, excludes non-numeric values.

    Returns:
        dict: Flattened dictionary with dot-separated keys.
    """
    if not isinstance(d, dict):
        raise TypeError("Expected a dictionary for flatten_metrics input")

    flat: dict[str, Union[str, int, float]] = {}

    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        logger.debug(f"[flatten_metrics] Visiting: {full_key} (type: {type(v).__name__})")

        if isinstance(v, dict):
            flat.update(flatten_metrics(v, prefix=full_key, numeric_only=numeric_only))
        elif isinstance(v, (int, float)):
            flat[full_key] = v
            logger.debug(f"[flatten_metrics] Added numeric: {full_key} = {v}")
        elif not numeric_only:
            flat[full_key] = v
            logger.debug(f"[flatten_metrics] Added non-numeric: {full_key} = {v}")
        else:
            logger.debug(f"[flatten_metrics] Skipped non-numeric: {full_key} (type: {type(v).__name__})")

    return flat
