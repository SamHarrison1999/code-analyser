
import logging
from collections.abc import Mapping
from typing import Dict, Any

# ✅ Best Practice: Configure logging for debug visibility during development
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def merge_nested_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested per-tool metric dictionaries into a single flat dictionary.

    This ensures that tool outputs like {'flake8': {'number_of_unused_imports': 2}}
    become {'flake8.number_of_unused_imports': 2} for consistent flattening and charting.

    Args:
        metrics_dict (Dict[str, Any]): Raw combined metrics from multiple tools.

    Returns:
        Dict[str, Any]: Merged dictionary with dot-notated keys.
    """
    merged = {}
    for section, subdict in metrics_dict.items():
        if isinstance(subdict, Mapping):
            for key, value in subdict.items():
                merged[f"{section}.{key}"] = value
                logger.debug(f"Merged: {section}.{key} = {value}")
        else:
            merged[section] = subdict
            logger.debug(f"Added non-nested: {section} = {subdict}")
    return merged

def flatten_metrics(d: dict, prefix: str = "", numeric_only: bool = False) -> dict:
    """Recursively flatten nested metric dictionaries into a flat key-value mapping.

    Args:
        d (dict): Nested input dictionary.
        prefix (str): Optional prefix for nested keys.
        numeric_only (bool): If True, include only numeric values (int, float).

    Returns:
        dict: Flattened dictionary with keys like "outer.inner.key".
    """
    if not isinstance(d, dict):
        raise TypeError("Expected input to be a dictionary")

    flat = {}

    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        logger.debug(f"Processing key: {full_key} (type: {type(v).__name__})")

        if isinstance(v, dict):
            logger.debug(f"Descending into nested dictionary at: {full_key}")
            flat.update(flatten_metrics(v, full_key, numeric_only))
        elif isinstance(v, (int, float)):
            flat[full_key] = v
            logger.debug(f"Added numeric value: {full_key} = {v}")
        elif not numeric_only:
            flat[full_key] = v
            logger.debug(f"Added non-numeric value: {full_key} = {v}")
        else:
            logger.debug(f"Excluded non-numeric value at: {full_key} (type: {type(v).__name__})")

    return flat
