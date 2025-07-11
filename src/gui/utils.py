# gui/utils.py
def flatten_metrics(d: dict, prefix: str = "", numeric_only: bool = False) -> dict:
    """Recursively flatten nested metric dictionaries.

    Args:
        d (dict): Nested input dictionary.
        prefix (str): Prefix to apply for keys in nested structures.
        numeric_only (bool): If True, include only int/float values.

    Returns:
        dict: Flattened dictionary with optional filtering.
    """
    flat = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            flat[full_key] = v
        elif isinstance(v, dict):
            flat.update(flatten_metrics(v, full_key, numeric_only))
        elif not numeric_only:
            flat[full_key] = v
    return flat

