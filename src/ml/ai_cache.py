import json
import hashlib
from pathlib import Path
from .config import AI_CACHE_DIR


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of the file contents.

    Args:
        file_path (str): Path to the source file

    Returns:
        str: SHA-256 hexadecimal digest
    """
    with open(file_path, "rb") as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()


def save_annotation_to_cache(file_path: str, annotation: dict) -> None:
    """
    Save annotated JSON output to cache path using file hash.

    Args:
        file_path (str): Path to source file
        annotation (dict): Annotation JSON object
    """
    file_hash = compute_file_hash(file_path)
    cache_path = Path(AI_CACHE_DIR) / f"{file_hash}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)


def load_cached_annotation(file_path: str) -> dict | None:
    """
    Load cached annotation if available.

    Args:
        file_path (str): Path to source file

    Returns:
        dict | None: Cached annotation or None if not found
    """
    file_hash = compute_file_hash(file_path)
    cache_path = Path(AI_CACHE_DIR) / f"{file_hash}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
