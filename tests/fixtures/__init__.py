"""
This package contains test fixtures used across multiple test modules.
Fixtures may include:
  - Static code examples (safe and vulnerable)
  - Synthetic AST representations
  - Mock tool outputs (e.g., Bandit, Flake8, etc.)
  - Git metadata simulation

Ensure fixture files are small, self-contained, and easy to modify or label.
"""

import os
import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent.resolve()

def load_fixture(filename: str, as_json: bool = False) -> str | dict:
    """
    Load a fixture file by name from the fixtures directory.

    Args:
        filename (str): Name of the fixture file.
        as_json (bool): Whether to parse the file as JSON.

    Returns:
        str | dict: File content or parsed JSON object.
    """
    path = FIXTURE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {filename}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f) if as_json else f.read()


def list_fixtures(extension: str = ".py") -> list[str]:
    """
    List all fixture files in the directory with the given extension.

    Args:
        extension (str): Filter by file extension (default: .py).

    Returns:
        list[str]: List of matching filenames.
    """
    return [f.name for f in FIXTURE_DIR.glob(f'*{extension}') if f.is_file()]
