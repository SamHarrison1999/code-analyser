"""
Initialise the test suite as a package and configure shared test settings.

This includes:
  - Enabling safe logging for debugging test runs
  - Initialising environment variables (e.g., disabling real API access)
  - Exposing reusable imports for fixtures

This file is imported when pytest or unittest discovers the test suite.
"""

import os
import logging

# ✅ Best Practice: Avoid running real API calls or mutations in test environments
# Set environment to 'test' so application logic can adjust accordingly
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DISABLE_NETWORK_CALLS", "1")

# ✅ Best Practice: Set up logging for test debugging (only stderr output)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Optional: import fixture helpers or monkeypatch plugins globally
from . import fixtures  # noqa: F401
