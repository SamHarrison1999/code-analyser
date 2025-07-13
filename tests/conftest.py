# File: tests/conftest.py
import sys
from pathlib import Path

# AAdd src/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
