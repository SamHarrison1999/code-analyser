# File: tests/conftest.py
import sys
import types
from unittest.mock import MagicMock
import os

# Mock torch and submodules early before any test files are imported
mock_torch = types.ModuleType("torch")
mock_torch.Tensor = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch._tensor"] = types.ModuleType("torch._tensor")
sys.modules["torch.overrides"] = types.ModuleType("torch.overrides")
sys.modules["torch.utils"] = types.ModuleType("torch.utils")
sys.modules["torch.utils._pytree"] = types.ModuleType("torch.utils._pytree")

# Optional: mock transformers/tokenizers if needed
for mod in ["transformers", "transformers.models", "transformers.utils", "tokenizers"]:
    sys.modules[mod] = MagicMock()


# ✅ Add 'src' directory to sys.path so imports like `from gui import ...` work in tests
# ✅ This avoids needing to run pytest with PYTHONPATH or from project root
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
