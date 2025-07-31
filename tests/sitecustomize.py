# ✅ sitecustomize.py for mocking torch modules in test environment

import sys
import types
from unittest.mock import MagicMock

# ✅ Create base mock module for torch
mock_torch = types.ModuleType("torch")

# ✅ Mock torch.nn with minimal structure
mock_nn = types.ModuleType("torch.nn")
mock_nn.Module = type("Module", (), {})  # Minimal class to satisfy inheritance
mock_nn.Linear = MagicMock(name="Linear")
mock_nn.ReLU = MagicMock(name="ReLU")
mock_torch.nn = mock_nn

# ✅ Mock torch.nn.functional as needed
mock_functional = MagicMock(name="functional")
mock_torch.nn.functional = mock_functional

# ✅ Mock other commonly used torch modules
mock_torch.optim = MagicMock(name="optim")
mock_torch.utils = types.SimpleNamespace(_pytree=MagicMock(name="_pytree"))
mock_torch.Tensor = MagicMock(name="Tensor")
mock_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
mock_torch.device = lambda x: f"mock_device:{x}"
mock_torch.save = MagicMock(name="save")

# ✅ Register all mocks in sys.modules to intercept imports
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_nn
sys.modules["torch.nn.functional"] = mock_functional
sys.modules["torch.optim"] = mock_torch.optim
sys.modules["torch.utils"] = mock_torch.utils
sys.modules["torch.utils._pytree"] = mock_torch.utils._pytree
sys.modules["torch.cuda"] = mock_torch.cuda
sys.modules["torch._tensor"] = MagicMock(name="_tensor")
sys.modules["torch.overrides"] = MagicMock(name="overrides")

# ✅ TensorBoard mocking for torch.utils.tensorboard.SummaryWriter
mock_tensorboard = types.ModuleType("torch.utils.tensorboard")
mock_tensorboard.SummaryWriter = MagicMock(name="SummaryWriter")
sys.modules["torch.utils.tensorboard"] = mock_tensorboard
