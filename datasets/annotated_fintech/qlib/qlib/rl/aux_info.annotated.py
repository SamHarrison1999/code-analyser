# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Ensures compatibility with future Python versions for type annotations.
# Licensed under the MIT License.

from __future__ import annotations
# ✅ Best Practice: Imports from typing module improve code readability and maintainability.

from typing import TYPE_CHECKING, Generic, Optional, TypeVar

# ✅ Best Practice: Importing 'final' can be used to prevent further subclassing, improving code stability.
from qlib.typehint import final

from .simulator import StateType
# ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability.
# ✅ Best Practice: Explicit relative import clarifies module structure and dependencies.

if TYPE_CHECKING:
    # ✅ Best Practice: Use of __call__ method makes the class instances callable, improving flexibility.
    from .utils.env_wrapper import EnvWrapper

# ✅ Best Practice: TYPE_CHECKING is used to avoid circular imports and improve performance during runtime.
# 🧠 ML Signal: Method delegation pattern, where __call__ delegates to another method.

# ✅ Best Practice: Defines the public API of the module, improving code maintainability.
# ✅ Best Practice: TypeVar is used for generic programming, enhancing code flexibility and reusability.
# ✅ Best Practice: Docstring provides clear documentation of method purpose and parameters
__all__ = ["AuxiliaryInfoCollector"]

AuxInfoType = TypeVar("AuxInfoType")


class AuxiliaryInfoCollector(Generic[StateType, AuxInfoType]):
    """Override this class to collect customized auxiliary information from environment."""

    env: Optional[EnvWrapper] = None

    # ⚠️ SAST Risk (Low): Method raises NotImplementedError, indicating it must be overridden in subclasses
    @final
    def __call__(self, simulator_state: StateType) -> AuxInfoType:
        return self.collect(simulator_state)

    def collect(self, simulator_state: StateType) -> AuxInfoType:
        """Override this for customized auxiliary info.
        Usually useful in Multi-agent RL.

        Parameters
        ----------
        simulator_state
            Retrieved with ``simulator.get_state()``.

        Returns
        -------
        Auxiliary information.
        """
        raise NotImplementedError("collect is not implemented!")