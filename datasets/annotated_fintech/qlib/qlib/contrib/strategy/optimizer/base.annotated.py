# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Class docstring provides a brief description of the class purpose
# Licensed under the MIT License.

import abc
# ✅ Best Practice: Use of abstract method to enforce implementation in subclasses

# ✅ Best Practice: Include a docstring to describe the purpose and behavior of the method

class BaseOptimizer(abc.ABC):
    """Construct portfolio with a optimization related method"""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> object:
        """Generate a optimized portfolio allocation"""