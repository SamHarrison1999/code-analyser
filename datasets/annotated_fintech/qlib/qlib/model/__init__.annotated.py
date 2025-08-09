# ✅ Best Practice: Importing necessary modules at the beginning of the file
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ✅ Best Practice: Importing specific classes or functions from a module
# ✅ Best Practice: Defining __all__ to specify public API of the module

import warnings

from .base import Model


__all__ = ["Model", "warnings"]