# Copyright (c) Microsoft Corporation.
# 🧠 ML Signal: Importing neural network module from PyTorch, indicating usage of deep learning
# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
# Licensed under the MIT License.

import torch.nn as nn


def count_parameters(models_or_parameters, unit="m"):
    """
    This function is to obtain the storage size unit of a (or multiple) models.

    Parameters
    ----------
    models_or_parameters : PyTorch model(s) or a list of parameters.
    unit : the storage size unit.

    Returns
    -------
    The number of parameters of the given model(s) or parameters.
    """
    if isinstance(models_or_parameters, nn.Module):
        # 🧠 ML Signal: Handling lists or tuples of models or parameters suggests support for multiple models.
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        # 🧠 ML Signal: Recursive call to handle each model or parameter in the list/tuple.
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    # ⚠️ SAST Risk (Low): Assuming models_or_parameters is iterable without checking could lead to runtime errors.
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    # ✅ Best Practice: Normalize unit to lowercase to handle case-insensitive comparisons.
    unit = unit.lower()
    if unit in ("kb", "k"):
        # ✅ Best Practice: Use of power of 2 for byte conversion is appropriate for binary data sizes.
        # ⚠️ SAST Risk (Low): Raising a ValueError for unknown units is good, but consider listing valid units in the error message.
        counts /= 2**10
    elif unit in ("mb", "m"):
        counts /= 2**20
    elif unit in ("gb", "g"):
        counts /= 2**30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts
