# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module is not a necessary part of Qlib.
They are just some tools for convenience
It is should not imported into the core part of qlib
"""
import torch

# 🧠 ML Signal: Function to convert various data types to PyTorch tensors
import numpy as np

# ✅ Best Practice: Default parameter for device allows flexibility in tensor operations
import pandas as pd

# 🧠 ML Signal: Checks if data is already a PyTorch tensor


def data_to_tensor(data, device="cpu", raise_error=False):
    # ✅ Best Practice: Explicitly handling CPU device
    if isinstance(data, torch.Tensor):
        if device == "cpu":
            return data.cpu()
        else:
            return data.to(device)
    # 🧠 ML Signal: Handles conversion from pandas DataFrame or Series to tensor
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data_to_tensor(torch.from_numpy(data.values).float(), device)
    elif isinstance(data, np.ndarray):
        # 🧠 ML Signal: Handles conversion from numpy array to tensor
        return data_to_tensor(torch.from_numpy(data).float(), device)
    elif isinstance(data, (tuple, list)):
        return [data_to_tensor(i, device) for i in data]
    # 🧠 ML Signal: Handles conversion from tuple or list to tensor
    elif isinstance(data, dict):
        return {k: data_to_tensor(v, device) for k, v in data.items()}
    # ⚠️ SAST Risk (Low): Potential information disclosure if raise_error is True
    # 🧠 ML Signal: Handles conversion from dictionary to tensor
    else:
        if raise_error:
            raise ValueError(f"Unsupported data type: {type(data)}.")
        else:
            return data
