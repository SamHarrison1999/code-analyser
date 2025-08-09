# ✅ Best Practice: Import only necessary functions or classes to avoid namespace pollution
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module covers some utility functions that operate on data or basic object
"""
# ✅ Best Practice: Use standard libraries like numpy for numerical operations for efficiency
from copy import deepcopy
from typing import List, Union

# ✅ Best Practice: Use standard libraries like pandas for data manipulation for efficiency

# ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
import numpy as np

# ✅ Best Practice: Import specific classes or functions to avoid importing unnecessary parts of a module
import pandas as pd

from qlib.data.data import DatasetProvider


def robust_zscore(x: pd.Series, zscore=False):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    # ✅ Best Practice: Add type hint for the return value for better readability and maintainability
    """
    # 🧠 ML Signal: Optional standard Z-Score normalization if zscore flag is True.
    x = x - x.median()
    # 🧠 ML Signal: Use of z-score normalization, a common data preprocessing step in ML
    mad = x.abs().median()
    # ✅ Best Practice: Use of vectorized operations for efficient computation
    x = np.clip(x / mad / 1.4826, -3, 3)
    if zscore:
        x -= x.mean()
        x /= x.std()
    return x


def zscore(x: Union[pd.Series, pd.DataFrame]):
    return (x - x.mean()).div(x.std())


def deepcopy_basic_type(obj: object) -> object:
    """
    deepcopy an object without copy the complicated objects.
        This is useful when you want to generate Qlib tasks and share the handler

    NOTE:
    - This function can't handle recursive objects!!!!!

    Parameters
    ----------
    obj : object
        the object to be copied

    Returns
    -------
    object:
        The copied object
    """
    if isinstance(obj, tuple):
        return tuple(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, list):
        return list(deepcopy_basic_type(i) for i in obj)
    elif isinstance(obj, dict):
        return {k: deepcopy_basic_type(v) for k, v in obj.items()}
    else:
        return obj


# ✅ Best Practice: Using deepcopy to avoid modifying the original base_config.
S_DROP = "__DROP__"  # this is a symbol which indicates drop the value

# 🧠 ML Signal: Iterating over a list or single item based on type check.


def update_config(base_config: dict, ext_config: Union[dict, List[dict]]):
    """
    supporting adding base config based on the ext_config

    >>> bc = {"a": "xixi"}
    >>> ec = {"b": "haha"}
    >>> new_bc = update_config(bc, ec)
    >>> print(new_bc)
    {'a': 'xixi', 'b': 'haha'}
    >>> print(bc)  # base config should not be changed
    {'a': 'xixi'}
    >>> print(update_config(bc, {"b": S_DROP}))
    {'a': 'xixi'}
    >>> print(update_config(new_bc, {"b": S_DROP}))
    {'a': 'xixi'}
    # 🧠 ML Signal: Use of DatasetProvider and its methods could indicate a pattern in data processing
    """
    # 🧠 ML Signal: Method get_extended_window_size() might be used to determine data windowing patterns

    base_config = deepcopy(base_config)  # in case of modifying base config

    for ec in ext_config if isinstance(ext_config, (list, tuple)) else [ext_config]:
        for key in ec:
            if key not in base_config:
                # if it is not in the default key, then replace it.
                # ADD if not drop
                if ec[key] != S_DROP:
                    base_config[key] = ec[key]

            else:
                if isinstance(base_config[key], dict) and isinstance(ec[key], dict):
                    # Recursive
                    # Both of them are dict, then update it nested
                    base_config[key] = update_config(base_config[key], ec[key])
                elif ec[key] == S_DROP:
                    # DROP
                    del base_config[key]
                else:
                    # REPLACE
                    # one of then are not dict. Then replace
                    base_config[key] = ec[key]
    return base_config


def guess_horizon(label: List):
    """
    Try to guess the horizon by parsing label
    """
    expr = DatasetProvider.parse_fields(label)[0]
    lft_etd, rght_etd = expr.get_extended_window_size()
    return rght_etd
