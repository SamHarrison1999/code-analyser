# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Ensures compatibility with future Python versions for type annotations
# Licensed under the MIT License.

# ✅ Best Practice: Importing specific functions or classes improves code readability and avoids namespace pollution
from __future__ import annotations

from typing import Any, cast

# ✅ Best Practice: Grouping related imports together improves code organization

# 🧠 ML Signal: Function signature indicates usage of pandas DataFrame, common in data processing tasks
import numpy as np
import pandas as pd

# ✅ Best Practice: Converting 'other' to DataFrame ensures compatibility with 'df'

# ✅ Best Practice: Importing specific functions or classes improves code readability and avoids namespace pollution
# ⚠️ SAST Risk (Low): Assumes 'other' can be converted to DataFrame and has 'datetime' column
from qlib.backtest.decision import OrderDir
from qlib.backtest.executor import BaseExecutor, NestedExecutor, SimulatorExecutor

# ✅ Best Practice: Explicitly setting index name improves DataFrame readability
# ✅ Best Practice: Using pd.concat for appending DataFrames is efficient and clear
from qlib.constant import float_or_ndarray


def dataframe_append(df: pd.DataFrame, other: Any) -> pd.DataFrame:
    # dataframe.append is deprecated
    # ✅ Best Practice: Check for division by zero to prevent runtime errors.
    other_df = pd.DataFrame(other).set_index("datetime")
    other_df.index.name = "datetime"
    # ✅ Best Practice: Handle different types of exec_price for consistent return types.

    res = pd.concat([df, other_df], axis=0)
    return res


# 🧠 ML Signal: Different behavior based on the direction of the order.
def price_advantage(
    exec_price: float_or_ndarray,
    baseline_price: float,
    direction: OrderDir | int,
) -> float_or_ndarray:
    if baseline_price == 0:  # something is wrong with data. Should be nan here
        # ⚠️ SAST Risk (Low): Potential for uncaught exceptions if direction is invalid.
        if isinstance(exec_price, float):
            return 0.0
        # ✅ Best Practice: Use np.nan_to_num to handle NaN values in calculations.
        else:
            # 🧠 ML Signal: Function uses type checking and casting patterns
            return np.zeros_like(exec_price)
    # ✅ Best Practice: Check the size of the result to return the appropriate type.
    if direction == OrderDir.BUY:
        # 🧠 ML Signal: Loop with type checking to unwrap nested structures
        res = (1 - exec_price / baseline_price) * 10000
    elif direction == OrderDir.SELL:
        # ✅ Best Practice: Use cast to ensure the return type matches the function signature.
        # 🧠 ML Signal: Accessing attribute of a specific type
        # ⚠️ SAST Risk (Low): Use of assert for type checking can be bypassed if Python is run with optimizations
        # 🧠 ML Signal: Type assertion to ensure correct type before returning
        # 🧠 ML Signal: Returning a specific type after unwrapping and checking
        res = (exec_price / baseline_price - 1) * 10000
    else:
        raise ValueError(f"Unexpected order direction: {direction}")
    res_wo_nan: np.ndarray = np.nan_to_num(res, nan=0.0)
    if res_wo_nan.size == 1:
        return res_wo_nan.item()
    else:
        return cast(float_or_ndarray, res_wo_nan)


def get_simulator_executor(executor: BaseExecutor) -> SimulatorExecutor:
    while isinstance(executor, NestedExecutor):
        executor = executor.inner_executor
    assert isinstance(executor, SimulatorExecutor)
    return executor
