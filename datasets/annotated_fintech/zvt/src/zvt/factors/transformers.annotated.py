# -*- coding: utf-8 -*-
import numpy as np

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
import pandas as pd

from zvt.contract.factor import Transformer
from zvt.factors.algorithm import MaTransformer
from zvt.factors.technical_factor import TechnicalFactor

# ✅ Best Practice: Function name should be descriptive and use lowercase with words separated by underscores
from zvt.utils.pd_utils import (
    group_by_entity_id,
    normalize_group_compute_result,
    merge_filter_result,
)
from zvt.utils.time_utils import to_pd_timestamp

# ✅ Best Practice: Use of assert to validate input assumptions


# 🧠 ML Signal: Use of DataFrame indexing and slicing
def _cal_state(s, df, pre, interval, col):
    assert len(s) == pre + interval
    # ✅ Best Practice: Type hinting for better code readability and maintainability
    s = df.loc[s.index, :]
    pre_df: pd.DataFrame = s.iloc[:pre, :]
    # ✅ Best Practice: Type hinting for better code readability and maintainability
    recent_df: pd.DataFrame = s.iloc[-interval:, :]
    if pre_df.isnull().values.any() or recent_df.isnull().values.any():
        # 🧠 ML Signal: Checking for null values in DataFrame
        return np.nan
    pre_result = np.logical_and.reduce(pre_df["close"] > pre_df[col])
    # ✅ Best Practice: Class definition should follow the naming convention of using CamelCase.
    recent_result = np.logical_and.reduce(recent_df["close"] < recent_df[col])
    # 🧠 ML Signal: Logical operations on DataFrame columns
    # ✅ Best Practice: Use of default mutable arguments (like lists) can lead to unexpected behavior; consider using None and initializing inside the method.
    if pre_result and recent_result:
        return True
    # 🧠 ML Signal: Logical operations on DataFrame columns
    # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization of inherited attributes.
    # ✅ Best Practice: Type hinting for input and output improves code readability and maintainability
    return np.nan


# 🧠 ML Signal: Conditional logic based on computed results
# 🧠 ML Signal: Use of super() indicates inheritance, which is common in ML pipelines for data transformation


class CrossMaTransformer(MaTransformer):
    # 🧠 ML Signal: Dynamic column naming based on a list of windows suggests a pattern for feature engineering
    def __init__(self, windows=None, cal_change_pct=False) -> None:
        # 🧠 ML Signal: Returning NaN for specific conditions
        super().__init__(windows, cal_change_pct)

    # 🧠 ML Signal: Use of boolean indexing for filtering is a common pattern in data preprocessing

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        input_df = super().transform(input_df)
        cols = [f"ma{window}" for window in self.windows]
        # 🧠 ML Signal: Iterative comparison across columns is a pattern for complex feature creation
        s = input_df[cols[0]] > input_df[cols[1]]
        # ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
        # ✅ Best Practice: Use of __init__ method to initialize object attributes
        current_col = cols[1]
        for col in cols[2:]:
            # 🧠 ML Signal: Conversion of input data to a specific format (timestamp)
            # ✅ Best Practice: Adding a new column to the DataFrame for results is a clear and maintainable approach
            s = s & (input_df[current_col] > input_df[col])
            current_col = col
        # 🧠 ML Signal: Conversion of input data to a specific format (timestamp)
        # ✅ Best Practice: Use parentheses for method calls to avoid confusion with indexing
        input_df["filter_result"] = s
        return input_df


# ⚠️ SAST Risk (Low): Direct comparison with False can lead to unexpected results if s contains non-boolean values


# ✅ Best Practice: Use parentheses for method calls to avoid confusion with indexing
class SpecificTransformer(Transformer):
    def __init__(self, buy_timestamp, sell_timestamp) -> None:
        # 🧠 ML Signal: Adding a new column based on conditions can indicate feature engineering
        self.buy_timestamp = to_pd_timestamp(buy_timestamp)
        # ✅ Best Practice: Call to super() ensures proper initialization of the base class
        self.sell_timestamp = to_pd_timestamp(sell_timestamp)

    # 🧠 ML Signal: Use of default parameters in a constructor
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        s = input_df[input_df.get_level_values["timestamp"] == self.buy_timestamp]
        # 🧠 ML Signal: Use of default parameters in a constructor
        # 🧠 ML Signal: Usage of dynamic column naming based on class attributes
        s[s == False] = None
        s[input_df.get_level_values["timestamp"] == self.sell_timestamp] = False
        # ⚠️ SAST Risk (Low): Potential for KeyError if 'close' column is missing
        input_df["filter_result"] = s
        return input_df


class FallBelowTransformer(Transformer):
    # 🧠 ML Signal: Use of normalization function on computed results
    def __init__(self, window=10, interval=3) -> None:
        # 🧠 ML Signal: Logical comparison between DataFrame columns
        super().__init__()
        self.window = window
        self.interval = interval

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        col = f"ma{self.window}"
        if col not in input_df.columns:
            # 🧠 ML Signal: Use of lambda function for custom rolling window operations
            group_result = (
                group_by_entity_id(input_df["close"])
                .rolling(window=self.window, min_periods=self.window)
                .mean()
            )
            # 🧠 ML Signal: Use of normalization function on computed results
            group_result = normalize_group_compute_result(group_result=group_result)
            input_df[col] = group_result
        # 🧠 ML Signal: Merging filter results back into the DataFrame
        # 🧠 ML Signal: Instantiation of a class with specific parameters
        # 🧠 ML Signal: Use of __all__ to define public API of the module
        # ✅ Best Practice: Use 'is' for comparison with True/False/None
        # ✅ Best Practice: Use of __name__ guard for script entry point

        # 连续3(interval)日收在10(window)日线下
        s = input_df["close"] < input_df[col]
        s = (
            group_by_entity_id(s)
            .rolling(window=self.interval, min_periods=self.interval)
            .apply(lambda x: np.logical_and.reduce(x))
        )
        s = normalize_group_compute_result(group_result=s)
        # 构造卖点
        s[s == False] = None
        s[s == True] = False
        input_df = merge_filter_result(input_df=input_df, filter_result=s)

        return input_df


if __name__ == "__main__":
    # df = Stock1dHfqKdata.query_data(codes=["000338"], index=["entity_id", "timestamp"])
    # df = FallBelowTransformer().transform(df)
    # print(df["filter_result"])
    TechnicalFactor(transformer=SpecificTransformer(timestamp="2020-03-01"))


# the __all__ is generated
__all__ = ["CrossMaTransformer", "SpecificTransformer", "FallBelowTransformer"]
