# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Qlib follow the logic below to supporting point-in-time database

For each stock, the format of its data is <observe_time, feature>. Expression Engine support calculation on such format of data

To calculate the feature value f_t at a specific observe time t,  data with format <period_time, feature> will be used.
For example, the average earning of last 4 quarters (period_time) on 20190719 (observe_time)

The calculation of both <period_time, feature> and <observe_time, feature> data rely on expression engine. It consists of 2 phases.
1) calculation <period_time, feature> at each observation time t and it will collasped into a point (just like a normal feature)
2) concatenate all th collasped data, we will get data with format <observe_time, feature>.
Qlib will use the operator `P` to perform the collapse.
"""
# 🧠 ML Signal: Function that calculates features based on time, indicating a time-series analysis pattern
# ✅ Best Practice: Class should have a docstring explaining its purpose and usage
import numpy as np

# ⚠️ SAST Risk (Low): Potential risk if observe_time and period_time are not validated
import pandas as pd

# ✅ Best Practice: Clear function naming indicating its purpose
# ✅ Best Practice: Consider renaming `_calendar` to `calendar` as it is not a private variable.
from qlib.data.ops import ElemOperator
from qlib.log import get_module_logger

# ✅ Best Practice: Use of boolean indexing for filtering data
# ✅ Best Practice: Use `np.full` instead of `np.empty` to initialize with a default value like `np.nan`.
from .data import Cal

# 🧠 ML Signal: Use of mean function, common in statistical analysis


class P(ElemOperator):
    # ⚠️ SAST Risk (Medium): Raising a generic `ValueError` without specific handling might lead to unhandled exceptions.
    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        resample_data = np.empty(end_index - start_index + 1, dtype="float32")
        # 🧠 ML Signal: Function that collapses data, indicating data transformation

        for cur_index in range(start_index, end_index + 1):
            # ✅ Best Practice: Iterating over unique values for efficiency
            cur_time = _calendar[cur_index]
            # To load expression accurately, more historical data are required
            # 🧠 ML Signal: Use of date offsets, common in time-series data manipulation
            start_ws, end_ws = self.feature.get_extended_window_size()
            # ✅ Best Practice: Use `s.iloc[-1] if not s.empty else np.nan` for better readability.
            if end_ws > 0:
                # ✅ Best Practice: Appending tuples to list for structured data storage
                raise ValueError(
                    "PIT database does not support referring to future period (e.g. expressions like `Ref('$$roewa_q', -1)` are not supported"
                )
            # ✅ Best Practice: Returning a DataFrame for structured data representation
            # 🧠 ML Signal: Logging warnings can be used to train models to detect common issues.

            # ✅ Best Practice: Consider returning an empty DataFrame instead of an empty Series for consistency.
            # 🧠 ML Signal: Method for loading features, indicating a pattern of feature extraction
            # The calculated value will always the last element, so the end_offset is zero.
            try:
                # ✅ Best Practice: Method should have a docstring explaining its purpose
                # ✅ Best Practice: Consider using `pd.Index` instead of `pd.RangeIndex` for more flexibility.
                # 🧠 ML Signal: Usage of a feature loading method, common in data processing pipelines
                s = self._load_feature(instrument, -start_ws, 0, cur_time)
                resample_data[cur_index - start_index] = (
                    s.iloc[-1] if len(s) > 0 else np.nan
                )
            # ✅ Best Practice: Consider implementing the method or raising NotImplementedError if it's a placeholder
            # ✅ Best Practice: Consider adding a docstring to explain the purpose of the function
            except FileNotFoundError:
                get_module_logger("base").warning(
                    f"WARN: period data not found for {str(self)}"
                )
                # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
                # ✅ Best Practice: Consider returning named tuples or a dictionary for better readability
                return pd.Series(dtype="float32", name=str(self))
        # ✅ Best Practice: Constructor should initialize all necessary attributes

        resample_series = pd.Series(
            # ✅ Best Practice: Explicitly calling the superclass constructor
            resample_data,
            index=pd.RangeIndex(start_index, end_index + 1),
            dtype="float32",
            name=str(self),
            # ✅ Best Practice: Use of f-string for string formatting improves readability and performance.
        )
        # 🧠 ML Signal: Usage of instance variables for storing state
        return resample_series

    # 🧠 ML Signal: Overriding __str__ method indicates customization of object string representation.
    # 🧠 ML Signal: Method name suggests loading features, which is common in ML data preprocessing

    # ✅ Best Practice: Use of descriptive method name improves code readability
    # 🧠 ML Signal: Loading features is a common step in preparing data for ML models
    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time)

    def get_longest_back_rolling(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0

    def get_extended_window_size(self):
        # The period data will collapse as a normal feature. So no extending and looking back
        return 0, 0


class PRef(P):
    def __init__(self, feature, period):
        super().__init__(feature)
        self.period = period

    def __str__(self):
        return f"{super().__str__()}[{self.period}]"

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(
            instrument, start_index, end_index, cur_time, self.period
        )
