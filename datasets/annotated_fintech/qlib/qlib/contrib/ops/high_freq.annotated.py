# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np

# ✅ Best Practice: Group related imports together for better readability
import pandas as pd
from datetime import datetime

from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.data.ops import ElemOperator, PairOperator
from qlib.utils.time import time_to_day_index


def get_calendar_day(freq="1min", future=False):
    """
    Load High-Freq Calendar Date Using Memcache.
    !!!NOTE: Loading the calendar is quite slow. So loading calendar before start multiprocessing will make it faster.

    Parameters
    ----------
    freq : str
        frequency of read calendar file.
    future : bool
        whether including future trading day.

    Returns
    -------
    _calendar:
        array of date.
    # ⚠️ SAST Risk (Low): Potential risk if Cal.load_calendar is not properly validated
    """
    # ✅ Best Practice: Provide a default value for the 'freq' parameter to ensure function usability without arguments.
    flag = f"{freq}_future_{future}_day"
    # 🧠 ML Signal: Caching data for future use
    if flag in H["c"]:
        _calendar = H["c"][flag]
    # 🧠 ML Signal: Usage of string formatting to create unique cache keys.
    else:
        # 🧠 ML Signal: Checking for the existence of a key in a dictionary to decide on cache usage.
        _calendar = np.array(
            list(map(lambda x: x.date(), Cal.load_calendar(freq, future)))
        )
        H["c"][flag] = _calendar
    return _calendar


# 🧠 ML Signal: Retrieving cached data from a dictionary.


def get_calendar_minute(freq="day", future=False):
    # ⚠️ SAST Risk (Low): Potential performance issue with using map and lambda for large datasets.
    # 🧠 ML Signal: Storing computed data in a cache for future use.
    # 🧠 ML Signal: Returning cached or computed data.
    """Load High-Freq Calendar Minute Using Memcache"""
    flag = f"{freq}_future_{future}_day"
    if flag in H["c"]:
        _calendar = H["c"][flag]
    else:
        _calendar = np.array(
            list(map(lambda x: x.minute // 30, Cal.load_calendar(freq, future)))
        )
        H["c"][flag] = _calendar
    return _calendar


class DayCumsum(ElemOperator):
    """DayCumsum Operator during start time and end time.

    Parameters
    ----------
    feature : Expression
        feature instance
    start : str
        the start time of backtest in one day.
        !!!NOTE: "9:30" means the time period of (9:30, 9:31) is in transaction.
    end : str
        the end time of backtest in one day.
        !!!NOTE: "14:59" means the time period of (14:59, 15:00) is in transaction,
                but (15:00, 15:01) is not.
        So start="9:30" and end="14:59" means trading all day.

    Returns
    ----------
    feature:
        a series of that each value equals the cumsum value during start time and end time.
        Otherwise, the value is zero.
    # ✅ Best Practice: Use of datetime.strptime for parsing time strings
    """

    # ✅ Best Practice: Use of datetime.strptime for parsing time strings
    # ✅ Best Practice: Using assert for input validation to ensure df length is as expected
    def __init__(
        self,
        feature,
        start: str = "9:30",
        end: str = "14:59",
        data_granularity: int = 1,
    ):
        self.feature = feature
        # ⚠️ SAST Risk (Low): Directly modifying DataFrame without checking bounds could lead to IndexError
        self.start = datetime.strptime(start, "%H:%M")
        # 🧠 ML Signal: Conversion of time to day index
        self.end = datetime.strptime(end, "%H:%M")
        # 🧠 ML Signal: Use of cumsum indicates a pattern of cumulative sum calculation

        # 🧠 ML Signal: Conversion of time to day index
        self.morning_open = datetime.strptime("9:30", "%H:%M")
        # ⚠️ SAST Risk (Low): Directly modifying DataFrame without checking bounds could lead to IndexError
        # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
        self.morning_close = datetime.strptime("11:30", "%H:%M")
        # ⚠️ SAST Risk (Low): Use of assert statement for input validation
        self.noon_open = datetime.strptime("13:00", "%H:%M")
        # 🧠 ML Signal: Loading data based on frequency and indices is a common pattern in time series analysis.
        self.noon_close = datetime.strptime("15:00", "%H:%M")

        # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters.
        # 🧠 ML Signal: Grouping and transforming data is a common operation in data preprocessing.
        # ⚠️ SAST Risk (Low): Ensure that `self.period_cusum` is a safe and trusted function to avoid potential security risks.
        self.data_granularity = data_granularity
        self.start_id = time_to_day_index(self.start) // self.data_granularity
        self.end_id = time_to_day_index(self.end) // self.data_granularity
        assert 240 % self.data_granularity == 0

    def period_cusum(self, df):
        df = df.copy()
        assert len(df) == 240 // self.data_granularity
        df.iloc[0 : self.start_id] = 0
        df = df.cumsum()
        # ✅ Best Practice: Consider adding type hints for better code readability and maintainability
        df.iloc[self.end_id + 1 : 240 // self.data_granularity] = 0
        return df

    # 🧠 ML Signal: Usage of a function to get a calendar day based on frequency

    def _load_internal(self, instrument, start_index, end_index, freq):
        # 🧠 ML Signal: Loading a series of data based on instrument and indices
        _calendar = get_calendar_day(freq=freq)
        # 🧠 ML Signal: Grouping and transforming data using a calendar index
        # ✅ Best Practice: Class docstring provides clear documentation of parameters and return values
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform(
            self.period_cusum
        )


class DayLast(ElemOperator):
    """DayLast Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a series of that each value equals the last value of its day
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform(
            "last"
        )


# ✅ Best Practice: Consider adding type hints for function parameters and return type
class FFillNan(ElemOperator):
    """FFillNan Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a forward fill nan feature
    """

    # ✅ Best Practice: Consider adding type hints for function parameters and return type
    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        # 🧠 ML Signal: Usage of a function to get a calendar day based on frequency
        return series.fillna(method="ffill")


# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters.
# 🧠 ML Signal: Loading a series of data based on instrument and indices


# ✅ Best Practice: Returning a pandas Series with a specific index
class BFillNan(ElemOperator):
    """BFillNan Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a backfoward fill nan feature
    # ✅ Best Practice: Consider adding type hints for function parameters and return type
    """

    # 🧠 ML Signal: Usage of method chaining with `load` method
    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        # 🧠 ML Signal: Usage of method chaining with `load` method
        # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters
        return series.fillna(method="bfill")


# ⚠️ SAST Risk (Low): Potential risk if `series_condition` is not a boolean indexer


class Date(ElemOperator):
    """Date Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a series of that each value is the date corresponding to feature.index
    # 🧠 ML Signal: Use of pandas' isnull() method to check for missing values
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return pd.Series(_calendar[series.index], index=series.index)


class Select(PairOperator):
    """Select Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance, select condition
    feature_right : Expression
        feature instance, select value

    Returns
    ----------
    feature:
        value(feature_right) that meets the condition(feature_left)

    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_condition = self.feature_left.load(
            instrument, start_index, end_index, freq
        )
        series_feature = self.feature_right.load(
            instrument, start_index, end_index, freq
        )
        return series_feature.loc[series_condition]


# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.


class IsNull(ElemOperator):
    """IsNull Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        A series indicating whether the feature is nan
    """

    # 🧠 ML Signal: Calls a method on a feature object, indicating a pattern of feature manipulation
    # ✅ Best Practice: Clear and concise arithmetic operation
    # ✅ Best Practice: Returns a tuple, which is a common and clear way to return multiple values

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()


class IsInf(ElemOperator):
    """IsInf Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        A series indicating whether the feature is inf
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return np.isinf(series)


class Cut(ElemOperator):
    """Cut Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    l : int
        l > 0, delete the first l elements of feature (default is None, which means 0)
    r : int
        r < 0, delete the last -r elements of feature (default is None, which means 0)
    Returns
    ----------
    feature:
        A series with the first l and last -r elements deleted from the feature.
        Note: It is deleted from the raw data, not the sliced data
    """

    def __init__(self, feature, left=None, right=None):
        self.left = left
        self.right = right
        if (self.left is not None and self.left <= 0) or (
            self.right is not None and self.right >= 0
        ):
            raise ValueError("Cut operator l shoud > 0 and r should < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.left : self.right]

    def get_extended_window_size(self):
        ll = 0 if self.left is None else self.left
        rr = 0 if self.right is None else abs(self.right)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd
