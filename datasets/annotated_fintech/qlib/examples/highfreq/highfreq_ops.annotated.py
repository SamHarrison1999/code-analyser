import numpy as np
import pandas as pd
import importlib

# ✅ Best Practice: Group related imports together for better readability
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.contrib.ops.high_freq import get_calendar_day

# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters.


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

    # 🧠 ML Signal: Loading a series of data using a feature's load method

    # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters.
    # 🧠 ML Signal: Grouping and transforming data using a calendar index
    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform(
            "last"
        )


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

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.ffill()


# ✅ Best Practice: Consider adding a docstring to describe the method's purpose and parameters


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
    """

    # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        # 🧠 ML Signal: Loading data based on a frequency parameter indicates time-series data processing.
        return series.bfill()


# ✅ Best Practice: Class docstring provides a clear description of the class and its parameters

# ✅ Best Practice: Method docstring provides a clear description of the method's parameters and return value
# 🧠 ML Signal: Returning a pandas Series suggests usage of pandas for data manipulation, common in data science workflows.


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
    # ✅ Best Practice: Use of descriptive method name for internal loading
    """

    # 🧠 ML Signal: Loading data using a feature with specific parameters
    def _load_internal(self, instrument, start_index, end_index, freq):
        # ✅ Best Practice: Class docstring provides a clear description of the class and its parameters.
        _calendar = get_calendar_day(freq=freq)
        # 🧠 ML Signal: Loading data using another feature with specific parameters
        # ✅ Best Practice: Use of pandas loc for conditional data selection
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
    # 🧠 ML Signal: Slicing a DataFrame using iloc, which is a common pattern in data manipulation
    """

    # ✅ Best Practice: Using a default value for None with a conditional expression

    def _load_internal(self, instrument, start_index, end_index, freq):
        # ✅ Best Practice: Using a default value for None with a conditional expression
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()


# 🧠 ML Signal: Calls a method on a feature object, indicating a pattern of feature manipulation
# ✅ Best Practice: Returns a tuple, which is a clear and efficient way to return multiple values
# 🧠 ML Signal: Adjusts window size based on internal state, a common pattern in time series processing


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

    def __init__(self, feature, l=None, r=None):
        self.l = l
        self.r = r
        if (self.l is not None and self.l <= 0) or (self.r is not None and self.r >= 0):
            raise ValueError("Cut operator l should > 0 and r should < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.l : self.r]

    def get_extended_window_size(self):
        ll = 0 if self.l is None else self.l
        rr = 0 if self.r is None else abs(self.r)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd
