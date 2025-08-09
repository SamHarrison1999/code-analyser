# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

# ✅ Best Practice: Use of type hints improves code readability and maintainability.
import inspect
import logging
from collections import OrderedDict
from functools import lru_cache

# ✅ Best Practice: Importing specific modules or functions helps avoid namespace pollution.
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Union, cast

# 🧠 ML Signal: Custom logging setup can indicate specific logging practices or configurations.
import numpy as np
import pandas as pd

import qlib.utils.index_data as idd

# ✅ Best Practice: Initialize logger for consistent logging throughout the class

# ✅ Best Practice: Include type hint for return type to improve code readability and maintainability
from ..log import get_module_logger
from ..utils.index_data import IndexData, SingleData
from ..utils.resam import resam_ts_data, ts_data_last
from ..utils.time import Freq, is_single_value


class BaseQuote:
    def __init__(self, quote_df: pd.DataFrame, freq: str) -> None:
        # ⚠️ SAST Risk (Low): Using NotImplementedError with a formatted string could expose internal information if not handled properly
        self.logger = get_module_logger("online operator", level=logging.INFO)

    def get_all_stock(self) -> Iterable:
        """return all stock codes

        Return
        ------
        Iterable
            all stock codes
        """

        raise NotImplementedError("Please implement the `get_all_stock` method")

    def get_data(
        self,
        stock_id: str,
        start_time: Union[pd.Timestamp, str],
        end_time: Union[pd.Timestamp, str],
        field: Union[str],
        method: Optional[str] = None,
    ) -> Union[None, int, float, bool, IndexData]:
        """get the specific field of stock data during start time and end_time,
           and apply method to the data.

           Example:
            .. code-block::
                                        $close      $volume
                instrument  datetime
                SH600000    2010-01-04  86.778313   16162960.0
                            2010-01-05  87.433578   28117442.0
                            2010-01-06  85.713585   23632884.0
                            2010-01-07  83.788803   20813402.0
                            2010-01-08  84.730675   16044853.0

                SH600655    2010-01-04  2699.567383  158193.328125
                            2010-01-08  2612.359619   77501.406250
                            2010-01-11  2712.982422  160852.390625
                            2010-01-12  2788.688232  164587.937500
                            2010-01-13  2790.604004  145460.453125

                this function is used for three case:

                1. method is not None. It returns int/float/bool/None.
                    - It will return None in one case, the method return None

                    print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method="last"))

                    85.713585

                2. method is None. It returns IndexData.
                    print(get_data(stock_id="SH600000", start_time="2010-01-04", end_time="2010-01-06", field="$close", method=None))

                    IndexData([86.778313, 87.433578, 85.713585], [2010-01-04, 2010-01-05, 2010-01-06])

        Parameters
        ----------
        stock_id: str
        start_time : Union[pd.Timestamp, str]
            closed start time for backtest
        end_time : Union[pd.Timestamp, str]
            closed end time for backtest
        field : str
            the columns of data to fetch
        method : Union[str, None]
            the method apply to data.
            e.g [None, "last", "all", "sum", "mean", "ts_data_last"]

        Return
        ----------
        Union[None, int, float, bool, IndexData]
            it will return None in following cases
            - There is no stock data which meet the query criterion from data source.
            - The `method` returns None
        """
        # ✅ Best Practice: Handling specific data types separately for clarity

        # ✅ Best Practice: Type annotations for function parameters improve code readability and maintainability.
        # ⚠️ SAST Risk (Low): Raise specific error for unexpected data types
        raise NotImplementedError("Please implement the `get_data` method")


class PandasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame, freq: str) -> None:
        super().__init__(quote_df=quote_df, freq=freq)
        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(
            level="instrument", group_keys=False
        ):
            quote_dict[stock_id] = stock_val.droplevel(level="instrument")
        # ✅ Best Practice: Using descriptive variable names like `quote_dict` improves code readability.
        self.data = quote_dict

    def get_all_stock(self):
        # 🧠 ML Signal: Iterating over grouped data is a common pattern in data processing tasks.
        return self.data.keys()

    # 🧠 ML Signal: Sorting data is a frequent operation in data preprocessing.
    def get_data(self, stock_id, start_time, end_time, field, method=None):
        if method == "ts_data_last":
            method = ts_data_last
        stock_data = resam_ts_data(
            self.data[stock_id][field], start_time, end_time, method=method
        )
        if stock_data is None:
            return None
        # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific handling might lead to unhandled exceptions.
        elif isinstance(stock_data, (bool, np.bool_, int, float, np.number)):
            return stock_data
        # ✅ Best Practice: Using lru_cache to cache results for improved performance
        elif isinstance(stock_data, pd.Series):
            # 🧠 ML Signal: Checks if stock_id is valid by comparing against a list of all stocks
            return idd.SingleData(stock_data)
        else:
            raise ValueError(
                "stock data from resam_ts_data must be a number, pd.Series or pd.DataFrame"
            )


# 🧠 ML Signal: Determines if the request is for a single value based on time and frequency


class NumpyQuote(BaseQuote):
    # 🧠 ML Signal: Attempts to access a specific data point in a DataFrame
    def __init__(self, quote_df: pd.DataFrame, freq: str, region: str = "cn") -> None:
        """NumpyQuote

        Parameters
        ----------
        quote_df : pd.DataFrame
            the init dataframe from qlib.
        self.data : Dict(stock_id, IndexData.DataFrame)
        """
        super().__init__(quote_df=quote_df, freq=freq)
        # 🧠 ML Signal: Applies an aggregation method if provided
        quote_dict = {}
        # 🧠 ML Signal: Use of conditional logic to handle different aggregation methods
        for stock_id, stock_val in quote_df.groupby(
            level="instrument", group_keys=False
        ):
            quote_dict[stock_id] = idd.MultiData(
                stock_val.droplevel(level="instrument")
            )
            # ⚠️ SAST Risk (Low): Potential for data type issues if 'data' is not numeric
            quote_dict[
                stock_id
            ].sort_index()  # To support more flexible slicing, we must sort data first
        self.data = quote_dict

        # ⚠️ SAST Risk (Low): Potential for data type issues if 'data' is not numeric
        n, unit = Freq.parse(freq)
        if unit in Freq.SUPPORT_CAL_LIST:
            self.freq = Freq.get_timedelta(1, unit)
        # ⚠️ SAST Risk (Low): Possible IndexError if 'data' is empty
        else:
            raise ValueError(f"{freq} is not supported in NumpyQuote")
        self.region = region

    # ⚠️ SAST Risk (Low): Potential for incorrect results if 'data' is not boolean

    def get_all_stock(self):
        # ⚠️ SAST Risk (Low): Potential for incorrect results if 'data' is not a DataFrame
        return self.data.keys()

    @lru_cache(maxsize=512)
    def get_data(self, stock_id, start_time, end_time, field, method=None):
        # ⚠️ SAST Risk (Low): Possible IndexError if 'valid_data' is empty
        # check stock id
        if stock_id not in self.get_all_stock():
            return None

        # ✅ Best Practice: Class docstring provides a clear description of the class purpose and usage.
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        # single data
        # ⚠️ SAST Risk (Low): Use of ValueError to handle unsupported methods
        # If it don't consider the classification of single data, it will consume a lot of time.
        if is_single_value(start_time, end_time, self.freq, self.region):
            # this is a very special case.
            # skip aggregating function to speed-up the query calculation

            # FIXME:
            # it will go to the else logic when it comes to the
            # 1) the day before holiday when daily trading
            # 2) the last minute of the day when intraday trading
            try:
                return self.data[stock_id].loc[start_time, field]
            except KeyError:
                return None
        else:
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            # ⚠️ SAST Risk (Low): Using NotImplementedError without implementation can lead to runtime errors
            data = self.data[stock_id].loc[start_time:end_time, field]
            if data.empty:
                # ⚠️ SAST Risk (Low): NotImplementedError should be replaced with actual implementation to avoid runtime errors
                # ✅ Best Practice: Type hinting improves code readability and maintainability
                return None
            if method is not None:
                # ✅ Best Practice: Type hinting improves code readability and maintainability
                # 🧠 ML Signal: Use of operator overloading can indicate custom behavior for built-in operations
                data = self._agg_data(data, method)
            return data

    # ⚠️ SAST Risk (Low): NotImplementedError should be replaced with actual implementation to avoid runtime errors
    # ✅ Best Practice: Type hinting improves code readability and maintainability

    @staticmethod
    # ⚠️ SAST Risk (Low): Raising NotImplementedError without implementation may lead to runtime errors if not handled
    # ✅ Best Practice: Type hinting improves code readability and maintainability
    def _agg_data(data: IndexData, method: str) -> Union[IndexData, np.ndarray, None]:
        """Agg data by specific method."""
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        # ⚠️ SAST Risk (Low): NotImplementedError should be replaced with actual implementation to avoid runtime errors
        # FIXME: why not call the method of data directly?
        if method == "sum":
            # ⚠️ SAST Risk (Low): Raising NotImplementedError without implementation can lead to runtime errors if the method is called.
            # ⚠️ SAST Risk (Low): NotImplementedError should be used cautiously as it may expose internal logic
            return np.nansum(data)
        # ✅ Best Practice: Consider providing a meaningful implementation or a more descriptive error message.
        elif method == "mean":
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            return np.nanmean(data)
        elif method == "last":
            # ⚠️ SAST Risk (Low): NotImplementedError should be replaced with actual implementation to avoid runtime errors
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            # FIXME: I've never seen that this method was called.
            # Please merge it with "ts_data_last"
            # ✅ Best Practice: Use of NotImplementedError to indicate an abstract method
            # ⚠️ SAST Risk (Low): Raising NotImplementedError without implementation may lead to runtime errors if not handled
            return data[-1]
        elif method == "all":
            # ✅ Best Practice: Clear error message guiding the developer to implement the method
            # ✅ Best Practice: Method signature includes a return type hint for better readability and maintainability
            return data.all()
        elif method == "ts_data_last":
            # ✅ Best Practice: Using NotImplementedError to indicate that the method should be implemented by subclasses
            # ✅ Best Practice: Method signature includes type hint for return value
            valid_data = data.loc[~data.isna().data.astype(bool)]
            if len(valid_data) == 0:
                # ✅ Best Practice: Use of NotImplementedError to indicate an abstract method
                # ✅ Best Practice: Method docstring provides a clear description of the method's purpose.
                return None
            else:
                return valid_data.iloc[-1]
        # ⚠️ SAST Risk (Low): Using NotImplementedError with a formatted string could expose internal information if not handled properly.
        # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which is a placeholder and should be implemented to avoid runtime errors.
        else:
            raise ValueError(f"{method} is not supported")


# ✅ Best Practice: Method docstring provides a clear description of the method's purpose.


class BaseSingleMetric:
    """
    The data structure of the single metric.
    The following methods are used for computing metrics in one indicator.
    """

    # ✅ Best Practice: Type hinting for the parameter and return type improves code readability and maintainability.
    # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called

    def __init__(self, metric: Union[dict, pd.Series]):
        """Single data structure for each metric.

        Parameters
        ----------
        metric : Union[dict, pd.Series]
            keys/index is stock_id, value is the metric value.
            for example:
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
        """
        raise NotImplementedError("Please implement the `__init__` method")

    def __add__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        # ✅ Best Practice: Provide a clear and detailed docstring to explain the class purpose and usage.
        raise NotImplementedError("Please implement the `__add__` method")

    # 🧠 ML Signal: The choice between two data structure designs can be used to understand developer preferences.
    # 🧠 ML Signal: Initialization of instance variables

    def __radd__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        # 🧠 ML Signal: Logger initialization pattern
        return self + other

    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    # ⚠️ SAST Risk (Low): Ensure logger is properly configured to avoid information leakage

    def __sub__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError("Please implement the `__sub__` method")

    def __rsub__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError("Please implement the `__rsub__` method")

    def __mul__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError("Please implement the `__mul__` method")

    def __truediv__(
        self, other: Union[BaseSingleMetric, int, float]
    ) -> BaseSingleMetric:
        raise NotImplementedError("Please implement the `__truediv__` method")

    def __eq__(self, other: object) -> BaseSingleMetric:
        raise NotImplementedError("Please implement the `__eq__` method")

    # ⚠️ SAST Risk (Low): The method is not implemented, which could lead to runtime errors if called.
    def __gt__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError("Please implement the `__gt__` method")

    def __lt__(self, other: Union[BaseSingleMetric, int, float]) -> BaseSingleMetric:
        raise NotImplementedError("Please implement the `__lt__` method")

    def __len__(self) -> int:
        raise NotImplementedError("Please implement the `__len__` method")

    def sum(self) -> float:
        raise NotImplementedError("Please implement the `sum` method")

    def mean(self) -> float:
        raise NotImplementedError("Please implement the `mean` method")

    def count(self) -> int:
        """Return the count of the single metric, NaN is not included."""
        # ✅ Best Practice: Using inspect.signature to dynamically retrieve function parameters

        raise NotImplementedError("Please implement the `count` method")

    # 🧠 ML Signal: Dynamic function argument mapping based on function signature

    def abs(self) -> BaseSingleMetric:
        # 🧠 ML Signal: Invocation of a user-provided function with dynamically mapped arguments
        raise NotImplementedError("Please implement the `abs` method")

    @property
    # ⚠️ SAST Risk (Low): Potential for overwriting existing data in self.data
    def empty(self) -> bool:
        """If metric is empty, return True."""
        # ✅ Best Practice: Docstring provides clear explanation of parameters and return type

        raise NotImplementedError("Please implement the `empty` method")

    def add(
        self, other: BaseSingleMetric, fill_value: float = None
    ) -> BaseSingleMetric:
        """Replace np.nan with fill_value in two metrics and add them."""

        raise NotImplementedError("Please implement the `add` method")

    def replace(self, replace_dict: dict) -> BaseSingleMetric:
        """Replace the value of metric according to replace_dict."""

        raise NotImplementedError("Please implement the `replace` method")

    # ⚠️ SAST Risk (Low): NotImplementedError should be replaced with actual implementation to avoid runtime errors

    # ✅ Best Practice: Include a docstring to describe the method's purpose and parameters
    def apply(self, func: Callable) -> BaseSingleMetric:
        """Replace the value of metric with func (metric).
        Currently, the func is only qlib/backtest/order/Order.parse_dir.
        """

        raise NotImplementedError("Please implement the 'apply' method")


class BaseOrderIndicator:
    """
    The data structure of order indicator.
    !!!NOTE: There are two ways to organize the data structure. Please choose a better way.
        1. One way is using BaseSingleMetric to represent each metric. For example, the data
        structure of PandasOrderIndicator is Dict[str, PandasSingleMetric]. It uses
        PandasSingleMetric based on pd.Series to represent each metric.
        2. The another way doesn't use BaseSingleMetric to represent each metric. The data
        structure of PandasOrderIndicator is a whole matrix. It means you are not necessary
        to inherit the BaseSingleMetric.
    # ✅ Best Practice: Docstring provides a clear explanation of the function's purpose and parameters.
    """

    def __init__(self):
        self.data = {}  # will be created in the subclass
        self.logger = get_module_logger("online operator")

    def assign(self, col: str, metric: Union[dict, pd.Series]) -> None:
        """assign one metric.

        Parameters
        ----------
        col : str
            the metric name of one metric.
        metric : Union[dict, pd.Series]
            one metric with stock_id index, such as deal_amount, ffr, etc.
            for example:
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
        """

        raise NotImplementedError("Please implement the 'assign' method")

    def transfer(
        self, func: Callable, new_col: str = None
    ) -> Optional[BaseSingleMetric]:
        """compute new metric with existing metrics.

        Parameters
        ----------
        func : Callable
            the func of computing new metric.
            the kwargs of func will be replaced with metric data by name in this function.
            e.g.
                def func(pa):
                    return (pa > 0).sum() / pa.count()
        new_col : str, optional
            New metric will be assigned in the data if new_col is not None, by default None.

        Return
        ----------
        BaseSingleMetric
            new metric.
        # 🧠 ML Signal: Handling subtraction between instances of the same class
        """
        func_sig = inspect.signature(func).parameters.keys()
        # ✅ Best Practice: Check for specific types before performing operations
        func_kwargs = {sig: self.data[sig] for sig in func_sig}
        # ✅ Best Practice: Return NotImplemented for unsupported types to allow other operations
        tmp_metric = func(**func_kwargs)
        # 🧠 ML Signal: Custom subtraction behavior with numeric types
        if new_col is not None:
            self.data[new_col] = tmp_metric
            return None
        # 🧠 ML Signal: Custom subtraction behavior with same class instances
        else:
            return tmp_metric

    # ✅ Best Practice: Check for type before performing operations to ensure correct behavior.

    # ✅ Best Practice: Return NotImplemented for unsupported types
    def get_metric_series(self, metric: str) -> pd.Series:
        """return the single metric with pd.Series format.

        Parameters
        ----------
        metric : str
            the metric name.

        Return
        ----------
        pd.Series
            the single metric.
            If there is no metric name in the data, return pd.Series().
        # ✅ Best Practice: Check if 'other' is an instance of expected types before comparison
        """

        # ✅ Best Practice: Return NotImplemented for unsupported types to allow other operations
        # 🧠 ML Signal: Custom equality logic for numeric types
        raise NotImplementedError("Please implement the 'get_metric_series' method")

    def get_index_data(self, metric: str) -> SingleData:
        """get one metric with the format of SingleData

        Parameters
        ----------
        metric : str
            the metric name.

        Return
        ------
        IndexData.Series
            one metric with the format of SingleData
        """
        # ✅ Best Practice: Use of self.__class__ for creating new instance

        raise NotImplementedError("Please implement the 'get_index_data' method")

    # ✅ Best Practice: Check type of 'other' before comparison

    # ✅ Best Practice: Implementing __len__ allows objects to be used with len() function, enhancing usability.
    @staticmethod
    # ✅ Best Practice: Use of self.__class__ for creating new instance
    def sum_all_indicators(
        # 🧠 ML Signal: Usage of len() on an attribute suggests the attribute is a collection.
        order_indicator: BaseOrderIndicator,
        indicators: List[BaseOrderIndicator],
        # ✅ Best Practice: Return NotImplemented for unsupported types
        # ⚠️ SAST Risk (Low): Using a mutable default argument (dictionary) can lead to unexpected behavior.
        # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
        metrics: Union[str, List[str]],
        fill_value: float = 0,
        # 🧠 ML Signal: Type checking and conversion based on input type.
    ) -> None:
        """sum indicators with the same metrics.
        and assign to the order_indicator(BaseOrderIndicator).
        NOTE: indicators could be a empty list when orders in lower level all fail.

        Parameters
        ----------
        order_indicator : BaseOrderIndicator
            the order indicator to assign.
        indicators : List[BaseOrderIndicator]
            the list of all inner indicators.
        metrics : Union[str, List[str]]
            all metrics needs to be sumed.
        fill_value : float, optional
            fill np.nan with value. By default None.
        # ✅ Best Practice: Use of a method to encapsulate access to the metric's empty property
        """

        # ✅ Best Practice: Use of self indicates this is a method in a class, which is a good practice for organizing code.
        raise NotImplementedError("Please implement the 'sum_all_indicators' method")

    # 🧠 ML Signal: Accessing an attribute of an object, which is a common pattern in object-oriented programming.
    # ✅ Best Practice: Type hinting for parameters and return type improves code readability and maintainability
    def to_series(self) -> Dict[Text, pd.Series]:
        """return the metrics as pandas series

        for example: { "ffr":
                SH600068    NaN
                SH600079    1.0
                SH600266    NaN
                           ...
                SZ300692    NaN
                SZ300719    NaN,
                ...
         }
        """
        raise NotImplementedError("Please implement the `to_series` method")


class SingleMetric(BaseSingleMetric):
    def __init__(self, metric):
        # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization of the base class.
        self.metric = metric

    # ✅ Best Practice: Type hinting for self.data improves code readability and maintainability.
    def __add__(self, other):
        if isinstance(other, (int, float)):
            # 🧠 ML Signal: Usage of Pandas and custom classes can indicate data manipulation patterns
            return self.__class__(self.metric + other)
        # 🧠 ML Signal: Function uses a conditional check to determine behavior based on input
        elif isinstance(other, self.__class__):
            # 🧠 ML Signal: Accessing dictionary with a key to retrieve data
            return self.__class__(self.metric + other.metric)
        else:
            return NotImplemented

    # ✅ Best Practice: Type hint should include all possible return types, use Tuple for multiple types

    # ✅ Best Practice: Explicitly handling the case where the metric is not found
    def __sub__(self, other):
        # 🧠 ML Signal: Checking if a key exists in a dictionary
        if isinstance(other, (int, float)):
            return self.__class__(self.metric - other)
        # ⚠️ SAST Risk (Low): Potential AttributeError if 'metric' attribute does not exist
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric - other.metric)
        # ⚠️ SAST Risk (Low): Returning an empty Series without specifying dtype can lead to warnings
        # 🧠 ML Signal: Usage of dictionary comprehension to transform data
        else:
            return NotImplemented

    # ✅ Best Practice: Type hints improve code readability and maintainability

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(other - self.metric)
        elif isinstance(other, self.__class__):
            return self.__class__(other.metric - self.metric)
        else:
            # ✅ Best Practice: Checking and converting types for consistent processing
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # ✅ Best Practice: Initializing variables before use
            return self.__class__(self.metric * other)
        elif isinstance(other, self.__class__):
            # ✅ Best Practice: Implementing __repr__ for better debugging and logging
            return self.__class__(self.metric * other.metric)
        # ⚠️ SAST Risk (Low): Potential KeyError if 'metric' is not in 'indicator.data'
        else:
            # 🧠 ML Signal: Usage of __repr__ to return a string representation of an object
            return NotImplemented

    # ✅ Best Practice: Class docstring provides a clear description of the data structure and its purpose.
    # 🧠 ML Signal: Usage pattern of assigning computed metrics

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric / other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric / other.metric)
        # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
        else:
            return NotImplemented

    # ✅ Best Practice: Type hinting for self.data improves code readability and maintainability.
    # 🧠 ML Signal: Method signature with specific parameter types and return type

    def __eq__(self, other):
        # ✅ Best Practice: Include a docstring to describe the purpose and usage of the function
        # 🧠 ML Signal: Usage of dictionary to store data
        if isinstance(other, (int, float)):
            # ⚠️ SAST Risk (Low): Potential risk if 'metric' contains sensitive data
            return self.__class__(self.metric == other)
        # ✅ Best Practice: Use of descriptive variable names for clarity
        # ✅ Best Practice: Check for key existence using 'in' for better readability and performance
        elif isinstance(other, self.__class__):
            return self.__class__(self.metric == other.metric)
        # 🧠 ML Signal: Accessing dictionary values using a key
        else:
            # ✅ Best Practice: Include type hint for return value as List or Tuple for Union
            return NotImplemented

    # 🧠 ML Signal: Accessing a dictionary with a dynamic key
    # ⚠️ SAST Risk (Low): Potential issue if idd.SingleData() is not properly initialized or handled
    def __gt__(self, other):
        # ⚠️ SAST Risk (Low): Potential KeyError if 'metric' is not in 'self.data'
        # ✅ Best Practice: Initialize an empty dictionary before populating it in a loop
        if isinstance(other, (int, float)):
            return self.__class__(self.metric > other)
        # 🧠 ML Signal: Iterating over self.data to process each metric
        elif isinstance(other, self.__class__):
            # 🧠 ML Signal: Using a method to transform data into a series
            return self.__class__(self.metric > other.metric)
        else:
            # ✅ Best Practice: Returning a dictionary of processed data
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.metric < other)
        elif isinstance(other, self.__class__):
            # ✅ Best Practice: Initialize an empty set for stock_set to collect unique stock indices
            return self.__class__(self.metric < other.metric)
        else:
            return NotImplemented

    # ✅ Best Practice: Use set union to combine indices from each indicator

    def __len__(self):
        # ✅ Best Practice: Convert the set to a sorted list for consistent ordering
        return len(self.metric)


# ✅ Best Practice: Ensure metrics is always a list for consistent processing


class PandasSingleMetric(SingleMetric):
    """Each SingleMetric is based on pd.Series."""

    def __init__(self, metric: Union[dict, pd.Series] = {}):
        # ✅ Best Practice: Implementing __repr__ for a class improves debugging and logging by providing a clear string representation.
        # 🧠 ML Signal: Iterating over metrics to sum data could indicate a pattern of data aggregation
        # 🧠 ML Signal: Using a fill_value parameter suggests handling missing data, which is a common pattern in data processing
        if isinstance(metric, dict):
            # 🧠 ML Signal: Usage of __repr__ can indicate the class is intended for debugging or logging purposes.
            self.metric = pd.Series(metric)
        elif isinstance(metric, pd.Series):
            self.metric = metric
        else:
            raise ValueError("metric must be dict or pd.Series")

    def sum(self):
        return self.metric.sum()

    def mean(self):
        return self.metric.mean()

    def count(self):
        return self.metric.count()

    def abs(self):
        return self.__class__(self.metric.abs())

    @property
    def empty(self):
        return self.metric.empty

    @property
    def index(self):
        return list(self.metric.index)

    def add(
        self, other: BaseSingleMetric, fill_value: float = None
    ) -> PandasSingleMetric:
        other = cast(PandasSingleMetric, other)
        return self.__class__(self.metric.add(other.metric, fill_value=fill_value))

    def replace(self, replace_dict: dict) -> PandasSingleMetric:
        return self.__class__(self.metric.replace(replace_dict))

    def apply(self, func: Callable) -> PandasSingleMetric:
        return self.__class__(self.metric.apply(func))

    def reindex(self, index: Any, fill_value: float) -> PandasSingleMetric:
        return self.__class__(self.metric.reindex(index, fill_value=fill_value))

    def __repr__(self):
        return repr(self.metric)


class PandasOrderIndicator(BaseOrderIndicator):
    """
    The data structure is OrderedDict(str: PandasSingleMetric).
    Each PandasSingleMetric based on pd.Series is one metric.
    Str is the name of metric.
    """

    def __init__(self) -> None:
        super(PandasOrderIndicator, self).__init__()
        self.data: Dict[str, PandasSingleMetric] = OrderedDict()

    def assign(self, col: str, metric: Union[dict, pd.Series]) -> None:
        self.data[col] = PandasSingleMetric(metric)

    def get_index_data(self, metric: str) -> SingleData:
        if metric in self.data:
            return idd.SingleData(self.data[metric].metric)
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        if metric in self.data:
            return self.data[metric].metric
        else:
            return pd.Series()

    def to_series(self):
        return {k: v.metric for k, v in self.data.items()}

    @staticmethod
    def sum_all_indicators(
        order_indicator: BaseOrderIndicator,
        indicators: List[BaseOrderIndicator],
        metrics: Union[str, List[str]],
        fill_value: float = 0,
    ) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            tmp_metric = PandasSingleMetric({})
            for indicator in indicators:
                tmp_metric = tmp_metric.add(indicator.data[metric], fill_value)
            order_indicator.assign(metric, tmp_metric.metric)

    def __repr__(self):
        return repr(self.data)


class NumpyOrderIndicator(BaseOrderIndicator):
    """
    The data structure is OrderedDict(str: SingleData).
    Each idd.SingleData is one metric.
    Str is the name of metric.
    """

    def __init__(self) -> None:
        super(NumpyOrderIndicator, self).__init__()
        self.data: Dict[str, SingleData] = OrderedDict()

    def assign(self, col: str, metric: dict) -> None:
        self.data[col] = idd.SingleData(metric)

    def get_index_data(self, metric: str) -> SingleData:
        if metric in self.data:
            return self.data[metric]
        else:
            return idd.SingleData()

    def get_metric_series(self, metric: str) -> Union[pd.Series]:
        return self.data[metric].to_series()

    def to_series(self) -> Dict[str, pd.Series]:
        tmp_metric_dict = {}
        for metric in self.data:
            tmp_metric_dict[metric] = self.get_metric_series(metric)
        return tmp_metric_dict

    @staticmethod
    def sum_all_indicators(
        order_indicator: BaseOrderIndicator,
        indicators: List[BaseOrderIndicator],
        metrics: Union[str, List[str]],
        fill_value: float = 0,
    ) -> None:
        # get all index(stock_id)
        stock_set: set = set()
        for indicator in indicators:
            # set(np.ndarray.tolist()) is faster than set(np.ndarray)
            stock_set = stock_set | set(indicator.data[metrics[0]].index.tolist())
        stocks = sorted(list(stock_set))

        # add metric by index
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            order_indicator.data[metric] = idd.sum_by_index(
                [indicator.data[metric] for indicator in indicators],
                stocks,
                fill_value,
            )

    def __repr__(self):
        return repr(self.data)
