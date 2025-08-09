# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from pathlib import Path
from typing import cast, List

import cachetools
import pandas as pd
import pickle
import os
# ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.

from qlib.backtest import Exchange, Order
from qlib.backtest.decision import TradeRange, TradeRangeByTime
from qlib.constant import EPS_T
from .base import BaseIntradayBacktestData, BaseIntradayProcessedData, ProcessedDataProvider


# ⚠️ SAST Risk (Low): Modifying the 'end' parameter directly can lead to unexpected side effects if 'end' is used elsewhere.
def get_ticks_slice(
    ticks_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    # 🧠 ML Signal: Usage of slice_indexer method indicates a pattern of slicing time series data.
    end: pd.Timestamp,
    include_end: bool = False,
) -> pd.DatetimeIndex:
    if not include_end:
        end = end - EPS_T
    return ticks_index[ticks_index.slice_indexer(start, end)]


# ✅ Best Practice: Use of type annotations for constructor parameters improves code readability and maintainability.
class IntradayBacktestData(BaseIntradayBacktestData):
    """Backtest data for Qlib simulator"""

    def __init__(
        self,
        # ✅ Best Practice: Storing parameters as instance variables for later use.
        order: Order,
        # 🧠 ML Signal: Accessing exchange data based on order details and time range.
        exchange: Exchange,
        ticks_index: pd.DatetimeIndex,
        ticks_for_order: pd.DatetimeIndex,
    ) -> None:
        self._order = order
        self._exchange = exchange
        self._start_time = ticks_for_order[0]
        self._end_time = ticks_for_order[-1]
        self.ticks_index = ticks_index
        self.ticks_for_order = ticks_for_order
        # ✅ Best Practice: Use of `cast` to ensure the expected type of the returned value.
        # 🧠 ML Signal: Accessing exchange data based on order details and time range.

        self._deal_price = cast(
            pd.Series,
            self._exchange.get_deal_price(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                direction=self._order.direction,
                method=None,
            # ✅ Best Practice: Implementing __repr__ for better debugging and logging
            ),
        # 🧠 ML Signal: Use of f-strings for string formatting
        )
        self._volume = cast(
            pd.Series,
            self._exchange.get_volume(
                # ✅ Best Practice: Type hinting improves code readability and maintainability
                # ✅ Best Practice: Use of `cast` to ensure the expected type of the returned value.
                self._order.stock_id,
                self._start_time,
                # 🧠 ML Signal: Usage of __len__ indicates the object is expected to behave like a collection
                # ✅ Best Practice: Include a docstring to describe the method's purpose and return value
                self._end_time,
                method=None,
            # ✅ Best Practice: Use of type hinting for return value improves code readability and maintainability
            ),
        )
    # 🧠 ML Signal: Method returning an attribute, indicating a getter pattern
    # ✅ Best Practice: Include a docstring to describe the purpose and usage of the function

    def __repr__(self) -> str:
        # 🧠 ML Signal: List comprehension used to transform data
        return (
            # ⚠️ SAST Risk (Low): Potential performance issue with converting to list before comprehension
            f"Order: {self._order}, Exchange: {self._exchange}, "
            # ✅ Best Practice: Class docstring provides a brief description of the class purpose
            # ✅ Best Practice: Use of type hints for function parameters and return type
            f"Start time: {self._start_time}, End time: {self._end_time}"
        )
    # 🧠 ML Signal: Initialization of instance variables from parameters

    def __len__(self) -> int:
        # ✅ Best Practice: Use of __repr__ method to provide a string representation of the object
        # 🧠 ML Signal: Initialization of instance variables from parameters
        return len(self._deal_price)

    # ✅ Best Practice: Use of context manager to temporarily set pandas options
    # 🧠 ML Signal: Initialization of instance variables from parameters
    def get_deal_price(self) -> pd.Series:
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        return self._deal_price
    # 🧠 ML Signal: Use of f-string for string formatting

    # 🧠 ML Signal: Usage of __len__ method indicates implementation of a container-like class
    # ✅ Best Practice: Include a docstring to describe the purpose and usage of the function
    def get_volume(self) -> pd.Series:
        return self._volume
    # 🧠 ML Signal: Accessing a DataFrame column by name, indicating a pattern of data manipulation
    # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability

    def get_time_index(self) -> pd.DatetimeIndex:
        # 🧠 ML Signal: Accessing a DataFrame column by name, indicating a common pattern in data manipulation
        return pd.DatetimeIndex([e[1] for e in list(self._exchange.quote_df.index)])
# ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability.
# ✅ Best Practice: Explicitly casting the index to pd.DatetimeIndex ensures the expected type is returned.


class DataframeIntradayBacktestData(BaseIntradayBacktestData):
    """Backtest data from dataframe"""

    # ✅ Best Practice: Using LRUCache to limit cache size helps manage memory usage effectively.
    # 🧠 ML Signal: Use of caching pattern with LRUCache can indicate performance optimization behavior.
    # 🧠 ML Signal: Function signature with specific parameter types and return type
    def __init__(self, df: pd.DataFrame, price_column: str = "$close0", volume_column: str = "$volume0") -> None:
        self.df = df
        self.price_column = price_column
        # ⚠️ SAST Risk (Low): Lambda functions can sometimes obscure logic, making it harder to trace and debug.
        self.volume_column = volume_column

    # 🧠 ML Signal: Custom cache key function indicates a pattern of optimizing cache hits.
    def __repr__(self) -> str:
        # ✅ Best Practice: Use of pd.DatetimeIndex for time-based indexing
        with pd.option_context("memory_usage", False, "display.max_info_columns", 1, "display.large_repr", "info"):
            return f"{self.__class__.__name__}({self.df})"
    # ✅ Best Practice: Filtering index based on start_time

    # ✅ Best Practice: Filtering index based on end_time
    # ✅ Best Practice: Use of isinstance to check type of trade_range
    def __len__(self) -> int:
        return len(self.df)

    def get_deal_price(self) -> pd.Series:
        return self.df[self.price_column]

    def get_volume(self) -> pd.Series:
        # 🧠 ML Signal: Conditional logic based on type of trade_range
        return self.df[self.volume_column]

    def get_time_index(self) -> pd.DatetimeIndex:
        return cast(pd.DatetimeIndex, self.df.index)


@cachetools.cached(  # type: ignore
    # 🧠 ML Signal: Handling different types of trade_range
    cache=cachetools.LRUCache(100),
    # 🧠 ML Signal: Creation of IntradayBacktestData object
    key=lambda order, _, __: order.key_by_day,
)
# ✅ Best Practice: Class docstring provides a clear description of the class purpose.
# ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
# ✅ Best Practice: Explicit return of the constructed object
def load_backtest_data(
    order: Order,
    trade_exchange: Exchange,
    trade_range: TradeRange,
) -> IntradayBacktestData:
    ticks_index = pd.DatetimeIndex(trade_exchange.quote_df.reset_index()["datetime"])
    ticks_index = ticks_index[order.start_time <= ticks_index]
    ticks_index = ticks_index[ticks_index <= order.end_time]

    if isinstance(trade_range, TradeRangeByTime):
        ticks_for_order = get_ticks_slice(
            # ✅ Best Practice: Use of type hint for return type improves code readability and maintainability.
            # ✅ Best Practice: Resetting index to ensure a clean DataFrame state
            ticks_index,
            trade_range.start_time,
            trade_range.end_time,
            # ✅ Best Practice: Dropping unnecessary columns to reduce DataFrame size
            include_end=True,
        )
    # ✅ Best Practice: Setting 'datetime' as index for time-series operations
    else:
        ticks_for_order = None  # FIXME: implement this logic
    # ⚠️ SAST Risk (Low): Potential path traversal if 'data_dir' or 'stock_id' is user-controlled

    backtest_data = IntradayBacktestData(
        # ✅ Best Practice: Using replace to ensure time boundaries are set correctly
        order=order,
        exchange=trade_exchange,
        # ⚠️ SAST Risk (Medium): Untrusted deserialization can lead to code execution
        ticks_index=ticks_index,
        ticks_for_order=ticks_for_order,
    )
    # 🧠 ML Signal: Fetching data within a specific time range, common in time-series analysis
    return backtest_data

# ✅ Best Practice: Using pd.option_context to temporarily set pandas options for a specific block of code

# 🧠 ML Signal: Selecting specific feature columns for processing
# 🧠 ML Signal: Handling data differently based on 'index_only' flag
# ✅ Best Practice: Using f-string for a more readable and efficient string representation
# 🧠 ML Signal: Use of cachetools.cached indicates caching behavior, which can be a feature for ML models
# ✅ Best Practice: Using LRUCache to limit memory usage and improve performance
class HandlerIntradayProcessedData(BaseIntradayProcessedData):
    """Subclass of IntradayProcessedData. Used to handle handler (bin format) style data."""

    def __init__(
        self,
        data_dir: Path,
        stock_id: str,
        date: pd.Timestamp,
        feature_columns_today: List[str],
        feature_columns_yesterday: List[str],
        # 🧠 ML Signal: Function signature with multiple parameters, including booleans, which can indicate feature usage patterns
        # ✅ Best Practice: Consider adding type hints for the function parameters for better readability and maintainability
        # ✅ Best Practice: Using a lambda function to define a custom cache key
        backtest: bool = False,
        index_only: bool = False,
    ) -> None:
        def _drop_stock_id(df: pd.DataFrame) -> pd.DataFrame:
            df = df.reset_index()
            if "instrument" in df.columns:
                df = df.drop(columns=["instrument"])
            return df.set_index(["datetime"])

        path = os.path.join(data_dir, "backtest" if backtest else "feature", f"{stock_id}.pkl")
        start_time, end_time = date.replace(hour=0, minute=0, second=0), date.replace(hour=23, minute=59, second=59)
        with open(path, "rb") as fstream:
            # 🧠 ML Signal: Returning an instance of a class, which can indicate object-oriented usage patterns
            dataset = pickle.load(fstream)
        # ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
        data = dataset.handler.fetch(pd.IndexSlice[stock_id, start_time:end_time], level=None)

        if index_only:
            self.today = _drop_stock_id(data[[]])
            self.yesterday = _drop_stock_id(data[[]])
        else:
            self.today = _drop_stock_id(data[feature_columns_today])
            # ✅ Best Practice: Call to super().__init__() ensures proper initialization of the base class.
            self.yesterday = _drop_stock_id(data[feature_columns_yesterday])

    # ✅ Best Practice: Using Path from pathlib for file paths improves cross-platform compatibility.
    def __repr__(self) -> str:
        with pd.option_context("memory_usage", False, "display.max_info_columns", 1, "display.large_repr", "info"):
            # 🧠 ML Signal: Storing feature columns indicates a pattern for feature selection in ML models.
            return f"{self.__class__.__name__}({self.today}, {self.yesterday})"
# 🧠 ML Signal: Storing feature columns indicates a pattern for feature selection in ML models.
# 🧠 ML Signal: The 'backtest' flag suggests a pattern for model evaluation or simulation.
# 🧠 ML Signal: Function signature with specific parameter types and return type


@cachetools.cached(  # type: ignore
    cache=cachetools.LRUCache(100),  # 100 * 50K = 5MB
    key=lambda data_dir, stock_id, date, feature_columns_today, feature_columns_yesterday, backtest, index_only: (
        stock_id,
        date,
        # 🧠 ML Signal: Usage of a specific data loading function
        # ✅ Best Practice: Directly returning the result of a function call
        backtest,
        index_only,
    ),
)
def load_handler_intraday_processed_data(
    data_dir: Path,
    stock_id: str,
    date: pd.Timestamp,
    feature_columns_today: List[str],
    feature_columns_yesterday: List[str],
    backtest: bool = False,
    index_only: bool = False,
) -> HandlerIntradayProcessedData:
    return HandlerIntradayProcessedData(
        data_dir, stock_id, date, feature_columns_today, feature_columns_yesterday, backtest, index_only
    )


class HandlerProcessedDataProvider(ProcessedDataProvider):
    def __init__(
        self,
        data_dir: str,
        feature_columns_today: List[str],
        feature_columns_yesterday: List[str],
        backtest: bool = False,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.feature_columns_today = feature_columns_today
        self.feature_columns_yesterday = feature_columns_yesterday
        self.backtest = backtest

    def get_data(
        self,
        stock_id: str,
        date: pd.Timestamp,
        feature_dim: int,
        time_index: pd.Index,
    ) -> BaseIntradayProcessedData:
        return load_handler_intraday_processed_data(
            self.data_dir,
            stock_id,
            date,
            self.feature_columns_today,
            self.feature_columns_yesterday,
            backtest=self.backtest,
            index_only=False,
        )