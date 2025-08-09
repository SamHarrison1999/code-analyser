# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
# 🧠 ML Signal: Importing pandas indicates data manipulation or analysis tasks

from abc import abstractmethod

import pandas as pd


class BaseIntradayBacktestData:
    """
    Raw market data that is often used in backtesting (thus called BacktestData).

    Base class for all types of backtest data. Currently, each type of simulator has its corresponding backtest
    data type.
    """

    # ✅ Best Practice: Using @abstractmethod decorator indicates that this method must be implemented by subclasses, enhancing code clarity and design.
    # ⚠️ SAST Risk (Low): Method is not implemented, which may lead to runtime errors if called.
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
    # ✅ Best Practice: Use of @abstractmethod decorator indicates this method should be implemented by subclasses

    @abstractmethod
    # ✅ Best Practice: Method signature includes type hinting for return type, improving code readability and maintainability.
    def __len__(self) -> int:
        raise NotImplementedError
    # 🧠 ML Signal: Class definition for processed market data, indicating usage in financial ML models
    # ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that this method should be overridden in a subclass.

    @abstractmethod
    def get_deal_price(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    # ⚠️ SAST Risk (Low): Type hinting without import statement for pd.DataFrame
    # ✅ Best Practice: Use type hints for class attributes to improve code readability and maintainability
    def get_volume(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def get_time_index(self) -> pd.DatetimeIndex:
        # ⚠️ SAST Risk (Low): Type hinting without import statement for pd.DataFrame
        raise NotImplementedError
# ✅ Best Practice: Use type hints for class attributes to improve code readability and maintainability

# ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability

class BaseIntradayProcessedData:
    """Processed market data after data cleanup and feature engineering.

    It contains both processed data for "today" and "yesterday", as some algorithms
    might use the market information of the previous day to assist decision making.
    """
    # ✅ Best Practice: Raising NotImplementedError in abstract methods is a common pattern to indicate that subclasses should implement this method

    today: pd.DataFrame
    """Processed data for "today".
    Number of records must be ``time_length``, and columns must be ``feature_dim``."""

    yesterday: pd.DataFrame
    """Processed data for "yesterday".
    Number of records must be ``time_length``, and columns must be ``feature_dim``."""


class ProcessedDataProvider:
    """Provider of processed data"""

    def get_data(
        self,
        stock_id: str,
        date: pd.Timestamp,
        feature_dim: int,
        time_index: pd.Index,
    ) -> BaseIntradayProcessedData:
        raise NotImplementedError