# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
from dateutil.tz import tzlocal

# ⚠️ SAST Risk (Low): Using __file__ can be risky if the script is frozen by a tool like PyInstaller.
import qlib
from qlib.data import D
# 🧠 ML Signal: Modifying sys.path to include parent directories is a common pattern for dynamic module loading.
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    get_us_stock_symbols,
    # ⚠️ SAST Risk (Medium): Hardcoded URL can be a security risk if not properly validated or sanitized.
    # ✅ Best Practice: Class-level constants should be documented or named descriptively.
    get_in_stock_symbols,
    get_br_stock_symbols,
    generate_minutes_calendar_from_daily,
    calc_adjusted_price,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class YahooCollector(BaseCollector):
    retry = 5  # Configuration attribute.  How many times will it try to re-request the data if the network fails.

    def __init__(
        # ✅ Best Practice: Docstring provides clear parameter descriptions and defaults
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(YahooCollector, self).__init__(
            # 🧠 ML Signal: Initialization of instance variables and method calls in constructor
            # 🧠 ML Signal: Conditional logic based on interval type can indicate usage patterns for different intervals
            save_dir=save_dir,
            start=start,
            # ✅ Best Practice: Use of max function to ensure start_datetime is not before a default value
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific handling could lead to unhandled exceptions
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        # ✅ Best Practice: Consider specifying the expected return type for better readability and maintainability
        # ✅ Best Practice: Converting datetime to a specific timezone for consistency
        )

        # ✅ Best Practice: Converting datetime to a specific timezone for consistency
        self.init_datetime()
    # ⚠️ SAST Risk (Low): Potential timezone-related issues if timezone is not validated
    # 🧠 ML Signal: Usage of pd.Timestamp for datetime conversion

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            # 🧠 ML Signal: Conversion of timestamp to local timezone
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        # ✅ Best Practice: Consider logging the exception for better debugging
        else:
            # ✅ Best Practice: Ensure the function returns a consistent type
            # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which could lead to unhandled exceptions if not properly overridden.
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        # 🧠 ML Signal: Use of f-string for error message formatting
        # ✅ Best Practice: Use of @staticmethod decorator indicates that the method does not depend on instance-specific data.
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    # ✅ Best Practice: Use of a function to encapsulate logging logic for reusability and clarity
    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        # ⚠️ SAST Risk (Low): Potential exposure of sensitive information in logs
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        # 🧠 ML Signal: Use of try-except block to handle exceptions
        except ValueError as e:
            pass
        # 🧠 ML Signal: Use of external library function call
        return dt

    # 🧠 ML Signal: DataFrame manipulation pattern
    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        def _show_logging_func():
            if interval == YahooCollector.INTERVAL_1min and show_1min_logging:
                # ⚠️ SAST Risk (Low): Generic exception handling may hide specific errors
                # ⚠️ SAST Risk (Low): Potential exposure of sensitive information in logs
                logger.warning(f"{error_msg}:{_resp}")

        interval = "1m" if interval in ["1m", "1min"] else interval
        try:
            # ✅ Best Practice: Use of a decorator to handle retries indicates a robust design for network operations
            _resp = Ticker(symbol, asynchronous=False).history(interval=interval, start=start, end=end)
            # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
            if isinstance(_resp, pd.DataFrame):
                return _resp.reset_index()
            # ⚠️ SAST Risk (Low): Potential use of uninitialized variable 'interval'
            elif isinstance(_resp, dict):
                _temp_data = _resp.get(symbol, {})
                if isinstance(_temp_data, str) or (
                    isinstance(_resp, dict) and _temp_data.get("indicators", {}).get("quote", None) is None
                ):
                    _show_logging_func()
            else:
                # ⚠️ SAST Risk (Low): Check for None or empty response to handle potential errors
                _show_logging_func()
        except Exception as e:
            logger.warning(
                f"get data error: {symbol}--{start}--{end}"
                + "Your data request fails. This may be caused by your firewall (e.g. GFW). Please switch your network if you want to access Yahoo! data"
            )

    # 🧠 ML Signal: Conditional logic based on 'interval' value
    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple(start_, end_):
            self.sleep()
            # ✅ Best Practice: Consider logging the exception for debugging purposes
            _remote_interval = "1m" if interval == self.INTERVAL_1min else interval
            resp = self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                # 🧠 ML Signal: Looping pattern with date range
                start=start_,
                end=end_,
            )
            if resp is None or resp.empty:
                raise ValueError(
                    f"get data error: {symbol}--{start_}--{end_}" + "The stock may be delisted, please check"
                )
            return resp
        # ✅ Best Practice: Consider logging the exception for debugging purposes

        _result = None
        if interval == self.INTERVAL_1d:
            # ✅ Best Practice: Consider using a more descriptive docstring to explain the method's purpose and behavior.
            try:
                # ✅ Best Practice: Use of super() is a good practice for calling methods from a parent class.
                # 🧠 ML Signal: Data concatenation and sorting pattern
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                # ✅ Best Practice: Method docstring provides a brief description of the method's purpose
                pass
        # ⚠️ SAST Risk (Low): Unhandled interval values could lead to unexpected behavior
        elif interval == self.INTERVAL_1min:
            # ✅ Best Practice: Use of @abc.abstractmethod indicates this method must be implemented by subclasses, which is a good design practice for abstract classes.
            _res = []
            # ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that a method should be overridden
            # 🧠 ML Signal: Return pattern based on condition
            _start = self.start_datetime
            # ✅ Best Practice: Inheriting from ABC indicates that this class is intended to be abstract.
            # 🧠 ML Signal: Logging usage pattern for monitoring and debugging
            while _start < self.end_datetime:
                _tmp_end = min(_start + pd.Timedelta(days=7), self.end_datetime)
                # 🧠 ML Signal: Logging usage pattern for monitoring and debugging
                try:
                    _resp = _get_simple(_start, _tmp_end)
                    # 🧠 ML Signal: Function call pattern for retrieving stock symbols
                    _res.append(_resp)
                except ValueError as e:
                    # 🧠 ML Signal: Logging usage pattern for monitoring and debugging
                    # 🧠 ML Signal: Function that processes and normalizes input data
                    pass
                _start = _tmp_end
            # 🧠 ML Signal: Return statement pattern for function output
            # ✅ Best Practice: Use of f-string for readability and performance
            if _res:
                _result = pd.concat(_res, sort=False).sort_values(["symbol", "date"])
        # ✅ Best Practice: Consider using a constant or configuration for timezone values to improve maintainability.
        else:
            raise ValueError(f"cannot support {self.interval}")
        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(YahooCollector, self).collector_data()
        # 🧠 ML Signal: Iterating over a dictionary of index names and codes
        self.download_index_data()

    # 🧠 ML Signal: Logging information about the index being processed
    @abc.abstractmethod
    # ⚠️ SAST Risk (Medium): No timeout specified for requests.get, which can lead to hanging connections
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")


class YahooCollectorCN(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    # 🧠 ML Signal: Logging warning when an exception occurs
    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        # ✅ Best Practice: Explicitly setting DataFrame column names for clarity
        return symbol

    # ✅ Best Practice: Converting date strings to datetime objects for better manipulation
    @property
    def _timezone(self):
        # ✅ Best Practice: Using astype with errors="ignore" to safely convert data types
        return "Asia/Shanghai"


# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage
class YahooCollectorCN1d(YahooCollectorCN):
    # ✅ Best Practice: Method name should be descriptive of its functionality
    def download_index_data(self):
        # ✅ Best Practice: Checking if a file exists before reading it
        # TODO: from MSN
        # 🧠 ML Signal: Use of inheritance and method overriding
        _format = "%Y%m%d"
        # ✅ Best Practice: Define a method to download index data, but currently it's a placeholder with no implementation.
        _begin = self.start_datetime.strftime(_format)
        # ✅ Best Practice: Using pd.concat to append new data to existing DataFrame
        # ✅ Best Practice: Returning a modified list of symbols
        _end = self.end_datetime.strftime(_format)
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903", "csi500": "000905"}.items():
            # ✅ Best Practice: Saving DataFrame to CSV without the index for cleaner output
            # ✅ Best Practice: Inheriting from ABC indicates this class is intended to be abstract
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            # 🧠 ML Signal: Logging usage pattern for monitoring and debugging
            try:
                # ⚠️ SAST Risk (Low): Fixed sleep duration can lead to inefficient waiting
                # 🧠 ML Signal: Function call pattern for data retrieval
                df = pd.DataFrame(
                    map(
                        lambda x: x.split(","),
                        requests.get(
                            INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end), timeout=None
                        ).json()["data"]["klines"],
                    )
                # ✅ Best Practice: Define a method with a clear purpose, even if not yet implemented
                # 🧠 ML Signal: Logging usage pattern for monitoring and debugging
                )
            except Exception as e:
                # ✅ Best Practice: Explicitly returning a value from a function
                # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
                logger.warning(f"get {_index_name} error: {e}")
                # 🧠 ML Signal: Usage of helper function to transform data
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
            # ✅ Best Practice: Consider using a constant or configuration for timezone values to improve maintainability.
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            # ✅ Best Practice: Use of inheritance to extend functionality from a parent class
            df["adjclose"] = df["close"]
            df["symbol"] = f"sh{_index_code}"
            # ✅ Best Practice: Use of inheritance to extend functionality of YahooCollectorUS
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                # ✅ Best Practice: Inheriting from both YahooCollector and ABC indicates use of abstract base classes for interface enforcement.
                _old_df = pd.read_csv(_path)
                # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
                df = pd.concat([_old_df, df], sort=False)
            df.to_csv(_path, index=False)
            # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
            time.sleep(5)

# 🧠 ML Signal: Function call pattern for retrieving stock symbols

# ✅ Best Practice: Define a method to download index data, even if not yet implemented
class YahooCollectorCN1min(YahooCollectorCN):
    # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
    def get_instrument_list(self):
        # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
        # ✅ Best Practice: Use 'pass' to indicate an intentional no-operation placeholder
        symbols = super(YahooCollectorCN1min, self).get_instrument_list()
        # ✅ Best Practice: Explicitly returning the result of a function
        # 🧠 ML Signal: Usage of helper function to transform data
        return symbols + ["000300.ss", "000905.ss", "000903.ss"]

    # ✅ Best Practice: Consider using a constant or configuration for timezone values to improve maintainability.
    def download_index_data(self):
        pass
# ✅ Best Practice: Use of inheritance to extend functionality of YahooCollectorIN


# ✅ Best Practice: Use of inheritance to extend functionality from YahooCollectorIN
class YahooCollectorUS(YahooCollector, ABC):
    def get_instrument_list(self):
        # ✅ Best Practice: Inheriting from both YahooCollector and ABC suggests this class is meant to be abstract.
        # ✅ Best Practice: Use of 'pass' to indicate intentional empty class definition
        logger.info("get US stock symbols......")
        symbols = get_us_stock_symbols() + [
            # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and usage.
            "^GSPC",
            "^NDX",
            "^DJI",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    # ⚠️ SAST Risk (Low): Method raises NotImplementedError, indicating it's intended to be overridden.
    def _timezone(self):
        # 🧠 ML Signal: Logging usage pattern for tracking function execution
        return "America/New_York"
# 🧠 ML Signal: Function call pattern for data retrieval


class YahooCollectorUS1d(YahooCollectorUS):
    pass

# ✅ Best Practice: Define a method to download index data, but currently it's a placeholder with no implementation.
# 🧠 ML Signal: Logging usage pattern for tracking function execution

class YahooCollectorUS1min(YahooCollectorUS):
    # ✅ Best Practice: Explicitly returning the result of the function
    pass
# ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.


# ✅ Best Practice: Consider using a constant or configuration for timezone values to improve maintainability.
# 🧠 ML Signal: Usage of string manipulation methods like upper() can indicate data normalization patterns.
class YahooCollectorIN(YahooCollector, ABC):
    def get_instrument_list(self):
        # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.
        # ✅ Best Practice: Using @property decorator is a good practice for creating read-only attributes.
        logger.info("get INDIA stock symbols......")
        symbols = get_in_stock_symbols()
        # ✅ Best Practice: Class variables should be documented to explain their purpose.
        # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.
        logger.info(f"get {len(symbols)} symbols.")
        return symbols
    # ✅ Best Practice: Class variables should be documented to explain their purpose.

    def download_index_data(self):
        # 🧠 ML Signal: Use of class-level constants for configuration
        pass

    # 🧠 ML Signal: Use of static method for utility function
    def normalize_symbol(self, symbol):
        # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame
        return code_to_fname(symbol).upper()

    # 🧠 ML Signal: Filling missing values with forward fill method
    @property
    def _timezone(self):
        # 🧠 ML Signal: Shifting series to calculate change
        return "Asia/Kolkata"

# ⚠️ SAST Risk (Low): Potential IndexError if DataFrame is empty

class YahooCollectorIN1d(YahooCollectorIN):
    # 🧠 ML Signal: Calculation of percentage change
    # ⚠️ SAST Risk (Low): Use of iloc can lead to IndexError if DataFrame is empty
    # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
    pass


class YahooCollectorIN1min(YahooCollectorIN):
    pass


class YahooCollectorBR(YahooCollector, ABC):
    # ✅ Best Practice: Early return for empty DataFrame improves code readability and efficiency.
    def retry(cls):  # pylint: disable=E0213
        """
        The reason to use retry=2 is due to the fact that
        Yahoo Finance unfortunately does not keep track of some
        Brazilian stocks.

        Therefore, the decorator deco_retry with retry argument
        set to 5 will keep trying to get the stock data up to 5 times,
        which makes the code to download Brazilians stocks very slow.

        In future, this may change, but for now
        I suggest to leave retry argument to 1 or 2 in
        order to improve download speed.

        To achieve this goal an abstract attribute (retry)
        was added into YahooCollectorBR base class
        """
        # ✅ Best Practice: Removing duplicate indices to ensure data integrity.
        raise NotImplementedError
    # ✅ Best Practice: Reindexing to align with a given calendar list.

    def get_instrument_list(self):
        logger.info("get BR stock symbols......")
        symbols = get_br_stock_symbols() + [
            "^BVSP",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    # ✅ Best Practice: Sorting the index to maintain chronological order.
    def download_index_data(self):
        pass
    # ⚠️ SAST Risk (Low): Potential for division by zero or NaN values in volume, ensure proper handling.

    def normalize_symbol(self, symbol):
        # 🧠 ML Signal: Calculating change could be a feature for ML models.
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        # 🧠 ML Signal: Repeated calculation of change could indicate a pattern for ML models.
        return "Brazil/East"


class YahooCollectorBR1d(YahooCollectorBR):
    retry = 2

# ⚠️ SAST Risk (Low): Ensure that division by 100 does not lead to unintended data corruption.
# 🧠 ML Signal: Method signature with type hints indicating input and output types

class YahooCollectorBR1min(YahooCollectorBR):
    # 🧠 ML Signal: Chaining method calls for data transformation
    retry = 2

# 🧠 ML Signal: Returning a DataFrame after processing
# ✅ Best Practice: Method docstring is present, providing a brief description of the method.
# ⚠️ SAST Risk (Low): Potential for logging sensitive information, ensure proper logging practices.

class YahooNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    # ✅ Best Practice: Use of abstract method to enforce implementation in subclasses
    # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called.
    # ✅ Best Practice: Class name should follow CamelCase naming convention
    DAILY_FORMAT = "%Y-%m-%d"

    # ✅ Best Practice: Constants should be in uppercase to distinguish them from variables
    # 🧠 ML Signal: Final change calculation could be a feature for ML models.
    @staticmethod
    # ✅ Best Practice: Check for empty DataFrame to avoid unnecessary processing
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        # ⚠️ SAST Risk (Low): Ensure that setting NaN values does not lead to data loss.
        _tmp_series = df["close"].fillna(method="ffill")
        # ✅ Best Practice: Use copy to avoid modifying the original DataFrame
        _tmp_shift_series = _tmp_series.shift(1)
        # ✅ Best Practice: Ensuring the symbol field is consistently set.
        if last_close is not None:
            # ✅ Best Practice: Set index for easier time-series manipulation
            _tmp_shift_series.iloc[0] = float(last_close)
        # ✅ Best Practice: Naming the index for clarity.
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series
    # ✅ Best Practice: Calculate adjustment factor for price adjustments
    # ✅ Best Practice: Resetting the index to return a DataFrame with a default integer index.

    @staticmethod
    # ✅ Best Practice: Forward fill to handle missing values in adjustment factor
    def normalize_yahoo(
        df: pd.DataFrame,
        calendar_list: list = None,
        # ✅ Best Practice: Default factor to 1 when 'adjclose' is not present
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    # ✅ Best Practice: Check if column exists before processing
    ):
        if df.empty:
            return df
        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        # ✅ Best Practice: Adjust volume by dividing with factor
        columns = copy.deepcopy(YahooNormalize.COLUMNS)
        # 🧠 ML Signal: Use of inheritance and method overriding, which is a common pattern in object-oriented programming.
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        # ✅ Best Practice: Adjust price-related columns by multiplying with factor
        # ✅ Best Practice: Ensure index name is set correctly for clarity
        # 🧠 ML Signal: Chaining method calls on a DataFrame, indicating a data transformation process.
        # ✅ Best Practice: Explicit return of the DataFrame makes the function's behavior clear.
        # ✅ Best Practice: Docstring provides a clear explanation of the method's purpose and usage.
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                # ✅ Best Practice: Reset index to return DataFrame to original structure
                pd.DataFrame(index=calendar_list)
                # ✅ Best Practice: Using .loc with first_valid_index() ensures that the DataFrame is sliced correctly from the first valid 'close' value.
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    # ⚠️ SAST Risk (Low): Assumes 'close' column always has at least one valid entry; potential IndexError if DataFrame is empty or all NaN.
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                # ✅ Best Practice: Check for empty DataFrame to avoid unnecessary processing
                # 🧠 ML Signal: Returns the first non-zero 'close' value, which could be a feature in financial models.
                .index
            )
        df.sort_index(inplace=True)
        # ✅ Best Practice: Use copy to avoid modifying the original DataFrame
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

        # ✅ Best Practice: Sort DataFrame to ensure operations are performed in the correct order
        change_series = YahooNormalize.calc_change(df, last_close)
        # NOTE: The data obtained by Yahoo finance sometimes has exceptions
        # ✅ Best Practice: Set index for easier data manipulation
        # WARNING: If it is normal for a `symbol(exchange)` to differ by a factor of *89* to *111* for consecutive trading days,
        # WARNING: the logic in the following line needs to be modified
        # 🧠 ML Signal: Usage of a helper function to retrieve specific data
        _count = 0
        while True:
            # NOTE: may appear unusual for many days in a row
            # ✅ Best Practice: Use of continue to skip unnecessary iterations
            change_series = YahooNormalize.calc_change(df, last_close)
            _mask = (change_series >= 89) & (change_series <= 111)
            # 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
            if not _mask.any():
                # ⚠️ SAST Risk (Low): Potential for integer overflow if volume and _close are large
                break
            _tmp_cols = ["high", "close", "low", "open", "adjclose"]
            df.loc[_mask, _tmp_cols] = df.loc[_mask, _tmp_cols] / 100
            # ⚠️ SAST Risk (Low): Division by zero risk if _close is zero
            # ✅ Best Practice: Reset index to maintain original DataFrame structure
            # ✅ Best Practice: Docstring provides clear parameter descriptions and default values
            _count += 1
            if _count >= 10:
                _symbol = df.loc[df[symbol_field_name].first_valid_index()]["symbol"]
                logger.warning(
                    f"{_symbol} `change` is abnormal for {_count} consecutive days, please check the specific data file carefully"
                )

        df["change"] = YahooNormalize.calc_change(df, last_close)

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan
        # ✅ Best Practice: Explicitly calling the superclass constructor

        df[symbol_field_name] = symbol
        # ✅ Best Practice: Type hinting for function parameters improves code readability and maintainability
        # 🧠 ML Signal: Usage of financial data column names
        df.index.names = [date_field_name]
        return df.reset_index()
    # ✅ Best Practice: Converting path to string and resolving it ensures consistent path format
    # 🧠 ML Signal: Loading data from a specified directory

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # ⚠️ SAST Risk (Low): Initialization with external data directory could lead to data integrity issues if not validated
        # normalize
        df = self.normalize_yahoo(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # 🧠 ML Signal: Usage of D.features suggests data extraction for ML model training or analysis
        # adjusted price
        # 🧠 ML Signal: Use of inheritance and method overriding
        df = self.adjusted_price(df)
        # ✅ Best Practice: Explicitly setting DataFrame columns improves code clarity
        return df
    # ✅ Best Practice: Setting index for DataFrame for efficient data manipulation

    @abc.abstractmethod
    # 🧠 ML Signal: Accessing DataFrame columns using dynamic field names
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """adjusted price"""
        # 🧠 ML Signal: Use of unique values from DataFrame for conditional logic
        raise NotImplementedError("rewrite adjusted_price")


# ✅ Best Practice: Resetting index before returning DataFrame
class YahooNormalize1d(YahooNormalize, ABC):
    DAILY_FORMAT = "%Y-%m-%d"
    # 🧠 ML Signal: Conditional data selection based on symbol name

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # 🧠 ML Signal: Use of DataFrame indexing to find latest date
        if df.empty:
            return df
        # 🧠 ML Signal: Slicing DataFrame based on date
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        # 🧠 ML Signal: Accessing first row of DataFrame for comparison
        if "adjclose" in df:
            df["factor"] = df["adjclose"] / df["close"]
            # 🧠 ML Signal: Accessing specific row in DataFrame for comparison
            # ✅ Best Practice: Constants should be defined at the class level for clarity and maintainability
            df["factor"] = df["factor"].fillna(method="ffill")
        else:
            # ✅ Best Practice: Constants should be defined at the class level for clarity and maintainability
            df["factor"] = 1
        for _col in self.COLUMNS:
            # ⚠️ SAST Risk (Low): Potential division by zero if new_latest_data[col] is zero
            # ✅ Best Practice: Constants should be defined at the class level for clarity and maintainability
            if _col not in df.columns:
                continue
            if _col == "volume":
                # ⚠️ SAST Risk (Low): Potential division by zero if new_latest_data[col] is zero
                # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
                # ✅ Best Practice: Dropping the first row and resetting index before returning DataFrame
                # ✅ Best Practice: Constants should be defined at the class level for clarity and maintainability
                df[_col] = df[_col] / df["factor"]
            else:
                df[_col] = df[_col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(YahooNormalize1d, self).normalize(df)
        df = self._manual_adj_data(df)
        return df

    # ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization of the base class.
    def _get_first_close(self, df: pd.DataFrame) -> float:
        """get first close value

        Notes
        -----
            For incremental updates(append) to Yahoo 1D data, user need to use a close that is not 0 on the first trading day of the existing data
        # 🧠 ML Signal: Use of getattr to check for cached attribute
        """
        # ✅ Best Practice: Use of @property decorator to define a method as a property, improving code readability and encapsulation.
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        # 🧠 ML Signal: Lazy initialization pattern
        return _close

    # 🧠 ML Signal: Use of setattr to cache computed value
    # 🧠 ML Signal: Method signature and parameter types can be used to infer function usage patterns
    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 🧠 ML Signal: Function call with specific parameters can indicate common usage patterns
        """manual adjust data: All fields (except change) are standardized according to the close of the first day"""
        if df.empty:
            # ✅ Best Practice: Use of descriptive parameter names improves code readability
            return df
        df = df.copy()
        # 🧠 ML Signal: Method signature with type hints can be used to infer data processing patterns
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df)
        for _col in df.columns:
            # NOTE: retain original adjclose, required for incremental updates
            if _col in [self._symbol_field_name, "adjclose", "change"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            # ✅ Best Practice: Returning the DataFrame directly after processing
            else:
                df[_col] = df[_col] / _close
        return df.reset_index()
# ✅ Best Practice: Use of abstract method to enforce implementation in subclasses
# ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that a method should be overridden.


# ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
# ✅ Best Practice: Using abc.abstractmethod enforces that subclasses must implement this method.
class YahooNormalize1dExtend(YahooNormalize1d):
    def __init__(
        # ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that a method should be overridden
        self, old_qlib_data_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    # 🧠 ML Signal: Method with a specific return type hint indicating expected output
    ):
        """

        Parameters
        ----------
        old_qlib_data_dir: str, Path
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and parameters of the function
        # 🧠 ML Signal: Calls an external function, indicating a dependency
        super(YahooNormalize1dExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        # 🧠 ML Signal: Usage of a helper function to transform data
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)
    # ✅ Best Practice: Class docstring should be added to describe the purpose and usage of the class
    # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        # 🧠 ML Signal: Function calls with specific string arguments can indicate usage patterns
        # ✅ Best Practice: Use of multiple inheritance can lead to complex class hierarchies; ensure that the parent classes are compatible and necessary.
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        # 🧠 ML Signal: Class attribute that might be used to configure behavior
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function
        return df

    # ⚠️ SAST Risk (Low): Raising a generic ValueError without context can make debugging difficult
    # 🧠 ML Signal: Method name suggests a pattern of retrieving calendar data
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(YahooNormalize1dExtend, self).normalize(df)
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and parameters of the function
        # 🧠 ML Signal: Usage of a specific calendar identifier "IN_ALL"
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        # 🧠 ML Signal: Function calls can indicate common usage patterns and dependencies
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        # 🧠 ML Signal: Function calls with specific string arguments can indicate usage patterns
        # ✅ Best Practice: Use of multiple inheritance to combine functionality from YahooNormalizeCN and YahooNormalize1d
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        # ✅ Best Practice: Use of class inheritance to extend functionality from multiple parent classes
        df = df.loc[latest_date:]
        new_latest_data = df.iloc[0]
        # ✅ Best Practice: Class definition should follow the naming conventions, using CamelCase.
        old_latest_data = old_df.loc[latest_date]
        for col in self.column_list[:-1]:
            # 🧠 ML Signal: Constants like AM_RANGE and PM_RANGE can be used to identify trading session times.
            if col == "volume":
                # ✅ Best Practice: Type hinting improves code readability and maintainability
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            # 🧠 ML Signal: Constants like AM_RANGE and PM_RANGE can be used to identify trading session times.
            else:
                # 🧠 ML Signal: Method chaining and function calls can indicate common usage patterns
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        # 🧠 ML Signal: Checks if a string contains a specific character
        return df.drop(df.index[0]).reset_index()

# 🧠 ML Signal: Extracts a substring from a string

class YahooNormalize1min(YahooNormalize, ABC):
    # 🧠 ML Signal: Conditional assignment based on string properties
    """Normalised to 1min using local 1d data"""
    # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability

    # 🧠 ML Signal: String concatenation and slicing
    AM_RANGE = None  # type: tuple  # eg: ("09:30:00", "11:29:00")
    # 🧠 ML Signal: Function calls with string literals can indicate feature usage patterns
    PM_RANGE = None  # type: tuple  # eg: ("13:00:00", "14:59:00")
    # ✅ Best Practice: Returns a value from a function
    # 🧠 ML Signal: Method definition with a specific return type hint

    # Whether the trading day of 1min data is consistent with 1d
    # 🧠 ML Signal: Function call with a hardcoded string argument
    # ✅ Best Practice: Use of multiple inheritance to combine functionality from YahooNormalizeBR and YahooNormalize1d
    CONSISTENT_1d = True
    CALC_PAUSED_NUM = True

    # ✅ Best Practice: Class attribute CALC_PAUSED_NUM is defined, indicating a shared state or configuration for instances.
    def __init__(
        # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and parameters
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            the qlib data to be updated for yahoo, usually from: Normalised to 1min using local 1d data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(YahooNormalize1min, self).__init__(date_field_name, symbol_field_name)
        qlib.init(provider_uri=qlib_data_1d_dir)
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return list(D.calendar(freq="day"))

    @property
    def calendar_list_1d(self):
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            # ✅ Best Practice: Explicitly calling the superclass constructor for proper initialization
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        # 🧠 ML Signal: Storing configuration values in instance variables
        # 🧠 ML Signal: Method constructs a class name dynamically based on region and interval.
        return calendar_list_1d

    # ✅ Best Practice: Method name should be descriptive of its action
    def generate_1min_from_daily(self, calendars: Iterable) -> pd.Index:
        # 🧠 ML Signal: Usage of f-string for string formatting
        # ✅ Best Practice: Use of @property decorator for defining a read-only attribute.
        return generate_minutes_calendar_from_daily(
            calendars, freq="1min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        # ✅ Best Practice: Specify the return type as a Union of Path and str for clarity.
        )

    # ⚠️ SAST Risk (Low): Returning a global variable directly can lead to unintended side effects if the variable is mutable.
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="1min",
            consistent_1d=self.CONSISTENT_1d,
            calc_paused=self.CALC_PAUSED_NUM,
            _1d_data_all=self.all_1d_data,
        )
        return df

    @abc.abstractmethod
    def symbol_to_yahoo(self, symbol):
        raise NotImplementedError("rewrite symbol_to_yahoo")

    @abc.abstractmethod
    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        raise NotImplementedError("rewrite _get_1d_calendar_list")


class YahooNormalizeUS:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        return get_calendar_list("US_ALL")


class YahooNormalizeUS1d(YahooNormalizeUS, YahooNormalize1d):
    pass


class YahooNormalizeUS1dExtend(YahooNormalizeUS, YahooNormalize1dExtend):
    pass


# ⚠️ SAST Risk (Low): Potential timezone issues with datetime.now() if not handled properly
class YahooNormalizeUS1min(YahooNormalizeUS, YahooNormalize1min):
    CALC_PAUSED_NUM = False
    # ⚠️ SAST Risk (Low): Error message may expose sensitive information if not handled properly

    # ✅ Best Practice: Use of super() to call a method from the parent class
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("US_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class YahooNormalizeIN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("IN_ALL")


class YahooNormalizeIN1d(YahooNormalizeIN, YahooNormalize1d):
    pass


class YahooNormalizeIN1min(YahooNormalizeIN, YahooNormalize1min):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("IN_ALL")
    # ⚠️ SAST Risk (Low): Potential NoneType dereference if qlib_data_1d_dir is None

    # ⚠️ SAST Risk (Low): Error message could expose internal logic or paths
    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class YahooNormalizeCN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # ✅ Best Practice: Use of super() to call parent class method ensures maintainability and proper inheritance
        # TODO: from MSN
        return get_calendar_list("ALL")

# ✅ Best Practice: Docstring provides detailed information about the function's purpose, parameters, and usage.

class YahooNormalizeCN1d(YahooNormalizeCN, YahooNormalize1d):
    pass


class YahooNormalizeCN1dExtend(YahooNormalizeCN, YahooNormalize1dExtend):
    pass


class YahooNormalizeCN1min(YahooNormalizeCN, YahooNormalize1min):
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_1min_from_daily(self.calendar_list_1d)

    def symbol_to_yahoo(self, symbol):
        if "." not in symbol:
            _exchange = symbol[:2]
            _exchange = ("ss" if _exchange.islower() else "SS") if _exchange.lower() == "sh" else _exchange
            symbol = symbol[2:] + "." + _exchange
        return symbol
    # 🧠 ML Signal: Dynamic attribute access using getattr, indicating potential use of reflection or dynamic class loading.
    # 🧠 ML Signal: Instantiation of a class with multiple parameters, indicating a complex object creation pattern.

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")


class YahooNormalizeBR:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("BR_ALL")


# 🧠 ML Signal: Method call on an object, indicating a potential action or process being executed.
class YahooNormalizeBR1d(YahooNormalizeBR, YahooNormalize1d):
    pass


class YahooNormalizeBR1min(YahooNormalizeBR, YahooNormalize1min):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("BR_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US", "BR"], default "CN"
        # ✅ Best Practice: Use of pd.Timedelta for date arithmetic is clear and effective.
        # 🧠 ML Signal: Method call with parameters could indicate a pattern for data downloading.
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"YahooCollector{self.region.upper()}{self.interval}"

    # 🧠 ML Signal: Date formatting pattern could be useful for ML models to understand date handling.
    @property
    def normalize_class_name(self):
        return f"YahooNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"; closed interval(including start)
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        if self.interval == "1d" and pd.Timestamp(end) > pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d")):
            raise ValueError(f"end_date: {end} is greater than the current date.")

        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        # ✅ Best Practice: Use of descriptive variable names
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, normalize the last date saved (including end_date); if None, it will ignore this parameter; by default None
        qlib_data_1d_dir: str
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
                )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    def normalize_data_1d_extend(
        self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """normalize data extend; extending yahoo qlib data(from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)

        Notes
        -----
            Steps to extend yahoo qlib data:

                1. download qlib data: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data; save to <dir1>

                2. collector source data: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#collector-data; save to <dir2>

                3. normalize new source data(from step 2): python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d

                4. dump data: python scripts/dump_bin.py dump_update --csv_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

                5. update instrument(eg. csi300): python python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments

        Parameters
        ----------
        old_qlib_data_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()

    def download_today_data(
        self,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """download today data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            Download today's data:
                start_time = datetime.datetime.now().date(); closed interval(including start)
                end_time = pd.Timestamp(start_time + pd.Timedelta(days=1)).date(); open interval(excluding end)

            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1m
        """
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
        exists_skip: bool = False,
    ):
        """update yahoo data to bin

        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        exists_skip: bool
            exists skip, by default False
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
        """

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        # download qlib 1d data
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        # start/end date
        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # download data from yahoo
        # NOTE: when downloading data from YahooFinance, max_workers is recommended to be 1
        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)
        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        # normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)

        # dump bin
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        _region = self.region.lower()
        if _region not in ["cn", "us"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        index_list = ["CSI100", "CSI300"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index, market_index=f"{_region}_index")


if __name__ == "__main__":
    fire.Fire(Run)