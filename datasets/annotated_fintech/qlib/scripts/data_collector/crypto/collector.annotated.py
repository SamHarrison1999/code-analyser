import abc
import sys
import datetime
from abc import ABC
from pathlib import Path

import fire
import pandas as pd
from loguru import logger
from dateutil.tz import tzlocal

# ⚠️ SAST Risk (Low): Modifying sys.path can lead to import conflicts or security issues if not handled carefully.

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import deco_retry

from pycoingecko import CoinGeckoAPI
from time import mktime

# 🧠 ML Signal: Global variables like _CG_CRYPTO_SYMBOLS can indicate shared state or configuration.
from datetime import datetime as dt

# ✅ Best Practice: Provide a clear and concise docstring for the function.
import time


_CG_CRYPTO_SYMBOLS = None

# 🧠 ML Signal: Use of global variables can indicate shared state or configuration patterns.


def get_cg_crypto_symbols(qlib_data_path: [str, Path] = None) -> list:
    """get crypto symbols in coingecko

    Returns
    -------
        crypto symbols in given exchanges list of coingecko
    """
    # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
    global _CG_CRYPTO_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    # ⚠️ SAST Risk (Low): Accessing DataFrame columns without checking if they exist
    def _get_coingecko():
        try:
            cg = CoinGeckoAPI()
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
            resp = pd.DataFrame(cg.get_coins_markets(vs_currency="usd"))
        except Exception as e:
            raise ValueError("request error") from e
        try:
            # ⚠️ SAST Risk (Low): Potential race condition if _CG_CRYPTO_SYMBOLS is modified elsewhere
            _symbols = resp["id"].to_list()
        # ✅ Best Practice: Use set to remove duplicates before sorting
        # ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is followed here.
        except Exception as e:
            logger.warning(f"request error: {e}")
            raise
        return _symbols

    if _CG_CRYPTO_SYMBOLS is None:
        _all_symbols = _get_coingecko()

        _CG_CRYPTO_SYMBOLS = sorted(set(_all_symbols))

    return _CG_CRYPTO_SYMBOLS


# ✅ Best Practice: Use of docstring to describe parameters and their default values


class CryptoCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=1,  # delay need to be one
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            crypto save dir
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
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None
        """
        # 🧠 ML Signal: Use of max function to ensure start_datetime is not before a default value
        super(CryptoCollector, self).__init__(
            # 🧠 ML Signal: Handling of specific interval case with no action
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            # ⚠️ SAST Risk (Low): Potential for unhandled interval values leading to exceptions
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            # 🧠 ML Signal: Conversion of datetime to a specific timezone
            # ✅ Best Practice: Use a more specific exception handling instead of a broad exception clause
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            # 🧠 ML Signal: Conversion of datetime to a specific timezone
            # ⚠️ SAST Risk (Medium): Potential timezone-related issues if timezone is not validated
        )

        # ⚠️ SAST Risk (Medium): Potential timezone-related issues if tzlocal() is not validated
        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            # ✅ Best Practice: Consider logging the exception for better debugging
            self.start_datetime = max(
                self.start_datetime, self.DEFAULT_START_DATETIME_1MIN
            )
        # ✅ Best Practice: Use of @property decorator for defining a read-only property
        # ⚠️ SAST Risk (Low): Raising NotImplementedError without handling can lead to unhandled exceptions.
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            # ✅ Best Practice: Use of @abc.abstractmethod to define an abstract method in a base class
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        # 🧠 ML Signal: Usage of external API (CoinGeckoAPI) to fetch data
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    # ✅ Best Practice: Initialize DataFrame with predefined columns for consistency
    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        # ✅ Best Practice: Use list comprehension for concise and efficient data processing
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        # ✅ Best Practice: Use list comprehension for concise and efficient data processing
        except ValueError as e:
            pass
        # ✅ Best Practice: Ensure date column is in datetime format for accurate filtering
        return dt

    # ✅ Best Practice: Convert datetime to date for easier comparison
    @property
    @abc.abstractmethod
    # ✅ Best Practice: Filter DataFrame using boolean indexing for clarity and performance
    def _timezone(self):
        # ✅ Best Practice: Consider using a more specific return type hint instead of a list of DataFrames
        raise NotImplementedError("rewrite get_timezone")

    @staticmethod
    # ✅ Best Practice: Reset index after filtering to maintain DataFrame integrity
    def get_data_from_remote(symbol, interval, start, end):
        # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
        error_msg = f"{symbol}-{interval}-{start}-{end}"
        # ✅ Best Practice: Ensure the return value is a DataFrame with a reset index
        try:
            # ⚠️ SAST Risk (Low): Generic exception handling may hide specific errors
            # ⚠️ SAST Risk (Low): Use of potentially undefined variable 'interval' if not set elsewhere in the class
            cg = CoinGeckoAPI()
            data = cg.get_coin_market_chart_by_id(
                id=symbol, vs_currency="usd", days="max"
            )
            _resp = pd.DataFrame(columns=["date"] + list(data.keys()))
            _resp["date"] = [
                dt.fromtimestamp(mktime(time.localtime(x[0] / 1000)))
                for x in data["prices"]
            ]
            for key in data.keys():
                _resp[key] = [x[1] for x in data[key]]
            _resp["date"] = pd.to_datetime(_resp["date"])
            _resp["date"] = [x.date() for x in _resp["date"]]
            _resp = _resp[
                (_resp["date"] < pd.to_datetime(end).date())
                & (_resp["date"] > pd.to_datetime(start).date())
            ]
            # 🧠 ML Signal: Conditional logic based on specific values of 'interval'
            if _resp.shape[0] != 0:
                _resp = _resp.reset_index()
            # ✅ Best Practice: Inheriting from ABC indicates that this class is intended to be abstract.
            if isinstance(_resp, pd.DataFrame):
                # ✅ Best Practice: Method name should be descriptive of its action and purpose
                return _resp.reset_index()
        # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific handling or logging
        except Exception as e:
            # 🧠 ML Signal: Logging usage pattern for monitoring and debugging
            logger.warning(f"{error_msg}:{e}")

    # 🧠 ML Signal: Function call pattern to external API or service
    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
        # 🧠 ML Signal: Logging usage pattern for monitoring and debugging
        # 🧠 ML Signal: Simple function returning input, could indicate a placeholder or stub
    ) -> [pd.DataFrame]:
        def _get_simple(start_, end_):
            # ✅ Best Practice: Explicitly returning the result of a function
            # ✅ Best Practice: Use of a private method to encapsulate functionality
            self.sleep()
            # ✅ Best Practice: Use of @property decorator for getter method
            _remote_interval = interval
            # 🧠 ML Signal: Returns a hardcoded timezone string
            return self.get_data_from_remote(
                # ✅ Best Practice: Use of @staticmethod for methods that do not access instance or class data
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )

        if interval == self.INTERVAL_1d:
            _result = _get_simple(start_datetime, end_datetime)
        # ✅ Best Practice: Check for empty DataFrame to avoid unnecessary processing
        else:
            raise ValueError(f"cannot support {interval}")
        return _result


# ✅ Best Practice: Use copy to avoid modifying the original DataFrame


# ✅ Best Practice: Set index for efficient time series operations
class CryptoCollector1d(CryptoCollector, ABC):
    def get_instrument_list(self):
        # ✅ Best Practice: Convert index to datetime for time series operations
        # ✅ Best Practice: Remove duplicate indices to ensure data integrity
        logger.info("get coingecko crypto symbols......")
        symbols = get_cg_crypto_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        return symbol

    # ✅ Best Practice: Reindex to align with a given calendar
    @property
    # ✅ Best Practice: Use pd.Timestamp for consistent datetime operations
    def _timezone(self):
        return "Asia/Shanghai"


# ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability


# 🧠 ML Signal: Method chaining pattern with DataFrame operations
class CryptoNormalize(BaseNormalize):
    # ✅ Best Practice: Use descriptive variable names for better readability
    DAILY_FORMAT = "%Y-%m-%d"
    # ✅ Best Practice: Sort index to maintain chronological order
    # ✅ Best Practice: Consider adding a docstring to describe the purpose and return value of the function

    # ✅ Best Practice: Return the result directly after processing
    @staticmethod
    # ✅ Best Practice: Set index name for clarity
    # ✅ Best Practice: Returning None explicitly can be useful for readability, but consider if this is the intended behavior
    # ✅ Best Practice: Class definition should inherit from a base class for reusability and structure
    def normalize_crypto(
        df: pd.DataFrame,
        # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
        # ✅ Best Practice: Reset index to return a DataFrame with default integer index
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        if df.empty:
            return df
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            df = df.reindex(
                # ✅ Best Practice: Calling the superclass constructor to ensure proper initialization
                pd.DataFrame(index=calendar_list)
                .loc[
                    # ✅ Best Practice: Method name should be descriptive of its purpose
                    pd.Timestamp(df.index.min())
                    .date() : pd.Timestamp(df.index.max())
                    .date()
                    # 🧠 ML Signal: Usage of f-string for string formatting
                    # ✅ Best Practice: Use of @property decorator for creating read-only attributes
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
                # 🧠 ML Signal: Method returns a formatted string based on an instance attribute
            )
        df.sort_index(inplace=True)
        # ⚠️ SAST Risk (Low): The function returns a variable CUR_DIR which is not defined within the function, leading to potential misuse if CUR_DIR is not properly defined elsewhere.

        # ✅ Best Practice: Use of @property decorator for creating a read-only attribute
        # ✅ Best Practice: The return type hint [Path, str] is not a valid type hint. Use Union[Path, str] from the typing module instead.
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.normalize_crypto(
            df, self._calendar_list, self._date_field_name, self._symbol_field_name
        )
        return df


class CryptoNormalize1d(CryptoNormalize):
    def _get_calendar_list(self):
        return None


class Run(BaseRun):
    def __init__(
        self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"
    ):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1
        interval: str
            freq, value from [1min, 1d], default 1d
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    # ✅ Best Practice: Use of super() to call a method from the parent class ensures proper inheritance and method resolution.
    def collector_class_name(self):
        return f"CryptoCollector{self.interval}"

    # ✅ Best Practice: Use of default parameter values for flexibility and ease of use

    @property
    def normalize_class_name(self):
        return f"CryptoNormalize{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        # ✅ Best Practice: Calling superclass method to ensure base functionality is preserved
        delay=0,
        # 🧠 ML Signal: Use of command-line interface for executing functions
        # ⚠️ SAST Risk (Low): Potential command injection if user input is not properly sanitized
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1d, currently only supprot 1d
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``
        check_data_length: int # if this param useful?
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/crypto_data/source/1d --start 2015-01-01 --end 2021-11-30 --delay 1 --interval 1d
        """

        super(Run, self).download_data(
            max_collector_count, delay, start, end, check_data_length, limit_nums
        )

    def normalize_data(
        self, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/crypto_data/source/1d --normalize_dir ~/.qlib/crypto_data/source/1d_nor --interval 1d --date_field_name date
        """
        super(Run, self).normalize_data(date_field_name, symbol_field_name)


if __name__ == "__main__":
    fire.Fire(Run)
