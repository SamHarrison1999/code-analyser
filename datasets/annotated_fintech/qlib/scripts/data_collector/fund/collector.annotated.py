# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import datetime
import json
from abc import ABC
from pathlib import Path

import fire
import requests
import pandas as pd
# ⚠️ SAST Risk (Low): Modifying sys.path can lead to import conflicts or security issues if not handled carefully.
from loguru import logger
from dateutil.tz import tzlocal
from qlib.constant import REG_CN as REGION_CN

# ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security risks if the URL is not trusted or if it changes.
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))
from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import get_calendar_list, get_en_fund_symbols

INDEX_BENCH_URL = "http://api.fund.eastmoney.com/f10/lsjz?callback=jQuery_&fundCode={index_code}&pageIndex=1&pageSize={numberOfHistoricalDaysToCrawl}&startDate={startDate}&endDate={endDate}"


class FundCollector(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        # ✅ Best Practice: Use of docstring to describe parameters and their defaults
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
            fund save dir
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
        super(FundCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            # 🧠 ML Signal: Initialization of datetime-related functionality
            # 🧠 ML Signal: Usage of conditional logic to handle different intervals
            interval=interval,
            max_workers=max_workers,
            # 🧠 ML Signal: Use of max function to ensure start_datetime is not before a default value
            max_collector_count=max_collector_count,
            # 🧠 ML Signal: Handling of specific interval case with no operation
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        # ⚠️ SAST Risk (Low): Raising a generic ValueError without specific handling

        self.init_datetime()
    # ✅ Best Practice: Type hinting with a list of types is not standard; consider using Union for clarity.

    # 🧠 ML Signal: Conversion of datetime to a specific timezone
    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            # 🧠 ML Signal: Conversion of datetime to a specific timezone
            # ⚠️ SAST Risk (Low): Potential timezone-related issues if timezone is not validated.
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            # ⚠️ SAST Risk (Low): Potential timezone-related issues if tzlocal() is not correctly handled.
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        # ✅ Best Practice: Consider logging the exception for better debugging.
        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        # ✅ Best Practice: Using @property with @abc.abstractmethod is a common pattern for defining abstract properties.
        # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which could lead to runtime errors if not properly overridden.
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    # 🧠 ML Signal: Use of formatted strings for error messages
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            # ⚠️ SAST Risk (Medium): Potentially unsafe URL construction without validation
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt
    # ⚠️ SAST Risk (Medium): No timeout specified for requests.get, can lead to hanging

    @property
    # ✅ Best Practice: Check for HTTP response status
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")
    # ⚠️ SAST Risk (Low): Potential for JSON parsing errors

    @staticmethod
    # 🧠 ML Signal: Checking specific keys in JSON response
    def get_data_from_remote(symbol, interval, start, end):
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        # ✅ Best Practice: Consider specifying the return type as List[pd.DataFrame] for clarity
        # 🧠 ML Signal: Use of pandas for data manipulation
        try:
            # TODO: numberOfHistoricalDaysToCrawl should be bigger enough
            url = INDEX_BENCH_URL.format(
                # ✅ Best Practice: Check type before operations
                index_code=symbol, numberOfHistoricalDaysToCrawl=10000, startDate=start, endDate=end
            # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters.
            )
            resp = requests.get(url, headers={"referer": "http://fund.eastmoney.com/110022.html"}, timeout=None)
            # 🧠 ML Signal: Logging exceptions with context
            # 🧠 ML Signal: Usage of a sleep function indicates a delay or rate-limiting pattern.
            # ⚠️ SAST Risk (Low): Use of a global or outer-scope variable without clear definition in the function.

            if resp.status_code != 200:
                raise ValueError("request error")

            data = json.loads(resp.text.split("(")[-1].split(")")[0])
            # ⚠️ SAST Risk (Low): Use of a global or outer-scope variable without clear definition in the function.

            # Some funds don't show the net value, example: http://fundf10.eastmoney.com/jjjz_010288.html
            SYType = data["Data"]["SYType"]
            if SYType in {"每万份收益", "每百份收益", "每百万份收益"}:
                raise ValueError("The fund contains 每*份收益")

            # ✅ Best Practice: Inheriting from ABC indicates that this class is intended to be abstract.
            # 🧠 ML Signal: Conditional logic based on interval values can indicate time-based data processing.
            # TODO: should we sort the value by datetime?
            # ✅ Best Practice: Method name should be descriptive of its functionality
            _resp = pd.DataFrame(data["Data"]["LSJZList"])

            # 🧠 ML Signal: Logging usage pattern
            if isinstance(_resp, pd.DataFrame):
                # ⚠️ SAST Risk (Low): Potential information disclosure through exception message.
                return _resp.reset_index()
        # 🧠 ML Signal: Function call pattern
        except Exception as e:
            # ✅ Best Practice: Ensure that the function or method that contains this code has a clear return type.
            logger.warning(f"{error_msg}:{e}")
    # 🧠 ML Signal: Logging usage pattern with dynamic data
    # 🧠 ML Signal: Function that returns input as output, indicating a potential placeholder or default behavior

    def get_data(
        # 🧠 ML Signal: Return statement pattern
        # ✅ Best Practice: Consider using a constant or configuration for timezone values to improve maintainability.
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    # ✅ Best Practice: Use of @property decorator for defining a read-only attribute
    ) -> [pd.DataFrame]:
        # ✅ Best Practice: Use of inheritance to extend functionality from a parent class
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = interval
            # ✅ Best Practice: Use of a static method for utility functions that do not require class or instance data
            return self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )

        if interval == self.INTERVAL_1d:
            # ✅ Best Practice: Check for empty DataFrame to avoid unnecessary processing
            _result = _get_simple(start_datetime, end_datetime)
        else:
            raise ValueError(f"cannot support {interval}")
        # ✅ Best Practice: Use copy to avoid modifying the original DataFrame
        return _result

# ✅ Best Practice: Set index for efficient time series operations

class FundollectorCN(FundCollector, ABC):
    # ✅ Best Practice: Convert index to datetime for time series operations
    # ✅ Best Practice: Remove duplicate indices to ensure data integrity
    def get_instrument_list(self):
        logger.info("get cn fund symbols......")
        symbols = get_en_fund_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        return symbol
    # ✅ Best Practice: Reindex to align with a given calendar

    # ✅ Best Practice: Use pd.Timestamp for consistent datetime operations
    @property
    def _timezone(self):
        # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability
        return "Asia/Shanghai"

# 🧠 ML Signal: Method chaining pattern with DataFrame operations

# ✅ Best Practice: Use descriptive variable names for better readability
# ✅ Best Practice: Use of inheritance to extend functionality of a base class
class FundCollectorCN1d(FundollectorCN):
    # ✅ Best Practice: Sort index to maintain chronological order
    pass
# 🧠 ML Signal: Returning a DataFrame after processing

# ✅ Best Practice: Explicitly set index names for clarity
# ✅ Best Practice: Use of a private method name indicates internal use within the class

class FundNormalize(BaseNormalize):
    # ✅ Best Practice: Use of multiple inheritance to combine functionality from FundNormalizeCN and FundNormalize1d
    # ✅ Best Practice: Reset index to convert index back to a column
    # 🧠 ML Signal: Function call with a constant string argument
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    # ✅ Best Practice: Class definition should inherit from a base class for reusability and structure.
    def normalize_fund(
        # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
        df: pd.DataFrame,
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
                pd.DataFrame(index=calendar_list)
                .loc[
                    # ✅ Best Practice: Calling the superclass's __init__ method to ensure proper initialization
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                # 🧠 ML Signal: Storing configuration or state information in instance variables
                # 🧠 ML Signal: Use of f-string for dynamic string formatting
                ]
                # 🧠 ML Signal: Use of string manipulation methods
                .index
            )
        df.sort_index(inplace=True)
        # 🧠 ML Signal: Method for generating class names based on attributes

        df.index.names = [date_field_name]
        # ✅ Best Practice: Specify the return type as a Union of Path and str for clarity.
        return df.reset_index()

    # ⚠️ SAST Risk (Low): Returning a global variable like CUR_DIR can expose internal state.
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_fund(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        return df


class FundNormalize1d(FundNormalize):
    pass

# ✅ Best Practice: Docstring provides clear documentation of parameters and usage

class FundNormalizeCN:
    def _get_calendar_list(self):
        return get_calendar_list("ALL")


class FundNormalizeCN1d(FundNormalizeCN, FundNormalize1d):
    pass


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, interval="1d", region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 4
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"FundCollector{self.region.upper()}{self.interval}"

    @property
    # 🧠 ML Signal: Method overriding in a class, indicating use of inheritance
    def normalize_class_name(self):
        # 🧠 ML Signal: Use of fire.Fire for command-line interface generation
        # ⚠️ SAST Risk (Low): Using fire.Fire can execute arbitrary code if user input is not properly sanitized
        return f"FundNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
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
            freq, value from [1min, 1d], default 1d
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
            $ python collector.py download_data --source_dir ~/.qlib/fund_data/source/cn_data --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
        """

        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol"):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/fund_data/source/cn_data --normalize_dir ~/.qlib/fund_data/source/cn_1d_nor --region CN --interval 1d --date_field_name FSRQ
        """
        super(Run, self).normalize_data(date_field_name, symbol_field_name)


if __name__ == "__main__":
    fire.Fire(Run)