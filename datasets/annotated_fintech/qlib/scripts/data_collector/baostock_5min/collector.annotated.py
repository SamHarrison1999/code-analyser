# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import sys
import copy
import fire
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from pathlib import Path
from loguru import logger
# ⚠️ SAST Risk (Low): Modifying sys.path can lead to import conflicts or security issues if not handled carefully.
from typing import Iterable, List

import qlib
from qlib.data import D
# 🧠 ML Signal: Inheritance from a base class indicates a design pattern for code reuse and extension

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import generate_minutes_calendar_from_daily, calc_adjusted_price


class BaostockCollectorHS3005min(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        # ✅ Best Practice: Use of docstring to describe parameters and their default values
        start=None,
        end=None,
        interval="5min",
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
            freq, value from [5min], default 5min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        bs.login()
        super(BaostockCollectorHS3005min, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            # ⚠️ SAST Risk (Low): Ensure that the 'bs.query_trade_dates' function handles input validation and sanitization to prevent potential injection attacks.
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            # ✅ Best Practice: Use 'and' instead of '&' for logical operations to improve readability and avoid confusion with bitwise operations.
            limit_nums=limit_nums,
        )
    # 🧠 ML Signal: Appending data to a list in a loop is a common pattern that can be used to identify data collection or aggregation behavior.

    def get_trade_calendar(self):
        # 🧠 ML Signal: Creating a DataFrame from a list of data is a common pattern in data processing tasks.
        _format = "%Y-%m-%d"
        # 🧠 ML Signal: Function processes input based on specific string values, useful for learning conditional logic
        start = self.start_datetime.strftime(_format)
        # 🧠 ML Signal: Filtering a DataFrame based on a condition is a common data manipulation pattern.
        end = self.end_datetime.strftime(_format)
        # 🧠 ML Signal: Returns a dictionary based on input, useful for learning data transformation patterns
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        # 🧠 ML Signal: Returning specific columns or values from a DataFrame is a common pattern in data extraction tasks.
        calendar_list = []
        # ✅ Best Practice: Use elif for mutually exclusive conditions to improve readability
        while (rs.error_code == "0") & rs.next():
            calendar_list.append(rs.get_row_data())
        calendar_df = pd.DataFrame(calendar_list, columns=rs.fields)
        # 🧠 ML Signal: Usage of a method to fetch data from a remote source
        # ✅ Best Practice: Use the @staticmethod decorator to indicate that the method does not modify the class or instance state.
        # 🧠 ML Signal: Returns a dictionary based on input, useful for learning data transformation patterns
        trade_calendar_df = calendar_df[~calendar_df["is_trading_day"].isin(["0"])]
        return trade_calendar_df["calendar_date"].values

    @staticmethod
    # ✅ Best Practice: Explicitly setting DataFrame column names for clarity
    def process_interval(interval: str):
        if interval == "1d":
            # ✅ Best Practice: Converting string to datetime for proper time handling
            return {"interval": "d", "fields": "date,code,open,high,low,close,volume,amount,adjustflag"}
        if interval == "5min":
            # ✅ Best Practice: Formatting datetime for consistency
            return {"interval": "5", "fields": "date,time,code,open,high,low,close,volume,amount,adjustflag"}
    # ⚠️ SAST Risk (Low): Potential timezone issues when subtracting fixed time deltas

    def get_data(
        # ✅ Best Practice: Dropping unnecessary columns to save memory
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.get_data_from_remote(
            # ✅ Best Practice: Normalizing symbol format for consistency
            # ✅ Best Practice: Initialize the DataFrame to ensure it is always defined, even if the query fails.
            symbol=symbol, interval=interval, start_datetime=start_datetime, end_datetime=end_datetime
        # 🧠 ML Signal: Usage of external API to fetch data, which can be a pattern for data retrieval tasks.
        # 🧠 ML Signal: Dynamic field selection based on interval, indicating a pattern of flexible data requests.
        )
        df.columns = ["date", "time", "symbol", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
        df["date"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["date"] = df["date"].map(lambda x: pd.Timestamp(x) - pd.Timedelta(minutes=5))
        df.drop(["time"], axis=1, inplace=True)
        df["symbol"] = df["symbol"].map(lambda x: str(x).replace(".", "").upper())
        return df
    # ✅ Best Practice: Explicit conversion of datetime to string for API compatibility.

    @staticmethod
    # 🧠 ML Signal: Dynamic interval processing, indicating a pattern of flexible data requests.
    def get_data_from_remote(
        symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        # ⚠️ SAST Risk (Low): No error handling for API call failures other than checking error_code.
        rs = bs.query_history_k_data_plus(
            symbol,
            # 🧠 ML Signal: Usage of tqdm for progress tracking indicates a pattern of processing large datasets
            BaostockCollectorHS3005min.process_interval(interval=interval)["fields"],
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            # ✅ Best Practice: Construct DataFrame with specified columns for clarity and structure.
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            # ⚠️ SAST Risk (Low): Potential for large data retrieval without error handling for network issues
            frequency=BaostockCollectorHS3005min.process_interval(interval=interval)["interval"],
            # ✅ Best Practice: Return a DataFrame, ensuring consistent return type.
            adjustflag="3",
        # ⚠️ SAST Risk (Low): Loop may become infinite if error_code is never "0"
        )
        if rs.error_code == "0" and len(rs.data) > 0:
            # 🧠 ML Signal: Appending data to a list in a loop is a common pattern for data collection
            # ✅ Best Practice: Method name should be descriptive of its functionality
            data_list = rs.data
            columns = rs.fields
            # 🧠 ML Signal: Logging usage pattern for tracking execution flow
            # 🧠 ML Signal: Progress bar update in a loop indicates iterative processing
            df = pd.DataFrame(data_list, columns=columns)
        return df
    # 🧠 ML Signal: Method call pattern for retrieving data
    # ✅ Best Practice: Using a set comprehension to remove duplicates before sorting

    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
    def get_hs300_symbols(self) -> List[str]:
        # 🧠 ML Signal: Logging usage pattern for tracking execution flow
        hs300_stocks = []
        # 🧠 ML Signal: Usage of string manipulation methods to normalize input
        # 🧠 ML Signal: Class definition with specific naming pattern, useful for classifying or identifying domain-specific classes
        trade_calendar = self.get_trade_calendar()
        # 🧠 ML Signal: Return statement pattern for method output
        # ✅ Best Practice: Ensure input is a string before processing
        with tqdm(total=len(trade_calendar)) as p_bar:
            # 🧠 ML Signal: Use of class-level constants, indicating a pattern of configuration or fixed parameters
            for date in trade_calendar:
                rs = bs.query_hs300_stocks(date=date)
                # 🧠 ML Signal: Time range constants, indicating a pattern of time-based data processing
                while rs.error_code == "0" and rs.next():
                    hs300_stocks.append(rs.get_row_data())
                p_bar.update()
        # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
        # 🧠 ML Signal: Time range constants, indicating a pattern of time-based data processing
        return sorted({e[1] for e in hs300_stocks})

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = self.get_hs300_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str):
        return str(symbol).replace(".", "").upper()


# ⚠️ SAST Risk (Medium): Ensure that the login credentials are securely managed and not hardcoded.
class BaostockNormalizeHS3005min(BaseNormalize):
    # ⚠️ SAST Risk (Low): Validate the input path to prevent potential path traversal issues.
    COLUMNS = ["open", "close", "high", "low", "volume"]
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")
    # 🧠 ML Signal: Collecting features from all instruments could be used to train models on financial data.
    # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame

    def __init__(
        # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
        # 🧠 ML Signal: Filling missing values with forward fill indicates time series data handling
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            the qlib data to be updated for yahoo, usually from: Normalised to 5min using local 1d data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        # 🧠 ML Signal: Lazy loading pattern
        """
        # 🧠 ML Signal: Use of setattr to cache attribute
        bs.login()
        qlib.init(provider_uri=qlib_data_1d_dir)
        # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")
        super(BaostockNormalizeHS3005min, self).__init__(date_field_name, symbol_field_name)

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].fillna(method="ffill")
        _tmp_shift_series = _tmp_series.shift(1)
        # ✅ Best Practice: Early return for empty DataFrame improves code readability and efficiency.
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        # 🧠 ML Signal: Extracting a symbol from the DataFrame could indicate a pattern of interest for ML models.
        return change_series

    # ⚠️ SAST Risk (Low): Deep copying can be resource-intensive; ensure it's necessary.
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_5min_from_daily(self.calendar_list_1d)
    # ✅ Best Practice: Copying the DataFrame before modifying it to avoid side effects on the original data.

    # ✅ Best Practice: Setting the index to a specific field improves data manipulation and access efficiency.
    @property
    def calendar_list_1d(self):
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            # ✅ Best Practice: Removing duplicate indices to maintain data integrity.
            # ✅ Best Practice: Converting index to datetime ensures proper date operations and comparisons.
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d

    # ✅ Best Practice: Reindexing with a calendar list ensures the DataFrame aligns with expected dates.
    @staticmethod
    def normalize_baostock(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        # 🧠 ML Signal: Method signature and parameter types can be used to infer function behavior and usage patterns
        symbol_field_name: str = "symbol",
        # ✅ Best Practice: Sorting the DataFrame by index ensures chronological order.
        # 🧠 ML Signal: Function call with specific parameters can indicate common usage patterns
        last_close: float = None,
    ):
        # ⚠️ SAST Risk (Low): Directly manipulating DataFrame values; ensure proper handling of NaN and zero values.
        if df.empty:
            # ✅ Best Practice: Using named parameters improves code readability and maintainability
            # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
            return df
        # 🧠 ML Signal: Calculating change could be a feature of interest for ML models.
        # ⚠️ SAST Risk (Low): Reassigning NaN values based on conditions; ensure this logic is correct and safe.
        # 🧠 ML Signal: Usage of a DataFrame suggests data manipulation, which is common in ML pipelines
        # ✅ Best Practice: Reassigning the DataFrame to the same variable name can help in reducing memory usage
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(BaostockNormalizeHS3005min.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            # 🧠 ML Signal: Usage of self attributes suggests object-oriented design patterns
            # 🧠 ML Signal: Assigning a constant symbol to a column could be a feature of interest for ML models.
            # ✅ Best Practice: Naming the index improves code readability and data manipulation.
            # 🧠 ML Signal: Named arguments indicate a pattern of using keyword arguments for clarity
            df = df.reindex(
                # 🧠 ML Signal: Usage of self attributes suggests object-oriented design patterns
                # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
                pd.DataFrame(index=calendar_list)
                # ✅ Best Practice: Resetting the index to return a DataFrame with a default integer index.
                .loc[pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date() + pd.Timedelta(days=1)]
                # 🧠 ML Signal: Specific frequency setting indicates time-series data processing
                # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability
                # ⚠️ SAST Risk (Low): Direct use of external library function without input validation or error handling
                .index
            )
        # 🧠 ML Signal: Usage of self attributes suggests object-oriented design patterns
        # 🧠 ML Signal: Method chaining pattern with DataFrame operations
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan
        # 🧠 ML Signal: Method chaining pattern with DataFrame operations

        # ✅ Best Practice: Returning the DataFrame directly is clear and concise
        # ✅ Best Practice: Class docstring is missing, consider adding one to describe the purpose and usage of the class.
        df["change"] = BaostockNormalizeHS3005min.calc_change(df, last_close)
        # ✅ Best Practice: Explicit return of the DataFrame for clarity
        # ✅ Best Practice: Use of default parameter values for flexibility and ease of use

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        # ✅ Best Practice: Calling the superclass's __init__ method to ensure proper initialization
        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        # 🧠 ML Signal: Storing configuration or state information in instance variables
        # 🧠 ML Signal: Method that constructs a class name based on attributes
        return df.reset_index()
    # 🧠 ML Signal: Usage of f-string for dynamic string formatting

    def generate_5min_from_daily(self, calendars: Iterable) -> pd.Index:
        # 🧠 ML Signal: Method for generating class names based on attributes
        return generate_minutes_calendar_from_daily(
            # ✅ Best Practice: Use of f-string for string formatting
            calendars, freq="5min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        )
    # ✅ Best Practice: Specify the return type as a Union of Path and str for clarity.

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # ⚠️ SAST Risk (Low): CUR_DIR should be validated to ensure it is a safe and expected path.
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="5min",
            _1d_data_all=self.all_1d_data,
        )
        return df

    # ✅ Best Practice: Docstring provides clear documentation for the function's purpose and usage.
    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return list(D.calendar(freq="day"))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_baostock(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # adjusted price
        df = self.adjusted_price(df)
        return df

# 🧠 ML Signal: Usage of `super()` indicates inheritance and method overriding, useful for understanding class hierarchies.

class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="5min", region="HS300"):
        """
        Changed the default value of: scripts.data_collector.base.BaseRun.
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"BaostockCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        # ⚠️ SAST Risk (Low): Potential risk if qlib_data_1d_dir is not validated properly
        return CUR_DIR

    def download_data(
        self,
        # 🧠 ML Signal: Usage of super() indicates inheritance and method overriding
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        # 🧠 ML Signal: Entry point for command-line interface
        # 🧠 ML Signal: Usage of fire.Fire for command-line interface generation
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Baostock

        Notes
        -----
            check_data_length, example:
                hs300 5min, a week: 4 * 60 * 5

        Examples
        ---------
            # get hs300 5min data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --start 2022-01-01 --end 2022-01-30 --interval 5min --region HS300
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Attention
        ---------
        qlib_data_1d_dir cannot be None, normalize 5min needs to use 1d data;

            qlib_data_1d can be obtained like this:
                $ python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
            or:
                download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
        """
        if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
            raise ValueError(
                "If normalize 5min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
            )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )


if __name__ == "__main__":
    fire.Fire(Run)