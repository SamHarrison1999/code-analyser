# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from functools import partial
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fire
import requests

# ⚠️ SAST Risk (Low): Modifying sys.path can lead to import conflicts or security issues if not handled carefully.
import pandas as pd
from tqdm import tqdm
from loguru import logger


# ⚠️ SAST Risk (Low): Hardcoded URLs can lead to security risks if the URL changes or is compromised.
# 🧠 ML Signal: Dictionary mapping for index names to URLs, useful for pattern recognition in ML models.
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_trading_date_by_shift,
)
from data_collector.utils import get_instruments

# ✅ Best Practice: Class names should follow the CapWords convention for readability.


# ✅ Best Practice: Class variables should be documented or initialized with meaningful default values.
WIKI_URL = "https://en.wikipedia.org/wiki"

WIKI_INDEX_NAME_MAP = {
    "NASDAQ100": "NASDAQ-100",
    "SP500": "List_of_S%26P_500_companies",
    "SP400": "List_of_S%26P_400_companies",
    "DJIA": "Dow_Jones_Industrial_Average",
}
# ✅ Best Practice: Call to super() ensures proper initialization of the base class


class WIKIIndex(IndexBase):
    # 🧠 ML Signal: Use of f-string for URL construction
    # NOTE: The US stock code contains "PRN", and the directory cannot be created on Windows system, use the "_" prefix
    # https://superuser.com/questions/613313/why-cant-we-make-con-prn-null-folder-in-windows
    INST_PREFIX = ""

    # ✅ Best Practice: Use of @property decorator for defining a read-only attribute
    # ✅ Best Practice: Include type hints for return values to improve code readability and maintainability
    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
        # ⚠️ SAST Risk (Low): Using NotImplementedError without implementation can lead to runtime errors if not properly handled
    ):
        # ✅ Best Practice: Use of @abc.abstractmethod to enforce implementation in subclasses
        # ✅ Best Practice: Docstring provides a clear description of the method's purpose and return type
        super(WIKIIndex, self).__init__(
            index_name=index_name,
            qlib_dir=qlib_dir,
            freq=freq,
            request_retry=request_retry,
            retry_sleep=retry_sleep,
        )

        self._target_url = f"{WIKI_URL}/{WIKI_INDEX_NAME_MAP[self.index_name.upper()]}"

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        Returns
        -------
            index start date
        # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called
        """
        raise NotImplementedError("rewrite bench_start_date")

    @abc.abstractmethod
    def get_changes(self) -> pd.DataFrame:
        """get companies changes

        Returns
        -------
            pd.DataFrame:
                symbol      date        type
                SH600000  2019-11-11    add
                SH600000  2020-11-10    remove
            dtypes:
                symbol: str
                date: pd.Timestamp
                type: str, value from ["add", "remove"]
        """
        raise NotImplementedError("rewrite get_changes")

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """formatting the datetime in an instrument

        Parameters
        ----------
        inst_df: pd.DataFrame
            inst_df.columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        Returns
        -------

        """
        # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame
        # ⚠️ SAST Risk (Low): Raising a generic exception without logging the error details
        if self.freq != "day":
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                # 🧠 ML Signal: Returning the response object from an HTTP request
                # 🧠 ML Signal: Stripping whitespace from string fields is a common data cleaning step
                lambda x: (
                    pd.Timestamp(x) + pd.Timedelta(hours=23, minutes=59)
                ).strftime("%Y-%m-%d %H:%M:%S")
            )
        # 🧠 ML Signal: Setting default values for date fields is a common pattern
        return inst_df

    # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
    # 🧠 ML Signal: Setting default values for date fields is a common pattern
    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """get history trading date

        Returns
        -------
            calendar list
        # ✅ Best Practice: Check for None and empty DataFrame before processing
        """
        _calendar_list = getattr(self, "_calendar_list", None)
        if _calendar_list is None:
            # ✅ Best Practice: Consistent column naming for DataFrame
            # ✅ Best Practice: Method signature includes type hints for better readability and maintainability
            _calendar_list = list(
                filter(
                    lambda x: x >= self.bench_start_date, get_calendar_list("US_ALL")
                )
            )
            setattr(self, "_calendar_list", _calendar_list)
        # 🧠 ML Signal: Usage of default settings for data processing
        # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which could lead to runtime errors if not properly implemented
        # ✅ Best Practice: Class definition should follow PEP 8 naming conventions
        return _calendar_list

    # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
    # ✅ Best Practice: Constants should be in uppercase and follow naming conventions

    def _request_new_companies(self) -> requests.Response:
        resp = requests.get(self._target_url, timeout=None)
        # ✅ Best Practice: Return early to avoid unnecessary processing
        if resp.status_code != 200:
            raise ValueError(f"request error: {self._target_url}")
        # ✅ Best Practice: Check for DataFrame length to ensure it has enough data to process
        # ⚠️ SAST Risk (Low): Hardcoded URL can lead to security risks if not validated or sanitized

        # ✅ Best Practice: Use .copy() to avoid modifying the original DataFrame
        # 🧠 ML Signal: Usage of a URL pattern for accessing external resources
        return resp

    # ✅ Best Practice: Constants should be in uppercase and follow naming conventions
    def set_default_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        # ✅ Best Practice: Use of pd.Timestamp for date handling ensures consistency and compatibility with pandas operations.
        # 🧠 ML Signal: Usage of a constant to define the number of workers for parallel processing
        _df = df.copy()
        _df[self.SYMBOL_FIELD_NAME] = _df[self.SYMBOL_FIELD_NAME].str.strip()
        _df[self.START_DATE_FIELD] = self.bench_start_date
        # ✅ Best Practice: Converting trade_date to a string format for consistent usage
        _df[self.END_DATE_FIELD] = self.DEFAULT_END_DATE
        return _df.loc[:, self.INSTRUMENTS_COLUMNS]

    # ✅ Best Practice: Using pathlib for file path operations

    # 🧠 ML Signal: Conditional logic for cache usage
    def get_new_companies(self):
        logger.info(f"get new companies {self.index_name} ......")
        _data = deco_retry(retry=self._request_retry, retry_sleep=self._retry_sleep)(
            self._request_new_companies
        )()
        # 🧠 ML Signal: Reading from cache
        df_list = pd.read_html(_data.text)
        for _df in df_list:
            _df = self.filter_df(_df)
            # 🧠 ML Signal: Constructing URL for API request
            if (_df is not None) and (not _df.empty):
                _df.columns = [self.SYMBOL_FIELD_NAME]
                # ⚠️ SAST Risk (Medium): No timeout specified for requests.post, which can lead to hanging connections
                _df = self.set_default_date_range(_df)
                logger.info(f"end of get new companies {self.index_name} ......")
                # ⚠️ SAST Risk (Low): Basic error handling for HTTP response
                return _df

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 🧠 ML Signal: Processing JSON response into DataFrame
        raise NotImplementedError("rewrite filter_df")


# ✅ Best Practice: Adding a new column to DataFrame


# 🧠 ML Signal: Usage of tqdm for progress tracking
class NASDAQ100Index(WIKIIndex):
    # ✅ Best Practice: Renaming DataFrame columns for consistency
    HISTORY_COMPANIES_URL = (
        # 🧠 ML Signal: Conditional logic for saving to cache
        # 🧠 ML Signal: Usage of ThreadPoolExecutor for concurrent execution
        # ⚠️ SAST Risk (Low): Potential for race conditions or thread safety issues
        "https://indexes.nasdaqomx.com/Index/WeightingData?id=NDX&tradeDate={trade_date}T00%3A00%3A00.000&timeOfDay=SOD"
    )
    MAX_WORKERS = 16
    # 🧠 ML Signal: Writing to cache

    # 🧠 ML Signal: Returning DataFrame
    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) >= 100 and "Ticker" in df.columns:
            return df.loc[:, ["Ticker"]].copy()

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2003-01-02")

    @deco_retry
    # ✅ Best Practice: Method chaining improves readability by reducing the need for intermediate variables.
    def _request_history_companies(
        self, trade_date: pd.Timestamp, use_cache: bool = True
    ) -> pd.DataFrame:
        trade_date = trade_date.strftime("%Y-%m-%d")
        # ✅ Best Practice: Use of @property decorator for getter method
        # ⚠️ SAST Risk (Low): Potential for ValueError if all_history is empty
        # 🧠 ML Signal: Usage of self indicates this is a method within a class, which is common in object-oriented programming.
        cache_path = self.cache_dir.joinpath(f"{trade_date}_history_companies.pkl")
        if cache_path.exists() and use_cache:
            # ✅ Best Practice: Use of type hint for return type improves code readability and maintainability
            df = pd.read_pickle(cache_path)
        else:
            # ✅ Best Practice: Define the function with a clear purpose and return type for better readability and maintainability.
            # 🧠 ML Signal: Consistent use of fixed start date could indicate a pattern in data processing
            url = self.HISTORY_COMPANIES_URL.format(trade_date=trade_date)
            resp = requests.post(url, timeout=None)
            # ✅ Best Practice: Use 'pass' to indicate an unimplemented function, making it clear that the function is intentionally left blank.
            if resp.status_code != 200:
                # ✅ Best Practice: Check if "Symbol" is in columns to avoid KeyError
                raise ValueError(f"request error: {url}")
            df = pd.DataFrame(resp.json()["aaData"])
            # ✅ Best Practice: Use .copy() to avoid modifying the original DataFrame
            df[self.DATE_FIELD_NAME] = trade_date
            df.rename(
                columns={"Name": "name", "Symbol": self.SYMBOL_FIELD_NAME}, inplace=True
            )
            # 🧠 ML Signal: Usage of lambda function for string manipulation
            # ✅ Best Practice: Method name should reflect its purpose; consider renaming if it doesn't parse instruments.
            if not df.empty:
                df.to_pickle(cache_path)
        # ⚠️ SAST Risk (Low): Logging warning messages can expose sensitive information if not handled properly.
        return df

    # 🧠 ML Signal: Constant URL for data source, useful for web scraping pattern detection

    def get_history_companies(self):
        # ✅ Best Practice: Type hinting for return type improves code readability and maintainability
        logger.info("start get history companies......")
        all_history = []
        # 🧠 ML Signal: Hardcoded date values can indicate fixed starting points or baselines in data processing
        error_list = []
        # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
        with tqdm(total=len(self.calendar_list)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                # ⚠️ SAST Risk (Low): External data source without validation or sanitization
                for _trading_date, _df in zip(
                    self.calendar_list,
                    executor.map(self._request_history_companies, self.calendar_list),
                ):
                    # ✅ Best Practice: Explicitly setting DataFrame column names for clarity
                    if _df.empty:
                        error_list.append(_trading_date)
                    # ✅ Best Practice: Converting date strings to datetime objects for consistency
                    else:
                        all_history.append(_df)
                    p_bar.update()

        if error_list:
            # ✅ Best Practice: Adding a new column to indicate the type of change
            logger.warning(f"get error: {error_list}")
        logger.info(f"total {len(self.calendar_list)}, error {len(error_list)}")
        logger.info("end of get history companies.")
        return pd.concat(all_history, sort=False)

    # ✅ Best Practice: Dropping rows with NaN values in specific columns

    def get_changes(self):
        return self.get_changes_with_history_companies(self.get_history_companies())


# 🧠 ML Signal: Conditional logic based on type of change


class DJIAIndex(WIKIIndex):
    @property
    # 🧠 ML Signal: Checks for the presence of a specific column in a DataFrame
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")

    # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame

    def get_changes(self) -> pd.DataFrame:
        # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
        # ✅ Best Practice: Use of @property decorator for getter method to provide a read-only attribute
        # ✅ Best Practice: Use of type hint for return type improves code readability and maintainability
        pass

    # ✅ Best Practice: Using pd.concat to combine DataFrames
    # 🧠 ML Signal: Hardcoded date values can indicate fixed starting points or baselines in data processing
    # 🧠 ML Signal: Method signature with return type hint indicating expected output type
    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Symbol" in df.columns:
            # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
            _df = df.loc[:, ["Symbol"]].copy()
            _df["Symbol"] = _df["Symbol"].apply(lambda x: x.split(":")[-1])
            # ✅ Best Practice: Checking if a column exists before accessing it prevents runtime errors
            return _df

    # 🧠 ML Signal: Usage of DataFrame column filtering pattern
    # ⚠️ SAST Risk (Low): Logging warning messages without context can lead to confusion during debugging.
    def parse_instruments(self):
        # ✅ Best Practice: Using .copy() to avoid SettingWithCopyWarning and ensure a new DataFrame is returned
        logger.warning("No suitable data source has been found!")


# ✅ Best Practice: Use the standard Python idiom for checking if a script is run as the main program.
# 🧠 ML Signal: Usage of the 'fire' library indicates a command-line interface pattern.
# ⚠️ SAST Risk (Low): Using 'fire.Fire' can execute arbitrary code if user input is not properly sanitized.
# ✅ Best Practice: Use 'partial' from 'functools' to fix certain arguments of a function, improving code readability.


class SP500Index(WIKIIndex):
    WIKISP500_CHANGES_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("1999-01-01")

    def get_changes(self) -> pd.DataFrame:
        logger.info("get sp500 history changes......")
        # NOTE: may update the index of the table
        changes_df = pd.read_html(self.WIKISP500_CHANGES_URL)[-1]
        changes_df = changes_df.iloc[:, [0, 1, 3]]
        changes_df.columns = [self.DATE_FIELD_NAME, self.ADD, self.REMOVE]
        changes_df[self.DATE_FIELD_NAME] = pd.to_datetime(
            changes_df[self.DATE_FIELD_NAME]
        )
        _result = []
        for _type in [self.ADD, self.REMOVE]:
            _df = changes_df.copy()
            _df[self.CHANGE_TYPE_FIELD] = _type
            _df[self.SYMBOL_FIELD_NAME] = _df[_type]
            _df.dropna(subset=[self.SYMBOL_FIELD_NAME], inplace=True)
            if _type == self.ADD:
                _df[self.DATE_FIELD_NAME] = _df[self.DATE_FIELD_NAME].apply(
                    lambda x: get_trading_date_by_shift(self.calendar_list, x, 0)
                )
            else:
                _df[self.DATE_FIELD_NAME] = _df[self.DATE_FIELD_NAME].apply(
                    lambda x: get_trading_date_by_shift(self.calendar_list, x, -1)
                )
            _result.append(
                _df[
                    [
                        self.DATE_FIELD_NAME,
                        self.CHANGE_TYPE_FIELD,
                        self.SYMBOL_FIELD_NAME,
                    ]
                ]
            )
        logger.info("end of get sp500 history changes.")
        return pd.concat(_result, sort=False)

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Symbol" in df.columns:
            return df.loc[:, ["Symbol"]].copy()


class SP400Index(WIKIIndex):
    @property
    def bench_start_date(self) -> pd.Timestamp:
        return pd.Timestamp("2000-01-01")

    def get_changes(self) -> pd.DataFrame:
        pass

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Ticker symbol" in df.columns:
            return df.loc[:, ["Ticker symbol"]].copy()

    def parse_instruments(self):
        logger.warning("No suitable data source has been found!")


if __name__ == "__main__":
    fire.Fire(partial(get_instruments, market_index="us_index"))
