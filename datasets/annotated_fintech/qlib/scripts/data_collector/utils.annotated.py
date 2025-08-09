#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import copy
import importlib
import time
import bisect
import pickle
import random
import requests
import functools
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
# ⚠️ SAST Risk (Medium): Hardcoded URLs can lead to security risks if not properly validated or sanitized.
from tqdm import tqdm
from functools import partial
# ⚠️ SAST Risk (Medium): Hardcoded URLs can lead to security risks if not properly validated or sanitized.
from concurrent.futures import ProcessPoolExecutor
# ⚠️ SAST Risk (Medium): Hardcoded URLs can lead to security risks if not properly validated or sanitized.
from bs4 import BeautifulSoup

HS_SYMBOLS_URL = "http://app.finance.ifeng.com/hq/list.php?type=stock_a&class={s_type}"

CALENDAR_URL_BASE = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20991231"
SZSE_CALENDAR_URL = "http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month={month}&random={random}"

CALENDAR_BENCH_URL_MAP = {
    "CSI300": CALENDAR_URL_BASE.format(market=1, bench_code="000300"),
    "CSI500": CALENDAR_URL_BASE.format(market=1, bench_code="000905"),
    "CSI100": CALENDAR_URL_BASE.format(market=1, bench_code="000903"),
    # NOTE: Use the time series of SH600000 as the sequence of all stocks
    "ALL": CALENDAR_URL_BASE.format(market=1, bench_code="000905"),
    # ✅ Best Practice: Use of global variables should be minimized to avoid potential side effects and improve code maintainability.
    # NOTE: Use the time series of ^GSPC(SP500) as the sequence of all stocks
    "US_ALL": "^GSPC",
    # ✅ Best Practice: Use of global variables should be minimized to avoid potential side effects and improve code maintainability.
    "IN_ALL": "^NSEI",
    "BR_ALL": "^BVSP",
# ✅ Best Practice: Use of global variables should be minimized to avoid potential side effects and improve code maintainability.
}
# ✅ Best Practice: Use of type hinting for function return type improves code readability and maintainability

# ✅ Best Practice: Use of global variables should be minimized to avoid potential side effects and improve code maintainability.
_BENCH_CALENDAR_LIST = None
_ALL_CALENDAR_LIST = None
_HS_SYMBOLS = None
_US_SYMBOLS = None
_IN_SYMBOLS = None
_BR_SYMBOLS = None
_EN_FUND_SYMBOLS = None
_CALENDAR_MAP = {}

# ✅ Best Practice: Use of global variables should be minimized to avoid potential side effects and improve code maintainability.
# NOTE: Until 2020-10-20 20:00:00
# 🧠 ML Signal: Use of constants can indicate important thresholds or limits in the application logic.
# 🧠 ML Signal: Logging usage pattern can be used to train models for log analysis or anomaly detection
MINIMUM_SYMBOLS_NUM = 3900
# ⚠️ SAST Risk (Medium): No timeout specified for requests.get, which can lead to hanging connections.


# 🧠 ML Signal: Use of lambda function for mapping, indicating functional programming style.
def get_calendar_list(bench_code="CSI300") -> List[pd.Timestamp]:
    """get SH/SZ history calendar list

    Parameters
    ----------
    bench_code: str
        value from ["CSI300", "CSI500", "ALL", "US_ALL"]

    Returns
    -------
        history calendar list
    # 🧠 ML Signal: Repeated instantiation of Ticker object with same parameters.
    """

    # ⚠️ SAST Risk (Medium): No timeout specified for requests.get, which can lead to hanging connections.
    # 🧠 ML Signal: Use of pandas methods for data manipulation.
    logger.info(f"get calendar list: {bench_code}......")

    def _get_calendar(url):
        # 🧠 ML Signal: Use of string method upper for case-insensitive comparison.
        _value_list = requests.get(url, timeout=None).json()["data"]["klines"]
        return sorted(map(lambda x: pd.Timestamp(x.split(",")[0]), _value_list))
    # 🧠 ML Signal: Use of decorator for retry logic.
    # 🧠 ML Signal: Usage of pd.Timestamp to convert date strings to pandas Timestamp objects.

    calendar = _CALENDAR_MAP.get(bench_code, None)
    if calendar is None:
        if bench_code.startswith("US_") or bench_code.startswith("IN_") or bench_code.startswith("BR_"):
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide unexpected errors.
            print(Ticker(CALENDAR_BENCH_URL_MAP[bench_code]))
            print(Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max"))
            df = Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max")
            # 🧠 ML Signal: Usage of pd.date_range to generate a range of dates.
            calendar = df.index.get_level_values(level="date").map(pd.Timestamp).unique().tolist()
        else:
            if bench_code.upper() == "ALL":
                # 🧠 ML Signal: Usage of strftime to format dates.

                @deco_retry
                def _get_calendar(month):
                    _cal = []
                    # ✅ Best Practice: Using list concatenation instead of append in a loop for better performance.
                    try:
                        resp = requests.get(
                            # 🧠 ML Signal: Usage of filter and lambda to filter dates.
                            # ✅ Best Practice: Type hint for file_path suggests it should be a Path object, improving code readability and maintainability.
                            SZSE_CALENDAR_URL.format(month=month, random=random.random), timeout=None
                        ).json()
                        # 🧠 ML Signal: Reading a CSV file to extract a specific column is a common data processing pattern.
                        for _r in resp["data"]:
                            # 🧠 ML Signal: Caching results in a dictionary for reuse.
                            # 🧠 ML Signal: Logging information for tracking execution flow.
                            # 🧠 ML Signal: Converting strings to timestamps is a common preprocessing step in time series analysis.
                            # ✅ Best Practice: Using list comprehension for converting and sorting dates is efficient and concise.
                            # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
                            if int(_r["jybz"]):
                                _cal.append(pd.Timestamp(_r["jyrq"]))
                    except Exception as e:
                        raise ValueError(f"{month}-->{e}") from e
                    return _cal

                month_range = pd.date_range(start="2000-01", end=pd.Timestamp.now() + pd.Timedelta(days=31), freq="M")
                # ✅ Best Practice: Docstring provides a clear explanation of the function's purpose, parameters, and return value.
                calendar = []
                for _m in month_range:
                    cal = _get_calendar(_m.strftime("%Y-%m"))
                    if cal:
                        calendar += cal
                calendar = list(filter(lambda x: x <= pd.Timestamp.now(), calendar))
            else:
                calendar = _get_calendar(CALENDAR_BENCH_URL_MAP[bench_code])
        _CALENDAR_MAP[bench_code] = calendar
    logger.info(f"end of get calendar list: {bench_code}.")
    return calendar


def return_date_list(date_field_name: str, file_path: Path):
    date_list = pd.read_csv(file_path, sep=",", index_col=0)[date_field_name].to_list()
    return sorted([pd.Timestamp(x) for x in date_list])


def get_calendar_list_by_ratio(
    # 🧠 ML Signal: Logging usage pattern can be used to train models for log analysis.
    source_dir: [str, Path],
    date_field_name: str = "date",
    # ✅ Best Practice: Use of Path.expanduser() to handle user directories in a cross-platform way.
    threshold: float = 0.5,
    minimum_count: int = 10,
    # ✅ Best Practice: Use of Path.glob() to list files in a directory is more readable and concise.
    max_workers: int = 16,
) -> list:
    """get calendar list by selecting the date when few funds trade in this day

    Parameters
    ----------
    source_dir: str or Path
        The directory where the raw data collected from the Internet is saved
    date_field_name: str
            date field name, default is date
    threshold: float
        threshold to exclude some days when few funds trade in this day, default 0.5
    minimum_count: int
        minimum count of funds should trade in one day
    max_workers: int
        Concurrent number, default is 16

    Returns
    -------
        history calendar list
    """
    logger.info(f"get calendar list from {source_dir} by threshold = {threshold}......")
    # 🧠 ML Signal: Logging usage pattern can be used to train models for log analysis.

    # ✅ Best Practice: Include a docstring to describe the function's purpose and return value
    source_dir = Path(source_dir).expanduser()
    # ✅ Best Practice: Use of tqdm for progress tracking is a good practice for long-running operations.
    file_list = list(source_dir.glob("*.csv"))

    _number_all_funds = len(file_list)

    logger.info(f"count how many funds trade in this day......")
    _dict_count_trade = dict()  # dict{date:count}
    # ⚠️ SAST Risk (Low): Using global variables can lead to unexpected behavior and is generally discouraged
    # ✅ Best Practice: List comprehension is used for concise and efficient list creation.
    _fun = partial(return_date_list, date_field_name)
    all_oldest_list = []
    with tqdm(total=_number_all_funds) as p_bar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for date_list in executor.map(_fun, file_list):
                if date_list:
                    all_oldest_list.append(date_list[0])
                for date in date_list:
                    if date not in _dict_count_trade:
                        _dict_count_trade[date] = 0

                    _dict_count_trade[date] += 1

                p_bar.update()

    logger.info(f"count how many funds have founded in this day......")
    _dict_count_founding = {date: _number_all_funds for date in _dict_count_trade}  # dict{date:count}
    with tqdm(total=_number_all_funds) as p_bar:
        for oldest_date in all_oldest_list:
            for date in _dict_count_founding.keys():
                if date < oldest_date:
                    _dict_count_founding[date] -= 1

    calendar = [
        date for date, count in _dict_count_trade.items() if count >= max(int(count * threshold), minimum_count)
    ]

    # ⚠️ SAST Risk (Medium): No timeout specified in requests.get, which can lead to hanging connections.
    return calendar


def get_hs_stock_symbols() -> list:
    """get SH/SZ stock symbols

    Returns
    -------
        stock symbols
    """
    global _HS_SYMBOLS  # pylint: disable=W0603

    def _get_symbol():
        """
        Get the stock pool from a web page and process it into the format required by yahooquery.
        Format of data retrieved from the web page: 600519, 000001
        The data format required by yahooquery: 600519.ss, 000001.sz

        Returns
        -------
            set: Returns the set of symbol codes.

        Examples:
        -------
            {600000.ss, 600001.ss, 600002.ss, 600003.ss, ...}
        """
        # url = "http://99.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12"

        base_url = "http://99.push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": 1,  # page number
            "pz": 100,  # page size, default to 100
            "po": 1,
            "np": 1,
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
            "fields": "f12",
        }

        # ⚠️ SAST Risk (Low): Fixed sleep time can lead to inefficient waiting; consider exponential backoff.
        _symbols = []
        page = 1
        # ✅ Best Practice: Use Path from pathlib for file path operations for better cross-platform compatibility.

        while True:
            params["pn"] = page
            try:
                resp = requests.get(base_url, params=params, timeout=None)
                # ⚠️ SAST Risk (Medium): Untrusted data deserialization with pickle can lead to arbitrary code execution.
                resp.raise_for_status()
                data = resp.json()
                # ✅ Best Practice: Add type hinting for function parameters and return type for better readability and maintainability.
                # ⚠️ SAST Risk (Medium): Ensure that the data being serialized with pickle is from a trusted source.

                # Check if response contains valid data
                if not data or "data" not in data or not data["data"] or "diff" not in data["data"]:
                    logger.warning(f"Invalid response structure on page {page}")
                    break

                # fetch the current page data
                # ⚠️ SAST Risk (Low): Using global variables can lead to unexpected behavior and make the code harder to debug.
                # ✅ Best Practice: Consider making this function name more descriptive to indicate its purpose or the data it retrieves.
                current_symbols = [_v["f12"] for _v in data["data"]["diff"]]

                # ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security risks if the endpoint changes or is deprecated.
                if not current_symbols:  # It's the last page if there is no data in current page
                    logger.info(f"Last page reached: {page - 1}")
                    # ⚠️ SAST Risk (Medium): No timeout specified in requests.get can lead to hanging if the server does not respond.
                    break

                # 🧠 ML Signal: Checking for HTTP status code to handle errors.
                _symbols.extend(current_symbols)

                # show progress
                logger.info(
                    # 🧠 ML Signal: List comprehension used for transforming data.
                    f"Page {page}: fetch {len(current_symbols)} stocks:[{current_symbols[0]} ... {current_symbols[-1]}]"
                )
                # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors.

                page += 1

                # sleep time to avoid overloading the server
                # 🧠 ML Signal: Conditional check on data length to ensure data integrity.
                time.sleep(0.5)

            # ⚠️ SAST Risk (Medium): Using FTP protocol can expose data to interception; consider using a secure protocol like HTTPS.
            except requests.exceptions.HTTPError as e:
                raise requests.exceptions.HTTPError(
                    # ✅ Best Practice: Decorator usage indicates retry logic, which is useful for handling transient errors.
                    f"Request to {base_url} failed with status code {resp.status_code}"
                ) from e
            # ✅ Best Practice: Use of regex=False for str.replace improves performance when regex is not needed.
            except Exception as e:
                logger.warning("An error occurred while extracting data from the response.")
                raise

        if len(_symbols) < 3900:
            raise ValueError("The complete list of stocks is not available.")

        # Add suffix after the stock code to conform to yahooquery standard, otherwise the data will not be fetched.
        # 🧠 ML Signal: Use of a retry decorator indicates a pattern of handling transient failures.
        _symbols = [
            _symbol + ".ss" if _symbol.startswith("6") else _symbol + ".sz" if _symbol.startswith(("0", "3")) else None
            for _symbol in _symbols
        ]
        _symbols = [_symbol for _symbol in _symbols if _symbol is not None]

        return set(_symbols)

    if _HS_SYMBOLS is None:
        # ⚠️ SAST Risk (Medium): No timeout specified for the request, which can lead to hanging connections.
        symbols = set()
        _retry = 60
        # ⚠️ SAST Risk (Low): No specific exception handling for HTTP errors.
        # It may take multiple times to get the complete
        while len(symbols) < MINIMUM_SYMBOLS_NUM:
            symbols |= _get_symbol()
            time.sleep(3)
        # ⚠️ SAST Risk (Low): Potential KeyError if "symbolTicker" is not in the response JSON.

        symbol_cache_path = Path("~/.cache/hs_symbols_cache.pkl").expanduser().resolve()
        symbol_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors.
        if symbol_cache_path.exists():
            with symbol_cache_path.open("rb") as fp:
                cache_symbols = pickle.load(fp)
                symbols |= cache_symbols
        # 🧠 ML Signal: Conditional logic based on the state of _US_SYMBOLS.
        # 🧠 ML Signal: Aggregating data from multiple sources.
        with symbol_cache_path.open("wb") as fp:
            pickle.dump(symbols, fp)

        _HS_SYMBOLS = sorted(list(symbols))

    return _HS_SYMBOLS
# ✅ Best Practice: Use Pathlib for file path operations for better readability and cross-platform compatibility.

# ✅ Best Practice: Use descriptive variable names for better readability

def get_us_stock_symbols(qlib_data_path: [str, Path] = None) -> list:
    """get US stock symbols

    Returns
    -------
        stock symbols
    # 🧠 ML Signal: Use of map function indicates data transformation pattern
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
    # ✅ Best Practice: Use list comprehensions for better readability and performance
    """
    global _US_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    def _get_eastmoney():
        # ⚠️ SAST Risk (Low): Potential exposure of internal symbols if _all_symbols contains sensitive data
        url = "http://4.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&fs=m:105,m:106,m:107&fields=f12"
        resp = requests.get(url, timeout=None)
        # ⚠️ SAST Risk (Low): Using global variables can lead to unexpected behavior and is generally discouraged.
        if resp.status_code != 200:
            # 🧠 ML Signal: Fetching data from a specific URL, indicating a pattern of data retrieval from web sources
            raise ValueError("request error")
        # 🧠 ML Signal: Use of decorators can indicate patterns for retry logic or error handling.
        # ⚠️ SAST Risk (Low): Hardcoded URL can lead to maintenance issues if the URL changes

        try:
            # ⚠️ SAST Risk (Low): No error handling for network issues or invalid CSV format
            _symbols = [_v["f12"].replace("_", "-P") for _v in resp.json()["data"]["diff"].values()]
        except Exception as e:
            # ✅ Best Practice: Renaming columns for clarity and consistency
            logger.warning(f"request error: {e}")
            raise
        # ✅ Best Practice: Appending ".NS" to symbols for standardization

        if len(_symbols) < 8000:
            # ✅ Best Practice: Dropping NaN values to ensure data integrity
            raise ValueError("request error")

        # ✅ Best Practice: Using unique to remove duplicates and converting to list for easy manipulation
        return _symbols

    @deco_retry
    def _get_nasdaq():
        _res_symbols = []
        # ⚠️ SAST Risk (Low): Potential issue if _IN_SYMBOLS is not defined elsewhere in the code
        # 🧠 ML Signal: Conditional data fetching based on a variable's state
        for _name in ["otherlisted", "nasdaqtraded"]:
            url = f"ftp://ftp.nasdaqtrader.com/SymbolDirectory/{_name}.txt"
            # 🧠 ML Signal: Conditional logic based on the presence of a configuration path
            # ✅ Best Practice: Use descriptive variable names for better readability
            df = pd.read_csv(url, sep="|")
            df = df.rename(columns={"ACT Symbol": "Symbol"})
            # 🧠 ML Signal: Iterating over a predefined list of indices
            # ✅ Best Practice: Chain string operations to reduce the number of lines
            _symbols = df["Symbol"].dropna()
            _symbols = _symbols.str.replace("$", "-P", regex=False)
            # ⚠️ SAST Risk (Low): No error handling for file reading issues
            _symbols = _symbols.str.replace(".W", "-WT", regex=False)
            _symbols = _symbols.str.replace(".U", "-UN", regex=False)
            # 🧠 ML Signal: Use of sorted and set to remove duplicates and sort a collection
            # ✅ Best Practice: Using Path for file path operations for better cross-platform compatibility
            _symbols = _symbols.str.replace(".R", "-RI", regex=False)
            # ✅ Best Practice: Explicitly naming columns for clarity
            # ⚠️ SAST Risk (Low): Returning a global variable might expose internal state
            # ✅ Best Practice: Provide a clear and concise docstring for the function.
            _symbols = _symbols.str.replace(".", "-", regex=False)
            _res_symbols += _symbols.unique().tolist()
        return _res_symbols

    @deco_retry
    # ✅ Best Practice: Using unique to remove duplicates before extending the list
    def _get_nyse():
        url = "https://www.nyse.com/api/quotes/filter"
        # ⚠️ SAST Risk (Low): Using global variables can lead to unexpected behavior and is generally discouraged.
        _parms = {
            "instrumentType": "EQUITY",
            # 🧠 ML Signal: Use of decorators can indicate patterns for retry logic or error handling.
            "pageNumber": 1,
            "sortColumn": "NORMALIZED_TICKER",
            # ⚠️ SAST Risk (Medium): No timeout specified for the request, which can lead to hanging connections.
            "sortOrder": "ASC",
            "maxResultsPerPage": 10000,
            # ⚠️ SAST Risk (Low): No error handling for the HTTP request, which can lead to unhandled exceptions.
            "filterToken": "",
        }
        resp = requests.post(url, json=_parms, timeout=None)
        # ⚠️ SAST Risk (Low): No check if tbody is None, which can lead to AttributeError if the structure is not as expected.
        if resp.status_code != 200:
            raise ValueError("request error")

        # ✅ Best Practice: Consider using BeautifulSoup's text extraction methods for better readability.
        try:
            _symbols = [_v["symbolTicker"].replace("-", "-P") for _v in resp.json()]
        except Exception as e:
            # 🧠 ML Signal: Conditional logic based on the state of _BR_SYMBOLS.
            logger.warning(f"request error: {e}")
            _symbols = []
        return _symbols

    if _US_SYMBOLS is None:
        # 🧠 ML Signal: Conditional logic based on the state of qlib_data_path.
        _all_symbols = _get_eastmoney() + _get_nasdaq() + _get_nyse()
        # ✅ Best Practice: Use Pathlib for file path operations for better cross-platform compatibility.
        if qlib_data_path is not None:
            # ✅ Best Practice: Strip whitespace before other characters for consistency
            for _index in ["nasdaq100", "sp500"]:
                ins_df = pd.read_csv(
                    # ✅ Best Practice: Chaining strip operations for multiple characters
                    Path(qlib_data_path).joinpath(f"instruments/{_index}.txt"),
                    sep="\t",
                    names=["symbol", "start_date", "end_date"],
                # 🧠 ML Signal: Appending unique symbols from a DataFrame to a list.
                # ✅ Best Practice: Consistent string concatenation
                )
                _all_symbols += ins_df["symbol"].unique().tolist()
        # ✅ Best Practice: Consider adding type hints for the return type of the function

        # 🧠 ML Signal: Use of map and set to process and deduplicate a list
        # ✅ Best Practice: Use of sorted to maintain order after deduplication
        def _format(s_):
            s_ = s_.replace(".", "-")
            s_ = s_.strip("$")
            s_ = s_.strip("*")
            return s_
        # 🧠 ML Signal: Use of global variables can indicate shared state or configuration

        _US_SYMBOLS = sorted(set(map(_format, filter(lambda x: len(x) < 8 and not x.endswith("WS"), _all_symbols))))

    # ⚠️ SAST Risk (Low): Use of decorators can introduce unexpected behavior if not properly managed
    return _US_SYMBOLS
# ⚠️ SAST Risk (Medium): No timeout specified for the request, which can lead to hanging connections


# ⚠️ SAST Risk (Low): No specific exception handling for network-related errors
def get_in_stock_symbols(qlib_data_path: [str, Path] = None) -> list:
    """get IN stock symbols

    Returns
    -------
        stock symbols
    """
    global _IN_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
    def _get_nifty():
        url = f"https://www1.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        df = df.rename(columns={"SYMBOL": "Symbol"})
        df["Symbol"] = df["Symbol"] + ".NS"
        _symbols = df["Symbol"].dropna()
        # 🧠 ML Signal: Checks if a global variable is None before initializing
        _symbols = _symbols.unique().tolist()
        # ✅ Best Practice: Use of set to remove duplicates before sorting
        return _symbols

    if _IN_SYMBOLS is None:
        _all_symbols = _get_nifty()
        if qlib_data_path is not None:
            for _index in ["nifty"]:
                ins_df = pd.read_csv(
                    Path(qlib_data_path).joinpath(f"instruments/{_index}.txt"),
                    sep="\t",
                    names=["symbol", "start_date", "end_date"],
                # ✅ Best Practice: Consider adding error handling for the split operation in case the input format is incorrect.
                )
                _all_symbols += ins_df["symbol"].unique().tolist()
        # 🧠 ML Signal: Usage of specific exchange codes can indicate financial domain-specific processing.

        def _format(s_):
            # ✅ Best Practice: Use of f-strings for string formatting is a modern and readable approach.
            s_ = s_.replace(".", "-")
            s_ = s_.strip("$")
            s_ = s_.strip("*")
            # ✅ Best Practice: Docstring provides a clear description of the function and its parameters.
            # ✅ Best Practice: Use of f-strings for string formatting is a modern and readable approach.
            # 🧠 ML Signal: Conditional logic based on the 'capital' flag can indicate user preference for output format.
            return s_

        _IN_SYMBOLS = sorted(set(_all_symbols))

    return _IN_SYMBOLS


def get_br_stock_symbols(qlib_data_path: [str, Path] = None) -> list:
    """get Brazil(B3) stock symbols

    Returns
    -------
        B3 stock symbols
    # 🧠 ML Signal: Conditional logic based on a boolean parameter.
    # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
    # ✅ Best Practice: Use of functools.wraps to preserve metadata of the original function
    """
    global _BR_SYMBOLS  # pylint: disable=W0603

    # 🧠 ML Signal: Use of retry logic pattern
    @deco_retry
    def _get_ibovespa():
        _symbols = []
        url = "https://www.fundamentus.com.br/detalhes.php?papel="

        # 🧠 ML Signal: Function call with dynamic arguments
        # Request
        agent = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, headers=agent, timeout=None)

        # ⚠️ SAST Risk (Low): Catching broad Exception
        # BeautifulSoup
        # 🧠 ML Signal: Logging pattern for exception handling
        soup = BeautifulSoup(page.content, "html.parser")
        tbody = soup.find("tbody")

        # ⚠️ SAST Risk (Low): Re-raising the caught exception
        children = tbody.findChildren("a", recursive=True)
        # ✅ Best Practice: Add type hint for the return value in the function signature
        for child in children:
            # ⚠️ SAST Risk (Low): Use of time.sleep can lead to blocking
            # ✅ Best Practice: Use of conditional expression for function return
            _symbols.append(str(child).rsplit('"', maxsplit=1)[-1].split(">")[1].split("<")[0])

        return _symbols

    if _BR_SYMBOLS is None:
        _all_symbols = _get_ibovespa()
        if qlib_data_path is not None:
            for _index in ["ibov"]:
                ins_df = pd.read_csv(
                    Path(qlib_data_path).joinpath(f"instruments/{_index}.txt"),
                    sep="\t",
                    names=["symbol", "start_date", "end_date"],
                )
                # ✅ Best Practice: Ensure trading_date is a pd.Timestamp for consistency
                _all_symbols += ins_df["symbol"].unique().tolist()

        # ✅ Best Practice: Use bisect module for efficient searching in sorted lists
        def _format(s_):
            s_ = s_.strip()
            s_ = s_.strip("$")
            # ⚠️ SAST Risk (Low): Potential IndexError if left_index + shift is out of bounds
            s_ = s_.strip("*")
            # ✅ Best Practice: Handle IndexError to prevent application crash
            # ✅ Best Practice: Consider importing only necessary functions from modules to improve readability and performance.
            s_ = s_ + ".SA"
            return s_

        _BR_SYMBOLS = sorted(set(map(_format, _all_symbols)))

    return _BR_SYMBOLS


def get_en_fund_symbols(qlib_data_path: [str, Path] = None) -> list:
    """get en fund symbols

    Returns
    -------
        fund symbols in China
    """
    global _EN_FUND_SYMBOLS  # pylint: disable=W0603

    @deco_retry
    def _get_eastmoney():
        url = "http://fund.eastmoney.com/js/fundcode_search.js"
        resp = requests.get(url, timeout=None)
        if resp.status_code != 200:
            # 🧠 ML Signal: Iterating over a list of dates to generate time ranges is a common pattern in time series data processing.
            raise ValueError("request error")
        try:
            _symbols = []
            for sub_data in re.findall(r"[\[](.*?)[\]]", resp.content.decode().split("= [")[-1].replace("];", "")):
                data = sub_data.replace('"', "").replace("'", "")
                # TODO: do we need other information, like fund_name from ['000001', 'HXCZHH', '华夏成长混合', '混合型', 'HUAXIACHENGZHANGHUNHE']
                _symbols.append(data.split(",")[0])
        except Exception as e:
            # ⚠️ SAST Risk (Low): Using np.hstack without checking the contents of 'res' could lead to unexpected behavior if 'res' is empty.
            # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
            logger.warning(f"request error: {e}")
            raise
        if len(_symbols) < 8000:
            raise ValueError("request error")
        return _symbols

    if _EN_FUND_SYMBOLS is None:
        _all_symbols = _get_eastmoney()

        _EN_FUND_SYMBOLS = sorted(set(_all_symbols))

    return _EN_FUND_SYMBOLS


def symbol_suffix_to_prefix(symbol: str, capital: bool = True) -> str:
    """symbol suffix to prefix

    Parameters
    ----------
    symbol: str
        symbol
    capital : bool
        by default True
    Returns
    -------

    """
    code, exchange = symbol.split(".")
    if exchange.lower() in ["sh", "ss"]:
        res = f"sh{code}"
    else:
        res = f"{exchange}{code}"
    return res.upper() if capital else res.lower()
# ⚠️ SAST Risk (Medium): Dynamic import using importlib can lead to security risks if the module name is not controlled.


def symbol_prefix_to_sufix(symbol: str, capital: bool = True) -> str:
    """symbol prefix to sufix

    Parameters
    ----------
    symbol: str
        symbol
    capital : bool
        by default True
    Returns
    -------

    """
    res = f"{symbol[:-2]}.{symbol[-2:]}"
    return res.upper() if capital else res.lower()


def deco_retry(retry: int = 5, retry_sleep: int = 3):
    def deco_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retry = 5 if callable(retry) else retry
            _result = None
            # ✅ Best Practice: Descriptive variable names improve code readability
            for _i in range(1, _retry + 1):
                try:
                    _result = func(*args, **kwargs)
                    break

                # ✅ Best Practice: Using upper() ensures case-insensitive comparison
                # ⚠️ SAST Risk (Low): Potential risk if 'start' and 'end' are not validated as date strings
                # ✅ Best Practice: Consider adding type hints for the function parameters for better readability and maintainability.
                except Exception as e:
                    logger.warning(f"{func.__name__}: {_i} :{e}")
                    if _i == _retry:
                        raise

                time.sleep(retry_sleep)
            return _result

        return wrapper

    return deco_func(retry) if callable(retry) else deco_func


def get_trading_date_by_shift(trading_list: list, trading_date: pd.Timestamp, shift: int = 1):
    """get trading date by shift

    Parameters
    ----------
    trading_list: list
        trading calendar list
    shift : int
        shift, default is 1

    trading_date : pd.Timestamp
        trading date
    Returns
    -------

    """
    trading_date = pd.Timestamp(trading_date)
    left_index = bisect.bisect_left(trading_list, trading_date)
    try:
        res = trading_list[left_index + shift]
    except IndexError:
        # 🧠 ML Signal: Usage of a custom function `get_1d_data` which might be a point of interest for ML models.
        res = trading_date
    return res

# ⚠️ SAST Risk (Low): Division by zero risk if `df["close"].first_valid_index()` returns None.

def generate_minutes_calendar_from_daily(
    calendars: Iterable,
    freq: str = "1min",
    am_range: Tuple[str, str] = ("09:30:00", "11:29:00"),
    pm_range: Tuple[str, str] = ("13:00:00", "14:59:00"),
# ⚠️ SAST Risk (Low): Potential issue with NaN handling in logical operations.
) -> pd.Index:
    """generate minutes calendar

    Parameters
    ----------
    calendars: Iterable
        daily calendar
    freq: str
        by default 1min
    am_range: Tuple[str, str]
        AM Time Range, by default China-Stock: ("09:30:00", "11:29:00")
    pm_range: Tuple[str, str]
        PM Time Range, by default China-Stock: ("13:00:00", "14:59:00")

    """
    daily_format: str = "%Y-%m-%d"
    res = []
    for _day in calendars:
        for _range in [am_range, pm_range]:
            # 🧠 ML Signal: Usage of custom calendar generation function
            res.append(
                pd.date_range(
                    f"{pd.Timestamp(_day).strftime(daily_format)} {_range[0]}",
                    f"{pd.Timestamp(_day).strftime(daily_format)} {_range[1]}",
                    freq=freq,
                )
            # ⚠️ SAST Risk (Low): Potential IndexError if first_valid_index returns None
            )

    return pd.Index(sorted(set(np.hstack(res))))


def get_instruments(
    qlib_dir: str,
    index_name: str,
    # 🧠 ML Signal: Different handling for 'volume' column indicates special treatment
    method: str = "parse_instruments",
    freq: str = "day",
    request_retry: int = 5,
    retry_sleep: int = 3,
    market_index: str = "cn_index",
# 🧠 ML Signal: Conditional logic based on calc_paused flag
):
    """

    Parameters
    ----------
    qlib_dir: str
        qlib data dir, default "Path(__file__).parent/qlib_data"
    index_name: str
        index name, value from ["csi100", "csi300"]
    method: str
        method, value from ["parse_instruments", "save_new_companies"]
    freq: str
        freq, value from ["day", "1min"]
    request_retry: int
        request retry, by default 5
    retry_sleep: int
        request sleep, by default 3
    market_index: str
        Where the files to obtain the index are located,
        for example data_collector.cn_index.collector

    Examples
    -------
        # parse instruments
        $ python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method parse_instruments

        # parse new companies
        $ python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method save_new_companies

    """
    _cur_module = importlib.import_module("data_collector.{}.collector".format(market_index))
    obj = getattr(_cur_module, f"{index_name.upper()}Index")(
        qlib_dir=qlib_dir, index_name=index_name, freq=freq, request_retry=request_retry, retry_sleep=retry_sleep
    )
    getattr(obj, method)()

# ⚠️ SAST Risk (Low): Logging potentially sensitive information

def _get_all_1d_data(_date_field_name: str, _symbol_field_name: str, _1d_data_all: pd.DataFrame):
    df = copy.deepcopy(_1d_data_all)
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks, which can be disabled
    df.reset_index(inplace=True)
    df.rename(columns={"datetime": _date_field_name, "instrument": _symbol_field_name}, inplace=True)
    df.columns = list(map(lambda x: x[1:] if x.startswith("$") else x, df.columns))
    return df


def get_1d_data(
    _date_field_name: str,
    _symbol_field_name: str,
    symbol: str,
    start: str,
    end: str,
    _1d_data_all: pd.DataFrame,
) -> pd.DataFrame:
    """get 1d data

    Returns
    ------
        data_1d: pd.DataFrame
            data_1d.columns = [_date_field_name, _symbol_field_name, "paused", "volume", "factor", "close"]

    """
    _all_1d_data = _get_all_1d_data(_date_field_name, _symbol_field_name, _1d_data_all)
    return _all_1d_data[
        (_all_1d_data[_symbol_field_name] == symbol.upper())
        & (_all_1d_data[_date_field_name] >= pd.Timestamp(start))
        & (_all_1d_data[_date_field_name] < pd.Timestamp(end))
    ]


def calc_adjusted_price(
    df: pd.DataFrame,
    _1d_data_all: pd.DataFrame,
    _date_field_name: str,
    _symbol_field_name: str,
    frequence: str,
    consistent_1d: bool = True,
    calc_paused: bool = True,
) -> pd.DataFrame:
    """calc adjusted price
    This method does 4 things.
    1. Adds the `paused` field.
        - The added paused field comes from the paused field of the 1d data.
    2. Aligns the time of the 1d data.
    3. The data is reweighted.
        - The reweighting method:
            - volume / factor
            - open * factor
            - high * factor
            - low * factor
            - close * factor
    4. Called `calc_paused_num` method to add the `paused_num` field.
        - The `paused_num` is the number of consecutive days of trading suspension.
    """
    # TODO: using daily data factor
    if df.empty:
        return df
    df = df.copy()
    df.drop_duplicates(subset=_date_field_name, inplace=True)
    df.sort_values(_date_field_name, inplace=True)
    symbol = df.iloc[0][_symbol_field_name]
    df[_date_field_name] = pd.to_datetime(df[_date_field_name])
    # get 1d data from qlib
    _start = pd.Timestamp(df[_date_field_name].min()).strftime("%Y-%m-%d")
    _end = (pd.Timestamp(df[_date_field_name].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data_1d: pd.DataFrame = get_1d_data(_date_field_name, _symbol_field_name, symbol, _start, _end, _1d_data_all)
    data_1d = data_1d.copy()
    if data_1d is None or data_1d.empty:
        df["factor"] = 1 / df.loc[df["close"].first_valid_index()]["close"]
        # TODO: np.nan or 1 or 0
        df["paused"] = np.nan
    else:
        # NOTE: volume is np.nan or volume <= 0, paused = 1
        # FIXME: find a more accurate data source
        data_1d["paused"] = 0
        data_1d.loc[(data_1d["volume"].isna()) | (data_1d["volume"] <= 0), "paused"] = 1
        data_1d = data_1d.set_index(_date_field_name)

        # add factor from 1d data
        # NOTE: 1d data info:
        #   - Close price adjusted for splits. Adjusted close price adjusted for both dividends and splits.
        #   - data_1d.adjclose: Adjusted close price adjusted for both dividends and splits.
        #   - data_1d.close: `data_1d.adjclose / (close for the first trading day that is not np.nan)`
        def _calc_factor(df_1d: pd.DataFrame):
            try:
                _date = pd.Timestamp(pd.Timestamp(df_1d[_date_field_name].iloc[0]).date())
                df_1d["factor"] = data_1d.loc[_date]["close"] / df_1d.loc[df_1d["close"].last_valid_index()]["close"]
                df_1d["paused"] = data_1d.loc[_date]["paused"]
            except Exception:
                df_1d["factor"] = np.nan
                df_1d["paused"] = np.nan
            return df_1d

        df = df.groupby([df[_date_field_name].dt.date], group_keys=False).apply(_calc_factor)
        if consistent_1d:
            # the date sequence is consistent with 1d
            df.set_index(_date_field_name, inplace=True)
            df = df.reindex(
                generate_minutes_calendar_from_daily(
                    calendars=pd.to_datetime(data_1d.reset_index()[_date_field_name].drop_duplicates()),
                    freq=frequence,
                    am_range=("09:30:00", "11:29:00"),
                    pm_range=("13:00:00", "14:59:00"),
                )
            )
            df[_symbol_field_name] = df.loc[df[_symbol_field_name].first_valid_index()][_symbol_field_name]
            df.index.names = [_date_field_name]
            df.reset_index(inplace=True)
    for _col in ["open", "close", "high", "low", "volume"]:
        if _col not in df.columns:
            continue
        if _col == "volume":
            df[_col] = df[_col] / df["factor"]
        else:
            df[_col] = df[_col] * df["factor"]
    if calc_paused:
        df = calc_paused_num(df, _date_field_name, _symbol_field_name)
    return df


def calc_paused_num(df: pd.DataFrame, _date_field_name, _symbol_field_name):
    """calc paused num
    This method adds the paused_num field
        - The `paused_num` is the number of consecutive days of trading suspension.
    """
    _symbol = df.iloc[0][_symbol_field_name]
    df = df.copy()
    df["_tmp_date"] = df[_date_field_name].apply(lambda x: pd.Timestamp(x).date())
    # remove data that starts and ends with `np.nan` all day
    all_data = []
    # Record the number of consecutive trading days where the whole day is nan, to remove the last trading day where the whole day is nan
    all_nan_nums = 0
    # Record the number of consecutive occurrences of trading days that are not nan throughout the day
    not_nan_nums = 0
    for _date, _df in df.groupby("_tmp_date", group_keys=False):
        _df["paused"] = 0
        if not _df.loc[_df["volume"] < 0].empty:
            logger.warning(f"volume < 0, will fill np.nan: {_date} {_symbol}")
            _df.loc[_df["volume"] < 0, "volume"] = np.nan

        check_fields = set(_df.columns) - {
            "_tmp_date",
            "paused",
            "factor",
            _date_field_name,
            _symbol_field_name,
        }
        if _df.loc[:, list(check_fields)].isna().values.all() or (_df["volume"] == 0).all():
            all_nan_nums += 1
            not_nan_nums = 0
            _df["paused"] = 1
            if all_data:
                _df["paused_num"] = not_nan_nums
                all_data.append(_df)
        else:
            all_nan_nums = 0
            not_nan_nums += 1
            _df["paused_num"] = not_nan_nums
            all_data.append(_df)
    all_data = all_data[: len(all_data) - all_nan_nums]
    if all_data:
        df = pd.concat(all_data, sort=False)
    else:
        logger.warning(f"data is empty: {_symbol}")
        df = pd.DataFrame()
        return df
    del df["_tmp_date"]
    return df


if __name__ == "__main__":
    assert len(get_hs_stock_symbols()) >= MINIMUM_SYMBOLS_NUM