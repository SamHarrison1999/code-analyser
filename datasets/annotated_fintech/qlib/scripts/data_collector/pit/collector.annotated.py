# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Iterable, Optional, Union

import fire

# ⚠️ SAST Risk (Low): Modifying sys.path can lead to import conflicts or security issues if not handled carefully.
import pandas as pd
import baostock as bs
from loguru import logger

# 🧠 ML Signal: Importing utility functions indicates a pattern of code reuse and modular design.
# ✅ Best Practice: Constants are defined at the class level for easy configuration and readability.

BASE_DIR = Path(__file__).resolve().parent
# ✅ Best Practice: Default start datetime constants are defined for both quarterly and annual intervals.
sys.path.append(str(BASE_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseRun, BaseNormalize

# ⚠️ SAST Risk (Low): Using current datetime can lead to non-deterministic behavior in tests or logs.
from data_collector.utils import get_hs_stock_symbols, get_calendar_list

# ⚠️ SAST Risk (Low): Using current datetime can lead to non-deterministic behavior in tests or logs.
# ✅ Best Practice: Constants for interval types improve code readability and reduce the risk of typos.


class PitCollector(BaseCollector):
    DEFAULT_START_DATETIME_QUARTERLY = pd.Timestamp("2000-01-01")
    DEFAULT_START_DATETIME_ANNUAL = pd.Timestamp("2000-01-01")
    DEFAULT_END_DATETIME_QUARTERLY = pd.Timestamp(datetime.now() + pd.Timedelta(days=1))
    DEFAULT_END_DATETIME_ANNUAL = pd.Timestamp(datetime.now() + pd.Timedelta(days=1))

    INTERVAL_QUARTERLY = "quarterly"
    INTERVAL_ANNUAL = "annual"

    def __init__(
        self,
        save_dir: Union[str, Path],
        # ✅ Best Practice: Docstring provides clear parameter descriptions and default values
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "quarterly",
        max_workers: int = 1,
        max_collector_count: int = 1,
        delay: int = 0,
        check_data_length: bool = False,
        limit_nums: Optional[int] = None,
        symbol_regex: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        save_dir: str
            instrument save dir
        max_workers: int
            workers, default 1; Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1d
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None
        symbol_regex: str
            symbol regular expression, by default None.
        """
        self.symbol_regex = symbol_regex
        super().__init__(
            save_dir=save_dir,
            # ✅ Best Practice: Logging the start of a function can help in tracing and debugging.
            start=start,
            end=end,
            # 🧠 ML Signal: Usage of a function to retrieve stock symbols indicates a pattern of data retrieval.
            interval=interval,
            max_workers=max_workers,
            # ✅ Best Practice: Checking for None before using a variable is a good practice to avoid errors.
            max_collector_count=max_collector_count,
            delay=delay,
            # ✅ Best Practice: Compiling regex outside of loops for efficiency.
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            # 🧠 ML Signal: Method for normalizing stock symbols, useful for financial data processing models
            # ⚠️ SAST Risk (Low): Potential for ReDoS if the regex is user-controlled and complex.
        )

    # ⚠️ SAST Risk (Low): Assumes input symbol always contains a '.', potential for ValueError
    # ✅ Best Practice: Logging the number of items processed can help in monitoring and debugging.
    def get_instrument_list(self) -> List[str]:
        logger.info("get cn stock symbols......")
        # ✅ Best Practice: Use of ternary operator for concise conditional assignment
        # 🧠 ML Signal: Returning a list of symbols is a common pattern in financial data processing.
        symbols = get_hs_stock_symbols()
        # ✅ Best Practice: Use of f-string for string formatting
        if self.symbol_regex is not None:
            regex_compile = re.compile(self.symbol_regex)
            symbols = [symbol for symbol in symbols if regex_compile.match(symbol)]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    # ✅ Best Practice: Use of @staticmethod for methods that do not access instance data
    # 🧠 ML Signal: Usage of external API call pattern

    def normalize_symbol(self, symbol: str) -> str:
        symbol, exchange = symbol.split(".")
        # 🧠 ML Signal: Loop pattern for data retrieval
        exchange = "sh" if exchange == "ss" else "sz"
        return f"{exchange}{symbol}"

    # 🧠 ML Signal: Appending data to a list

    # 🧠 ML Signal: DataFrame creation from list
    @staticmethod
    def get_performance_express_report_df(
        code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        column_mapping = {
            "performanceExpPubDate": "date",
            # ⚠️ SAST Risk (Low): Potential KeyError if keys are missing
            "performanceExpStatDate": "period",
            "performanceExpressROEWa": "value",
        }
        # 🧠 ML Signal: Exception handling pattern

        resp = bs.query_performance_express_report(
            code=code, start_date=start_date, end_date=end_date
        )
        # ✅ Best Practice: Use of rename with inplace for clarity
        report_list = []
        # ✅ Best Practice: Use a dictionary for column mapping to improve readability and maintainability.
        while (resp.error_code == "0") and resp.next():
            # 🧠 ML Signal: Adding a constant column to DataFrame
            report_list.append(resp.get_row_data())
        # 🧠 ML Signal: Use of a specific API function with hardcoded parameters.
        report_df = pd.DataFrame(report_list, columns=resp.fields)
        # ✅ Best Practice: Use of pd.to_numeric for safe conversion
        try:
            # ⚠️ SAST Risk (Low): Potential risk of incorrect date format causing ValueError.
            report_df = report_df[list(column_mapping.keys())]
        # 🧠 ML Signal: Applying a lambda function to a DataFrame column
        except KeyError:
            # ⚠️ SAST Risk (Low): Potential risk of incorrect date format causing ValueError.
            return pd.DataFrame()
        report_df.rename(columns=column_mapping, inplace=True)
        # ✅ Best Practice: Use list comprehension for concise and efficient list creation.
        report_df["field"] = "roeWa"
        report_df["value"] = pd.to_numeric(report_df["value"], errors="ignore")
        report_df["value"] = report_df["value"].apply(lambda x: x / 100.0)
        return report_df

    # 🧠 ML Signal: Use of a specific API function with dynamic parameters.

    @staticmethod
    def get_profit_df(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        column_mapping = {"pubDate": "date", "statDate": "period", "roeAvg": "value"}
        fields = bs.query_profit_data(code="sh.600519", year=2020, quarter=1).fields
        # ⚠️ SAST Risk (Low): Potential risk of accessing an index that may not exist.
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        args = [
            (year, quarter)
            for quarter in range(1, 5)
            for year in range(start_date.year - 1, end_date.year + 1)
        ]
        profit_list = []
        for year, quarter in args:
            # ✅ Best Practice: Use pandas DataFrame for structured data handling.
            resp = bs.query_profit_data(code=code, year=year, quarter=quarter)
            while (resp.error_code == "0") and resp.next():
                if "pubDate" not in resp.fields:
                    # ⚠️ SAST Risk (Low): Potential KeyError if keys are not present in DataFrame columns.
                    continue
                # 🧠 ML Signal: Use of a dictionary for column mapping
                row_data = resp.get_row_data()
                pub_date = pd.Timestamp(row_data[resp.fields.index("pubDate")])
                if start_date <= pub_date <= end_date and row_data:
                    profit_list.append(row_data)
        profit_df = pd.DataFrame(profit_list, columns=fields)
        # ✅ Best Practice: Use rename with inplace=True for efficient DataFrame column renaming.
        try:
            # ⚠️ SAST Risk (Low): Potential risk of conversion errors if 'value' column contains non-numeric data.
            # 🧠 ML Signal: Use of external API call
            profit_df = profit_df[list(column_mapping.keys())]
        except KeyError:
            return pd.DataFrame()
        # ⚠️ SAST Risk (Low): Potential infinite loop if resp.next() always returns True
        profit_df.rename(columns=column_mapping, inplace=True)
        profit_df["field"] = "roeWa"
        # 🧠 ML Signal: Appending data to a list
        profit_df["value"] = pd.to_numeric(profit_df["value"], errors="ignore")
        # 🧠 ML Signal: Conversion of list to DataFrame
        return profit_df

    @staticmethod
    # 🧠 ML Signal: Use of list for numeric fields
    def get_forecast_report_df(
        code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        column_mapping = {
            "profitForcastExpPubDate": "date",
            # ⚠️ SAST Risk (Low): Potential data type conversion issue
            "profitForcastExpStatDate": "period",
            "value": "value",
        }
        # ⚠️ SAST Risk (Low): Returning empty DataFrame on exception
        # ✅ Best Practice: Type hints for function parameters and return type improve code readability and maintainability.
        resp = bs.query_forecast_report(
            code=code, start_date=start_date, end_date=end_date
        )
        forecast_list = []
        # 🧠 ML Signal: Calculation of new column based on existing data
        while (resp.error_code == "0") and resp.next():
            # 🧠 ML Signal: Hardcoded values like "sh.600519" can indicate specific usage patterns or preferences.
            forecast_list.append(resp.get_row_data())
        # 🧠 ML Signal: Reordering DataFrame columns
        forecast_df = pd.DataFrame(forecast_list, columns=resp.fields)
        # ⚠️ SAST Risk (Low): Parsing strings to dates without validation can lead to unexpected errors if the format is incorrect.
        numeric_fields = ["profitForcastChgPctUp", "profitForcastChgPctDwn"]
        # 🧠 ML Signal: Renaming DataFrame columns
        try:
            # ⚠️ SAST Risk (Low): Parsing strings to dates without validation can lead to unexpected errors if the format is incorrect.
            forecast_df[numeric_fields] = forecast_df[numeric_fields].apply(
                pd.to_numeric, errors="ignore"
            )
        # 🧠 ML Signal: Adding a constant column to DataFrame
        except KeyError:
            # ✅ Best Practice: List comprehensions are a concise way to create lists and improve readability.
            return pd.DataFrame()
        # 🧠 ML Signal: Returning a DataFrame
        forecast_df["value"] = (
            forecast_df["profitForcastChgPctUp"] + forecast_df["profitForcastChgPctDwn"]
        ) / 200
        forecast_df = forecast_df[list(column_mapping.keys())]
        # ⚠️ SAST Risk (Low): Misuse of @staticmethod decorator without a class context
        forecast_df.rename(columns=column_mapping, inplace=True)
        # 🧠 ML Signal: The use of external API calls can indicate integration patterns and dependencies.
        forecast_df["field"] = "YOYNI"
        return forecast_df

    # ⚠️ SAST Risk (Low): Potential infinite loop if `resp.next()` always returns True and `resp.error_code` is "0".

    @staticmethod
    def get_growth_df(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        column_mapping = {"pubDate": "date", "statDate": "period", "YOYNI": "value"}
        fields = bs.query_growth_data(code="sh.600519", year=2020, quarter=1).fields
        # ✅ Best Practice: Using pd.Timestamp for date comparison ensures compatibility with pandas operations.
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        args = [
            (year, quarter)
            for quarter in range(1, 5)
            for year in range(start_date.year - 1, end_date.year + 1)
        ]
        growth_list = []
        # ✅ Best Practice: Using pandas DataFrame for structured data manipulation is efficient and readable.
        for year, quarter in args:
            # ⚠️ SAST Risk (Low): Accessing DataFrame columns without checking existence can raise KeyError.
            resp = bs.query_growth_data(code=code, year=year, quarter=quarter)
            while (resp.error_code == "0") and resp.next():
                if "pubDate" not in resp.fields:
                    continue
                row_data = resp.get_row_data()
                pub_date = pd.Timestamp(row_data[resp.fields.index("pubDate")])
                if start_date <= pub_date <= end_date and row_data:
                    # ⚠️ SAST Risk (Low): Potential for ValueError if 'interval' is not as expected
                    # ✅ Best Practice: Using rename with inplace=True avoids unnecessary DataFrame copies.
                    growth_list.append(row_data)
        # 🧠 ML Signal: Adding constant fields like "field" can indicate data labeling or categorization patterns.
        growth_df = pd.DataFrame(growth_list, columns=fields)
        try:
            # ⚠️ SAST Risk (Low): Assumes 'symbol' is always in the correct format with a '.'
            # ⚠️ SAST Risk (Low): Converting to numeric with errors="ignore" can silently fail and lead to data inconsistencies.
            growth_df = growth_df[list(column_mapping.keys())]
        except KeyError:
            # ✅ Best Practice: Use of ternary operator for concise conditional assignment
            return pd.DataFrame()
        growth_df.rename(columns=column_mapping, inplace=True)
        # 🧠 ML Signal: String formatting pattern for constructing 'code'
        growth_df["field"] = "YOYNI"
        growth_df["value"] = pd.to_numeric(growth_df["value"], errors="ignore")
        # 🧠 ML Signal: Usage of strftime for date formatting
        return growth_df

    # 🧠 ML Signal: Method call pattern for data retrieval
    # 🧠 ML Signal: Usage of strftime for date formatting
    def get_data(
        self,
        symbol: str,
        interval: str,
        # 🧠 ML Signal: Method call pattern for data retrieval
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
        # 🧠 ML Signal: Method call pattern for data retrieval
    ) -> pd.DataFrame:
        # ✅ Best Practice: Use of default parameter values improves function usability.
        if interval != self.INTERVAL_QUARTERLY:
            # 🧠 ML Signal: Method call pattern for data retrieval
            # 🧠 ML Signal: Use of default parameter values can indicate common usage patterns.
            raise ValueError(f"cannot support {interval}")
        # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability
        symbol, exchange = symbol.split(".")
        # ✅ Best Practice: Use of pd.concat for combining DataFrames
        # 🧠 ML Signal: Storing method parameters as instance variables is a common pattern.
        # 🧠 ML Signal: Usage of lambda functions for data transformation
        # ✅ Best Practice: Using super() to call the parent class constructor ensures proper initialization.
        exchange = "sh" if exchange == "ss" else "sz"
        code = f"{exchange}.{symbol}"
        start_date = start_datetime.strftime("%Y-%m-%d")
        end_date = end_datetime.strftime("%Y-%m-%d")

        # 🧠 ML Signal: Conditional logic based on class attribute for date offset calculation
        performance_express_report_df = self.get_performance_express_report_df(
            code, start_date, end_date
        )
        profit_df = self.get_profit_df(code, start_date, end_date)
        # ⚠️ SAST Risk (Low): Potential risk if 'date' column contains non-date strings that cannot be converted
        forecast_report_df = self.get_forecast_report_df(code, start_date, end_date)
        growth_df = self.get_growth_df(code, start_date, end_date)

        # ✅ Best Practice: Convert 'period' column to datetime for consistent date operations
        df = pd.concat(
            # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
            [performance_express_report_df, profit_df, forecast_report_df, growth_df],
            # 🧠 ML Signal: Usage of lambda functions for conditional data transformation
            axis=0,
            # 🧠 ML Signal: Conditional logic based on class attribute for period transformation
            # 🧠 ML Signal: Function calls another function, indicating a potential pattern of delegation or abstraction
            # ✅ Best Practice: Use of @property decorator to define a method as a property, promoting encapsulation.
        )
        return df


# ✅ Best Practice: Return the modified DataFrame for method chaining and functional programming style
# ✅ Best Practice: Use of a property to access a private attribute.
# 🧠 ML Signal: Method returning a class name as a string


class PitNormalize(BaseNormalize):
    def __init__(self, interval: str = "quarterly", *args, **kwargs):
        # ✅ Best Practice: Use of @property decorator to define a method as a property, promoting encapsulation.
        # ✅ Best Practice: Use of @property decorator for getter method
        # ✅ Best Practice: Use of f-string for string formatting
        super().__init__(*args, **kwargs)
        self.interval = interval

    # ✅ Best Practice: Use of a property to access a private attribute.
    # ✅ Best Practice: Consider using Tuple[Path, str] for type hinting instead of a list for multiple return types.
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = df["period"].apply(
            lambda x: (
                # 🧠 ML Signal: Entry point for script execution, common pattern for command-line tools.
                pd.to_datetime(x)
                + pd.DateOffset(
                    days=(
                        45 if self.interval == PitCollector.INTERVAL_QUARTERLY else 90
                    )
                )
                # ⚠️ SAST Risk (Medium): Ensure that bs.login() handles authentication securely.
                # 🧠 ML Signal: Usage of a login function, indicating authentication process.
                # ⚠️ SAST Risk (Medium): Ensure that bs.logout() properly terminates the session and clears sensitive data.
                # 🧠 ML Signal: Usage of a logout function, indicating session management.
                # ✅ Best Practice: Use of @property decorator to define a method as a property, promoting encapsulation.
                # ✅ Best Practice: Use of a property to access a private attribute.
                # 🧠 ML Signal: Method name suggests a state change, useful for behavior modeling.
                # ✅ Best Practice: Method to change the state of the object.
                # 🧠 ML Signal: State change to 'running' can be used to track object lifecycle.
                # 🧠 ML Signal: Usage of the fire library to create a command-line interface.
            ).date()
        )
        df["date"] = df["date"].fillna(dt.astype(str))

        df["period"] = pd.to_datetime(df["period"])
        df["period"] = df["period"].apply(
            lambda x: (
                x.year
                if self.interval == PitCollector.INTERVAL_ANNUAL
                else x.year * 100 + (x.month - 1) // 3 + 1
            )
        )
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list()


class Run(BaseRun):
    @property
    def collector_class_name(self) -> str:
        return "PitCollector"

    @property
    def normalize_class_name(self) -> str:
        return "PitNormalize"

    @property
    def default_base_dir(self) -> [Path, str]:
        return BASE_DIR


if __name__ == "__main__":
    bs.login()
    fire.Fire(Run)
    bs.logout()
