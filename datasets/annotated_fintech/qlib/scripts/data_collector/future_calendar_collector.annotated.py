# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import importlib
from pathlib import Path
from typing import Union, Iterable, List

import fire
# ✅ Best Practice: Class-level constants improve readability and maintainability
# ✅ Best Practice: Use of 'loguru' for logging provides advanced logging features and better readability.
import numpy as np
import pandas as pd
# ✅ Best Practice: Using a descriptive constant name for date format

# ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
# pip install baostock
import baostock as bs
from loguru import logger


class CollectorFutureCalendar:
    calendar_format = "%Y-%m-%d"

    def __init__(self, qlib_dir: Union[str, Path], start_date: str = None, end_date: str = None):
        """

        Parameters
        ----------
        qlib_dir:
            qlib data directory
        start_date
            start date
        end_date
            end date
        # ✅ Best Practice: Check if the file exists before attempting to read it
        """
        # ✅ Best Practice: Use of pd.Timestamp for date conversion ensures consistency and correctness.
        self.qlib_dir = Path(qlib_dir).expanduser().absolute()
        # ⚠️ SAST Risk (Low): Raising a generic ValueError without logging the error
        self.calendar_path = self.qlib_dir.joinpath("calendars/day.txt")
        # ✅ Best Practice: Use of pd.Timedelta for date arithmetic ensures consistency and correctness.
        self.future_path = self.qlib_dir.joinpath("calendars/day_future.txt")
        # 🧠 ML Signal: Reading a CSV file into a DataFrame
        self._calendar_list = self.calendar_list
        _latest_date = self._calendar_list[-1]
        # ✅ Best Practice: Assigning column names to the DataFrame for clarity
        # ✅ Best Practice: Type hint for datetime_d should use Union for better clarity
        self.start_date = _latest_date if start_date is None else pd.Timestamp(start_date)
        self.end_date = _latest_date + pd.Timedelta(days=365 * 2) if end_date is None else pd.Timestamp(end_date)
    # ✅ Best Practice: Convert input to a consistent type at the start of the function
    # 🧠 ML Signal: Converting a DataFrame column to datetime

    # ✅ Best Practice: Type hint for 'calendar' parameter improves code readability and maintainability
    @property
    # 🧠 ML Signal: Usage of strftime indicates a pattern of date formatting
    # 🧠 ML Signal: Converting a DataFrame column to a list
    def calendar_list(self) -> List[pd.Timestamp]:
        # 🧠 ML Signal: Use of list comprehension for data transformation
        # 🧠 ML Signal: Use of set to remove duplicates
        # load old calendar
        if not self.calendar_path.exists():
            # ✅ Best Practice: Include a docstring description for the method's purpose and return value
            raise ValueError(f"calendar does not exist: {self.calendar_path}")
        # ⚠️ SAST Risk (Low): Potential risk if 'self.future_path' is user-controlled, leading to file overwrite
        # 🧠 ML Signal: Use of numpy's savetxt for writing data to a file
        calendar_df = pd.read_csv(self.calendar_path, header=None)
        calendar_df.columns = ["date"]
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])
        return calendar_df["date"].to_list()
    # ✅ Best Practice: Use of abstractmethod indicates this method should be implemented by subclasses

    # ✅ Best Practice: Use NotImplementedError to indicate an abstract method that should be implemented by subclasses
    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        # ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
        datetime_d = pd.Timestamp(datetime_d)
        # 🧠 ML Signal: Login pattern with error handling
        return datetime_d.strftime(self.calendar_format)

    # ⚠️ SAST Risk (Low): Potential exposure of error messages
    def write_calendar(self, calendar: Iterable):
        calendars_list = [self._format_datetime(x) for x in sorted(set(self.calendar_list + calendar))]
        np.savetxt(self.future_path, calendars_list, fmt="%s", encoding="utf-8")

    # 🧠 ML Signal: Query pattern with error handling
    @abc.abstractmethod
    def collector(self) -> Iterable[pd.Timestamp]:
        """

        Returns
        -------

        # ✅ Best Practice: Use of while loop with condition for data collection
        """
        raise NotImplementedError(f"Please implement the `collector` method")
# ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.

# ✅ Best Practice: Use of pandas DataFrame for structured data handling

# ⚠️ SAST Risk (Low): Raising a generic exception without specific handling can lead to unhandled exceptions
class CollectorFutureCalendarCN(CollectorFutureCalendar):
    # ✅ Best Practice: Explicit type conversion for clarity
    # ✅ Best Practice: Use of pandas to_datetime for date conversion
    # ✅ Best Practice: Docstring provides clear documentation of parameters and usage.
    def collector(self) -> Iterable[pd.Timestamp]:
        lg = bs.login()
        if lg.error_code != "0":
            raise ValueError(f"login respond error_msg: {lg.error_msg}")
        rs = bs.query_trade_dates(
            start_date=self._format_datetime(self.start_date), end_date=self._format_datetime(self.end_date)
        )
        if rs.error_code != "0":
            raise ValueError(f"query_trade_dates respond error_msg: {rs.error_msg}")
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        calendar = pd.DataFrame(data_list, columns=rs.fields)
        calendar["is_trading_day"] = calendar["is_trading_day"].astype(int)
        return pd.to_datetime(calendar[calendar["is_trading_day"] == 1]["calendar_date"]).to_list()

# 🧠 ML Signal: Logging usage pattern for tracking execution and debugging.

class CollectorFutureCalendarUS(CollectorFutureCalendar):
    # ⚠️ SAST Risk (Medium): Dynamic import and attribute access can lead to code execution risks if inputs are not controlled.
    def collector(self) -> Iterable[pd.Timestamp]:
        # TODO: US future calendar
        # ⚠️ SAST Risk (Medium): Dynamic import and attribute access can lead to code execution risks if inputs are not controlled.
        raise ValueError("Us calendar is not supported")
# 🧠 ML Signal: Instantiation pattern of a class with dynamic attributes.
# 🧠 ML Signal: Method chaining pattern for executing class methods.
# 🧠 ML Signal: Entry point pattern for command-line interface applications.
# ⚠️ SAST Risk (Low): Using fire.Fire can execute arbitrary code if user input is not sanitized.


def run(qlib_dir: Union[str, Path], region: str = "cn", start_date: str = None, end_date: str = None):
    """Collect future calendar(day)

    Parameters
    ----------
    qlib_dir:
        qlib data directory
    region:
        cn/CN or us/US
    start_date
        start date
    end_date
        end date

    Examples
    -------
        # get cn future calendar
        $ python future_calendar_collector.py --qlib_data_1d_dir <user data dir> --region cn
    """
    logger.info(f"collector future calendar: region={region}")
    _cur_module = importlib.import_module("future_calendar_collector")
    _class = getattr(_cur_module, f"CollectorFutureCalendar{region.upper()}")
    collector = _class(qlib_dir=qlib_dir, start_date=start_date, end_date=end_date)
    collector.write_calendar(collector.collector())


if __name__ == "__main__":
    fire.Fire(run)