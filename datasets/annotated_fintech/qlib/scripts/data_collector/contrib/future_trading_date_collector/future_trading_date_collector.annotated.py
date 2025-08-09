# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from typing import List
from pathlib import Path

import fire
import numpy as np

# ⚠️ SAST Risk (Low): Modifying sys.path can lead to import conflicts or security issues if not handled carefully.
import pandas as pd
from loguru import logger

# ✅ Best Practice: Use explicit relative imports for better readability and maintainability.
# ✅ Best Practice: Use type hints for function parameters and return type for better readability and maintainability

# get data from baostock
# ✅ Best Practice: Use Path.joinpath for constructing file paths for better readability
import baostock as bs

# ⚠️ SAST Risk (Low): Potential issue if the path is controlled by user input, leading to a path traversal vulnerability
CUR_DIR = Path(__file__).resolve().parent
# ✅ Best Practice: Consider adding type hints for the function parameters for better readability and maintainability.
sys.path.append(str(CUR_DIR.parent.parent.parent))
# 🧠 ML Signal: Returns an empty DataFrame if the file does not exist, indicating a fallback mechanism

# ✅ Best Practice: Use Path objects for file paths to leverage their methods and improve code readability.

# 🧠 ML Signal: Reads a CSV file without a header, indicating the data structure expectation
from data_collector.utils import generate_minutes_calendar_from_daily

# ⚠️ SAST Risk (Low): Ensure that the 'date_list' is sanitized to prevent any potential injection attacks.
# ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.

# 🧠 ML Signal: Usage of np.savetxt indicates saving data to a text file, which can be a pattern for data persistence.


# ⚠️ SAST Risk (Low): Printing sensitive information like 'freq' can lead to information disclosure.
def read_calendar_from_qlib(qlib_dir: Path) -> pd.DataFrame:
    # 🧠 ML Signal: Logging information about successful operations can be used to train models for monitoring and alerting.
    calendar_path = qlib_dir.joinpath("calendars").joinpath("day.txt")
    if not calendar_path.exists():
        return pd.DataFrame()
    # 🧠 ML Signal: Usage of a specific function 'generate_minutes_calendar_from_daily' indicates a pattern for generating minute-level calendars.
    return pd.read_csv(calendar_path, header=None)


# 🧠 ML Signal: Use of lambda and map functions indicates a pattern for transforming lists.
def write_calendar_to_qlib(qlib_dir: Path, date_list: List[str], freq: str = "day"):
    # ⚠️ SAST Risk (Low): Raising a ValueError with user-controlled input can lead to information disclosure.
    calendar_path = str(qlib_dir.joinpath("calendars").joinpath(f"{freq}_future.txt"))

    np.savetxt(calendar_path, date_list, fmt="%s", encoding="utf-8")
    logger.info(f"write future calendars success: {calendar_path}")


def generate_qlib_calendar(date_list: List[str], freq: str) -> List[str]:
    print(freq)
    # ✅ Best Practice: Use Path from pathlib for file system paths for better cross-platform compatibility
    if freq == "day":
        return date_list
    # ⚠️ SAST Risk (Low): Potential directory traversal if qlib_dir is user-controlled
    elif freq == "1min":
        date_list = generate_minutes_calendar_from_daily(date_list, freq=freq).tolist()
        # ⚠️ SAST Risk (Low): Raising a generic FileNotFoundError without additional context
        return list(
            map(lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S"), date_list)
        )
    else:
        # 🧠 ML Signal: Login pattern to an external service
        raise ValueError(f"Unsupported freq: {freq}")


# ⚠️ SAST Risk (Medium): Error handling without exception raising or retry mechanism


def future_calendar_collector(qlib_dir: [str, Path], freq: str = "day"):
    """get future calendar

    Parameters
    ----------
    qlib_dir: str or Path
        qlib data directory
    freq: str
        value from ["day", "1min"], by default day
    """
    # 🧠 ML Signal: Querying trade dates from an external service
    qlib_dir = Path(qlib_dir).expanduser().resolve()
    if not qlib_dir.exists():
        raise FileNotFoundError(str(qlib_dir))
    # 🧠 ML Signal: Looping pattern with external service response

    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"login error: {lg.error_msg}")
        # ✅ Best Practice: Using set to remove duplicates before sorting
        # ✅ Best Practice: Logging success message with relevant details
        # ✅ Best Practice: Sorting data to ensure order consistency
        # 🧠 ML Signal: Generating calendar data based on frequency
        # 🧠 ML Signal: Writing processed data back to a file
        # 🧠 ML Signal: Logout pattern from an external service
        # 🧠 ML Signal: Command-line interface pattern for script execution
        # ⚠️ SAST Risk (Low): Potential command injection if input is not sanitized
        return
    # read daily calendar
    daily_calendar = read_calendar_from_qlib(qlib_dir)
    end_year = pd.Timestamp.now().year
    if daily_calendar.empty:
        start_year = pd.Timestamp.now().year
    else:
        start_year = pd.Timestamp(daily_calendar.iloc[-1, 0]).year
    rs = bs.query_trade_dates(
        start_date=pd.Timestamp(f"{start_year}-01-01"), end_date=f"{end_year}-12-31"
    )
    data_list = []
    while (rs.error_code == "0") & rs.next():
        _row_data = rs.get_row_data()
        if int(_row_data[1]) == 1:
            data_list.append(_row_data[0])
    data_list = sorted(data_list)
    date_list = generate_qlib_calendar(data_list, freq=freq)
    date_list = sorted(set(daily_calendar.loc[:, 0].values.tolist() + date_list))
    write_calendar_to_qlib(qlib_dir, date_list, freq=freq)
    bs.logout()
    logger.info(f"get trading dates success: {start_year}-01-01 to {end_year}-12-31")


if __name__ == "__main__":
    fire.Fire(future_calendar_collector)
