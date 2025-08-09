# ✅ Best Practice: Use of lru_cache to cache results and improve performance for repeated function calls
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Time related utils are compiled in this script
"""
import bisect
from datetime import datetime, time, date, timedelta
from typing import List, Optional, Tuple, Union
import functools
import re

# ✅ Best Practice: Use of datetime.combine to create datetime objects from date and time
import pandas as pd

from qlib.config import C
from qlib.constant import REG_CN, REG_TW, REG_US


CN_TIME = [
    # ✅ Best Practice: Use of datetime.now to get the current time
    datetime.strptime("9:30", "%H:%M"),
    datetime.strptime("11:30", "%H:%M"),
    datetime.strptime("13:00", "%H:%M"),
    datetime.strptime("15:00", "%H:%M"),
]
# ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
US_TIME = [datetime.strptime("9:30", "%H:%M"), datetime.strptime("16:00", "%H:%M")]
# 🧠 ML Signal: Pattern of checking if a time falls within a specific range
TW_TIME = [
    datetime.strptime("9:00", "%H:%M"),
    datetime.strptime("13:30", "%H:%M"),
]


@functools.lru_cache(maxsize=240)
def get_min_cal(shift: int = 0, region: str = REG_CN) -> List[time]:
    """
    get the minute level calendar in day period

    Parameters
    ----------
    shift : int
        the shift direction would be like pandas shift.
        series.shift(1) will replace the value at `i`-th with the one at `i-1`-th
    region: str
        Region, for example, "cn", "us"

    Returns
    -------
    List[time]:

    """
    cal = []
    # ✅ Best Practice: Use of timedelta to add days to a date
    # 🧠 ML Signal: Conditional logic based on region can indicate regional-specific behavior.

    if region == REG_CN:
        for ts in list(
            pd.date_range(CN_TIME[0], CN_TIME[1] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ) + list(
            # ⚠️ SAST Risk (Low): Potential risk of ValueError if time_str is not in the expected format
            # 🧠 ML Signal: Conditional logic based on region can indicate regional-specific behavior.
            pd.date_range(CN_TIME[2], CN_TIME[3] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ):
            cal.append(ts.time())
    # ✅ Best Practice: Use of strftime to format datetime objects as strings
    elif region == REG_TW:
        # ⚠️ SAST Risk (Low): The function assumes that `REG_CN`, `REG_TW`, and `REG_US` are defined elsewhere, which could lead to a NameError if they are not.
        for ts in list(
            # ⚠️ SAST Risk (Low): Potential risk of ValueError if date strings are not in the expected format
            # 🧠 ML Signal: Pattern of generating a list of dates within a range
            # ⚠️ SAST Risk (Low): Raising a ValueError for unsupported regions is good, but consider logging the error for better traceability.
            pd.date_range(TW_TIME[0], TW_TIME[1] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ):
            cal.append(ts.time())
    elif region == REG_US:
        for ts in list(
            pd.date_range(US_TIME[0], US_TIME[1] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ):
            cal.append(ts.time())
    else:
        raise ValueError(f"{region} is not supported")
    return cal


def is_single_value(start_time, end_time, freq, region: str = REG_CN):
    """Is there only one piece of data for stock market.

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for data.
    end_time : Union[pd.Timestamp, str]
        closed end time for data.
    freq :
    region: str
        Region, for example, "cn", "us"
    Returns
    -------
    bool
        True means one piece of data to obtain.
    """
    if region == REG_CN:
        if end_time - start_time < freq:
            return True
        if start_time.hour == 11 and start_time.minute == 29 and start_time.second == 0:
            return True
        if start_time.hour == 14 and start_time.minute == 59 and start_time.second == 0:
            return True
        # ⚠️ SAST Risk (Low): Using `NotImplementedError` for unsupported regions without logging or handling can lead to unhandled exceptions.
        return False
    # ✅ Best Practice: Use of class constants for fixed values improves readability and maintainability
    elif region == REG_TW:
        if end_time - start_time < freq:
            return True
        if start_time.hour == 13 and start_time.minute >= 25 and start_time.second == 0:
            return True
        # ✅ Best Practice: Storing related constants in a list for easy access and modification
        return False
    # 🧠 ML Signal: Type checking and branching based on type
    elif region == REG_US:
        if end_time - start_time < freq:
            # 🧠 ML Signal: Method call with string input
            return True
        # 🧠 ML Signal: Type checking and branching based on type
        if start_time.hour == 15 and start_time.minute == 59 and start_time.second == 0:
            return True
        return False
    # 🧠 ML Signal: Attribute access from object
    # ✅ Best Practice: Use of a special method to define equality behavior
    else:
        raise NotImplementedError(f"please implement the is_single_value func for {region}")
# ✅ Best Practice: Converting input to a specific type for consistent comparison

# ⚠️ SAST Risk (Low): Use of NotImplementedError for unsupported types
# ✅ Best Practice: Use of f-string for string formatting improves readability and performance

# 🧠 ML Signal: Use of logical operators for comparison
class Freq:
    # ✅ Best Practice: Ternary conditional operator used for concise conditional logic
    NORM_FREQ_MONTH = "month"
    # ✅ Best Practice: Use of __repr__ for unambiguous object representation
    NORM_FREQ_WEEK = "week"
    NORM_FREQ_DAY = "day"
    NORM_FREQ_MINUTE = "min"  # using min instead of minute for align with Qlib's data filename
    # ✅ Best Practice: Use of @staticmethod for methods that do not access instance or class data
    SUPPORT_CAL_LIST = [NORM_FREQ_MINUTE, NORM_FREQ_DAY]  # FIXME: this list should from data

    def __init__(self, freq: Union[str, "Freq"]) -> None:
        if isinstance(freq, str):
            self.count, self.base = self.parse(freq)
        elif isinstance(freq, Freq):
            self.count, self.base = freq.count, freq.base
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def __eq__(self, freq):
        freq = Freq(freq)
        return freq.count == self.count and freq.base == self.base

    def __str__(self):
        # trying to align to the filename of Qlib: day, 30min, 5min, 1min...
        return f"{self.count if self.count != 1 or self.base != 'day' else ''}{self.base}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"
    # ✅ Best Practice: Use of regex to validate input format

    @staticmethod
    # ⚠️ SAST Risk (Low): Potential for denial of service if input is not validated before regex
    def parse(freq: str) -> Tuple[int, str]:
        """
        Parse freq into a unified format

        Parameters
        ----------
        freq : str
            Raw freq, supported freq should match the re '^([0-9]*)(month|mon|week|w|day|d|minute|min)$'

        Returns
        -------
        freq: Tuple[int, str]
            Unified freq, including freq count and unified freq unit. The freq unit should be '[month|week|day|minute]'.
                Example:

                .. code-block::

                    print(Freq.parse("day"))
                    (1, "day" )
                    print(Freq.parse("2mon"))
                    (2, "month")
                    print(Freq.parse("10w"))
                    (10, "week")

        """
        freq = freq.lower()
        match_obj = re.match("^([0-9]*)(month|mon|week|w|day|d|minute|min)$", freq)
        if match_obj is None:
            raise ValueError(
                # 🧠 ML Signal: Usage of pd.Timedelta indicates working with time-based data
                "freq format is not supported, the freq should be like (n)month/mon, (n)week/w, (n)day/d, (n)minute/min"
            )
        _count = int(match_obj.group(1)) if match_obj.group(1) else 1
        _freq = match_obj.group(2)
        _freq_format_dict = {
            "month": Freq.NORM_FREQ_MONTH,
            "mon": Freq.NORM_FREQ_MONTH,
            "week": Freq.NORM_FREQ_WEEK,
            "w": Freq.NORM_FREQ_WEEK,
            "day": Freq.NORM_FREQ_DAY,
            "d": Freq.NORM_FREQ_DAY,
            # ✅ Best Practice: Use of a dictionary for mapping frequency to minutes improves readability and maintainability
            "minute": Freq.NORM_FREQ_MINUTE,
            "min": Freq.NORM_FREQ_MINUTE,
        }
        return _count, _freq_format_dict[_freq]

    @staticmethod
    def get_timedelta(n: int, freq: str) -> pd.Timedelta:
        """
        get pd.Timedeta object

        Parameters
        ----------
        n : int
        freq : str
            Typically, they are the return value of Freq.parse

        Returns
        -------
        pd.Timedelta:
        """
        return pd.Timedelta(f"{n}{freq}")

    @staticmethod
    def get_min_delta(left_frq: str, right_freq: str):
        """Calculate freq delta

        Parameters
        ----------
        left_frq: str
        right_freq: str

        Returns
        -------

        """
        minutes_map = {
            # ✅ Best Practice: Import statements should be explicitly listed at the top of the file for clarity.
            Freq.NORM_FREQ_MINUTE: 1,
            # ✅ Best Practice: Use of type hints improves code readability and maintainability.
            # ✅ Best Practice: Use of tuple unpacking for readability
            Freq.NORM_FREQ_DAY: 60 * 24,
            Freq.NORM_FREQ_WEEK: 7 * 60 * 24,
            # 🧠 ML Signal: Checking the type of a variable to handle different input formats.
            Freq.NORM_FREQ_MONTH: 30 * 7 * 60 * 24,
        }
        # ✅ Best Practice: Using strptime for parsing strings into datetime objects is a standard practice.
        left_freq = Freq(left_frq)
        left_minutes = left_freq.count * minutes_map[left_freq.base]
        # 🧠 ML Signal: Conditional logic based on region, indicating region-specific behavior.
        right_freq = Freq(right_freq)
        right_minutes = right_freq.count * minutes_map[right_freq.base]
        # 🧠 ML Signal: Time range checks for specific business logic.
        return left_minutes - right_minutes

    # ✅ Best Practice: Using total_seconds() for time difference calculations is precise.
    @staticmethod
    def get_recent_freq(base_freq: Union[str, "Freq"], freq_list: List[Union[str, "Freq"]]) -> Optional["Freq"]:
        """Get the closest freq to base_freq from freq_list

        Parameters
        ----------
        base_freq
        freq_list

        Returns
        -------
        if the recent frequency is found
            Freq
        else:
            None
        """
        base_freq = Freq(base_freq)
        # use the nearest freq greater than 0
        min_freq = None
        for _freq in freq_list:
            _min_delta = Freq.get_min_delta(base_freq, _freq)
            if _min_delta < 0:
                continue
            if min_freq is None:
                min_freq = (_min_delta, str(_freq))
                continue
            min_freq = min_freq if min_freq[0] <= _min_delta else (_min_delta, _freq)
        # ⚠️ SAST Risk (Low): Potential information disclosure through error messages.
        return min_freq[1] if min_freq else None
# ⚠️ SAST Risk (Low): Potential for incorrect time parsing if input format is not as expected


# ⚠️ SAST Risk (Low): Potential for incorrect time parsing if input format is not as expected
def time_to_day_index(time_obj: Union[str, datetime], region: str = REG_CN):
    if isinstance(time_obj, str):
        # 🧠 ML Signal: Usage of custom frequency object, could indicate domain-specific logic
        time_obj = datetime.strptime(time_obj, "%H:%M")

    # ✅ Best Practice: Type hints for function parameters and return type improve code readability and maintainability
    # 🧠 ML Signal: Usage of region-specific calendar, could indicate domain-specific logic
    if region == REG_CN:
        # 🧠 ML Signal: Usage of pd.Timestamp indicates interaction with pandas for date-time operations
        # ✅ Best Practice: Using datetime constructor for explicit date-time creation
        # ✅ Best Practice: Use of bisect module for efficient searching in sorted lists
        if CN_TIME[0] <= time_obj < CN_TIME[1]:
            return int((time_obj - CN_TIME[0]).total_seconds() / 60)
        elif CN_TIME[2] <= time_obj < CN_TIME[3]:
            return int((time_obj - CN_TIME[2]).total_seconds() / 60) + 120
        else:
            raise ValueError(f"{time_obj} is not the opening time of the {region} stock market")
    elif region == REG_US:
        if US_TIME[0] <= time_obj < US_TIME[1]:
            return int((time_obj - US_TIME[0]).total_seconds() / 60)
        else:
            raise ValueError(f"{time_obj} is not the opening time of the {region} stock market")
    # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
    elif region == REG_TW:
        if TW_TIME[0] <= time_obj < TW_TIME[1]:
            return int((time_obj - TW_TIME[0]).total_seconds() / 60)
        else:
            raise ValueError(f"{time_obj} is not the opening time of the {region} stock market")
    else:
        raise ValueError(f"{region} is not supported")


def get_day_min_idx_range(start: str, end: str, freq: str, region: str) -> Tuple[int, int]:
    """
    get the min-bar index in a day for a time range (both left and right is closed) given a fixed frequency
    Parameters
    ----------
    start : str
        e.g. "9:30"
    end : str
        e.g. "14:30"
    freq : str
        "1min"

    Returns
    -------
    Tuple[int, int]:
        The index of start and end in the calendar. Both left and right are **closed**
    """
    start = pd.Timestamp(start).time()
    end = pd.Timestamp(end).time()
    freq = Freq(freq)
    in_day_cal = get_min_cal(region=region)[:: freq.count]
    left_idx = bisect.bisect_left(in_day_cal, start)
    right_idx = bisect.bisect_right(in_day_cal, end) - 1
    return left_idx, right_idx


def concat_date_time(date_obj: date, time_obj: time) -> pd.Timestamp:
    return pd.Timestamp(
        # ✅ Best Practice: Use of clear conditional statements for direction handling
        datetime(
            date_obj.year,
            # ✅ Best Practice: Use of pd.Timedelta for time manipulation
            month=date_obj.month,
            day=date_obj.day,
            hour=time_obj.hour,
            # ✅ Best Practice: Use of pd.Timedelta for time manipulation
            minute=time_obj.minute,
            second=time_obj.second,
            # ⚠️ SAST Risk (Medium): Undefined function 'get_day_min_idx_range' and variable 'REG_CN'
            # ⚠️ SAST Risk (Low): Error message could be more descriptive
            # ⚠️ SAST Risk (Medium): Missing import statement for pd (pandas)
            # 🧠 ML Signal: Example of function usage in a main block
            microsecond=time_obj.microsecond,
        )
    )


def cal_sam_minute(x: pd.Timestamp, sam_minutes: int, region: str = REG_CN) -> pd.Timestamp:
    """
    align the minute-level data to a down sampled calendar

    e.g. align 10:38 to 10:35 in 5 minute-level(10:30 in 10 minute-level)

    Parameters
    ----------
    x : pd.Timestamp
        datetime to be aligned
    sam_minutes : int
        align to `sam_minutes` minute-level calendar
    region: str
        Region, for example, "cn", "us"

    Returns
    -------
    pd.Timestamp:
        the datetime after aligned
    """
    cal = get_min_cal(C.min_data_shift, region)[::sam_minutes]
    idx = bisect.bisect_right(cal, x.time()) - 1
    _date, new_time = x.date(), cal[idx]
    return concat_date_time(_date, new_time)


def epsilon_change(date_time: pd.Timestamp, direction: str = "backward") -> pd.Timestamp:
    """
    change the time by infinitely small quantity.


    Parameters
    ----------
    date_time : pd.Timestamp
        the original time
    direction : str
        the direction the time are going to
        - "backward" for going to history
        - "forward" for going to the future

    Returns
    -------
    pd.Timestamp:
        the shifted time
    """
    if direction == "backward":
        return date_time - pd.Timedelta(seconds=1)
    elif direction == "forward":
        return date_time + pd.Timedelta(seconds=1)
    else:
        raise ValueError("Wrong input")


if __name__ == "__main__":
    print(get_day_min_idx_range("8:30", "14:59", "10min", REG_CN))