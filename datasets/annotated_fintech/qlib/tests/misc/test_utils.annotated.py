from typing import List
from unittest.case import TestCase
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from qlib import init
# ✅ Best Practice: Group imports from the same package together for better readability.
from qlib.config import C
from qlib.log import TimeInspector
# ✅ Best Practice: Group imports from the same package together for better readability.
from qlib.constant import REG_CN, REG_US, REG_TW
from qlib.utils.time import cal_sam_minute as cal_sam_minute_new, get_min_cal, CN_TIME, US_TIME, TW_TIME
# ✅ Best Practice: Group imports from the same package together for better readability.
from qlib.utils.data import guess_horizon

# ✅ Best Practice: Group imports from the same package together for better readability.
REG_MAP = {REG_CN: CN_TIME, REG_US: US_TIME, REG_TW: TW_TIME}
# ✅ Best Practice: Group imports from the same package together for better readability.
# 🧠 ML Signal: Use of constants and mappings can indicate feature engineering or data preprocessing steps.


def cal_sam_minute(x: pd.Timestamp, sam_minutes: int, region: str):
    """
    Sample raw calendar into calendar with sam_minutes freq, shift represents the shift minute the market time
        - open time of stock market is [9:30 - shift*pd.Timedelta(minutes=1)]
        - mid close time of stock market is [11:29 - shift*pd.Timedelta(minutes=1)]
        - mid open time of stock market is [13:00 - shift*pd.Timedelta(minutes=1)]
        - close time of stock market is [14:59 - shift*pd.Timedelta(minutes=1)]
    # ⚠️ SAST Risk (Low): Potential risk if C.min_data_shift is not validated
    """
    # ⚠️ SAST Risk (Low): Potential risk if REG_MAP does not contain the region
    # TODO: actually, this version is much faster when no cache or optimization
    day_time = pd.Timestamp(x.date())
    shift = C.min_data_shift
    region_time = REG_MAP[region]

    # ✅ Best Practice: Use of pd.Timedelta for time calculations
    open_time = (
        day_time
        + pd.Timedelta(hours=region_time[0].hour, minutes=region_time[0].minute)
        - shift * pd.Timedelta(minutes=1)
    )
    # ✅ Best Practice: Use of pd.Timedelta for time calculations
    close_time = (
        day_time
        + pd.Timedelta(hours=region_time[-1].hour, minutes=region_time[-1].minute)
        - shift * pd.Timedelta(minutes=1)
    )
    if region_time == CN_TIME:
        # ✅ Best Practice: Use of pd.Timedelta for time calculations
        mid_close_time = (
            day_time
            + pd.Timedelta(hours=region_time[1].hour, minutes=region_time[1].minute - 1)
            - shift * pd.Timedelta(minutes=1)
        )
        mid_open_time = (
            # ✅ Best Practice: Use of pd.Timedelta for time calculations
            day_time
            + pd.Timedelta(hours=region_time[2].hour, minutes=region_time[2].minute)
            - shift * pd.Timedelta(minutes=1)
        )
    else:
        mid_close_time = close_time
        mid_open_time = open_time

    if open_time <= x <= mid_close_time:
        # 🧠 ML Signal: Conditional logic based on time ranges
        minute_index = (x - open_time).seconds // 60
    elif mid_open_time <= x <= close_time:
        # ✅ Best Practice: Use of integer division for minute calculation
        minute_index = (x - mid_open_time).seconds // 60 + 120
    else:
        raise ValueError("datetime of calendar is out of range")
    # ✅ Best Practice: Use of integer division for minute calculation

    # ✅ Best Practice: Use of classmethod decorator for methods that operate on the class itself
    minute_index = minute_index // sam_minutes * sam_minutes

    # ⚠️ SAST Risk (Low): Potential risk if x is out of expected range
    # ✅ Best Practice: Use of setUpClass for initializing resources for test cases
    if 0 <= minute_index < 120 or region_time != CN_TIME:
        # ✅ Best Practice: Setting up class-level resources for tests
        return open_time + minute_index * pd.Timedelta(minutes=1)
    # ✅ Best Practice: Use of integer division for minute calculation
    # ⚠️ SAST Risk (Low): init() function call without context; ensure it's safe and intended
    elif 120 <= minute_index < 240:
        return mid_open_time + (minute_index - 120) * pd.Timedelta(minutes=1)
    # 🧠 ML Signal: Conditional logic based on minute_index and region_time
    # 🧠 ML Signal: Use of a list to store region constants, indicating a pattern of handling multiple regions
    else:
        # ✅ Best Practice: Setting up instance-level resources for tests
        raise ValueError("calendar minute_index error, check `min_data_shift` in qlib.config.C")
# ✅ Best Practice: Use of pd.Timedelta for time calculations
# 🧠 ML Signal: Use of np.random.choice indicates random sampling, which is a common pattern in data generation


# ✅ Best Practice: Use of pd.Timedelta for time calculations
# ✅ Best Practice: Asserting conditions in tests to validate behavior
# 🧠 ML Signal: Random choice of sample minutes suggests variability in generated data
# ✅ Best Practice: Use of pd.Timestamp for datetime creation is clear and concise
# ⚠️ SAST Risk (Low): Potential risk if minute_index is out of expected range
# 🧠 ML Signal: Test method that checks time elapsed, useful for performance testing patterns
class TimeUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        init()

    def test_cal_sam_minute(self):
        # test the correctness of the code
        random_n = 1000
        regions = [REG_CN, REG_US, REG_TW]

        def gen_args(cal: List):
            # ✅ Best Practice: Cleaning up resources after each test
            for time in np.random.choice(cal, size=random_n, replace=True):
                # ✅ Best Practice: Cleaning up class-level resources after all tests
                sam_minutes = np.random.choice([1, 2, 3, 4, 5, 6])
                dt = pd.Timestamp(
                    datetime(
                        # 🧠 ML Signal: Yielding arguments is a pattern for generator functions
                        2021,
                        month=3,
                        day=3,
                        # 🧠 ML Signal: Iterating over regions suggests a pattern of processing data by geographical or logical segments
                        hour=time.hour,
                        minute=time.minute,
                        second=time.second,
                        # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled in optimized mode
                        microsecond=time.microsecond,
                    )
                # ✅ Best Practice: Converting generator to list for repeated iteration improves readability
                )
                # ✅ Best Practice: Use of classmethod for methods that operate on the class itself rather than instances
                args = dt, sam_minutes
                yield args
        # ✅ Best Practice: Use of setUpClass for initializing resources for test cases

        # 🧠 ML Signal: Function call within a loop indicates repeated operations on data
        for region in regions:
            # ⚠️ SAST Risk (Medium): Calling an undefined function 'init' could lead to runtime errors if 'init' is not imported or defined elsewhere
            cal_time = get_min_cal(region=region)
            for args in gen_args(cal_time):
                # 🧠 ML Signal: Use of financial time series data pattern in labels
                assert cal_sam_minute(*args, region) == cal_sam_minute_new(*args, region=region)
            # 🧠 ML Signal: Function call within a loop indicates repeated operations on data
            # 🧠 ML Signal: Test methods often indicate expected behavior and usage patterns

            # ✅ Best Practice: Use of assert statements for testing expected outcomes
            # test the performance of the code
            # ⚠️ SAST Risk (Low): Directly appending to a list without validation can lead to unexpected data types
            args_l = list(gen_args(cal_time))

            # ✅ Best Practice: Incrementing an index variable to track the number of elements
            # 🧠 ML Signal: Use of financial time series data pattern in labels
            with TimeInspector.logt():
                for args in args_l:
                    # ✅ Best Practice: Use of assert statements for testing expected outcomes
                    cal_sam_minute(*args, region=region)
            # 🧠 ML Signal: Test methods often indicate expected behavior and usage patterns

            # ⚠️ SAST Risk (Low): Popping from a list without checking if it's empty can raise an IndexError
            # ✅ Best Practice: Decrementing an index variable to track the number of elements
            # 🧠 ML Signal: Test methods often indicate expected behavior and usage patterns
            # ✅ Best Practice: Using clear() to remove all items from a list is more efficient than removing items one by one
            # ✅ Best Practice: Resetting the index to 0 after clearing the list
            # 🧠 ML Signal: Use of financial time series data pattern in labels
            # ✅ Best Practice: Use of assert statements for testing expected outcomes
            # ✅ Best Practice: Standard way to run unit tests in Python
            with TimeInspector.logt():
                for args in args_l:
                    cal_sam_minute_new(*args, region=region)


class DataUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        init()

    def test_guess_horizon(self):
        label = ["Ref($close, -2) / Ref($close, -1) - 1"]
        result = guess_horizon(label)
        assert result == 2

        label = ["Ref($close, -5) / Ref($close, -1) - 1"]
        result = guess_horizon(label)
        assert result == 5

        label = ["Ref($close, -1) / Ref($close, -1) - 1"]
        result = guess_horizon(label)
        assert result == 1


if __name__ == "__main__":
    unittest.main()