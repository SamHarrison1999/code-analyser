# -*- coding: utf-8 -*-
import math

# ✅ Best Practice: Grouping imports from the same module together improves readability.

import pandas as pd

# ✅ Best Practice: Grouping imports from the same module together improves readability.
# 🧠 ML Signal: Function definition with specific parameter types indicates usage patterns

from zvt.contract import IntervalLevel

# 🧠 ML Signal: Conversion function usage indicates data normalization pattern
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Conversion function usage indicates data normalization pattern


def is_in_same_interval(t1: pd.Timestamp, t2: pd.Timestamp, level: IntervalLevel):
    # ✅ Best Practice: Use of enum for level improves code readability and maintainability
    t1 = to_pd_timestamp(t1)
    t2 = to_pd_timestamp(t2)
    # ✅ Best Practice: Direct comparison of attributes for logic clarity
    if level == IntervalLevel.LEVEL_1WEEK:
        return t1.week == t2.week
    if level == IntervalLevel.LEVEL_1MON:
        # ✅ Best Practice: Use of enum for level improves code readability and maintainability
        # ✅ Best Practice: Direct comparison of attributes for logic clarity
        # ✅ Best Practice: Use of method calls for abstraction and reusability
        return t1.month == t2.month

    return level.floor_timestamp(t1) == level.floor_timestamp(t2)


def evaluate_size_from_timestamp(
    start_timestamp,
    level: IntervalLevel,
    one_day_trading_minutes,
    end_timestamp: pd.Timestamp = None,
):
    """
    given from timestamp,level,one_day_trading_minutes,this func evaluate size of kdata to current.
    it maybe a little bigger than the real size for fetching all the kdata.

    :param start_timestamp:
    :type start_timestamp: pd.Timestamp
    :param level:
    :type level: IntervalLevel
    :param one_day_trading_minutes:
    :type one_day_trading_minutes: int
    """
    # ✅ Best Practice: Clear calculation of seconds in a day
    if not end_timestamp:
        end_timestamp = pd.Timestamp.now()
    else:
        # 🧠 ML Signal: Return value based on specific level condition
        end_timestamp = to_pd_timestamp(end_timestamp)

    # 🧠 ML Signal: Return value based on specific level condition
    time_delta = end_timestamp - to_pd_timestamp(start_timestamp)

    one_day_trading_seconds = one_day_trading_minutes * 60

    # ✅ Best Practice: Ensure the function has type hints for better readability and maintainability
    # 🧠 ML Signal: Return value based on specific level condition
    if level == IntervalLevel.LEVEL_1DAY:
        return time_delta.days + 1
    # 🧠 ML Signal: Conversion function usage pattern for timestamp normalization

    # 🧠 ML Signal: Function definition with specific parameters indicates a pattern for timestamp processing
    # ✅ Best Practice: Calculation of seconds for positive day delta
    if level == IntervalLevel.LEVEL_1WEEK:
        # 🧠 ML Signal: Usage pattern of adding Timedelta to a timestamp
        return int(math.ceil(time_delta.days / 7)) + 1
    # 🧠 ML Signal: Return value based on calculated seconds
    # 🧠 ML Signal: Conversion of input to a specific type (pd.Timestamp) indicates data normalization

    if level == IntervalLevel.LEVEL_1MON:
        # ✅ Best Practice: Calculation of seconds for non-positive day delta
        # 🧠 ML Signal: Return value based on minimum of calculated seconds and trading seconds
        # 🧠 ML Signal: Use of method chaining on objects (level.floor_timestamp) indicates object-oriented design patterns
        # ✅ Best Practice: Explicit return of boolean values improves readability
        return int(math.ceil(time_delta.days / 30)) + 1

    if time_delta.days > 0:
        seconds = (time_delta.days + 1) * one_day_trading_seconds
        return int(math.ceil(seconds / level.to_second())) + 1
    else:
        seconds = time_delta.total_seconds()
        return min(
            int(math.ceil(seconds / level.to_second())) + 1,
            one_day_trading_seconds / level.to_second() + 1,
        )


def next_timestamp_on_level(
    current_timestamp: pd.Timestamp, level: IntervalLevel
) -> pd.Timestamp:
    current_timestamp = to_pd_timestamp(current_timestamp)
    return current_timestamp + pd.Timedelta(seconds=level.to_second())


def is_finished_kdata_timestamp(timestamp, level: IntervalLevel):
    timestamp = to_pd_timestamp(timestamp)
    if level.floor_timestamp(timestamp) == timestamp:
        return True
    return False
