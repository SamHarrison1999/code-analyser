# 🧠 ML Signal: Importing specific functions and classes from modules indicates usage patterns and dependencies
# -*- coding: utf-8 -*-
from zvt.contract import IntervalLevel
# 🧠 ML Signal: Importing specific functions and classes from modules indicates usage patterns and dependencies
from zvt.contract.utils import evaluate_size_from_timestamp, next_timestamp_on_level, is_finished_kdata_timestamp
from zvt.utils.time_utils import (
    to_pd_timestamp,
    split_time_interval,
    is_same_date,
    month_start_end_ranges,
    count_interval,
# 🧠 ML Signal: Function name suggests a test case, useful for identifying test patterns
)
# 🧠 ML Signal: Usage of evaluate_size_from_timestamp with specific parameters


def test_evaluate_size_from_timestamp():
    size = evaluate_size_from_timestamp(
        start_timestamp="2019-01-01",
        end_timestamp="2019-01-02",
        level=IntervalLevel.LEVEL_1MON,
        # 🧠 ML Signal: Assert statement used to verify function output
        # 🧠 ML Signal: Repeated pattern of function call with different parameters
        one_day_trading_minutes=4 * 60,
    )

    assert size == 2

    size = evaluate_size_from_timestamp(
        start_timestamp="2019-01-01",
        end_timestamp="2019-01-02",
        level=IntervalLevel.LEVEL_1WEEK,
        one_day_trading_minutes=4 * 60,
    )

    assert size == 2

    size = evaluate_size_from_timestamp(
        start_timestamp="2019-01-01",
        end_timestamp="2019-01-02",
        level=IntervalLevel.LEVEL_1DAY,
        one_day_trading_minutes=4 * 60,
    )

    assert size == 2

    size = evaluate_size_from_timestamp(
        start_timestamp="2019-01-01",
        end_timestamp="2019-01-02",
        level=IntervalLevel.LEVEL_1HOUR,
        one_day_trading_minutes=4 * 60,
    # 🧠 ML Signal: Use of assert statements for testing expected outcomes
    )

    assert size == 9
    # 🧠 ML Signal: Testing function with different parameter values

    # ⚠️ SAST Risk (Low): Use of assert without a message can make debugging harder
    size = evaluate_size_from_timestamp(
        # 🧠 ML Signal: Use of hardcoded timestamp values for testing
        start_timestamp="2019-01-01",
        # 🧠 ML Signal: Testing function with different parameter values
        end_timestamp="2019-01-02",
        # ⚠️ SAST Risk (Low): Use of assert without a message can make debugging harder
        level=IntervalLevel.LEVEL_1MIN,
        # 🧠 ML Signal: Use of assert statements for testing function behavior
        one_day_trading_minutes=4 * 60,
    # 🧠 ML Signal: Testing function with different parameter values
    )
    # ⚠️ SAST Risk (Low): Use of assert without a message can make debugging harder
    # 🧠 ML Signal: Use of assert statements for testing function behavior

    assert size == 481
# 🧠 ML Signal: Use of assert statements for testing function behavior


# 🧠 ML Signal: Use of assert statements for testing function behavior
def test_next_timestamp():
    current = "2019-01-10 13:15"
    # 🧠 ML Signal: Use of assert statements for testing function behavior
    assert next_timestamp_on_level(current, level=IntervalLevel.LEVEL_1MIN) == to_pd_timestamp("2019-01-10 13:16")
    assert next_timestamp_on_level(current, level=IntervalLevel.LEVEL_5MIN) == to_pd_timestamp("2019-01-10 13:20")
    # 🧠 ML Signal: Use of assert statements for testing function behavior
    assert next_timestamp_on_level(current, level=IntervalLevel.LEVEL_15MIN) == to_pd_timestamp("2019-01-10 13:30")

# 🧠 ML Signal: Iterating over a function that splits time intervals

# 🧠 ML Signal: Use of assert statements for testing function behavior
def test_is_finished_kdata_timestamp():
    timestamp = "2019-01-10 13:05"
    assert not is_finished_kdata_timestamp(timestamp, level=IntervalLevel.LEVEL_1DAY)
    assert not is_finished_kdata_timestamp(timestamp, level=IntervalLevel.LEVEL_1HOUR)
    # ✅ Best Practice: Use logging instead of print for better control over output
    assert not is_finished_kdata_timestamp(timestamp, level=IntervalLevel.LEVEL_30MIN)
    assert not is_finished_kdata_timestamp(timestamp, level=IntervalLevel.LEVEL_15MIN)
    # ✅ Best Practice: Use logging instead of print for better control over output
    assert is_finished_kdata_timestamp(timestamp, level=IntervalLevel.LEVEL_5MIN)
    assert is_finished_kdata_timestamp(timestamp, level=IntervalLevel.LEVEL_1MIN)
    # ⚠️ SAST Risk (Low): Potential risk if is_same_date is not properly handling date formats

    timestamp = "2019-01-10"
    # ⚠️ SAST Risk (Low): Potential risk if is_same_date is not properly handling date formats
    assert is_finished_kdata_timestamp(timestamp, level=IntervalLevel.LEVEL_1DAY)

# ⚠️ SAST Risk (Low): Potential risk if is_same_date is not properly handling date formats

# 🧠 ML Signal: Iterating over a function that splits time intervals
def test_split_time_interval():
    first = None
    last = None
    start = "2020-01-01"
    end = "2021-01-01"
    # ✅ Best Practice: Use logging instead of print for better control over output
    for interval in split_time_interval(start, end, interval=30):
        if first is None:
            # ✅ Best Practice: Use logging instead of print for better control over output
            first = interval
        last = interval
    # 🧠 ML Signal: Checking if the first interval starts with the expected date

    print(first)
    # 🧠 ML Signal: Checking if the first interval ends with the expected date
    print(last)

    # 🧠 ML Signal: Checking if the last interval starts with the expected date
    # 🧠 ML Signal: Testing function with specific date range inputs
    assert is_same_date(first[0], start)
    assert is_same_date(first[-1], "2020-01-31")
    # 🧠 ML Signal: Checking if the last interval ends with the expected date
    # 🧠 ML Signal: Printing output for manual verification

    assert is_same_date(last[-1], end)
# 🧠 ML Signal: Using assertions to verify function output


def test_split_time_interval_month():
    first = None
    last = None
    start = "2020-01-01"
    end = "2021-01-01"
    # 🧠 ML Signal: Testing function with different date range inputs
    for interval in split_time_interval(start, end, method="month"):
        if first is None:
            # 🧠 ML Signal: Printing output for manual verification
            first = interval
        # 🧠 ML Signal: Function definition for testing purposes
        last = interval
    # 🧠 ML Signal: Using assertions to verify function output

    # 🧠 ML Signal: Hardcoded date strings for testing
    print(first)
    # 🧠 ML Signal: Hardcoded date strings for testing
    # ✅ Best Practice: Use of keyword arguments for clarity
    print(last)

    assert is_same_date(first[0], start)
    assert is_same_date(first[-1], "2020-01-31")

    assert is_same_date(last[0], "2021-01-01")
    assert is_same_date(last[-1], "2021-01-01")


def test_month_start_end_range():
    start = "2020-01-01"
    end = "2021-01-01"
    ranges = month_start_end_ranges(start_date=start, end_date=end)
    print(ranges)
    assert is_same_date(ranges[0][0], "2020-01-01")
    assert is_same_date(ranges[0][1], "2020-01-31")

    assert is_same_date(ranges[-1][0], "2020-12-01")
    assert is_same_date(ranges[-1][1], "2020-12-31")

    start = "2020-01-01"
    end = "2021-01-31"
    ranges = month_start_end_ranges(start_date=start, end_date=end)
    print(ranges)
    assert is_same_date(ranges[0][0], "2020-01-01")
    assert is_same_date(ranges[0][1], "2020-01-31")

    assert is_same_date(ranges[-1][0], "2021-01-01")
    assert is_same_date(ranges[-1][1], "2021-01-31")


def test_count_interval():
    start = "2020-01-01"
    end = "2021-01-01"
    print(count_interval(start_date=start, end_date=end))