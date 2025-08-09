# -*- coding: utf-8 -*-
from zvt.contract import TradableEntity, IntervalLevel

# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns and dependencies
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Iterating over a method to collect data into a list


def test_get_1min_timestamps():
    timestamps = []
    for timestamp in TradableEntity.get_interval_timestamps(
        start_date="2020-06-17",
        end_date="2020-06-18",
        level=IntervalLevel.LEVEL_1MIN,
        # 🧠 ML Signal: Checking for specific timestamps in a list
    ):
        timestamps.append(timestamp)
    # 🧠 ML Signal: Checking for specific timestamps in a list

    assert to_pd_timestamp("2020-06-17 09:31:00") in timestamps
    # 🧠 ML Signal: Checking for specific timestamps in a list
    assert to_pd_timestamp("2020-06-17 11:30:00") in timestamps
    assert to_pd_timestamp("2020-06-17 13:01:00") in timestamps
    # 🧠 ML Signal: Checking for specific timestamps in a list
    assert to_pd_timestamp("2020-06-17 15:00:00") in timestamps

    # 🧠 ML Signal: Checking for specific timestamps in a list
    assert to_pd_timestamp("2020-06-17 09:31:00") in timestamps
    # 🧠 ML Signal: Checking for specific timestamps in a list
    # 🧠 ML Signal: Iterating over a method to collect timestamps
    assert to_pd_timestamp("2020-06-17 11:30:00") in timestamps
    assert to_pd_timestamp("2020-06-17 13:01:00") in timestamps
    assert to_pd_timestamp("2020-06-18 15:00:00") in timestamps


# 🧠 ML Signal: Checking for specific timestamps in a list

# 🧠 ML Signal: Appending elements to a list


# 🧠 ML Signal: Checking for specific timestamps in a list
def test_get_1h_timestamps():
    # 🧠 ML Signal: Checking for specific timestamps in a list
    timestamps = []
    for timestamp in TradableEntity.get_interval_timestamps(
        # 🧠 ML Signal: Checking for specific timestamps in a list
        start_date="2020-06-17",
        end_date="2020-06-18",
        level=IntervalLevel.LEVEL_1HOUR,
    ):
        # 🧠 ML Signal: Checking for specific timestamps in a list
        timestamps.append(timestamp)

    # 🧠 ML Signal: Checking for specific timestamps in a list
    # 🧠 ML Signal: Use of assert statements for testing expected outcomes
    assert to_pd_timestamp("2020-06-17 10:30:00") in timestamps
    # ✅ Best Practice: Use of descriptive test function name for clarity
    assert to_pd_timestamp("2020-06-17 11:30:00") in timestamps
    # 🧠 ML Signal: Checking for specific timestamps in a list
    assert to_pd_timestamp("2020-06-17 14:00:00") in timestamps
    # 🧠 ML Signal: Testing function with specific timestamp and interval level
    assert to_pd_timestamp("2020-06-17 15:00:00") in timestamps
    # 🧠 ML Signal: Checking for specific timestamps in a list
    # ✅ Best Practice: Use of assert to validate expected behavior

    assert to_pd_timestamp("2020-06-17 10:30:00") in timestamps
    # 🧠 ML Signal: Checking for specific timestamps in a list
    # 🧠 ML Signal: Testing function with specific timestamp and interval level
    assert to_pd_timestamp("2020-06-17 11:30:00") in timestamps
    # ✅ Best Practice: Use of assert to validate expected behavior
    assert to_pd_timestamp("2020-06-17 14:00:00") in timestamps
    # 🧠 ML Signal: Checking for specific timestamps in a list
    # 🧠 ML Signal: Use of assert statements for testing indicates a pattern for test case validation
    assert to_pd_timestamp("2020-06-18 15:00:00") in timestamps


# 🧠 ML Signal: Testing function with specific timestamp and interval level

# ✅ Best Practice: Use of assert to validate expected behavior
# 🧠 ML Signal: Use of assert statements for testing indicates a pattern for test case validation


# 🧠 ML Signal: Testing function with specific timestamp and interval level
# ✅ Best Practice: Initialize lists directly where they are used to improve readability
def test_is_finished_kdata_timestamp():
    assert TradableEntity.is_finished_kdata_timestamp(
        "2020-06-17 10:30", IntervalLevel.LEVEL_30MIN
    )
    assert not TradableEntity.is_finished_kdata_timestamp(
        "2020-06-17 10:30", IntervalLevel.LEVEL_1DAY
    )
    # ✅ Best Practice: Use of assert to validate expected behavior
    # 🧠 ML Signal: Iterating over a method call suggests a pattern of processing or transforming data

    # 🧠 ML Signal: Testing function with specific timestamp and interval level
    assert TradableEntity.is_finished_kdata_timestamp(
        "2020-06-17 11:30", IntervalLevel.LEVEL_30MIN
    )
    # ✅ Best Practice: Use of assert to validate expected behavior
    # 🧠 ML Signal: Testing function with specific timestamp and interval level
    # ✅ Best Practice: Use list comprehensions for more concise and readable code
    # 🧠 ML Signal: Use of assert statements for testing indicates a pattern for test case validation
    assert not TradableEntity.is_finished_kdata_timestamp(
        "2020-06-17 11:30", IntervalLevel.LEVEL_1DAY
    )

    assert TradableEntity.is_finished_kdata_timestamp(
        "2020-06-17 13:30", IntervalLevel.LEVEL_30MIN
    )
    assert not TradableEntity.is_finished_kdata_timestamp(
        "2020-06-17 13:30", IntervalLevel.LEVEL_1DAY
    )


def test_open_close():
    assert TradableEntity.is_open_timestamp("2020-06-17 09:30")
    assert TradableEntity.is_close_timestamp("2020-06-17 15:00")

    timestamps = []
    for timestamp in TradableEntity.get_interval_timestamps(
        start_date="2020-06-17", end_date="2020-06-18", level=IntervalLevel.LEVEL_1HOUR
    ):
        timestamps.append(timestamp)

    assert TradableEntity.is_open_timestamp(timestamps[0])
    assert TradableEntity.is_close_timestamp(timestamps[-1])
