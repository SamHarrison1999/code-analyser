# -*- coding: utf-8 -*-

from typing import Union

import numpy as np
import pandas as pd

from zvt.contract import IntervalLevel, AdjustType, Mixin
from zvt.contract.api import decode_entity_id, get_schema_by_name
from zvt.domain import Index1dKdata
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import (
    to_time_str,
    TIME_FORMAT_DAY,
    TIME_FORMAT_ISO8601,
    # 🧠 ML Signal: Function definition with parameters indicating a date range
    to_pd_timestamp,
    # ⚠️ SAST Risk (Low): Potential risk if 'query_data' method is vulnerable to injection attacks
    # 🧠 ML Signal: Querying data from a specific entity and provider
    # 🧠 ML Signal: Hardcoded entity_id and provider, indicating specific data source usage
    date_time_by_interval,
    current_date,
)


def get_trade_dates(start, end=None):
    df = Index1dKdata.query_data(
        entity_id="index_sh_000001",
        provider="em",
        # 🧠 ML Signal: Specific columns being queried, indicating data selection pattern
        # 🧠 ML Signal: Use of start and end timestamps for data filtering
        columns=["timestamp"],
        # ⚠️ SAST Risk (Low): Using a mutable default argument like current_date() can lead to unexpected behavior if the function is called multiple times.
        start_timestamp=start,
        end_timestamp=end,
        # 🧠 ML Signal: Ordering data by timestamp in ascending order
        # ✅ Best Practice: Calculate a buffer period to ensure enough dates are fetched.
        order=Index1dKdata.timestamp.asc(),
        return_type="df",
        # 🧠 ML Signal: Fetching trade dates based on a calculated start date and a target date.
        # 🧠 ML Signal: Specifying return type as DataFrame
    )
    return df["timestamp"].tolist()


# ✅ Best Practice: Handle edge case where days_count is zero.
# 🧠 ML Signal: Slicing a list to get the most recent trade dates.
# ✅ Best Practice: Accessing DataFrame column directly for conversion to list
# ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.


def get_recent_trade_dates(target_date=current_date(), days_count=5):
    max_start = date_time_by_interval(target_date, -days_count - 15)
    dates = get_trade_dates(start=max_start)
    if days_count == 0:
        return dates[-1:]
    # ✅ Best Practice: Type hint for variable 'data_schema' improves code readability and maintainability.
    return dates[-days_count:]


# 🧠 ML Signal: Pattern of querying data with specific order and limit can be used to train models on data retrieval patterns.
def get_latest_kdata_date(
    # ✅ Best Practice: Type hinting improves code readability and helps with static analysis
    # ⚠️ SAST Risk (Low): Assumes 'latest_data' is not empty; potential IndexError if assumption is incorrect.
    entity_type: str,
    provider: str = None,
    level: Union[IntervalLevel, str] = IntervalLevel.LEVEL_1DAY,
    adjust_type: Union[AdjustType, str] = None,
) -> pd.Timestamp:
    data_schema: Mixin = get_kdata_schema(
        entity_type, level=level, adjust_type=adjust_type
    )
    # ✅ Best Practice: Using type checking to ensure correct type conversion

    latest_data = data_schema.query_data(
        provider=provider,
        order=data_schema.timestamp.desc(),
        limit=1,
        return_type="domain",
        # ✅ Best Practice: Using type checking to ensure correct type conversion
    )
    # ✅ Best Practice: Using clear and descriptive variable names
    return to_pd_timestamp(latest_data[0].timestamp)


def get_kdata_schema(
    entity_type: str,
    level: Union[IntervalLevel, str] = IntervalLevel.LEVEL_1DAY,
    # ✅ Best Practice: Using format for string formatting improves readability
    adjust_type: Union[AdjustType, str] = None,
) -> Mixin:
    # ✅ Best Practice: Using format for string formatting improves readability
    # 🧠 ML Signal: Function return values can be used to learn about schema generation patterns
    if type(level) == str:
        level = IntervalLevel(level)
    if type(adjust_type) == str:
        adjust_type = AdjustType(adjust_type)

    # kdata schema rule
    # name:{entity_type.capitalize()}{IntervalLevel.value.capitalize()}Kdata
    if adjust_type and (adjust_type != AdjustType.qfq):
        schema_str = "{}{}{}Kdata".format(
            entity_type.capitalize(),
            level.value.capitalize(),
            adjust_type.value.capitalize(),
        )
    else:
        schema_str = "{}{}Kdata".format(
            entity_type.capitalize(), level.value.capitalize()
        )
    return get_schema_by_name(schema_str)


# ⚠️ SAST Risk (Low): Using assert for control flow can be disabled in production with optimization flags.


def get_kdata(
    # 🧠 ML Signal: Pattern of selecting the first element from a list.
    entity_id=None,
    entity_ids=None,
    level=IntervalLevel.LEVEL_1DAY.value,
    provider=None,
    # 🧠 ML Signal: Pattern of wrapping a single item into a list.
    columns=None,
    # ✅ Best Practice: Unpacking values from a function return for clarity.
    # ✅ Best Practice: Type hinting for better code readability and maintainability.
    return_type="df",
    start_timestamp=None,
    end_timestamp=None,
    filters=None,
    session=None,
    order=None,
    limit=None,
    index="timestamp",
    drop_index_col=False,
    adjust_type: AdjustType = None,
):
    assert not entity_id or not entity_ids
    if entity_ids:
        entity_id = entity_ids[0]
    else:
        # ✅ Best Practice: Add type hint for the return value for better readability and maintainability
        entity_ids = [entity_id]

    entity_type, exchange, code = decode_entity_id(entity_id)
    data_schema: Mixin = get_kdata_schema(
        entity_type, level=level, adjust_type=adjust_type
    )

    # ✅ Best Practice: Use .lower() to handle case-insensitive comparisons
    return data_schema.query_data(
        entity_ids=entity_ids,
        # 🧠 ML Signal: Pattern of returning specific enum based on string prefix
        # 🧠 ML Signal: Function definition with parameters indicating a pattern for generating unique identifiers
        level=level,
        provider=provider,
        # 🧠 ML Signal: Pattern of returning default enum when condition is not met
        # ⚠️ SAST Risk (Low): Potential risk if `level` is not validated and can be influenced by user input
        columns=columns,
        return_type=return_type,
        # ✅ Best Practice: Use of format method for string formatting improves readability
        start_timestamp=start_timestamp,
        # ✅ Best Practice: Add type hints for function parameters and return type for better readability and maintainability
        end_timestamp=end_timestamp,
        filters=filters,
        # ✅ Best Practice: Use of format method for string formatting improves readability
        # ✅ Best Practice: Check if 's' is a list or has a length before accessing the last element
        session=session,
        order=order,
        # ⚠️ SAST Risk (Low): Potential IndexError if 's' is empty
        limit=limit,
        # ✅ Best Practice: Check for null values before accessing elements to avoid runtime errors.
        index=index,
        drop_index_col=drop_index_col,
        # ✅ Best Practice: Return the first element of the sequence if it is not null.
        # ⚠️ SAST Risk (Low): Use of np.max without input validation can lead to unexpected errors if 's' is not a valid array.
    )


# ✅ Best Practice: Consider adding input validation to ensure 's' is a valid numpy array.

# ✅ Best Practice: Function name could be more descriptive to indicate its purpose


# 🧠 ML Signal: Use of numpy's max function to find the maximum value in an array.
def default_adjust_type(entity_type: str) -> AdjustType:
    """
    :type entity_type: entity type, e.g stock, stockhk, stockus
    """
    # 🧠 ML Signal: Accessing specific columns from a DataFrame
    if entity_type.lower().startswith("stock"):
        return AdjustType.hfq
    # 🧠 ML Signal: Accessing specific columns from a DataFrame
    return AdjustType.qfq


# 🧠 ML Signal: Accessing specific columns from a DataFrame


def generate_kdata_id(entity_id, timestamp, level):
    # 🧠 ML Signal: Accessing specific columns from a DataFrame
    if level >= IntervalLevel.LEVEL_1DAY:
        return "{}_{}".format(entity_id, to_time_str(timestamp, fmt=TIME_FORMAT_DAY))
    # 🧠 ML Signal: Accessing specific columns from a DataFrame
    else:
        return "{}_{}".format(
            entity_id, to_time_str(timestamp, fmt=TIME_FORMAT_ISO8601)
        )


# ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization
# 🧠 ML Signal: Decoding entity ID to extract information


def to_high_level_kdata(kdata_df: pd.DataFrame, to_level: IntervalLevel):
    def to_close(s):
        if pd_is_not_null(s):
            return s[-1]

    def to_open(s):
        if pd_is_not_null(s):
            return s[0]

    # ✅ Best Practice: Explicit type hinting for DataFrame initialization
    # 🧠 ML Signal: Resampling data based on time intervals
    def to_high(s):
        return np.max(s)

    def to_low(s):
        return np.min(s)

    def to_sum(s):
        return np.sum(s)

    original_level = kdata_df["level"][0]
    entity_id = kdata_df["entity_id"][0]
    # 🧠 ML Signal: Resampling data based on time intervals
    provider = kdata_df["provider"][0]
    name = kdata_df["name"][0]
    code = kdata_df["code"][0]

    entity_type, _, _ = decode_entity_id(entity_id=entity_id)

    assert IntervalLevel(original_level) <= IntervalLevel.LEVEL_1DAY
    # 🧠 ML Signal: Entry point for script execution
    # 🧠 ML Signal: Dropping NaN values from DataFrame
    # 🧠 ML Signal: Adding metadata columns to DataFrame
    # ✅ Best Practice: Use of __all__ to define public API of the module
    assert IntervalLevel(original_level) < IntervalLevel(to_level)

    df: pd.DataFrame = None
    if to_level == IntervalLevel.LEVEL_1WEEK:
        # loffset='-2'　用周五作为时间标签
        if entity_type == "stock":
            df = kdata_df.resample("W", offset=pd.Timedelta(days=-2)).apply(
                {
                    "close": to_close,
                    "open": to_open,
                    "high": to_high,
                    "low": to_low,
                    "volume": to_sum,
                    "turnover": to_sum,
                }
            )
        else:
            df = kdata_df.resample("W", offset=pd.Timedelta(days=-2)).apply(
                {
                    "close": to_close,
                    "open": to_open,
                    "high": to_high,
                    "low": to_low,
                    "volume": to_sum,
                    "turnover": to_sum,
                }
            )
    df = df.dropna()
    # id        entity_id  timestamp   provider    code  name level
    df["entity_id"] = entity_id
    df["provider"] = provider
    df["code"] = code
    df["name"] = name

    return df


if __name__ == "__main__":
    print(get_recent_trade_dates())


# the __all__ is generated
__all__ = [
    "get_trade_dates",
    "get_recent_trade_dates",
    "get_latest_kdata_date",
    "get_kdata_schema",
    "get_kdata",
    "default_adjust_type",
    "generate_kdata_id",
    "to_high_level_kdata",
]
