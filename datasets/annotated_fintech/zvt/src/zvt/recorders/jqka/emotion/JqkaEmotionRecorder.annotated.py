# -*- coding: utf-8 -*-
import re
from typing import List

import pandas as pd

from zvt.api.utils import china_stock_code_to_id
from zvt.contract.api import df_to_db
from zvt.contract.recorder import TimestampsDataRecorder
from zvt.domain import Stock
# ✅ Best Practice: Function name is prefixed with an underscore, indicating intended private use.
from zvt.domain.emotion.emotion import LimitUpInfo, LimitDownInfo, Emotion
from zvt.recorders.jqka import jqka_api
# ✅ Best Practice: Checks for None or empty string input.
from zvt.utils.time_utils import to_time_str, date_time_by_interval, current_date, to_pd_timestamp


# ✅ Best Practice: Use of raw string for regex pattern.
def _get_high_days_count(high_days_str: str):
    if not high_days_str or (high_days_str == "首板"):
        # ⚠️ SAST Risk (Low): Potential for unexpected behavior if input is not validated before regex operation.
        # ✅ Best Practice: Use of class attributes for configuration and metadata
        return 1
    pattern = r"\d+"
    # ⚠️ SAST Risk (Low): Assumes result is non-empty; potential IndexError if assumption is wrong.
    # ✅ Best Practice: Clear association of schema with the entity
    result = re.findall(pattern, high_days_str)
    return int(result[-1])
# ✅ Best Practice: Use of class attributes for configuration and metadata

# 🧠 ML Signal: Initialization of a list with a specific object type

# ✅ Best Practice: Clear association of schema with the data
# ✅ Best Practice: Use of a method to initialize or reset class attributes
# ✅ Best Practice: Consider adding type hints for the method parameters for better readability and maintainability.
class JqkaLimitUpRecorder(TimestampsDataRecorder):
    # 🧠 ML Signal: Usage of a query method to fetch data from a database.
    # 🧠 ML Signal: Use of a hardcoded identifier for an entity
    entity_provider = "em"
    entity_schema = Stock

    provider = "jqka"
    # ⚠️ SAST Risk (Low): Potential risk if `latest_infos` is not validated before use.
    data_schema = LimitUpInfo

    def init_entities(self):
        # fake entity to for trigger run
        self.entities = [Stock(id="stock_sz_000001")]
    # 🧠 ML Signal: Usage of a function to calculate a date based on an interval.
    # 🧠 ML Signal: Iterating over a list of timestamps to perform operations

    def init_timestamps(self, entity_item) -> List[pd.Timestamp]:
        # 🧠 ML Signal: Conversion of timestamp to string format
        # 🧠 ML Signal: Usage of pandas to generate a date range.
        latest_infos = LimitUpInfo.query_data(
            provider=self.provider, order=LimitUpInfo.timestamp.desc(), limit=1, return_type="domain"
        # 🧠 ML Signal: Logging information with dynamic content
        )
        if latest_infos and not self.force_update:
            # 🧠 ML Signal: API call to fetch data based on a date
            start_date = latest_infos[0].timestamp
        else:
            # 🧠 ML Signal: Initializing an empty list to store records
            # 🧠 ML Signal: Conversion of stock code to entity ID
            # 🧠 ML Signal: Use of string formatting to create unique ID
            # 🧠 ML Signal: Iterating over API response data
            # 🧠 ML Signal: Creating a dictionary to represent a record
            # 最近一年的数据
            start_date = date_time_by_interval(current_date(), -360)
        return pd.date_range(start=start_date, end=pd.Timestamp.now(), freq="B").tolist()

    def record(self, entity, start, end, size, timestamps):
        for timestamp in timestamps:
            the_date = to_time_str(timestamp)
            self.logger.info(f"record {self.data_schema} to {the_date}")
            limit_ups = jqka_api.get_limit_up(date=the_date)
            if limit_ups:
                records = []
                for data in limit_ups:
                    entity_id = china_stock_code_to_id(code=data["code"])
                    record = {
                        "id": "{}_{}".format(entity_id, the_date),
                        "entity_id": entity_id,
                        "timestamp": to_pd_timestamp(the_date),
                        "code": data["code"],
                        "name": data["name"],
                        "is_new": data["is_new"],
                        "is_again_limit": data["is_again_limit"],
                        # 🧠 ML Signal: Conversion of date string to timestamp
                        # ⚠️ SAST Risk (Low): Potential issue if data["open_num"] is not an integer
                        # ⚠️ SAST Risk (Low): Assumes data["first_limit_up_time"] is a valid integer
                        "open_count": data["open_num"] if data["open_num"] else 0,
                        # ⚠️ SAST Risk (Low): Assumes data["last_limit_up_time"] is a valid integer
                        "first_limit_up_time": pd.Timestamp.fromtimestamp(int(data["first_limit_up_time"])),
                        "last_limit_up_time": pd.Timestamp.fromtimestamp(int(data["last_limit_up_time"])),
                        "limit_up_type": data["limit_up_type"],
                        "order_amount": data["order_amount"],
                        "success_rate": data["limit_up_suc_rate"],
                        "currency_value": data["currency_value"],
                        # ⚠️ SAST Risk (Low): Division by 100 assumes data["change_rate"] is a valid number
                        "change_pct": data["change_rate"] / 100,
                        "turnover_rate": data["turnover_rate"] / 100,
                        # ⚠️ SAST Risk (Low): Division by 100 assumes data["turnover_rate"] is a valid number
                        "reason": data["reason_type"],
                        # ✅ Best Practice: Use of class attributes for configuration and metadata
                        "high_days": data["high_days"],
                        "high_days_count": _get_high_days_count(data["high_days"]),
                    # ✅ Best Practice: Use of class attributes for configuration and metadata
                    }
                    # 🧠 ML Signal: Function call to process high_days data
                    records.append(record)
                # ✅ Best Practice: Use of class attributes for configuration and metadata
                df = pd.DataFrame.from_records(records)
                # 🧠 ML Signal: Method initializes entities, indicating a setup or configuration pattern
                df_to_db(
                    # 🧠 ML Signal: Appending processed record to a list
                    # ✅ Best Practice: Consider adding type hints for the 'entity_item' parameter for better readability and maintainability.
                    # ✅ Best Practice: Use of class attributes for configuration and metadata
                    # ✅ Best Practice: Use of a method to initialize class attributes improves readability and maintainability
                    data_schema=self.data_schema,
                    # 🧠 ML Signal: Creating a DataFrame from a list of records
                    # 🧠 ML Signal: Usage of a query method to fetch data from a database.
                    # 🧠 ML Signal: Use of a list to store entities suggests a collection or aggregation pattern
                    df=df,
                    provider=self.provider,
                    force_update=True,
                    # 🧠 ML Signal: Storing DataFrame to a database
                    # ⚠️ SAST Risk (Low): Hardcoded stock ID may lead to inflexibility or errors if the ID changes
                    drop_duplicates=True,
                # ⚠️ SAST Risk (Low): Potential risk if 'latest_infos' is not validated for expected structure or content.
                )


class JqkaLimitDownRecorder(TimestampsDataRecorder):
    # ✅ Best Practice: Explicitly setting force_update to ensure data consistency
    entity_provider = "em"
    # 🧠 ML Signal: Usage of a date manipulation function to calculate a past date.
    # 🧠 ML Signal: Iterating over timestamps to process data for each date
    entity_schema = Stock
    # ✅ Best Practice: Dropping duplicates to maintain data integrity

    # 🧠 ML Signal: Usage of pandas to generate a date range.
    # 🧠 ML Signal: Conversion of timestamp to string format
    provider = "jqka"
    data_schema = LimitDownInfo
    # 🧠 ML Signal: Logging information with dynamic content

    def init_entities(self):
        # 🧠 ML Signal: API call to fetch data for a specific date
        # fake entity to for trigger run
        self.entities = [Stock(id="stock_sz_000001")]
    # 🧠 ML Signal: Conversion of stock code to entity ID
    # ✅ Best Practice: Initializing an empty list before appending records
    # 🧠 ML Signal: Enumerating over API response data

    def init_timestamps(self, entity_item) -> List[pd.Timestamp]:
        latest_infos = LimitDownInfo.query_data(
            provider=self.provider, order=LimitDownInfo.timestamp.desc(), limit=1, return_type="domain"
        )
        if latest_infos and not self.force_update:
            start_date = latest_infos[0].timestamp
        else:
            # 最近一年的数据
            start_date = date_time_by_interval(current_date(), -360)
        return pd.date_range(start=start_date, end=pd.Timestamp.now(), freq="B").tolist()

    # 🧠 ML Signal: Conversion of date string to timestamp
    # ✅ Best Practice: Using dictionary comprehension for record creation
    def record(self, entity, start, end, size, timestamps):
        for timestamp in timestamps:
            the_date = to_time_str(timestamp)
            self.logger.info(f"record {self.data_schema} to {the_date}")
            limit_downs = jqka_api.get_limit_down(date=the_date)
            if limit_downs:
                # ✅ Best Practice: Explicit conversion of percentage values
                records = []
                for idx, data in enumerate(limit_downs):
                    entity_id = china_stock_code_to_id(code=data["code"])
                    # ✅ Best Practice: Consider adding type hints for function parameters and return values for better readability and maintainability.
                    record = {
                        # 🧠 ML Signal: Appending processed record to a list
                        "id": "{}_{}".format(entity_id, the_date),
                        # ✅ Best Practice: Initialize variables before using them in a loop.
                        "entity_id": entity_id,
                        # 🧠 ML Signal: Creating a DataFrame from records
                        "timestamp": to_pd_timestamp(the_date),
                        "code": data["code"],
                        # ⚠️ SAST Risk (Low): Potential risk of SQL injection if data_schema or provider are user-controlled
                        # 🧠 ML Signal: Iterating over a dictionary to calculate aggregate values.
                        "name": data["name"],
                        "is_new": data["is_new"],
                        # ⚠️ SAST Risk (Low): Assumes 'height' key exists in every dictionary item, which may lead to KeyError.
                        "is_again_limit": data["is_again_limit"],
                        "currency_value": data["currency_value"],
                        # 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
                        "change_pct": data["change_rate"] / 100,
                        # ✅ Best Practice: Enforcing data update and duplicate handling
                        # ⚠️ SAST Risk (Low): Assumes 'height' and 'number' keys exist in every dictionary item, which may lead to KeyError.
                        "turnover_rate": data["turnover_rate"] / 100,
                    # 🧠 ML Signal: Use of class attributes for configuration
                    }
                    # ✅ Best Practice: Returning multiple values as a tuple.
                    records.append(record)
                # 🧠 ML Signal: Use of class attributes for configuration
                # ✅ Best Practice: Method name 'init_entities' suggests initialization, which is clear and descriptive.
                df = pd.DataFrame.from_records(records)
                df_to_db(
                    # 🧠 ML Signal: Use of class attributes for configuration
                    # 🧠 ML Signal: Usage of a list to store entities, indicating a collection of items.
                    # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability.
                    data_schema=self.data_schema,
                    # 🧠 ML Signal: Use of class attributes for configuration
                    # 🧠 ML Signal: Instantiation of a Stock object with a specific id, indicating a pattern of object creation.
                    # 🧠 ML Signal: Usage of a query method to retrieve data from a database.
                    df=df,
                    provider=self.provider,
                    force_update=True,
                    # 🧠 ML Signal: Use of provider and ordering parameters in a query.
                    drop_duplicates=True,
                )

# ✅ Best Practice: Checking for conditions before proceeding with logic is a good practice.

def _cal_power_and_max_height(continuous_limit_up: dict):
    # 🧠 ML Signal: Accessing the timestamp attribute of the latest information.
    max_height = 0
    # 🧠 ML Signal: Iterating over timestamps to process data for each date
    power = 0
    for item in continuous_limit_up:
        # ✅ Best Practice: Convert timestamp to string format for logging and API calls
        # ✅ Best Practice: Using a utility function to calculate a date based on an interval.
        if max_height < item["height"]:
            max_height = item["height"]
        # 🧠 ML Signal: Logging information about the recording process
        # ✅ Best Practice: Using pandas date_range for generating a list of dates is efficient and readable.
        power = power + item["height"] * item["number"]
    return max_height, power
# 🧠 ML Signal: Fetching limit stats for a specific date


# 🧠 ML Signal: Fetching continuous limit up data for a specific date
# 🧠 ML Signal: Calculating power and max height from continuous limit up data
# ⚠️ SAST Risk (Low): Potential issue if limit_stats is None or has unexpected structure
# 🧠 ML Signal: Hardcoded entity_id for stock data
class JqkaEmotionRecorder(TimestampsDataRecorder):
    entity_provider = "em"
    entity_schema = Stock

    provider = "jqka"
    data_schema = Emotion

    def init_entities(self):
        # fake entity to for trigger run
        self.entities = [Stock(id="stock_sz_000001")]

    def init_timestamps(self, entity_item) -> List[pd.Timestamp]:
        latest_infos = Emotion.query_data(
            # 🧠 ML Signal: Creating a record dictionary with financial statistics
            # ✅ Best Practice: Convert date string to pandas timestamp for consistency
            provider=self.provider, order=Emotion.timestamp.desc(), limit=1, return_type="domain"
        )
        if latest_infos and not self.force_update:
            start_date = latest_infos[0].timestamp
        else:
            # 最近一年的数据
            start_date = date_time_by_interval(current_date(), -365)
        return pd.date_range(start=start_date, end=pd.Timestamp.now(), freq="B").tolist()

    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Creating a DataFrame from the record for database insertion
        # ⚠️ SAST Risk (Low): Ensure df_to_db handles SQL injection and data validation
        # ⚠️ SAST Risk (Low): Ensure record_data method handles input validation and error handling
        # 🧠 ML Signal: Example of calling a method with specific parameters
        # 🧠 ML Signal: Defining module exports for specific recorder classes
        for timestamp in timestamps:
            the_date = to_time_str(timestamp)
            self.logger.info(f"record {self.data_schema} to {the_date}")
            limit_stats = jqka_api.get_limit_stats(date=the_date)
            continuous_limit_up = jqka_api.get_continuous_limit_up(date=the_date)
            max_height, continuous_power = _cal_power_and_max_height(continuous_limit_up=continuous_limit_up)

            if limit_stats:
                # 大盘
                entity_id = "stock_sh_000001"
                record = {
                    "id": "{}_{}".format(entity_id, the_date),
                    "entity_id": entity_id,
                    "timestamp": to_pd_timestamp(the_date),
                    "limit_up_count": limit_stats["limit_up_count"]["today"]["num"],
                    "limit_up_open_count": limit_stats["limit_up_count"]["today"]["open_num"],
                    "limit_up_success_rate": limit_stats["limit_up_count"]["today"]["rate"],
                    "limit_down_count": limit_stats["limit_down_count"]["today"]["num"],
                    "limit_down_open_count": limit_stats["limit_down_count"]["today"]["open_num"],
                    "limit_down_success_rate": limit_stats["limit_down_count"]["today"]["rate"],
                    "max_height": max_height,
                    "continuous_power": continuous_power,
                }
                df = pd.DataFrame.from_records([record])
                df_to_db(
                    data_schema=self.data_schema,
                    df=df,
                    provider=self.provider,
                    force_update=True,
                    drop_duplicates=True,
                )


if __name__ == "__main__":
    # JqkaLimitDownRecorder().run()
    LimitDownInfo.record_data(start_timestamp="2024-02-02", end_timestamp="2024-02-16", force_update=True)


# the __all__ is generated
__all__ = ["JqkaLimitUpRecorder", "JqkaLimitDownRecorder", "JqkaEmotionRecorder"]