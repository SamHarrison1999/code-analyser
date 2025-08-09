# -*- coding: utf-8 -*-
import argparse

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns

import pandas as pd

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from jqdatapy.api import get_token, get_bars

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt import init_log, zvt_config
from zvt.api.kdata import generate_kdata_id, get_kdata_schema, get_kdata

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
from zvt.contract import IntervalLevel
from zvt.contract.api import df_to_db

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.domain import Index, IndexKdataCommon

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.recorders.joinquant.common import to_jq_trading_level, to_jq_entity_id
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Importing specific classes from a module indicates selective usage patterns
# 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
from zvt.utils.time_utils import to_time_str, TIME_FORMAT_DAY, TIME_FORMAT_ISO8601

# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
# 🧠 ML Signal: Class attribute definition, useful for understanding schema usage


# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns
# 🧠 ML Signal: Class attribute definition, useful for understanding default configurations
# 🧠 ML Signal: Class attribute definition, useful for understanding schema usage
class JqChinaIndexKdataRecorder(FixedCycleDataRecorder):
    entity_provider = "joinquant"
    entity_schema = Index

    # 数据来自jq
    provider = "joinquant"

    # 只是为了把recorder注册到data_schema
    data_schema = IndexKdataCommon

    def __init__(
        self,
        force_update=True,
        sleeping_time=10,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        code=None,
        codes=None,
        day_data=False,
        entity_filters=None,
        # ✅ Best Practice: Convert level to IntervalLevel to ensure type consistency
        ignore_failed=True,
        real_time=False,
        # 🧠 ML Signal: Usage of get_kdata_schema indicates data schema retrieval pattern
        fix_duplicate_way="ignore",
        start_timestamp=None,
        # ⚠️ SAST Risk (Medium): Storing credentials in code can lead to security vulnerabilities
        # ✅ Best Practice: Use of super() to ensure proper inheritance and initialization
        # 🧠 ML Signal: Conversion to trading level suggests a pattern for data granularity
        end_timestamp=None,
        level=IntervalLevel.LEVEL_1DAY,
        kdata_use_begin_time=False,
        one_day_trading_minutes=24 * 60,
        return_unfinished=False,
    ) -> None:
        level = IntervalLevel(level)
        self.data_schema = get_kdata_schema(entity_type="index", level=level)
        self.jq_trading_level = to_jq_trading_level(level)
        get_token(zvt_config["jq_username"], zvt_config["jq_password"], force=True)
        super().__init__(
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            entity_ids,
            code,
            codes,
            day_data,
            entity_filters,
            ignore_failed,
            # ✅ Best Practice: Call to superclass method ensures proper initialization of inherited attributes
            real_time,
            # 🧠 ML Signal: Filtering entities based on specific codes indicates a pattern of exclusion
            fix_duplicate_way,
            start_timestamp,
            end_timestamp,
            # 🧠 ML Signal: List comprehension used for filtering, a common pattern in Python
            # 🧠 ML Signal: Function definition with parameters indicating a pattern for generating domain IDs
            level,
            kdata_use_begin_time,
            # 🧠 ML Signal: Usage of a helper function to generate IDs, indicating a common pattern
            one_day_trading_minutes,
            # ⚠️ SAST Risk (Low): Potential risk if `generate_kdata_id` is not properly validated or sanitized
            # ⚠️ SAST Risk (Low): Potential for NoneType error if self.end_timestamp is not initialized
            return_unfinished,
            # ✅ Best Practice: Directly passing parameters from one function to another improves readability
            # 🧠 ML Signal: Usage of external function get_bars with specific parameters
            # 🧠 ML Signal: Conversion of entity to a specific ID format
        )

    def init_entities(self):
        super().init_entities()
        # ignore no data index
        self.entities = [
            entity
            for entity in self.entities
            if entity.code not in ["310001", "310002", "310003", "310004"]
            # 🧠 ML Signal: Conversion of timestamp to string format
        ]

    def generate_domain_id(self, entity, original_data):
        return generate_kdata_id(
            entity_id=entity.id, timestamp=original_data["timestamp"], level=self.level
        )

    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Usage of external function get_bars with specific parameters
        # 🧠 ML Signal: Conversion of entity to a specific ID format
        if not self.end_timestamp:
            df = get_bars(
                to_jq_entity_id(entity),
                count=size,
                unit=self.jq_trading_level,
                # fields=['date', 'open', 'close', 'low', 'high', 'volume', 'money']
                # ⚠️ SAST Risk (Low): Potential for handling of null dataframes
            )
        else:
            # ✅ Best Practice: Adding metadata to dataframe for better traceability
            end_timestamp = to_time_str(self.end_timestamp)
            df = get_bars(
                # ✅ Best Practice: Renaming columns for consistency and clarity
                to_jq_entity_id(entity),
                count=size,
                # ✅ Best Practice: Adding metadata to dataframe for better traceability
                unit=self.jq_trading_level,
                # 🧠 ML Signal: Usage of DataFrame apply function with custom function
                # fields=['date', 'open', 'close', 'low', 'high', 'volume', 'money'],
                # ✅ Best Practice: Ensuring timestamp column is in datetime format
                end_date=end_timestamp,
                # 🧠 ML Signal: Dropping duplicates in DataFrame
            )
        # ✅ Best Practice: Adding metadata to dataframe for better traceability
        if pd_is_not_null(df):
            # ⚠️ SAST Risk (Low): Potential risk if df_to_db does not handle SQL injection or data validation
            df["name"] = entity.name
            # ✅ Best Practice: Adding metadata to dataframe for better traceability
            df.rename(columns={"money": "turnover", "date": "timestamp"}, inplace=True)

            # ✅ Best Practice: Adding metadata to dataframe for better traceability
            df["entity_id"] = entity.id
            # ✅ Best Practice: Use of argparse for command-line argument parsing
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["provider"] = "joinquant"
            # 🧠 ML Signal: Command-line argument parsing with default values and choices
            df["level"] = self.level.value
            df["code"] = entity.code
            # 🧠 ML Signal: Command-line argument parsing with default list and nargs

            def generate_kdata_id(se):
                # 🧠 ML Signal: Conversion of command-line argument to specific type
                # ✅ Best Practice: Logging initialization with dynamic filename
                # 🧠 ML Signal: Instantiation and execution of a class with specific parameters
                # 🧠 ML Signal: Function call with specific parameters
                # ✅ Best Practice: Use of __all__ to define public API of the module
                if self.level >= IntervalLevel.LEVEL_1DAY:
                    return "{}_{}".format(
                        se["entity_id"],
                        to_time_str(se["timestamp"], fmt=TIME_FORMAT_DAY),
                    )
                else:
                    return "{}_{}".format(
                        se["entity_id"],
                        to_time_str(se["timestamp"], fmt=TIME_FORMAT_ISO8601),
                    )

            df["id"] = df[["entity_id", "timestamp"]].apply(generate_kdata_id, axis=1)

            df = df.drop_duplicates(subset="id", keep="last")

            df_to_db(
                df=df,
                data_schema=self.data_schema,
                provider=self.provider,
                force_update=self.force_update,
            )

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        help="trading level",
        default="1d",
        choices=[item.value for item in IntervalLevel],
    )
    parser.add_argument("--codes", help="codes", default=["000001"], nargs="+")

    args = parser.parse_args()

    level = IntervalLevel(args.level)
    codes = args.codes

    init_log("jq_china_stock_{}_kdata.log".format(args.level))
    JqChinaIndexKdataRecorder(
        level=level, sleeping_time=0, codes=codes, real_time=False
    ).run()

    print(get_kdata(entity_id="index_sh_000001", limit=10))


# the __all__ is generated
__all__ = ["JqChinaIndexKdataRecorder"]
