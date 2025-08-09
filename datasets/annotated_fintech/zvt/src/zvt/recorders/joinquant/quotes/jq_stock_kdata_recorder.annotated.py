# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions from a module indicates selective usage patterns

import pandas as pd

# 🧠 ML Signal: Importing specific configurations and functions from a module indicates selective usage patterns
from jqdatapy.api import get_token, get_bars

from zvt import zvt_config
from zvt.api.kdata import generate_kdata_id, get_kdata_schema, get_kdata
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract.api import df_to_db
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.domain import Stock, StockKdataCommon, Stock1wkHfqKdata
from zvt.recorders.joinquant.common import to_jq_trading_level, to_jq_entity_id

# 🧠 ML Signal: Inheritance from a specific base class indicates a pattern of extending functionality
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import (
    to_time_str,
    now_pd_timestamp,
    TIME_FORMAT_DAY,
    TIME_FORMAT_ISO8601,
)

# 🧠 ML Signal: Hardcoded provider name could indicate a pattern of data source usage


# 🧠 ML Signal: Association with a specific schema suggests a pattern of data structure usage
# 🧠 ML Signal: Repeated provider name reinforces the pattern of data source usage
# 🧠 ML Signal: Use of a common data schema indicates a pattern of standardizing data formats
class JqChinaStockKdataRecorder(FixedCycleDataRecorder):
    entity_provider = "joinquant"
    entity_schema = Stock

    # 数据来自jq
    provider = "joinquant"

    # 只是为了把recorder注册到data_schema
    data_schema = StockKdataCommon

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
        ignore_failed=True,
        # 🧠 ML Signal: Use of default parameter values
        real_time=False,
        # ✅ Best Practice: Explicitly converting level to IntervalLevel
        fix_duplicate_way="ignore",
        start_timestamp=None,
        # ✅ Best Practice: Explicitly converting adjust_type to AdjustType
        end_timestamp=None,
        # 🧠 ML Signal: Use of super() to initialize parent class
        # 🧠 ML Signal: Use of a function to get a data schema
        # 🧠 ML Signal: Use of a function to convert level to trading level
        level=IntervalLevel.LEVEL_1DAY,
        kdata_use_begin_time=False,
        one_day_trading_minutes=24 * 60,
        adjust_type=AdjustType.qfq,
        return_unfinished=False,
    ) -> None:
        level = IntervalLevel(level)
        adjust_type = AdjustType(adjust_type)
        self.data_schema = get_kdata_schema(
            entity_type="stock", level=level, adjust_type=adjust_type
        )
        self.jq_trading_level = to_jq_trading_level(level)

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
            real_time,
            # ✅ Best Practice: Call to superclass method ensures proper initialization of inherited attributes or methods.
            fix_duplicate_way,
            # 🧠 ML Signal: Filtering entities based on a condition, indicating a pattern of data validation or cleaning.
            start_timestamp,
            end_timestamp,
            # ✅ Best Practice: List comprehension is used for concise and efficient filtering of entities.
            level,
            # 🧠 ML Signal: Function definition with parameters indicating a pattern for generating domain IDs
            # ⚠️ SAST Risk (Medium): Storing sensitive information like username and password in a config
            kdata_use_begin_time,
            # 🧠 ML Signal: Use of a function to get a token
            # 🧠 ML Signal: Use of conditional logic to filter entities, which can be a pattern for data processing.
            one_day_trading_minutes,
            # ⚠️ SAST Risk (Low): Potential risk if `now_pd_timestamp()` is not timezone-aware or consistent.
            # 🧠 ML Signal: Usage of a helper function to generate an ID, indicating a common pattern for ID generation
            return_unfinished,
            # ✅ Best Practice: Use of descriptive parameter names improves code readability
            # ✅ Best Practice: Check for division by zero to prevent runtime errors
        )
        # 🧠 ML Signal: Usage of a data retrieval function with specific parameters

        self.adjust_type = adjust_type

        get_token(zvt_config["jq_username"], zvt_config["jq_password"], force=True)

    def init_entities(self):
        super().init_entities()
        # 过滤掉退市的
        self.entities = [
            entity
            for entity in self.entities
            if (entity.end_date is None) or (entity.end_date > now_pd_timestamp())
        ]

    # 🧠 ML Signal: Logging information about the process and parameters
    def generate_domain_id(self, entity, original_data):
        return generate_kdata_id(
            entity_id=entity.id, timestamp=original_data["timestamp"], level=self.level
        )

    # 🧠 ML Signal: Pattern of adjusting financial data with a factor
    def recompute_qfq(self, entity, qfq_factor, last_timestamp):
        # 重新计算前复权数据
        if qfq_factor != 0:
            kdatas = get_kdata(
                # 🧠 ML Signal: Use of conditional logic to determine reference date based on adjustment type
                provider=self.provider,
                # ⚠️ SAST Risk (Low): Potential risk if kdatas contains untrusted data
                entity_id=entity.id,
                level=self.level.value,
                # ⚠️ SAST Risk (Low): Committing changes to the database without error handling
                order=self.data_schema.timestamp.asc(),
                return_type="domain",
                # 🧠 ML Signal: Conditional logic to determine if end_timestamp is used
                session=self.session,
                filters=[self.data_schema.timestamp < last_timestamp],
            )
            if kdatas:
                self.logger.info(
                    "recomputing {} qfq kdata,factor is:{}".format(
                        entity.code, qfq_factor
                    )
                )
                for kdata in kdatas:
                    kdata.open = round(kdata.open * qfq_factor, 2)
                    kdata.close = round(kdata.close * qfq_factor, 2)
                    kdata.high = round(kdata.high * qfq_factor, 2)
                    kdata.low = round(kdata.low * qfq_factor, 2)
                self.session.add_all(kdatas)
                self.session.commit()

    def record(self, entity, start, end, size, timestamps):
        if self.adjust_type == AdjustType.hfq:
            fq_ref_date = "2000-01-01"
        else:
            # 🧠 ML Signal: Checking if DataFrame is not null before processing
            fq_ref_date = to_time_str(now_pd_timestamp())

        if not self.end_timestamp:
            # ✅ Best Practice: Use of rename for clarity and consistency in column names
            df = get_bars(
                to_jq_entity_id(entity),
                count=size,
                # ✅ Best Practice: Converting timestamp to datetime for consistency
                unit=self.jq_trading_level,
                # fields=['date', 'open', 'close', 'low', 'high', 'volume', 'money'],
                fq_ref_date=fq_ref_date,
                # 🧠 ML Signal: Conditional logic based on adjustment type
            )
        else:
            end_timestamp = to_time_str(self.end_timestamp)
            df = get_bars(
                to_jq_entity_id(entity),
                count=size,
                unit=self.jq_trading_level,
                # fields=['date', 'open', 'close', 'low', 'high', 'volume', 'money'],
                end_date=end_timestamp,
                fq_ref_date=fq_ref_date,
            )
        if pd_is_not_null(df):
            df["name"] = entity.name
            df.rename(columns={"money": "turnover", "date": "timestamp"}, inplace=True)

            # 🧠 ML Signal: Checking if DataFrame is not null before processing
            df["entity_id"] = entity.id
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # ✅ Best Practice: Use of string formatting for constructing IDs
            df["provider"] = "joinquant"
            # ⚠️ SAST Risk (Low): Potential floating-point precision issue with rounding
            df["level"] = self.level.value
            df["code"] = entity.code

            # 判断是否需要重新计算之前保存的前复权数据
            # 🧠 ML Signal: Use of DataFrame apply method for row-wise operations
            if self.adjust_type == AdjustType.qfq:
                # 🧠 ML Signal: Recomputing factors based on condition
                check_df = df.head(1)
                # 🧠 ML Signal: Use of drop_duplicates to handle duplicate data
                check_date = check_df["timestamp"][0]
                current_df = get_kdata(
                    # 🧠 ML Signal: Use of a function to save DataFrame to a database
                    entity_id=entity.id,
                    # 🧠 ML Signal: Use of __all__ to define public API of the module
                    # 🧠 ML Signal: Use of a main guard to execute code conditionally
                    provider=self.provider,
                    start_timestamp=check_date,
                    end_timestamp=check_date,
                    limit=1,
                    level=self.level,
                    adjust_type=self.adjust_type,
                )
                if pd_is_not_null(current_df):
                    old = current_df.iloc[0, :]["close"]
                    new = check_df["close"][0]
                    # 相同时间的close不同，表明前复权需要重新计算
                    if round(old, 2) != round(new, 2):
                        qfq_factor = new / old
                        last_timestamp = pd.Timestamp(check_date)
                        self.recompute_qfq(
                            entity, qfq_factor=qfq_factor, last_timestamp=last_timestamp
                        )

            def generate_kdata_id(se):
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
    Stock1wkHfqKdata.record_data(codes=["300999"])


# the __all__ is generated
__all__ = ["JqChinaStockKdataRecorder"]
