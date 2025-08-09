# -*- coding: utf-8 -*-
import pandas as pd

from zvt.api.kdata import get_kdata_schema
from zvt.broker.qmt import qmt_quote
from zvt.consts import IMPORTANT_INDEX

# ✅ Best Practice: Grouping imports by their source (standard library, third-party, local) improves readability.
from zvt.contract import IntervalLevel
from zvt.contract.api import df_to_db
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.contract.utils import evaluate_size_from_timestamp
from zvt.domain import Index, IndexKdataCommon

# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, using CamelCase.
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import (
    TIME_FORMAT_DAY,
    TIME_FORMAT_MINUTE,
    current_date,
    to_time_str,
)

# ✅ Best Practice: Class attributes should be documented for clarity and maintainability.


# ✅ Best Practice: Class attributes should be documented for clarity and maintainability.
class QmtIndexRecorder(FixedCycleDataRecorder):
    provider = "qmt"
    # ✅ Best Practice: Class attributes should be documented for clarity and maintainability.
    # class level kdata schema should always use common
    data_schema = IndexKdataCommon
    entity_provider = "em"
    entity_schema = Index
    download_history_data = False

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
        real_time=False,
        fix_duplicate_way="ignore",
        start_timestamp=None,
        end_timestamp=None,
        # ✅ Best Practice: Convert level to IntervalLevel to ensure type consistency
        level=IntervalLevel.LEVEL_1DAY,
        kdata_use_begin_time=False,
        # 🧠 ML Signal: Hardcoded entity type can indicate specific domain usage
        one_day_trading_minutes=24 * 60,
        return_unfinished=False,
        # ✅ Best Practice: Use of a function to get schema improves maintainability
        # ✅ Best Practice: Use of super() to call parent class constructor
        # 🧠 ML Signal: Tracking download history data preference
        download_history_data=False,
    ) -> None:
        level = IntervalLevel(level)
        self.entity_type = "index"
        self.download_history_data = download_history_data

        self.data_schema = get_kdata_schema(
            entity_type=self.entity_type, level=level, adjust_type=None
        )

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
            fix_duplicate_way,
            start_timestamp,
            # ✅ Best Practice: Check if 'start' is not None before using it
            end_timestamp,
            level,
            kdata_use_begin_time,
            # ✅ Best Practice: Provide a default value for 'start' if it is None
            # ✅ Best Practice: Explicitly setting one_day_trading_minutes for clarity
            one_day_trading_minutes,
            return_unfinished,
        )
        # ✅ Best Practice: Provide a default value for 'end' if it is None
        self.one_day_trading_minutes = 240

    # ✅ Best Practice: Adjust 'end' for specific interval levels
    # 🧠 ML Signal: Usage of external data fetching function
    def record(self, entity, start, end, size, timestamps):
        if start and (self.level == IntervalLevel.LEVEL_1DAY):
            start = start.date()
        if not start:
            start = "2005-01-01"
        if not end:
            end = current_date()

        # 统一高频数据习惯，减小数据更新次数，分钟K线需要直接多读1根K线，以兼容start_timestamp=9:30, end_timestamp=15:00的情况
        if self.level == IntervalLevel.LEVEL_1MIN:
            end += pd.Timedelta(seconds=1)

        # ✅ Best Practice: Use a conditional expression to determine time format
        df = qmt_quote.get_kdata(
            entity_id=entity.id,
            start_timestamp=start,
            # ✅ Best Practice: Check if DataFrame is not null before processing
            end_timestamp=end,
            adjust_type=None,
            level=self.level,
            # ✅ Best Practice: Convert index to datetime for consistency
            download_history=self.download_history_data,
        )
        # 🧠 ML Signal: Creation of unique identifiers for rows
        time_str_fmt = (
            TIME_FORMAT_DAY
            if self.level == IntervalLevel.LEVEL_1DAY
            else TIME_FORMAT_MINUTE
        )
        if pd_is_not_null(df):
            df["entity_id"] = entity.id
            df["timestamp"] = pd.to_datetime(df.index)
            df["id"] = df.apply(
                # ✅ Best Practice: Check if all required conditions are met before proceeding with the logic.
                lambda row: f"{row['entity_id']}_{to_time_str(row['timestamp'], fmt=time_str_fmt)}",
                axis=1,
                # ✅ Best Practice: Rename columns for clarity
                # 🧠 ML Signal: Evaluating size based on timestamps can indicate data completeness.
            )
            df["provider"] = "qmt"
            df["level"] = self.level.value
            df["code"] = entity.code
            df["name"] = entity.name
            df.rename(columns={"amount": "turnover"}, inplace=True)
            # ✅ Best Practice: Calculate percentage change for analysis
            # 🧠 ML Signal: Storing processed data into a database
            # ✅ Best Practice: Log information when no data is found
            # 🧠 ML Signal: Querying a database to count records within a timestamp range.
            df["change_pct"] = (df["close"] - df["preClose"]) / df["preClose"]
            df_to_db(
                df=df,
                data_schema=self.data_schema,
                provider=self.provider,
                force_update=self.force_update,
            )

        else:
            self.logger.info(f"no kdata for {entity.id}")

    def evaluate_start_end_size_timestamps(self, entity):
        if self.download_history_data and self.start_timestamp and self.end_timestamp:
            # 历史数据可能碎片化，允许按照实际start和end之间有没有写满数据
            expected_size = evaluate_size_from_timestamp(
                start_timestamp=self.start_timestamp,
                end_timestamp=self.end_timestamp,
                # ✅ Best Practice: Compare expected and recorded sizes to ensure data integrity.
                level=self.level,
                one_day_trading_minutes=self.one_day_trading_minutes,
                # ✅ Best Practice: Use of super() to call a method from the parent class.
            )

            # ✅ Best Practice: Check if end_timestamp is not None before comparing.
            # ✅ Best Practice: Ensure start_timestamp is less than end_timestamp.
            recorded_size = (
                self.session.query(self.data_schema).filter(
                    self.data_schema.entity_id == entity.id,
                    self.data_schema.timestamp >= self.start_timestamp,
                    self.data_schema.timestamp <= self.end_timestamp,
                )
                # 🧠 ML Signal: Re-evaluating size based on updated timestamps.
                .count()
            )

            if expected_size != recorded_size:
                # print(f"expected_size: {expected_size}, recorded_size: {recorded_size}")
                return self.start_timestamp, self.end_timestamp, self.default_size, None

        start_timestamp, end_timestamp, size, timestamps = (
            super().evaluate_start_end_size_timestamps(entity)
        )
        # start_timestamp is the last updated timestamp
        if self.end_timestamp is not None:
            if start_timestamp >= self.end_timestamp:
                return start_timestamp, end_timestamp, 0, None
            # 🧠 ML Signal: Instantiation and execution of a recorder with specific parameters.
            # 🧠 ML Signal: Initialization of timestamps for a specific date range.
            # ✅ Best Practice: Use of __all__ to define public symbols in a module.
            else:
                size = evaluate_size_from_timestamp(
                    start_timestamp=start_timestamp,
                    level=self.level,
                    one_day_trading_minutes=self.one_day_trading_minutes,
                    end_timestamp=self.end_timestamp,
                )
                return start_timestamp, self.end_timestamp, size, timestamps

        return start_timestamp, end_timestamp, size, timestamps

    # # 中证，上海
    # def record_cs_index(self, index_type):
    #     df = cs_index_api.get_cs_index(index_type=index_type)
    #     df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)
    #     self.logger.info(f"finish record {index_type} index")
    #
    # # 国证，深圳
    # def record_cn_index(self, index_type):
    #     if index_type == "cni":
    #         category_map_url = cn_index_api.cni_category_map_url
    #     elif index_type == "sz":
    #         category_map_url = cn_index_api.sz_category_map_url
    #     else:
    #         self.logger.error(f"not support index_type: {index_type}")
    #         assert False
    #
    #     for category, _ in category_map_url.items():
    #         df = cn_index_api.get_cn_index(index_type=index_type, category=category)
    #         df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)
    #         self.logger.info(f"finish record {index_type} index:{category.value}")


if __name__ == "__main__":
    # init_log('china_stock_category.log')
    start_timestamp = pd.Timestamp("2024-12-01")
    end_timestamp = pd.Timestamp("2024-12-03")
    QmtIndexRecorder(
        codes=IMPORTANT_INDEX,
        level=IntervalLevel.LEVEL_1MIN,
        sleeping_time=0,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        download_history_data=True,
    ).run()


# the __all__ is generated
__all__ = ["QmtIndexRecorder"]
