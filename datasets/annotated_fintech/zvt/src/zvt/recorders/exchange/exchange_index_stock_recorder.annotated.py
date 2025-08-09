# -*- coding: utf-8 -*-

from typing import List

import pandas as pd

from zvt.contract.api import df_to_db
from zvt.contract.recorder import TimestampsDataRecorder

# 🧠 ML Signal: Inheritance from TimestampsDataRecorder indicates a pattern of extending functionality
from zvt.domain import Index, IndexStock

# 🧠 ML Signal: Use of class attributes for configuration suggests a pattern for setting static properties
from zvt.recorders.exchange.api import cs_index_stock_api, cn_index_stock_api

# ✅ Best Practice: Class name should be descriptive of its purpose and functionality
from zvt.utils.time_utils import pre_month_start_date
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Static configuration of entity_provider indicates a pattern for data source specification

# 🧠 ML Signal: Static configuration of entity_schema indicates a pattern for defining data structure
# 🧠 ML Signal: Static configuration of provider indicates a pattern for specifying data provider


class ExchangeIndexStockRecorder(TimestampsDataRecorder):
    entity_provider = "exchange"
    entity_schema = Index

    provider = "exchange"
    data_schema = IndexStock

    def __init__(
        self,
        force_update=False,
        sleeping_time=5,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        code=None,
        codes=None,
        day_data=False,
        # ✅ Best Practice: Call to super() ensures proper initialization of the base class.
        entity_filters=None,
        ignore_failed=True,
        real_time=False,
        fix_duplicate_way="add",
        start_timestamp=None,
        end_timestamp=None,
        record_history=False,
    ) -> None:
        super().__init__(
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            entity_ids,
            code,
            codes,
            day_data,
            # 🧠 ML Signal: Tracking the record_history parameter could indicate user preference for data persistence.
            # ✅ Best Practice: Consider adding type hints for the method parameters for better readability and maintainability
            entity_filters,
            ignore_failed,
            # ✅ Best Practice: Use descriptive variable names for better readability
            real_time,
            # 🧠 ML Signal: Conditional logic based on a boolean attribute (self.record_history) can indicate feature usage patterns
            fix_duplicate_way,
            start_timestamp,
            end_timestamp,
            # 🧠 ML Signal: List comprehension usage can indicate coding style and efficiency preferences
        )
        # 🧠 ML Signal: Conditional logic based on entity attributes
        self.record_history = record_history

    # 🧠 ML Signal: Returning a list with a single element can indicate a specific design choice or pattern
    def init_timestamps(self, entity_item) -> List[pd.Timestamp]:
        # 🧠 ML Signal: API call pattern with dynamic parameters
        last_valid_date = pre_month_start_date()
        if self.record_history:
            # 🧠 ML Signal: Data persistence pattern
            # 每个月记录一次
            return [
                to_pd_timestamp(item)
                for item in pd.date_range(
                    entity_item.list_date, last_valid_date, freq="M"
                )
            ]
        else:
            # 🧠 ML Signal: API call pattern with static parameters
            return [last_valid_date]

    # 🧠 ML Signal: Data persistence pattern
    # ⚠️ SAST Risk (Low): Direct execution of code without input validation
    # 🧠 ML Signal: Hardcoded parameters in function call
    # ✅ Best Practice: Define __all__ for explicit module exports

    def record(self, entity, start, end, size, timestamps):
        if entity.publisher == "cnindex":
            for timestamp in timestamps:
                df = cn_index_stock_api.get_cn_index_stock(
                    code=entity.code, timestamp=timestamp, name=entity.name
                )
                df_to_db(
                    data_schema=self.data_schema,
                    df=df,
                    provider=self.provider,
                    force_update=True,
                )
        elif entity.publisher == "csindex":
            # cs index not support history data
            df = cs_index_stock_api.get_cs_index_stock(
                code=entity.code, timestamp=None, name=entity.name
            )
            df_to_db(
                data_schema=self.data_schema,
                df=df,
                provider=self.provider,
                force_update=True,
            )


if __name__ == "__main__":
    # ExchangeIndexMetaRecorder().run()
    ExchangeIndexStockRecorder(codes=["399370"]).run()


# the __all__ is generated
__all__ = ["ExchangeIndexStockRecorder"]
