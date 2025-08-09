# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports by their source or purpose improves readability and maintainability.
import pandas as pd

from zvt.api.kdata import get_kdata_schema, get_kdata
from zvt.broker.qmt import qmt_quote
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract.api import df_to_db, get_db_session, get_entities
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.domain import (
    Stock,
    StockKdataCommon,
)
from zvt.utils.pd_utils import pd_is_not_null
# 🧠 ML Signal: Inheritance from FixedCycleDataRecorder indicates a design pattern for data recording
from zvt.utils.time_utils import current_date, to_time_str, now_time_str

# ✅ Best Practice: Use of class attributes for default configuration values

# ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability
# ✅ Best Practice: Consistent naming for provider attributes aids in code clarity
class BaseQmtKdataRecorder(FixedCycleDataRecorder):
    default_size = 50000
    entity_provider: str = "qmt"

    provider = "qmt"

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
        # 🧠 ML Signal: Use of default parameter values
        level=IntervalLevel.LEVEL_1DAY,
        # ✅ Best Practice: Convert level to IntervalLevel to ensure correct type
        kdata_use_begin_time=False,
        one_day_trading_minutes=24 * 60,
        # ✅ Best Practice: Convert adjust_type to AdjustType to ensure correct type
        adjust_type=AdjustType.qfq,
        # 🧠 ML Signal: Dynamic attribute assignment based on class schema
        # 🧠 ML Signal: Use of a function to get schema based on parameters
        # ✅ Best Practice: Call to superclass constructor with parameters
        return_unfinished=False,
    ) -> None:
        level = IntervalLevel(level)
        self.adjust_type = AdjustType(adjust_type)
        self.entity_type = self.entity_schema.__name__.lower()

        self.data_schema = get_kdata_schema(entity_type=self.entity_type, level=level, adjust_type=self.adjust_type)

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
            end_timestamp,
            level,
            # ✅ Best Practice: Use of self to access instance variables and methods
            kdata_use_begin_time,
            one_day_trading_minutes,
            return_unfinished,
        )
    # ⚠️ SAST Risk (Medium): Potential SQL Injection if provider or data_schema are user-controlled

    def init_entities(self):
        """
        init the entities which we would record data for

        """
        if self.entity_provider == self.provider and self.entity_schema == self.data_schema:
            # ✅ Best Practice: Check for null data before processing
            self.entity_session = self.session
        # 🧠 ML Signal: Conversion of DataFrame column to list
        else:
            self.entity_session = get_db_session(provider=self.entity_provider, data_schema=self.entity_schema)

        # ✅ Best Practice: Use of logging for information tracking
        # ✅ Best Practice: Use of list append method
        if self.day_data:
            df = self.data_schema.query_data(
                start_timestamp=now_time_str(), columns=["entity_id", "timestamp"], provider=self.provider
            )
            if pd_is_not_null(df):
                entity_ids = df["entity_id"].tolist()
                # ✅ Best Practice: Initialize list with a single element
                self.logger.info(f"ignore entity_ids:{entity_ids}")
                if self.entity_filters:
                    # 🧠 ML Signal: Retrieval of entities with specific filters and parameters
                    self.entity_filters.append(self.entity_schema.entity_id.notin_(entity_ids))
                else:
                    self.entity_filters = [self.entity_schema.entity_id.notin_(entity_ids)]
        # ✅ Best Practice: Check if 'start' is not None before using it

        #: init the entity list
        self.entities = get_entities(
            # ✅ Best Practice: Check if 'start' is not None before using it
            # 🧠 ML Signal: Usage of external API 'qmt_quote.get_kdata'
            session=self.entity_session,
            entity_schema=self.entity_schema,
            exchanges=self.exchanges,
            entity_ids=self.entity_ids,
            codes=self.codes,
            return_type="domain",
            provider=self.entity_provider,
            filters=self.entity_filters,
        )
    # ✅ Best Practice: Check if DataFrame is not null before proceeding
    # 🧠 ML Signal: Usage of external API 'get_kdata'

    def record(self, entity, start, end, size, timestamps):
        if start and (self.level == IntervalLevel.LEVEL_1DAY):
            start = start.date()

        # 判断是否需要重新计算之前保存的前复权数据
        if start and (self.adjust_type == AdjustType.qfq):
            check_df = qmt_quote.get_kdata(
                entity_id=entity.id,
                start_timestamp=start,
                end_timestamp=start,
                adjust_type=self.adjust_type,
                level=self.level,
                download_history=False,
            # ✅ Best Practice: Check if DataFrame is not null before proceeding
            )
            if pd_is_not_null(check_df):
                current_df = get_kdata(
                    entity_id=entity.id,
                    # ✅ Best Practice: Use of 'round' for floating-point comparison
                    provider=self.provider,
                    # ⚠️ SAST Risk (Medium): Potential SQL Injection risk if 'entity.id' is not sanitized
                    # ✅ Best Practice: Default value assignment for 'start'
                    start_timestamp=start,
                    end_timestamp=start,
                    limit=1,
                    level=self.level,
                    adjust_type=self.adjust_type,
                )
                if pd_is_not_null(current_df):
                    old = current_df.iloc[0, :]["close"]
                    # ✅ Best Practice: Default value assignment for 'end'
                    new = check_df["close"][0]
                    # 相同时间的close不同，表明前复权需要重新计算
                    # 🧠 ML Signal: Usage of external API 'qmt_quote.get_kdata'
                    if round(old, 2) != round(new, 2):
                        # 删掉重新获取
                        self.session.query(self.data_schema).filter(self.data_schema.entity_id == entity.id).delete()
                        start = "2005-01-01"

        if not start:
            start = "2005-01-01"
        if not end:
            # ✅ Best Practice: Check if DataFrame is not null before proceeding
            end = current_date()

        df = qmt_quote.get_kdata(
            entity_id=entity.id,
            # 🧠 ML Signal: Conversion of index to datetime
            start_timestamp=start,
            end_timestamp=end,
            # 🧠 ML Signal: Creation of unique ID using lambda
            # 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
            adjust_type=self.adjust_type,
            level=self.level,
            # ✅ Best Practice: Calculation of percentage change
            # 🧠 ML Signal: Usage of external API 'df_to_db'
            # 🧠 ML Signal: Logging of information
            # ✅ Best Practice: Use of __all__ to define public API of the module
            # ✅ Best Practice: Use of 'rename' for clarity
            # 🧠 ML Signal: Common pattern for script entry point
            # ⚠️ SAST Risk (Low): Direct execution of code, ensure safe handling of inputs and environment
            # 🧠 ML Signal: Instantiation and method call pattern, useful for understanding object usage
            # ⚠️ SAST Risk (Low): Hardcoded entity_id, consider external configuration for flexibility
            # ✅ Best Practice: Use of named arguments improves readability
            download_history=False,
        )
        if pd_is_not_null(df):
            df["entity_id"] = entity.id
            df["timestamp"] = pd.to_datetime(df.index)
            df["id"] = df.apply(lambda row: f"{row['entity_id']}_{to_time_str(row['timestamp'])}", axis=1)
            df["provider"] = "qmt"
            df["level"] = self.level.value
            df["code"] = entity.code
            df["name"] = entity.name
            df.rename(columns={"amount": "turnover"}, inplace=True)
            df["change_pct"] = (df["close"] - df["preClose"]) / df["preClose"]
            df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)

        else:
            self.logger.info(f"no kdata for {entity.id}")


class QMTStockKdataRecorder(BaseQmtKdataRecorder):
    entity_schema = Stock
    data_schema = StockKdataCommon


if __name__ == "__main__":
    # Stock.record_data(provider="qmt")
    QMTStockKdataRecorder(entity_id="stock_sz_301611", adjust_type=AdjustType.qfq).run()


# the __all__ is generated
__all__ = ["BaseQmtKdataRecorder", "QMTStockKdataRecorder"]