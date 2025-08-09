# -*- coding: utf-8 -*-

from zvt.api.kdata import get_kdata_schema

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from zvt.contract import IntervalLevel, AdjustType
from zvt.contract.api import df_to_db
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.domain import (
    Stock,
    Index,
    Block,
    StockKdataCommon,
    IndexKdataCommon,
    StockhkKdataCommon,
    StockusKdataCommon,
    BlockKdataCommon,
    Indexus,
    IndexusKdataCommon,
    Future,
    FutureKdataCommon,
    Currency,
    CurrencyKdataCommon,
)
from zvt.domain.meta.stockhk_meta import Stockhk
from zvt.domain.meta.stockus_meta import Stockus
from zvt.recorders.em import em_api

# 🧠 ML Signal: Class definition for a data recorder, useful for identifying patterns in class-based architecture
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Usage of utility functions like time_utils can indicate time-based operations or scheduling.
from zvt.utils.time_utils import count_interval, now_pd_timestamp, current_date

# 🧠 ML Signal: Default size attribute, could indicate typical data batch sizes


# 🧠 ML Signal: Entity provider attribute, useful for understanding data source patterns
class BaseEMStockKdataRecorder(FixedCycleDataRecorder):
    default_size = 50000
    entity_provider: str = "em"

    provider = "em"

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
        # ✅ Best Practice: Convert adjust_type to AdjustType to ensure type consistency
        one_day_trading_minutes=24 * 60,
        adjust_type=AdjustType.qfq,
        # ✅ Best Practice: Use of super() to ensure proper initialization of the base class
        # 🧠 ML Signal: Usage of entity schema name to determine entity type
        # 🧠 ML Signal: Dynamic schema retrieval based on entity type, level, and adjust type
        return_unfinished=False,
    ) -> None:
        level = IntervalLevel(level)
        self.adjust_type = AdjustType(adjust_type)
        self.entity_type = self.entity_schema.__name__.lower()

        self.data_schema = get_kdata_schema(
            entity_type=self.entity_type, level=level, adjust_type=self.adjust_type
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
            # 🧠 ML Signal: Function signature indicates a pattern of recording data with specific parameters
            start_timestamp,
            end_timestamp,
            level,
            kdata_use_begin_time,
            # ⚠️ SAST Risk (Medium): Potential risk if 'entity.id' is not validated or sanitized
            one_day_trading_minutes,
            return_unfinished,
            # ✅ Best Practice: Check for null data before proceeding with database operations
        )

    # 🧠 ML Signal: Usage of 'df_to_db' indicates a pattern of storing data in a database
    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Checking for missing attributes in an entity
        # ✅ Best Practice: Logging information when no data is found for traceability
        df = em_api.get_kdata(
            session=self.http_session,
            entity_id=entity.id,
            limit=size,
            adjust_type=self.adjust_type,
            level=self.level,
        )
        if pd_is_not_null(df):
            df_to_db(
                df=df,
                data_schema=self.data_schema,
                provider=self.provider,
                force_update=self.force_update,
            )
        else:
            self.logger.info(f"no kdata for {entity.id}")

    def on_finish_entity(self, entity):
        # 🧠 ML Signal: Handling non-empty query results
        # fill timestamp
        if not entity.timestamp or not entity.list_date:
            # ✅ Best Practice: Use f-string for logging messages
            # get the first
            kdatas = self.data_schema.query_data(
                provider=self.provider,
                # 🧠 ML Signal: Conditional assignment based on attribute presence
                entity_id=entity.id,
                order=self.data_schema.timestamp.asc(),
                # ✅ Best Practice: Class definition should follow PEP 8 naming conventions
                limit=1,
                # 🧠 ML Signal: Conditional assignment based on attribute presence
                return_type="domain",
                # ✅ Best Practice: Class attributes should be documented for clarity
            )
            # ⚠️ SAST Risk (Low): Potential risk of SQL injection if entity data is not sanitized
            if kdatas:
                # ✅ Best Practice: Class attributes should be documented for clarity
                # ✅ Best Practice: Call to superclass method ensures proper inheritance behavior.
                timestamp = kdatas[0].timestamp
                # ⚠️ SAST Risk (Low): Committing changes to the database without error handling

                # ✅ Best Practice: Check for holder_modified_date ensures logic only runs when necessary.
                self.logger.info(f"fill {entity.name} list_date as {timestamp}")

                # 🧠 ML Signal: Usage of external API to fetch data based on entity code.
                if not entity.timestamp:
                    entity.timestamp = timestamp
                # ✅ Best Practice: Use of get method to safely access dictionary keys.
                if not entity.list_date:
                    entity.list_date = timestamp
                self.entity_session.add(entity)
                self.entity_session.commit()


class EMStockKdataRecorder(BaseEMStockKdataRecorder):
    # 🧠 ML Signal: Updating entity attributes based on external data.
    entity_schema = Stock
    data_schema = StockKdataCommon
    # ⚠️ SAST Risk (Low): Directly adding and committing to session without exception handling.

    def on_finish_entity(self, entity):
        super().on_finish_entity(entity)
        # 🧠 ML Signal: Fetching additional statistics for the entity.
        # fill holder
        # 🧠 ML Signal: Inheritance from a base class indicates a common pattern for extending functionality.
        if not entity.holder_modified_date or (
            count_interval(entity.holder_modified_date, now_pd_timestamp()) > 30
        ):
            holder = em_api.get_controlling_shareholder(code=entity.code)
            # 🧠 ML Signal: Use of a string constant for provider name, useful for categorization.
            # ✅ Best Practice: Use of get method to safely access dictionary keys.
            if holder:
                entity.controlling_holder = holder.get("holder")
                # 🧠 ML Signal: Updating entity attributes based on external data.
                # 🧠 ML Signal: Assignment of schema class, indicating a pattern for data structure.
                # 🧠 ML Signal: Inheritance from a base class indicates a pattern of code reuse and specialization.
                if holder.get("parent"):
                    entity.controlling_holder_parent = holder.get("parent")
                # ⚠️ SAST Risk (Low): Directly adding and committing to session without exception handling.
                # 🧠 ML Signal: Assignment of data schema, indicating a pattern for data structure.
                # 🧠 ML Signal: Use of a string constant to define a provider, which could be used to categorize or filter data.
                else:
                    entity.controlling_holder_parent = holder.get("holder")
                # 🧠 ML Signal: Assignment of a schema class, indicating a pattern of structured data handling.
                # 🧠 ML Signal: Inheritance from a base class indicates a usage pattern for extending functionality.
                entity.holder_modified_date = current_date()
                # ✅ Best Practice: Class-level attributes provide a clear and consistent way to define static properties.
                self.entity_session.add(entity)
                # 🧠 ML Signal: Use of a common data schema, suggesting a standardized approach to data representation.
                self.entity_session.commit()
            # 🧠 ML Signal: Use of a string to define a provider can indicate a pattern for data source identification.
            holder_stats = em_api.get_top_ten_free_holder_stats(code=entity.code)
            # 🧠 ML Signal: Inheritance pattern indicating a specialized class
            if holder_stats:
                # 🧠 ML Signal: Assigning a schema to a class attribute suggests a pattern for data structure enforcement.
                # ✅ Best Practice: Class-level attributes for configuration and metadata
                entity.top_ten_ratio = holder_stats.get("ratio")
                entity.holder_modified_date = current_date()
                # 🧠 ML Signal: Use of a common data schema indicates a pattern for standardizing data handling.
                # 🧠 ML Signal: Static configuration of data source
                self.entity_session.add(entity)
                # 🧠 ML Signal: Inheritance from a base class indicates a usage pattern for extending functionality.
                self.entity_session.commit()


# 🧠 ML Signal: Static configuration of data schema
# ✅ Best Practice: Class attributes are defined at the top for clarity and easy access.


# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies
# 🧠 ML Signal: Use of a string to define a provider suggests a pattern for identifying data sources.
class EMStockusKdataRecorder(BaseEMStockKdataRecorder):
    entity_provider = "em"
    # 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
    # 🧠 ML Signal: Use of schema attributes indicates a pattern for data structure definition.
    entity_schema = Stockus
    data_schema = StockusKdataCommon


# 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
# 🧠 ML Signal: Use of schema attributes indicates a pattern for data structure definition.


# 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
class EMStockhkKdataRecorder(BaseEMStockKdataRecorder):
    entity_provider = "em"
    entity_schema = Stockhk
    # ✅ Best Practice: Use of __name__ == "__main__" to ensure code only runs when script is executed directly
    data_schema = StockhkKdataCommon


# 🧠 ML Signal: Querying data from a specific provider and filtering by exchange


class EMIndexKdataRecorder(BaseEMStockKdataRecorder):
    entity_provider = "em"
    # 🧠 ML Signal: Converting a DataFrame column to a list
    entity_schema = Index
    # 🧠 ML Signal: Instantiating a recorder with specific parameters
    # ✅ Best Practice: Use of __all__ to define public API of the module
    # 🧠 ML Signal: Running a recorder instance

    data_schema = IndexKdataCommon


class EMIndexusKdataRecorder(BaseEMStockKdataRecorder):
    entity_provider = "em"
    entity_schema = Indexus

    data_schema = IndexusKdataCommon


class EMBlockKdataRecorder(BaseEMStockKdataRecorder):
    entity_provider = "em"
    entity_schema = Block

    data_schema = BlockKdataCommon


class EMFutureKdataRecorder(BaseEMStockKdataRecorder):
    entity_provider = "em"
    entity_schema = Future

    data_schema = FutureKdataCommon


class EMCurrencyKdataRecorder(BaseEMStockKdataRecorder):
    entity_provider = "em"
    entity_schema = Currency

    data_schema = CurrencyKdataCommon


if __name__ == "__main__":
    df = Stock.query_data(filters=[Stock.exchange == "bj"], provider="em")
    entity_ids = df["entity_id"].tolist()
    recorder = EMStockKdataRecorder(
        level=IntervalLevel.LEVEL_1DAY,
        entity_ids=entity_ids,
        sleeping_time=0,
        adjust_type=AdjustType.hfq,
    )
    recorder.run()


# the __all__ is generated
__all__ = [
    "BaseEMStockKdataRecorder",
    "EMStockKdataRecorder",
    "EMStockusKdataRecorder",
    "EMStockhkKdataRecorder",
    "EMIndexKdataRecorder",
    "EMIndexusKdataRecorder",
    "EMBlockKdataRecorder",
    "EMFutureKdataRecorder",
    "EMCurrencyKdataRecorder",
]
