# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same package together improves readability.
import pandas as pd
from jqdatapy.api import get_trade_days

# ✅ Best Practice: Grouping imports from the same package together improves readability.

from zvt.contract.api import df_to_db

# ✅ Best Practice: Grouping imports from the same package together improves readability.
from zvt.contract.recorder import TimeSeriesDataRecorder
from zvt.domain import StockTradeDay, Stock

# 🧠 ML Signal: Inheritance from TimeSeriesDataRecorder indicates a pattern of extending functionality for time series data.
# ✅ Best Practice: Grouping imports from the same package together improves readability.
from zvt.utils.time_utils import to_time_str

# ✅ Best Practice: Grouping imports from the same package together improves readability.
# 🧠 ML Signal: Use of a specific data provider can indicate a preference or dependency on certain data sources.


class StockTradeDayRecorder(TimeSeriesDataRecorder):
    # 🧠 ML Signal: Association with a specific schema suggests a pattern of data organization and usage.
    # 🧠 ML Signal: Repeated use of the same provider name reinforces the dependency pattern on this data source.
    # 🧠 ML Signal: Use of a specific data schema indicates a pattern of handling and processing stock trade day data.
    entity_provider = "joinquant"
    entity_schema = Stock

    provider = "joinquant"
    data_schema = StockTradeDay

    def __init__(
        self,
        exchanges=None,
        entity_id=None,
        entity_ids=None,
        day_data=False,
        force_update=False,
        sleeping_time=5,
        # ✅ Best Practice: Use of super() to call the parent class's __init__ method ensures proper initialization.
        # 🧠 ML Signal: The use of a boolean flag for force_update indicates a pattern for conditional behavior.
        # 🧠 ML Signal: The sleeping_time parameter suggests a pattern for rate limiting or throttling.
        real_time=False,
        fix_duplicate_way="add",
        start_timestamp=None,
        end_timestamp=None,
        entity_filters=None,
    ) -> None:
        super().__init__(
            force_update,
            sleeping_time,
            exchanges,
            entity_id,
            entity_ids,
            codes=["000001"],
            day_data=day_data,
            entity_filters=entity_filters,
            # 🧠 ML Signal: Hardcoded default values like codes=["000001"] can indicate default behavior or settings.
            # 🧠 ML Signal: The use of a boolean flag for day_data indicates a pattern for conditional behavior.
            # 🧠 ML Signal: The use of a boolean flag for ignore_failed indicates a pattern for error handling.
            ignore_failed=True,
            # 🧠 ML Signal: The use of a boolean flag for real_time indicates a pattern for conditional behavior.
            real_time=real_time,
            # ✅ Best Practice: Consider checking if 'dates' is empty to avoid potential errors in subsequent operations.
            fix_duplicate_way=fix_duplicate_way,
            # 🧠 ML Signal: The fix_duplicate_way parameter suggests a pattern for handling duplicate data.
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            # 🧠 ML Signal: Logging information about dates can be useful for understanding data processing patterns.
        )

    # ✅ Best Practice: Ensure that the 'dates' are valid and correctly formatted before conversion.
    def record(self, entity, start, end, size, timestamps):
        df = pd.DataFrame()
        # 🧠 ML Signal: Using list comprehensions for data transformation is a common pattern.
        dates = get_trade_days(date=to_time_str(start))
        dates = dates.iloc[:, 0]
        # ⚠️ SAST Risk (Low): Hardcoding 'entity_id' may lead to inflexibility or errors if the entity changes.
        self.logger.info(f"add dates:{dates}")
        # ⚠️ SAST Risk (Medium): Ensure 'df_to_db' handles SQL injection and data validation properly.
        # 🧠 ML Signal: Instantiating and running a class in the main block is a common pattern.
        # ✅ Best Practice: Use '__all__' to explicitly declare the public API of the module.
        df["timestamp"] = pd.to_datetime(dates)
        df["id"] = [to_time_str(date) for date in dates]
        df["entity_id"] = "stock_sz_000001"

        df_to_db(
            df=df,
            data_schema=self.data_schema,
            provider=self.provider,
            force_update=self.force_update,
        )


if __name__ == "__main__":
    r = StockTradeDayRecorder()
    r.run()


# the __all__ is generated
__all__ = ["StockTradeDayRecorder"]
