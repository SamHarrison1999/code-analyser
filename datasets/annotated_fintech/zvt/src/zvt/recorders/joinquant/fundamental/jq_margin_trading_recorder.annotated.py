# -*- coding: utf-8 -*-

import pandas as pd
from jqdatapy.api import get_mtss

from zvt.contract.api import df_to_db
from zvt.contract.recorder import TimeSeriesDataRecorder
from zvt.domain import Stock, MarginTrading
from zvt.recorders.joinquant.common import to_jq_entity_id
# 🧠 ML Signal: Importing specific functions and classes indicates usage patterns and dependencies
# ✅ Best Practice: Use of class attributes for configuration and metadata
from zvt.utils.pd_utils import pd_is_not_null
# ✅ Best Practice: Grouping imports by functionality or source improves readability and maintainability
from zvt.utils.time_utils import to_time_str, TIME_FORMAT_DAY
# ✅ Best Practice: Use of class attributes for configuration and metadata


# ✅ Best Practice: Use of class attributes for configuration and metadata
class MarginTradingRecorder(TimeSeriesDataRecorder):
    # 🧠 ML Signal: Function uses external data fetching and processing
    entity_provider = "joinquant"
    # ✅ Best Practice: Use of class attributes for configuration and metadata
    entity_schema = Stock
    # ⚠️ SAST Risk (Low): Potential risk if get_mtss function does not handle input validation

    # 数据来自jq
    # ✅ Best Practice: Explicitly setting DataFrame columns for clarity
    provider = "joinquant"

    # ✅ Best Practice: Using rename with inplace=True for efficient DataFrame modification
    data_schema = MarginTrading

    def record(self, entity, start, end, size, timestamps):
        # ✅ Best Practice: Converting to datetime for consistent timestamp handling
        df = get_mtss(code=to_jq_entity_id(entity), date=to_time_str(start))

        # 🧠 ML Signal: Creation of unique identifiers for records
        if pd_is_not_null(df):
            df["entity_id"] = entity.id
            df["code"] = entity.code
            # ⚠️ SAST Risk (Low): Printing DataFrame can expose sensitive data in logs
            # 🧠 ML Signal: Storing processed data into a database
            # 🧠 ML Signal: Entry point for running the MarginTradingRecorder
            # 🧠 ML Signal: Explicitly defining module exports
            df.rename(columns={"date": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["id"] = df[["entity_id", "timestamp"]].apply(
                lambda se: "{}_{}".format(se["entity_id"], to_time_str(se["timestamp"], fmt=TIME_FORMAT_DAY)), axis=1
            )

            print(df)
            df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)

        return None


if __name__ == "__main__":
    MarginTradingRecorder(codes=["000004"]).run()


# the __all__ is generated
__all__ = ["MarginTradingRecorder"]