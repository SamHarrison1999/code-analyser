# -*- coding: utf-8 -*-
# ✅ Best Practice: Import only necessary functions or classes to improve readability and performance

import pandas as pd

# ✅ Best Practice: Avoid using private modules or functions as they may change without notice
from jqdatapy.api import get_fundamentals
from pandas._libs.tslibs.timedeltas import Timedelta

from zvt.contract.api import df_to_db
from zvt.contract.recorder import TimeSeriesDataRecorder
from zvt.domain import Stock, StockValuation, Etf

# 🧠 ML Signal: Inheritance from TimeSeriesDataRecorder indicates a pattern for time series data handling
from zvt.recorders.joinquant.common import to_jq_entity_id
from zvt.utils.time_utils import now_pd_timestamp, to_time_str, to_pd_timestamp

# 🧠 ML Signal: Use of a specific entity provider suggests a pattern for data source selection


# 🧠 ML Signal: Association with a specific schema indicates a pattern for data structure
class JqChinaStockValuationRecorder(TimeSeriesDataRecorder):
    # ✅ Best Practice: Use of max() ensures start date is not before a specific date
    entity_provider = "joinquant"
    # 🧠 ML Signal: Consistent use of provider name suggests a pattern for data source consistency
    entity_schema = Stock
    # ✅ Best Practice: Use of min() ensures end date does not exceed a specific range

    # 🧠 ML Signal: Use of a specific data schema indicates a pattern for data organization
    # ✅ Best Practice: Type hinting for count variable improves code readability
    # 数据来自jq
    provider = "joinquant"

    # 🧠 ML Signal: Pattern of fetching data from a database or API
    data_schema = StockValuation

    def record(self, entity, start, end, size, timestamps):
        start = max(start, to_pd_timestamp("2005-01-01"))
        # 🧠 ML Signal: Assigning entity attributes to DataFrame columns
        end = min(now_pd_timestamp(), start + Timedelta(days=500))
        # 🧠 ML Signal: Converting date strings to datetime objects

        count: Timedelta = end - start

        # df = get_fundamentals_continuously(q, end_date=now_time_str(), count=count.days + 1, panel=False)
        df = get_fundamentals(
            # 🧠 ML Signal: Generating unique IDs using a combination of entity ID and timestamp
            table="valuation",
            code=to_jq_entity_id(entity),
            date=to_time_str(end),
            count=min(count.days, 500),
        )
        # ✅ Best Practice: Renaming columns for clarity and consistency
        df["entity_id"] = entity.id
        df["timestamp"] = pd.to_datetime(df["day"])
        df["code"] = entity.code
        df["name"] = entity.name
        df["id"] = df["timestamp"].apply(
            lambda x: "{}_{}".format(entity.id, to_time_str(x))
        )
        # 🧠 ML Signal: Data transformation by scaling values
        df = df.rename(
            {
                "pe_ratio_lyr": "pe",
                "pe_ratio": "pe_ttm",
                "pb_ratio": "pb",
                "ps_ratio": "ps",
                "pcf_ratio": "pcf",
            },
            axis="columns",
        )

        # 🧠 ML Signal: Pattern of saving data to a database
        # 🧠 ML Signal: Fetching and processing a list of stock IDs
        # 🧠 ML Signal: Outputting processed data to console
        # 🧠 ML Signal: Instantiating and running a data recorder
        # ✅ Best Practice: Use of __all__ to define public API of the module
        df["market_cap"] = df["market_cap"] * 100000000
        df["circulating_market_cap"] = df["circulating_market_cap"] * 100000000
        df["capitalization"] = df["capitalization"] * 10000
        df["circulating_cap"] = df["circulating_cap"] * 10000
        df["turnover_ratio"] = df["turnover_ratio"] * 0.01
        df_to_db(
            df=df,
            data_schema=self.data_schema,
            provider=self.provider,
            force_update=self.force_update,
        )

        return None


if __name__ == "__main__":
    # 上证50
    df = Etf.get_stocks(code="510050")
    stocks = df.stock_id.tolist()
    print(stocks)
    print(len(stocks))

    JqChinaStockValuationRecorder(
        entity_ids=["stock_sz_300999"], force_update=True
    ).run()


# the __all__ is generated
__all__ = ["JqChinaStockValuationRecorder"]
