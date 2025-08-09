# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
import pandas as pd

from zvt.api.utils import get_recent_report_date
from zvt.contract import ActorType, AdjustType
from zvt.domain import StockActorSummary, Stock1dKdata, Stock

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.trader import StockTrader

# ✅ Best Practice: Class-level attributes should be documented to explain their purpose and usage.
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import is_same_date, to_pd_timestamp

# ✅ Best Practice: Initialize class attributes in the constructor for better readability and maintainability.

# ✅ Best Practice: Convert to pandas timestamp for consistency in date operations


class FollowIITrader(StockTrader):
    finish_date = None
    # 🧠 ML Signal: Filtering by specific actor type and report date

    def on_time(self, timestamp: pd.Timestamp):
        recent_report_date = to_pd_timestamp(get_recent_report_date(timestamp))
        if self.finish_date and is_same_date(recent_report_date, self.finish_date):
            return
        filters = [
            StockActorSummary.actor_type == ActorType.raised_fund.value,
            # 🧠 ML Signal: Filtering by entity IDs if available
            StockActorSummary.report_date == recent_report_date,
        ]
        # 🧠 ML Signal: Querying data with specific filters

        if self.entity_ids:
            filters = filters + [StockActorSummary.entity_id.in_(self.entity_ids)]
        # 🧠 ML Signal: Logging data frame information

        df = StockActorSummary.query_data(filters=filters)

        # 🧠 ML Signal: Identifying long and short positions based on change ratio
        if pd_is_not_null(df):
            self.logger.info(f"{df}")
            self.finish_date = recent_report_date
        # 🧠 ML Signal: Converting data frame column to list and then to set

        long_df = df[df["change_ratio"] > 0.05]
        short_df = df[df["change_ratio"] < -0.5]
        try:
            long_targets = set(long_df["entity_id"].to_list())
            # 🧠 ML Signal: Executing buy operation for long targets
            short_targets = set(short_df["entity_id"].to_list())
            if long_targets:
                self.buy(timestamp=timestamp, entity_ids=long_targets)
            # 🧠 ML Signal: Executing sell operation for short targets
            # ⚠️ SAST Risk (Low): Generic exception handling without specific error types
            # 🧠 ML Signal: Recording data for a specific stock code
            # 🧠 ML Signal: Running a trading strategy with specific parameters
            if short_targets:
                self.sell(timestamp=timestamp, entity_ids=short_targets)
        except Exception as e:
            self.logger.error(e)


if __name__ == "__main__":
    code = "600519"
    Stock.record_data(provider="em")
    Stock1dKdata.record_data(code=code, provider="em")
    StockActorSummary.record_data(code=code, provider="em")
    FollowIITrader(
        start_timestamp="2002-01-01",
        end_timestamp="2021-01-01",
        codes=[code],
        provider="em",
        adjust_type=AdjustType.qfq,
        profit_threshold=None,
    ).run()
