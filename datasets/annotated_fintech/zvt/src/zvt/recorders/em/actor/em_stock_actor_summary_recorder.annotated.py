# -*- coding: utf-8 -*-
from typing import List

# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.

import pandas as pd

from zvt.api.utils import to_report_period_type, value_to_pct
from zvt.contract import ActorType
from zvt.contract.api import df_to_db
from zvt.contract.recorder import TimestampsDataRecorder
from zvt.domain import Stock
from zvt.domain.actor.stock_actor import StockActorSummary

# 🧠 ML Signal: Inheritance from TimestampsDataRecorder indicates a pattern of extending functionality
from zvt.recorders.em.em_api import (
    get_ii_holder_report_dates,
    actor_type_to_org_type,
    get_ii_summary,
)
from zvt.utils.time_utils import to_pd_timestamp, to_time_str

# 🧠 ML Signal: Use of string literals for configuration or identification


# ✅ Best Practice: Consider adding a type hint for the return type of the function
# 🧠 ML Signal: Use of class attributes for schema definition
# [{'CHANGE_RATIO': -1.045966694333,
#   'IS_COMPLETE': '1',
# 🧠 ML Signal: Usage of external function get_ii_holder_report_dates
# 🧠 ML Signal: Consistent use of provider attribute for data source identification
#   'ORG_TYPE': '07',
#   'REPORT_DATE': '2021-03-31 00:00:00',
# 🧠 ML Signal: Use of class attributes for schema definition
#   'SECUCODE': '000338.SZ',
# 🧠 ML Signal: Iterating over timestamps to process data
# 🧠 ML Signal: List comprehension pattern
#   'SECURITY_CODE': '000338',
# 🧠 ML Signal: Usage of external function to_pd_timestamp
#   'TOTAL_FREE_SHARES': 2598718411,
# 🧠 ML Signal: Converting timestamp to string format
#   'TOTAL_MARKET_CAP': 49999342227.64,
#   'TOTAL_ORG_NUM': 5,
# 🧠 ML Signal: Logging information with entity code and date
#   'TOTAL_SHARES_RATIO': 29.51742666}]

# 🧠 ML Signal: Iterating over actor types for processing


class EMStockActorSummaryRecorder(TimestampsDataRecorder):
    entity_provider = "em"
    # ✅ Best Practice: Skipping specific actor types early in the loop
    entity_schema = Stock
    # 🧠 ML Signal: Fetching summary data based on entity code, date, and actor type
    # ✅ Best Practice: Checking if result is not empty before processing
    # 🧠 ML Signal: Creating a list of summaries from result data

    provider = "em"
    data_schema = StockActorSummary

    def init_timestamps(self, entity_item) -> List[pd.Timestamp]:
        result = get_ii_holder_report_dates(code=entity_item.code)
        if result:
            return [to_pd_timestamp(item["REPORT_DATE"]) for item in result]

    def record(self, entity, start, end, size, timestamps):
        for timestamp in timestamps:
            the_date = to_time_str(timestamp)
            self.logger.info(f"to {entity.code} {the_date}")
            for actor_type in ActorType:
                if (
                    actor_type == ActorType.private_equity
                    or actor_type == ActorType.individual
                ):
                    continue
                result = get_ii_summary(
                    code=entity.code,
                    report_date=the_date,
                    org_type=actor_type_to_org_type(actor_type),
                )
                # 🧠 ML Signal: Constructing unique ID for each summary
                # 🧠 ML Signal: Extracting and transforming data from result
                if result:
                    summary_list = [
                        {
                            "id": f"{entity.entity_id}_{the_date}_{actor_type.value}",
                            "entity_id": entity.entity_id,
                            "timestamp": timestamp,
                            "code": entity.code,
                            "name": entity.name,
                            "actor_type": actor_type.value,
                            "actor_count": item["TOTAL_ORG_NUM"],
                            # 🧠 ML Signal: Creating DataFrame from summary list
                            # 🧠 ML Signal: Running the recorder with specific stock codes
                            # 🧠 ML Signal: Storing DataFrame to database with specific parameters
                            # ⚠️ SAST Risk (Low): Code execution entry point
                            # 🧠 ML Signal: Defining module exports
                            "report_date": timestamp,
                            "report_period": to_report_period_type(timestamp),
                            "change_ratio": value_to_pct(
                                item["CHANGE_RATIO"], default=1
                            ),
                            "is_complete": item["IS_COMPLETE"],
                            "holding_numbers": item["TOTAL_FREE_SHARES"],
                            "holding_ratio": value_to_pct(
                                item["TOTAL_SHARES_RATIO"], default=0
                            ),
                            "holding_values": item["TOTAL_MARKET_CAP"],
                        }
                        for item in result
                    ]
                    df = pd.DataFrame.from_records(summary_list)
                    df_to_db(
                        data_schema=self.data_schema,
                        df=df,
                        provider=self.provider,
                        force_update=True,
                        drop_duplicates=True,
                    )


if __name__ == "__main__":
    EMStockActorSummaryRecorder(codes=["000338"]).run()


# the __all__ is generated
__all__ = ["EMStockActorSummaryRecorder"]
