# -*- coding: utf-8 -*-
from typing import List

# ✅ Best Practice: Grouping imports into standard library, third-party, and local can improve readability.

import pandas as pd

from zvt.api.utils import to_report_period_type, value_to_pct
from zvt.contract import ActorType
from zvt.contract.api import df_to_db
from zvt.contract.recorder import TimestampsDataRecorder
from zvt.domain import Stock, ActorMeta

# 🧠 ML Signal: Inheritance from TimestampsDataRecorder indicates a pattern of extending functionality
from zvt.domain.actor.stock_actor import StockInstitutionalInvestorHolder

# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
from zvt.recorders.em.em_api import (
    get_ii_holder_report_dates,
    get_ii_holder,
    actor_type_to_org_type,
)
from zvt.utils.time_utils import to_pd_timestamp, to_time_str

# 🧠 ML Signal: Class attributes define configuration or metadata, useful for pattern recognition


# 🧠 ML Signal: Association with a specific schema (Stock) indicates a pattern of data handling
# ✅ Best Practice: Specify the return type for better readability and maintainability
# {'END_DATE': '2021-03-31 00:00:00',
#   'HOLDER_CODE': '10015776',
# 🧠 ML Signal: Usage of external function get_ii_holder_report_dates
# 🧠 ML Signal: Class attributes define configuration or metadata, useful for pattern recognition
#   'HOLDER_CODE_OLD': '80010104',
#   'HOLDER_NAME': '香港中央结算代理人有限公司',
# 🧠 ML Signal: Association with a specific schema (StockInstitutionalInvestorHolder) indicates a pattern of data handling
#   'HOLDER_RANK': 1,
# 🧠 ML Signal: List comprehension pattern
#   'HOLD_NUM': 1938664086,
# 🧠 ML Signal: Usage of external function to_pd_timestamp
#   'HOLD_NUM_RATIO': 24.44,
#   'HOLD_RATIO_QOQ': '0.04093328',
# 🧠 ML Signal: Iterating over a fixed set of types (ActorType) to filter or process data
#   'IS_HOLDORG': '1',
#   'SECUCODE': '000338.SZ'}

# ⚠️ SAST Risk (Low): Potential exposure of sensitive data if entity.code or the_date contains sensitive information
#  {'END_DATE': '2021-03-31 00:00:00',
#   'FREE_HOLDNUM_RATIO': 0.631949916991,
#   'FREE_RATIO_QOQ': '-5.33046217',
#   'HOLDER_CODE': '161606',
# 🧠 ML Signal: Creating a structured data record from API results
#   'HOLDER_CODE_OLD': '161606',
#   'HOLDER_NAME': '交通银行-融通行业景气证券投资基金',
#   'HOLDER_RANK': 10,
#   'HOLD_NUM': 39100990,
#   'IS_HOLDORG': '1',
#   'SECUCODE': '000338.SZ'}


class EMStockIIRecorder(TimestampsDataRecorder):
    entity_provider = "em"
    entity_schema = Stock

    provider = "em"
    data_schema = StockInstitutionalInvestorHolder

    def init_timestamps(self, entity_item) -> List[pd.Timestamp]:
        result = get_ii_holder_report_dates(code=entity_item.code)
        if result:
            return [to_pd_timestamp(item["REPORT_DATE"]) for item in result]

    # ✅ Best Practice: Using pandas DataFrame for structured data manipulation
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
                # ⚠️ SAST Risk (Low): Ensure df_to_db handles SQL injection and data validation
                # 🧠 ML Signal: Creating a structured data record for actors
                result = get_ii_holder(
                    code=entity.code,
                    report_date=the_date,
                    org_type=actor_type_to_org_type(actor_type),
                )
                if result:
                    holders = [
                        {
                            "id": f'{entity.entity_id}_{the_date}_{actor_type.value}_cn_{item["HOLDER_CODE"]}',
                            "entity_id": entity.entity_id,
                            "timestamp": timestamp,
                            "code": entity.code,
                            "name": entity.name,
                            "actor_id": f'{actor_type.value}_cn_{item["HOLDER_CODE"]}',
                            "actor_type": actor_type.value,
                            "actor_code": item["HOLDER_CODE"],
                            "actor_name": f'{item["HOLDER_NAME"]}',
                            "report_date": timestamp,
                            "report_period": to_report_period_type(timestamp),
                            # ✅ Best Practice: Using pandas DataFrame for structured data manipulation
                            # ⚠️ SAST Risk (Low): Ensure df_to_db handles SQL injection and data validation
                            # 🧠 ML Signal: Entry point for running the script with specific parameters
                            # 🧠 ML Signal: Defining module exports for external use
                            "holding_numbers": item["TOTAL_SHARES"],
                            "holding_ratio": value_to_pct(item["FREESHARES_RATIO"], 0),
                            "holding_values": item["HOLD_VALUE"],
                        }
                        for item in result
                    ]
                    df = pd.DataFrame.from_records(holders)
                    df_to_db(
                        data_schema=self.data_schema,
                        df=df,
                        provider=self.provider,
                        force_update=True,
                        drop_duplicates=True,
                    )

                    # save the actors
                    actors = [
                        {
                            "id": f'{actor_type.value}_cn_{item["HOLDER_CODE"]}',
                            "entity_id": f'{actor_type.value}_cn_{item["HOLDER_CODE"]}',
                            "entity_type": actor_type.value,
                            "exchange": "cn",
                            "code": item["HOLDER_CODE"],
                            "name": f'{item["HOLDER_NAME"]}',
                        }
                        for item in result
                    ]
                    df1 = pd.DataFrame.from_records(actors)
                    df_to_db(
                        data_schema=ActorMeta,
                        df=df1,
                        provider=self.provider,
                        force_update=False,
                        drop_duplicates=True,
                    )


if __name__ == "__main__":
    EMStockIIRecorder(codes=["000562"]).run()


# the __all__ is generated
__all__ = ["EMStockIIRecorder"]
