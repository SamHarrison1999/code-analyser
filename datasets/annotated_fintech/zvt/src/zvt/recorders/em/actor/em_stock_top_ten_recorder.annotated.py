# -*- coding: utf-8 -*-
from typing import List
# ✅ Best Practice: Grouping imports into standard library, third-party, and local sections improves readability.

import pandas as pd

from zvt.api.utils import to_report_period_type, value_to_pct
from zvt.contract import ActorType
from zvt.contract.api import df_to_db
from zvt.contract.recorder import TimestampsDataRecorder
from zvt.domain import Stock, ActorMeta
from zvt.domain.actor.stock_actor import StockTopTenHolder, StockInstitutionalInvestorHolder
# 🧠 ML Signal: Inheritance from TimestampsDataRecorder indicates a pattern of extending functionality
from zvt.recorders.em.em_api import get_holder_report_dates, get_holders
from zvt.utils.time_utils import to_pd_timestamp, to_time_str
# 🧠 ML Signal: Use of class attributes for configuration


# 🧠 ML Signal: Use of class attributes for configuration
class EMStockTopTenRecorder(TimestampsDataRecorder):
    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
    entity_provider = "em"
    # 🧠 ML Signal: Use of class attributes for configuration
    entity_schema = Stock
    # 🧠 ML Signal: Usage of external function get_holder_report_dates with entity_item.code

    # 🧠 ML Signal: Use of class attributes for configuration
    provider = "em"
    # 🧠 ML Signal: List comprehension pattern for transforming data
    data_schema = StockTopTenHolder
    # 🧠 ML Signal: Usage of to_pd_timestamp function to convert dates
    # ⚠️ SAST Risk (Low): Potential risk if to_pd_timestamp does not handle invalid date formats

    def init_timestamps(self, entity_item) -> List[pd.Timestamp]:
        result = get_holder_report_dates(code=entity_item.code)
        if result:
            return [to_pd_timestamp(item["END_DATE"]) for item in result]

    # ⚠️ SAST Risk (Low): Potential SQL injection if filters are not properly sanitized
    def on_finish_entity(self, entity):
        super().on_finish_entity(entity)
        holders = StockTopTenHolder.query_data(
            entity_id=entity.id,
            filters=[StockTopTenHolder.holding_values == None],
            session=self.session,
            return_type="domain",
        )
        for holder in holders:
            ii = StockInstitutionalInvestorHolder.query_data(
                entity_id=entity.id,
                filters=[
                    # ⚠️ SAST Risk (Low): Potential SQL injection if filters are not properly sanitized
                    StockInstitutionalInvestorHolder.holding_values > 1,
                    StockInstitutionalInvestorHolder.holding_ratio > 0.01,
                    StockInstitutionalInvestorHolder.timestamp == holder.timestamp,
                # 🧠 ML Signal: Iterating over timestamps to process data for each timestamp
                # ⚠️ SAST Risk (Low): Division by zero risk if ii[0].holding_ratio is zero
                ],
                limit=1,
                # 🧠 ML Signal: Converting timestamp to string format
                # ✅ Best Practice: Ensure session is committed to save changes to the database
                return_type="domain",
            )
            # 🧠 ML Signal: Fetching data based on entity code and date
            if ii:
                holder.holding_values = holder.holding_ratio * ii[0].holding_values / ii[0].holding_ratio
        self.session.commit()

    # 🧠 ML Signal: Iterating over result items to process each holder
    def record(self, entity, start, end, size, timestamps):
        for timestamp in timestamps:
            the_date = to_time_str(timestamp)
            result = get_holders(code=entity.code, end_date=the_date)
            # 🧠 ML Signal: Querying data to check for existing domains
            if result:
                # 🧠 ML Signal: Creating new ActorMeta object for corporation
                holders = []
                new_actors = []
                for item in result:
                    # 机构
                    if item["IS_HOLDORG"] == "1":
                        domains: List[ActorMeta] = ActorMeta.query_data(
                            filters=[ActorMeta.code == item["HOLDER_CODE"]], return_type="domain"
                        )
                        if not domains:
                            actor_type = ActorType.corporation.value
                            actor = ActorMeta(
                                entity_id=f'{actor_type}_cn_{item["HOLDER_CODE"]}',
                                # 🧠 ML Signal: Using existing domain data
                                id=f'{actor_type}_cn_{item["HOLDER_CODE"]}',
                                entity_type=actor_type,
                                exchange="cn",
                                code=item["HOLDER_CODE"],
                                name=item["HOLDER_NAME"],
                            )
                        else:
                            actor = domains[0]
                    # 🧠 ML Signal: Creating new ActorMeta object for individual
                    else:
                        # 🧠 ML Signal: Collecting new actor data
                        actor_type = ActorType.individual.value
                        actor = ActorMeta(
                            entity_id=f'{actor_type}_cn_{item["HOLDER_NAME"]}',
                            id=f'{actor_type}_cn_{item["HOLDER_NAME"]}',
                            entity_type=actor_type,
                            exchange="cn",
                            code=item["HOLDER_NAME"],
                            name=item["HOLDER_NAME"],
                        )
                        new_actors.append(actor.__dict__)
                    holder = {
                        "id": f"{entity.entity_id}_{the_date}_{actor.entity_id}",
                        "entity_id": entity.entity_id,
                        "timestamp": timestamp,
                        "code": entity.code,
                        # 🧠 ML Signal: Constructing holder dictionary with detailed information
                        "name": entity.name,
                        "actor_id": actor.entity_id,
                        "actor_type": actor.entity_type,
                        "actor_code": actor.code,
                        "actor_name": actor.name,
                        "report_date": timestamp,
                        "report_period": to_report_period_type(timestamp),
                        "holding_numbers": item["HOLD_NUM"],
                        "holding_ratio": value_to_pct(item["HOLD_NUM_RATIO"], default=0),
                    # 🧠 ML Signal: Appending holder data to list
                    # 🧠 ML Signal: Converting holders list to DataFrame
                    # ⚠️ SAST Risk (Low): Potential SQL injection risk if df_to_db is not properly handling inputs
                    # 🧠 ML Signal: Running the recorder with specific codes
                    # ✅ Best Practice: Use of __name__ == "__main__" to allow or prevent parts of code from being run when the modules are imported
                    # ✅ Best Practice: Defining __all__ to specify public API of the module
                    }
                    holders.append(holder)
                if holders:
                    df = pd.DataFrame.from_records(holders)
                    df_to_db(data_schema=self.data_schema, df=df, provider=self.provider, force_update=True)
                if new_actors:
                    df = pd.DataFrame.from_records(new_actors)
                    df_to_db(data_schema=ActorMeta, df=df, provider=self.provider, force_update=False)


if __name__ == "__main__":
    EMStockTopTenRecorder(codes=["000002"]).run()


# the __all__ is generated
__all__ = ["EMStockTopTenRecorder"]