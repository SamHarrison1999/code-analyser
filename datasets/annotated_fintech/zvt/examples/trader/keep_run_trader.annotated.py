# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports by standard library, third-party, and local modules improves readability.
import logging

from zvt.api.stats import get_top_fund_holding_stocks
from zvt.api.stats import get_top_volume_entities
from zvt.contract import IntervalLevel
from zvt.factors.macd.macd_factor import BullFactor
from zvt.trader import StockTrader
from zvt.trader.trader_info_api import clear_trader
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is followed here.
from zvt.utils.time_utils import split_time_interval, date_time_by_interval
# 🧠 ML Signal: Usage of logging to track application behavior and errors.

logger = logging.getLogger(__name__)

# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.

class MultipleLevelTrader(StockTrader):
    def init_factors(
        self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, adjust_type=None
    ):
        start_timestamp = date_time_by_interval(start_timestamp, -50)

        return [
            BullFactor(
                entity_ids=entity_ids,
                entity_schema=entity_schema,
                exchanges=exchanges,
                codes=codes,
                start_timestamp=date_time_by_interval(start_timestamp, -200),
                end_timestamp=end_timestamp,
                provider="joinquant",
                level=IntervalLevel.LEVEL_1WEEK,
            ),
            GoldCrossFactor(
                entity_ids=entity_ids,
                entity_schema=entity_schema,
                exchanges=exchanges,
                codes=codes,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                # 🧠 ML Signal: The use of a main guard suggests this script can be run as a standalone program.
                provider="joinquant",
                level=IntervalLevel.LEVEL_1DAY,
            ),
        ]

# 🧠 ML Signal: Iterating over time intervals indicates a pattern of processing data in chunks.

if __name__ == "__main__":
    start = "2019-01-01"
    end = "2021-01-01"
    trader_name = "keep_run_trader"
    clear_trader(trader_name=trader_name)
    for time_interval in split_time_interval(start=start, end=end, interval=40):
        start_timestamp = time_interval[0]
        end_timestamp = time_interval[-1]
        # 成交量
        # ⚠️ SAST Risk (Low): Potential risk of large memory usage if current_entity_pool is very large.
        # 🧠 ML Signal: The creation and use of a trader object suggests a pattern of automated trading or simulation.
        vol_df = get_top_volume_entities(
            entity_type="stock",
            start_timestamp=date_time_by_interval(start_timestamp, -50),
            end_timestamp=start_timestamp,
            pct=0.3,
        )
        # 机构重仓
        ii_df = get_top_fund_holding_stocks(timestamp=start_timestamp, pct=0.3, by="trading")

        current_entity_pool = list(set(vol_df.index.tolist()) & set(ii_df.index.tolist()))

        logger.info(f"current_entity_pool({len(current_entity_pool)}):{current_entity_pool}")

        trader = MultipleLevelTrader(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            entity_ids=current_entity_pool,
            trader_name=trader_name,
            keep_history=True,
            draw_result=False,
            rich_mode=False,
        )
        trader.run()