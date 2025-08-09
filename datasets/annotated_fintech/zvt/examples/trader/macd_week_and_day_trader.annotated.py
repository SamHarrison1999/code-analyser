# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from typing import List, Tuple

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract import IntervalLevel
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which this does.
from zvt.factors.macd.macd_factor import GoldCrossFactor
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.trader.trader import StockTrader


# ✅ Best Practice: Returning a list directly without storing it in a variable improves readability.
# 🧠 ML Signal: Instantiation of GoldCrossFactor with specific parameters could indicate a pattern for ML models.
# 依赖数据
# dataschema: Stock1dHfqKdata Stock1wkHfqKdata
# provider: joinquant
class MultipleLevelTrader(StockTrader):
    def init_factors(
        self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, adjust_type=None
    ):
        # 同时使用周线和日线策略
        return [
            GoldCrossFactor(
                entity_ids=entity_ids,
                entity_schema=entity_schema,
                exchanges=exchanges,
                codes=codes,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                provider="joinquant",
                level=IntervalLevel.LEVEL_1WEEK,
            ),
            GoldCrossFactor(
                entity_ids=entity_ids,
                entity_schema=entity_schema,
                # 🧠 ML Signal: Repeated instantiation with different parameters could be used to identify variations in usage.
                exchanges=exchanges,
                # ✅ Best Practice: Use of type hints for return values improves code readability and maintainability.
                codes=codes,
                start_timestamp=start_timestamp,
                # ✅ Best Practice: Using the standard Python idiom for script entry point.
                end_timestamp=end_timestamp,
                # 🧠 ML Signal: Instantiation of a class with specific parameters could indicate a pattern for model training.
                # 🧠 ML Signal: Method invocation on an object could be used to identify common usage patterns.
                provider="joinquant",
                level=IntervalLevel.LEVEL_1DAY,
            ),
        ]

    def on_targets_selected_from_levels(self, timestamp) -> Tuple[List[str], List[str]]:
        # 过滤多级别做 多/空 的标的
        return super().on_targets_selected_from_levels(timestamp)


if __name__ == "__main__":
    trader = MultipleLevelTrader(start_timestamp="2019-01-01", end_timestamp="2020-01-01")
    trader.run()