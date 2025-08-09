# -*- coding: utf-8 -*-
from typing import List, Union
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.

import pandas as pd

from zvt.contract import IntervalLevel
# 🧠 ML Signal: Class definition for a custom factor, useful for model training on class-based patterns
from zvt.contract.factor import Factor, Transformer, Accumulator
# ⚠️ SAST Risk (Low): Using mutable default arguments like lists can lead to unexpected behavior.
from zvt.domain import Stock, DragonAndTiger
from zvt.trader import StockTrader


class DragonTigerFactor(Factor):
    def __init__(
        self,
        provider: str = "em",
        entity_provider: str = "em",
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = None,
        filters: List = [DragonAndTiger.dep1 == "机构专用"],
        order: object = None,
        limit: int = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
        keep_all_timestamp: bool = False,
        fill_method: str = "ffill",
        effective_number: int = None,
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        need_persist: bool = False,
        # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
    ) -> None:
        super().__init__(
            DragonAndTiger,
            Stock,
            provider,
            entity_provider,
            entity_ids,
            exchanges,
            codes,
            start_timestamp,
            end_timestamp,
            columns,
            filters,
            order,
            limit,
            level,
            category_field,
            time_field,
            keep_window,
            keep_all_timestamp,
            fill_method,
            effective_number,
            transformer,
            accumulator,
            need_persist,
            only_compute_factor,
            # 🧠 ML Signal: Method overriding in a class, useful for learning class behavior
            factor_name,
            clear_state,
            # 🧠 ML Signal: Setting a DataFrame column to a constant value, indicates data manipulation pattern
            only_load_factor,
        # 🧠 ML Signal: Use of super() to call a method from a parent class, useful for understanding inheritance patterns
        # 🧠 ML Signal: Inheritance from a base class, indicating a pattern of extending functionality
        )

    def compute_result(self):
        # 🧠 ML Signal: Function initializes factors with given parameters, indicating a pattern of data preparation.
        self.factor_df["filter_result"] = True
        super().compute_result()


class MyTrader(StockTrader):
    def init_factors(
        self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, adjust_type=None
    ):
        return [
            DragonTigerFactor(
                # ✅ Best Practice: Use of __name__ == "__main__" to ensure code only runs when script is executed directly.
                entity_ids=entity_ids,
                # 🧠 ML Signal: Instantiation of MyTrader with specific timestamps, indicating a pattern of usage.
                # 🧠 ML Signal: Calling run method on trader object, indicating a pattern of execution.
                exchanges=exchanges,
                codes=codes,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        ]


if __name__ == "__main__":
    trader = MyTrader(start_timestamp="2020-01-01", end_timestamp="2022-05-01")
    trader.run()