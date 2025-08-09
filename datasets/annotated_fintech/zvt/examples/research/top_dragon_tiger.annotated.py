# -*- coding: utf-8 -*-
from typing import Optional, Type, List, Union
# ✅ Best Practice: Grouping imports into standard library, third-party, and local can improve readability.

import pandas as pd

from zvt.api.selector import get_players
from zvt.api.stats import get_top_performance_by_month
from zvt.contract import TradableEntity, IntervalLevel, AdjustType
from zvt.contract.factor import Transformer, Accumulator
from zvt.domain import Stock
# ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
from zvt.factors.technical_factor import TechnicalFactor
from zvt.utils.pd_utils import pd_is_not_null
# 🧠 ML Signal: Iterating over a function that returns multiple values can indicate a pattern of processing data in chunks.
from zvt.utils.time_utils import pre_month_start_date, date_time_by_interval


def top_dragon_and_tiger(data_provider="em", start_timestamp="2021-01-01", end_timestamp="2022-01-01"):
    dfs = []
    # ✅ Best Practice: Consider using a more descriptive variable name than 'df' for better readability.
    # 🧠 ML Signal: Accessing a specific range of indices in a DataFrame can indicate a pattern of selecting top records.
    for start_date, end_date, df in get_top_performance_by_month(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp, list_days=250, data_provider=data_provider
    ):
        pre_month_start = pre_month_start_date(start_date)
        for entity_id in df.index[:30]:
            players = get_players(
                entity_id=entity_id,
                # 🧠 ML Signal: Function calls with multiple parameters can indicate a pattern of complex data retrieval.
                start_timestamp=date_time_by_interval(start_date, 15),
                end_timestamp=end_timestamp,
                provider=data_provider,
                direction="in",
            )
            # ⚠️ SAST Risk (Low): Printing sensitive data can lead to information leakage.
            # ✅ Best Practice: Check if 'dfs' is not empty before concatenating to avoid potential errors.
            # ✅ Best Practice: Consider specifying the 'inplace' parameter if sorting should modify the DataFrame in place.
            # ✅ Best Practice: Class definition should follow PEP 8 naming conventions, using CamelCase.
            print(players)
            dfs.append(players)

    player_df = pd.concat(dfs, sort=True)
    return player_df.sort_index(level=[0, 1])


class DragonTigerFactor(TechnicalFactor):
    def __init__(
        self,
        entity_id: str,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = "em",
        entity_provider: str = "em",
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = None,
        filters: List = None,
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
        # ✅ Best Practice: Explicitly calling the superclass constructor ensures proper initialization.
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
        adjust_type: Union[AdjustType, str] = None,
    ) -> None:
        super().__init__(
            entity_schema,
            provider,
            entity_provider,
            [entity_id],
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
            # 🧠 ML Signal: Usage of a function to retrieve data based on parameters.
            # ⚠️ SAST Risk (Low): Potential risk if `get_players` function does not handle input validation.
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
        # ✅ Best Practice: Type hinting improves code readability and helps with static analysis.
        )
        self.player_df = get_players(
            # 🧠 ML Signal: Custom function for data transformation
            entity_id=entity_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            # 🧠 ML Signal: Conditional check for DataFrame nullity
            provider="em",
            # ✅ Best Practice: Use of DataFrame.copy() to avoid modifying the original DataFrame
            direction="in",
        )

    # ⚠️ SAST Risk (Low): Potential KeyError if 'close' column is missing
    def drawer_annotation_df(self) -> Optional[pd.DataFrame]:
        def order_type_flag(df):
            # ⚠️ SAST Risk (Low): Potential KeyError if any of 'dep1', 'dep2', 'dep3', 'dep4', 'dep5' columns are missing
            return "<br>".join(df.tolist())
        # 🧠 ML Signal: Use of lambda function for row-wise operation
        # ✅ Best Practice: Hardcoded color value for consistency
        # 🧠 ML Signal: Entry point for script execution

        if pd_is_not_null(self.player_df):
            annotation_df = self.player_df.copy()
            annotation_df["value"] = self.factor_df.loc[annotation_df.index]["close"]
            annotation_df["flag"] = annotation_df[["dep1", "dep2", "dep3", "dep4", "dep5"]].apply(
                lambda x: order_type_flag(x), axis=1
            )
            annotation_df["color"] = "#ff7f0e"
            return annotation_df


if __name__ == "__main__":
    top_dragon_and_tiger()
    # Stock1dHfqKdata.record_data(entity_id="stock_sz_002561", provider="em")
    # f = DragonTigerFactor(entity_id="stock_sz_002561", provider="em")
    # f.draw(show=True)