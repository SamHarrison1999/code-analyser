# -*- coding: utf-8 -*-
from typing import List, Union

import pandas as pd

from zvt.contract import AdjustType
from zvt.contract import IntervalLevel, TradableEntity
from zvt.contract.drawer import Drawer
from zvt.contract.factor import Accumulator
from zvt.contract.factor import Transformer
from zvt.contract.reader import DataReader

# ✅ Best Practice: Grouping imports into standard library, third-party, and local application sections improves readability.
from zvt.domain import Stock, Stock1dKdata
from zvt.factors.technical_factor import TechnicalFactor

# ✅ Best Practice: Use of default parameter values for flexibility
from zvt.utils.time_utils import now_pd_timestamp

# ✅ Best Practice: Explicitly calling the superclass initializer
# 🧠 ML Signal: Method name 'transform' suggests a data transformation operation, common in data preprocessing for ML.


class TopBottomTransformer(Transformer):
    # 🧠 ML Signal: Tracking the initialization of instance variables
    # ✅ Best Practice: Using descriptive variable names like 'top_df' improves code readability.
    def __init__(self, window=20) -> None:
        super().__init__()
        # ✅ Best Practice: Resetting index with 'drop=True' to avoid unnecessary index column in the result.
        self.window = window

    # ✅ Best Practice: Directly assigning new columns to 'input_df' makes the transformation clear and concise.
    def transform(self, input_df) -> pd.DataFrame:
        top_df = (
            input_df["high"]
            .groupby(level=0)
            .rolling(window=self.window, min_periods=self.window)
            .max()
        )
        # ✅ Best Practice: Using descriptive variable names like 'bottom_df' improves code readability.
        top_df = top_df.reset_index(level=0, drop=True)
        # ✅ Best Practice: Resetting index with 'drop=True' to avoid unnecessary index column in the result.
        # ✅ Best Practice: Directly assigning new columns to 'input_df' makes the transformation clear and concise.
        # ✅ Best Practice: Returning the modified DataFrame allows for method chaining and functional programming style.
        input_df["top"] = top_df

        bottom_df = (
            input_df["low"]
            .groupby(level=0)
            .rolling(window=self.window, min_periods=self.window)
            .min()
        )
        bottom_df = bottom_df.reset_index(level=0, drop=True)
        input_df["bottom"] = bottom_df

        return input_df


class TopBottomFactor(TechnicalFactor):
    def __init__(
        self,
        entity_schema: TradableEntity = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = [
            "id",
            "entity_id",
            "timestamp",
            "level",
            "open",
            "close",
            "high",
            "low",
        ],
        filters: List = None,
        order: object = None,
        limit: int = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
        keep_all_timestamp: bool = False,
        fill_method: str = "ffill",
        # ✅ Best Practice: Initialize transformer with a clear and descriptive name
        effective_number: int = None,
        # ✅ Best Practice: Use of super() to call the parent class's __init__ method
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
        adjust_type: Union[AdjustType, str] = None,
        window=30,
    ) -> None:

        transformer = TopBottomTransformer(window=window)

        super().__init__(
            entity_schema,
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
            # 🧠 ML Signal: Example of instantiating a class with specific parameters
            fill_method,
            effective_number,
            transformer,
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
            # 🧠 ML Signal: Printing the factor's dataframe, indicating usage pattern
        )


# 🧠 ML Signal: Example of creating a DataReader with specific parameters
# 🧠 ML Signal: Example of using a Drawer to visualize data
# 🧠 ML Signal: Drawing a kline chart, indicating visualization usage
# ✅ Best Practice: Use of __all__ to define public API of the module


if __name__ == "__main__":
    factor = TopBottomFactor(
        codes=["601318"],
        start_timestamp="2005-01-01",
        end_timestamp=now_pd_timestamp(),
        level=IntervalLevel.LEVEL_1DAY,
        window=120,
    )
    print(factor.factor_df)

    data_reader1 = DataReader(
        data_schema=Stock1dKdata, entity_schema=Stock, codes=["601318"]
    )

    drawer = Drawer(
        main_df=data_reader1.data_df,
        factor_df_list=[factor.factor_df[["top", "bottom"]]],
    )
    drawer.draw_kline(show=True)


# the __all__ is generated
__all__ = ["TopBottomTransformer", "TopBottomFactor"]
