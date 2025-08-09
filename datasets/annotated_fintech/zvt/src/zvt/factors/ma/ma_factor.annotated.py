# -*- coding: utf-8 -*-
from typing import List, Union, Type
# ✅ Best Practice: Grouping imports into standard library, third-party, and local can improve readability.

import pandas as pd

from zvt.contract import IntervalLevel, TradableEntity, AdjustType
from zvt.contract.api import get_schema_by_name
from zvt.contract.factor import Accumulator
from zvt.contract.factor import Transformer
from zvt.domain import Stock
# ✅ Best Practice: Consider importing Union from typing for type hinting
from zvt.factors.algorithm import MaTransformer, MaAndVolumeTransformer
from zvt.factors.technical_factor import TechnicalFactor
# ✅ Best Practice: Use isinstance() instead of type() for type checking
from zvt.utils.time_utils import now_pd_timestamp

# ✅ Best Practice: Consider handling exceptions when converting string to IntervalLevel

# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which is CamelCase.
def get_ma_factor_schema(entity_type: str, level: Union[IntervalLevel, str] = IntervalLevel.LEVEL_1DAY):
    # 🧠 ML Signal: Usage of string formatting to create schema names
    # 🧠 ML Signal: Function call pattern to retrieve schema by name
    if type(level) == str:
        level = IntervalLevel(level)

    schema_str = "{}{}MaFactor".format(entity_type.capitalize(), level.value.capitalize())

    return get_schema_by_name(schema_str)


class MaFactor(TechnicalFactor):
    def __init__(
        self,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
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
        # ⚠️ SAST Risk (Low): Potential risk if `need_persist` is set to True without proper validation of `entity_schema`.
        effective_number: int = None,
        need_persist: bool = False,
        # 🧠 ML Signal: Usage of dynamic schema generation based on entity type and level.
        only_compute_factor: bool = False,
        factor_name: str = None,
        # ✅ Best Practice: Defaulting `windows` to a list of integers if not provided.
        clear_state: bool = False,
        only_load_factor: bool = False,
        # ✅ Best Practice: Using `super()` to ensure proper initialization of the base class.
        # 🧠 ML Signal: Initialization of a transformer with specific window sizes.
        adjust_type: Union[AdjustType, str] = None,
        windows=None,
    ) -> None:
        if need_persist:
            self.factor_schema = get_ma_factor_schema(entity_type=entity_schema.__name__, level=level)

        if not windows:
            windows = [5, 10, 34, 55, 89, 144, 120, 250]
        self.windows = windows
        transformer: Transformer = MaTransformer(windows=windows)

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
            fill_method,
            effective_number,
            # ✅ Best Practice: Call to superclass method ensures base class functionality is executed
            transformer,
            None,
            # 🧠 ML Signal: List comprehension used for dynamic column name generation
            need_persist,
            only_compute_factor,
            # 🧠 ML Signal: Boolean indexing pattern for DataFrame filtering
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
        # 🧠 ML Signal: Iterative boolean condition refinement
        )

# ⚠️ SAST Risk (Low): Use of print statement for debugging can expose data in production
# ✅ Best Practice: Converting boolean Series to DataFrame for consistent data handling

class CrossMaFactor(MaFactor):
    def compute_result(self):
        super().compute_result()
        cols = [f"ma{window}" for window in self.windows]
        s = self.factor_df[cols[0]] > self.factor_df[cols[1]]
        current_col = cols[1]
        for col in cols[2:]:
            s = s & (self.factor_df[current_col] > self.factor_df[col])
            current_col = col

        print(self.factor_df[s])
        self.result_df = s.to_frame(name="filter_result")


class VolumeUpMaFactor(TechnicalFactor):
    def __init__(
        self,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
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
        # ✅ Best Practice: Use of default mutable arguments (like lists) should be avoided to prevent shared state across instances.
        fill_method: str = "ffill",
        effective_number: int = None,
        accumulator: Accumulator = None,
        # ✅ Best Practice: Use of default mutable arguments (like lists) should be avoided to prevent shared state across instances.
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
        adjust_type: Union[AdjustType, str] = None,
        windows=None,
        vol_windows=None,
        # 🧠 ML Signal: Use of a specific transformer class indicates a pattern for data transformation.
        # 🧠 ML Signal: Use of a superclass constructor with many parameters suggests a pattern for object initialization.
        turnover_threshold=300000000,
        turnover_rate_threshold=0.02,
        up_intervals=40,
        over_mode="and",
    ) -> None:
        if not windows:
            windows = [250]
        if not vol_windows:
            vol_windows = [30]

        self.windows = windows
        self.vol_windows = vol_windows
        self.turnover_threshold = turnover_threshold
        self.turnover_rate_threshold = turnover_rate_threshold
        self.up_intervals = up_intervals
        self.over_mode = over_mode

        transformer: Transformer = MaAndVolumeTransformer(windows=windows, vol_windows=vol_windows)

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
            # ✅ Best Practice: Call to superclass method ensures base functionality is executed
            order,
            limit,
            # 🧠 ML Signal: Use of list comprehension to generate column names
            level,
            category_field,
            time_field,
            # 🧠 ML Signal: Use of boolean indexing for filtering data
            keep_window,
            keep_all_timestamp,
            # 🧠 ML Signal: Conditional logic based on attribute value
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
        )

    def compute_result(self):
        # 🧠 ML Signal: Conditional logic based on attribute presence
        super().compute_result()
        # 🧠 ML Signal: Use of list comprehension to generate column names

        # 价格刚上均线
        cols = [f"ma{window}" for window in self.windows]
        filter_up = (self.factor_df["close"] > self.factor_df[cols[0]]) & (
            self.factor_df["close"] < 1.15 * self.factor_df[cols[0]]
        )
        # 🧠 ML Signal: Use of boolean indexing for filtering data
        for col in cols[1:]:
            if self.over_mode == "and":
                filter_up = filter_up & (
                    (self.factor_df["close"] > self.factor_df[col])
                    # 🧠 ML Signal: Combination of multiple filters using logical AND
                    # ⚠️ SAST Risk (Low): Use of '==' to compare with False, consider using 'is False'
                    # 🧠 ML Signal: Use of groupby and fillna for forward filling missing values
                    # ⚠️ SAST Risk (Low): Use of 'isna()' to check for NaN values
                    # ✅ Best Practice: Storing result in a DataFrame for further processing
                    & (self.factor_df["close"] < 1.1 * self.factor_df[col])
                )
            else:
                filter_up = filter_up | (
                    (self.factor_df["close"] > self.factor_df[col])
                    & (self.factor_df["close"] < 1.1 * self.factor_df[col])
                )
        # 放量
        if self.vol_windows:
            vol_cols = [f"vol_ma{window}" for window in self.vol_windows]
            filter_vol = self.factor_df["volume"] > 2 * self.factor_df[vol_cols[0]]
            for col in vol_cols[1:]:
                filter_vol = filter_vol & (self.factor_df["volume"] > 2 * self.factor_df[col])

        # 成交额，换手率过滤
        filter_turnover = (self.factor_df["turnover"] > self.turnover_threshold) & (
            self.factor_df["turnover_rate"] > self.turnover_rate_threshold
        )
        s = filter_up & filter_vol & filter_turnover

        # 突破后的时间周期 up_intervals
        s[s == False] = None
        s = s.groupby(level=0).fillna(method="ffill", limit=self.up_intervals)
        s[s.isna()] = False

        # 还在均线附近
        # 1)刚突破
        # 2)突破后，回调到附近
        filter_result = filter_up & s & filter_turnover

        self.result_df = filter_result.to_frame(name="filter_result")
        # self.result_df = self.result_df.replace(False, None)


class CrossMaVolumeFactor(VolumeUpMaFactor):
    # ✅ Best Practice: Use of super() to call the parent class's __init__ method ensures proper initialization.
    def __init__(
        self,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
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
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
        adjust_type: Union[AdjustType, str] = None,
        windows=[5, 10, 250],
        vol_windows=None,
        turnover_threshold=300000000,
        turnover_rate_threshold=0.02,
        up_intervals=40,
        over_mode="and",
    ) -> None:
        # 🧠 ML Signal: Usage of dynamic column names based on a pattern
        super().__init__(
            entity_schema,
            # 🧠 ML Signal: Filtering data based on a condition
            provider,
            entity_provider,
            entity_ids,
            exchanges,
            # 🧠 ML Signal: Iterative filtering with logical AND
            codes,
            start_timestamp,
            end_timestamp,
            # 🧠 ML Signal: Additional filtering based on a threshold
            # 🧠 ML Signal: Conversion of filter result to DataFrame
            columns,
            filters,
            order,
            limit,
            level,
            category_field,
            time_field,
            keep_window,
            # ⚠️ SAST Risk (Low): Potential risk if `CrossMaVolumeFactor` is not properly validated
            keep_all_timestamp,
            # 🧠 ML Signal: Visualization of results
            # ✅ Best Practice: Use of __all__ to define public API of the module
            # ⚠️ SAST Risk (Low): Use of current timestamp can lead to non-deterministic behavior
            # ✅ Best Practice: Explicitly setting persistence behavior
            fill_method,
            effective_number,
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
            windows,
            vol_windows,
            turnover_threshold,
            turnover_rate_threshold,
            up_intervals,
            over_mode,
        )

    def compute_result(self):
        # 均线多头排列
        cols = [f"ma{window}" for window in self.windows]
        filter_se = self.factor_df[cols[0]] > self.factor_df[cols[1]]
        current_col = cols[1]
        for col in cols[2:]:
            filter_se = filter_se & (self.factor_df[current_col] > self.factor_df[col])
            current_col = col

        filter_se = filter_se & (self.factor_df["turnover"] > self.turnover_threshold)
        self.result_df = filter_se.to_frame(name="filter_result")
        # self.result_df = self.result_df.replace(False, None)


if __name__ == "__main__":

    factor = CrossMaVolumeFactor(
        entity_provider="em",
        provider="em",
        entity_ids=["stock_sz_000338"],
        start_timestamp="2020-01-01",
        end_timestamp=now_pd_timestamp(),
        need_persist=False,
    )
    factor.drawer().draw(show=True)


# the __all__ is generated
__all__ = ["get_ma_factor_schema", "MaFactor", "CrossMaFactor", "VolumeUpMaFactor", "CrossMaVolumeFactor"]