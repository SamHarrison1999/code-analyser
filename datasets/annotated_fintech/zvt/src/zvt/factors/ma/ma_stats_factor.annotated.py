# -*- coding: utf-8 -*-
from typing import List, Union, Type, Optional
# ✅ Best Practice: Grouping imports by standard, third-party, and local can improve readability.

import pandas as pd

from zvt.contract import IntervalLevel, TradableEntity, AdjustType
from zvt.contract.api import get_schema_by_name
from zvt.contract.factor import Accumulator
from zvt.domain import Stock
# 🧠 ML Signal: Function with default parameter value
from zvt.factors.algorithm import live_or_dead
from zvt.factors.technical_factor import TechnicalFactor
# ✅ Best Practice: Use isinstance() instead of type() for type checking
from zvt.utils.pd_utils import pd_is_not_null


# ✅ Best Practice: Use f-string for better readability and performance
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, which this does.
def get_ma_stats_factor_schema(entity_type: str, level: Union[IntervalLevel, str] = IntervalLevel.LEVEL_1DAY):
    if type(level) == str:
        # ✅ Best Practice: Use of type hints for function parameters and return type
        # 🧠 ML Signal: Function call with dynamic string argument
        level = IntervalLevel(level)

    # ✅ Best Practice: Explicitly calling the superclass's __init__ method
    schema_str = "{}{}MaStatsFactor".format(entity_type.capitalize(), level.value.capitalize())

    # 🧠 ML Signal: Storing parameters in instance variables
    # 🧠 ML Signal: Logging usage pattern for tracking function execution
    return get_schema_by_name(schema_str)

# 🧠 ML Signal: Storing parameters in instance variables
# ⚠️ SAST Risk (Low): Potential issue if pd_is_not_null is not defined or imported

class MaStatsAccumulator(Accumulator):
    def __init__(self, acc_window: int = 250, windows=None, vol_windows=None) -> None:
        # ⚠️ SAST Risk (Low): Potential issue if pd_is_not_null is not defined or imported
        super().__init__(acc_window)
        self.windows = windows
        # 🧠 ML Signal: Logging usage pattern for tracking data processing
        self.vol_windows = vol_windows
    # ✅ Best Practice: Using pd.concat for DataFrame concatenation

    def acc_one(self, entity_id, df: pd.DataFrame, acc_df: pd.DataFrame, state: dict) -> (pd.DataFrame, dict):
        self.logger.info(f"acc_one:{entity_id}")
        if pd_is_not_null(acc_df):
            # 🧠 ML Signal: Logging usage pattern for conditional branches
            df = df[df.index > acc_df.index[-1]]
            if pd_is_not_null(df):
                self.logger.info(f'compute from {df.iloc[0]["timestamp"]}')
                acc_df = pd.concat([acc_df, df])
            else:
                self.logger.info("no need to compute")
                # ✅ Best Practice: Using format for string formatting
                return acc_df, state
        else:
            # 🧠 ML Signal: Tracking indicators being appended
            acc_df = df

        # ✅ Best Practice: Using rolling mean for moving average calculation
        for window in self.windows:
            col = "ma{}".format(window)
            self.indicators.append(col)
            # ✅ Best Practice: Using apply with lambda for conditional column creation

            ma_df = acc_df["close"].rolling(window=window, min_periods=window).mean()
            acc_df[col] = ma_df

        # ✅ Best Practice: Using groupby and cumcount for sequence counting
        acc_df["live"] = (acc_df["ma5"] > acc_df["ma10"]).apply(lambda x: live_or_dead(x))
        # ✅ Best Practice: Class definition should follow the naming convention of using CamelCase.
        acc_df["distance"] = (acc_df["ma5"] - acc_df["ma10"]) / acc_df["close"]
        # 🧠 ML Signal: Tracking indicators being appended
        # ✅ Best Practice: Setting index with drop=False to retain column
        # ✅ Best Practice: Using groupby and cumsum for cumulative sum calculation
        # ✅ Best Practice: Using format for string formatting
        # ✅ Best Practice: Using rolling mean for moving average calculation

        live = acc_df["live"]
        acc_df["count"] = live * (live.groupby((live != live.shift()).cumsum()).cumcount() + 1)

        acc_df["bulk"] = (live != live.shift()).cumsum()
        area_df = acc_df[["distance", "bulk"]]
        acc_df["area"] = area_df.groupby("bulk").cumsum()

        for vol_window in self.vol_windows:
            col = "vol_ma{}".format(vol_window)
            self.indicators.append(col)

            vol_ma_df = acc_df["turnover"].rolling(window=vol_window, min_periods=vol_window).mean()
            acc_df[col] = vol_ma_df

        acc_df = acc_df.set_index("timestamp", drop=False)
        return acc_df, state


class MaStatsFactor(TechnicalFactor):
    def __init__(
        self,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        # ⚠️ SAST Risk (Low): Potential risk if `need_persist` is not properly validated before use
        end_timestamp: Union[str, pd.Timestamp] = None,
        filters: List = None,
        # 🧠 ML Signal: Usage of dynamic schema generation based on entity type and level
        order: object = None,
        limit: int = None,
        # ✅ Best Practice: Default values for `windows` should be set in the function signature
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
        # ✅ Best Practice: Default values for `vol_windows` should be set in the function signature
        keep_all_timestamp: bool = False,
        fill_method: str = "ffill",
        # 🧠 ML Signal: Use of accumulator pattern for statistical calculations
        # 🧠 ML Signal: Use of inheritance and super() for class initialization
        effective_number: int = None,
        need_persist: bool = True,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
        adjust_type: Union[AdjustType, str] = None,
        windows=None,
        vol_windows=None,
    ) -> None:
        if need_persist:
            self.factor_schema = get_ma_stats_factor_schema(entity_type=entity_schema.__name__, level=level)

        if not windows:
            windows = [5, 10, 34, 55, 89, 144, 120, 250]
        self.windows = windows

        if not vol_windows:
            vol_windows = [30]
        self.vol_windows = vol_windows

        columns: List = ["id", "entity_id", "timestamp", "level", "open", "close", "high", "low", "turnover"]

        accumulator: Accumulator = MaStatsAccumulator(windows=self.windows, vol_windows=self.vol_windows)

        super().__init__(
            entity_schema,
            provider,
            entity_provider,
            entity_ids,
            # ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
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
            None,
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
        )


class TFactor(MaStatsFactor):
    def __init__(
        self,
        # ✅ Best Practice: Use of super() to call the parent class's __init__ method
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
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
        need_persist: bool = True,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = True,
        adjust_type: Union[AdjustType, str] = None,
        windows=None,
        vol_windows=None,
    ) -> None:
        super().__init__(
            # ✅ Best Practice: Specify the return type as List[pd.DataFrame] instead of Optional since the function always returns a list.
            entity_schema,
            provider,
            # 🧠 ML Signal: Accessing a specific column from a DataFrame, indicating columnar data processing.
            entity_provider,
            # ⚠️ SAST Risk (Low): Potential KeyError if "area" column does not exist in factor_df.
            # ✅ Best Practice: Explicitly specify the columns to be included in the DataFrame to improve readability and maintainability.
            entity_ids,
            exchanges,
            codes,
            # 🧠 ML Signal: Usage of a main guard to execute code conditionally.
            start_timestamp,
            end_timestamp,
            # 🧠 ML Signal: Instantiation of a class with specific parameters.
            # 🧠 ML Signal: Method call on an object with specific parameters.
            # ✅ Best Practice: Use of __all__ to define the public interface of the module.
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
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
            windows,
            vol_windows,
        )

    def drawer_sub_df_list(self) -> Optional[List[pd.DataFrame]]:
        return [self.factor_df[["area"]]]

    def drawer_factor_df_list(self) -> Optional[List[pd.DataFrame]]:
        return [self.factor_df[["ma5", "ma10"]]]


if __name__ == "__main__":
    codes = ["000338"]

    f = TFactor(codes=codes, only_load_factor=False)

    # distribute(f.factor_df[['area']],'area')
    f.draw(show=True)


# the __all__ is generated
__all__ = ["get_ma_stats_factor_schema", "MaStatsAccumulator", "MaStatsFactor", "TFactor"]