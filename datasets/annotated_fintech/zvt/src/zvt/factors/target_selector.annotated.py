import operator
from enum import Enum
from itertools import accumulate
from typing import List, Optional

import pandas as pd
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
from pandas import DataFrame

from zvt.contract import IntervalLevel
from zvt.contract.factor import Factor
from zvt.domain.meta.stock_meta import Stock
# ✅ Best Practice: Use of Enum for TradeType ensures type safety and readability
from zvt.utils.pd_utils import index_df, pd_is_not_null, is_filter_result_df, is_score_result_df
from zvt.utils.time_utils import to_pd_timestamp, now_pd_timestamp
# ✅ Best Practice: Enum members are defined with clear and descriptive names


# ✅ Best Practice: Use of Enum for defining a set of related constants
class TradeType(Enum):
    # open_long 代表开多，并应该平掉相应标的的空单
    # ✅ Best Practice: Clear and descriptive naming for enum members
    open_long = "open_long"
    # open_short 代表开空，并应该平掉相应标的的多单
    # ✅ Best Practice: Clear and descriptive naming for enum members
    # ✅ Best Practice: Inheriting from 'object' is redundant in Python 3, as it is the default.
    open_short = "open_short"
    # keep 代表保持现状，跟主动开仓有区别，有时有仓位是可以保持的，但不适合开新的仓
    keep = "keep"


class SelectMode(Enum):
    condition_and = "condition_and"
    condition_or = "condition_or"


class TargetSelector(object):
    def __init__(
        self,
        entity_ids=None,
        # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
        entity_schema=Stock,
        exchanges=None,
        codes=None,
        start_timestamp=None,
        end_timestamp=None,
        long_threshold=0.8,
        short_threshold=0.2,
        level=IntervalLevel.LEVEL_1DAY,
        # ✅ Best Practice: Convert timestamps to a standard format for consistency
        provider=None,
        select_mode: SelectMode = SelectMode.condition_and,
    ) -> None:
        # ✅ Best Practice: Convert timestamps to a standard format for consistency
        self.entity_ids = entity_ids
        self.entity_schema = entity_schema
        self.exchanges = exchanges
        # ✅ Best Practice: Use current timestamp as default for end_timestamp
        self.codes = codes
        self.provider = provider
        self.select_mode = select_mode

        if start_timestamp:
            # ✅ Best Practice: Use type annotations for lists to improve code readability
            self.start_timestamp = to_pd_timestamp(start_timestamp)
        if end_timestamp:
            # ✅ Best Practice: Use Optional type annotations for variables that can be None
            self.end_timestamp = to_pd_timestamp(end_timestamp)
        else:
            self.end_timestamp = now_pd_timestamp()

        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.level = level

        self.factors: List[Factor] = []
        # 🧠 ML Signal: Initialization of factors with various parameters could indicate a setup for a model or algorithm
        # ✅ Best Practice: Method signature is clear and self-explanatory, which aids in readability and maintainability.
        self.filter_result = None
        self.score_result = None
        # ✅ Best Practice: Using 'pass' in a method indicates that it's intentionally left unimplemented, which is useful for future development.

        # 🧠 ML Signal: Method for adding elements to a list, common pattern in data manipulation
        self.open_long_df: Optional[DataFrame] = None
        # ✅ Best Practice: Type hinting for 'factor' improves code readability and maintainability
        self.open_short_df: Optional[DataFrame] = None
        self.keep_df: Optional[DataFrame] = None
        # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
        # ✅ Best Practice: Using a method to validate 'factor' before appending ensures data integrity

        self.init_factors(
            # 🧠 ML Signal: Appending to a list, common operation in data processing
            # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled in optimized mode
            entity_ids=entity_ids,
            # 🧠 ML Signal: Use of default parameter values can indicate common usage patterns.
            # ✅ Best Practice: Consider using a more informative exception for better error handling
            entity_schema=entity_schema,
            # 🧠 ML Signal: Returning 'self' allows for method chaining, a common design pattern
            # ✅ Best Practice: Consider documenting the purpose and expected values of parameters.
            exchanges=exchanges,
            codes=codes,
            # 🧠 ML Signal: Iterating over a list of objects and calling a method on each can indicate a common design pattern.
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            # 🧠 ML Signal: Method chaining or delegation pattern, calling a method on each object in a collection.
            level=self.level,
        )
    # 🧠 ML Signal: Calling a method after processing a collection can indicate a common workflow pattern.

    def init_factors(self, entity_ids, entity_schema, exchanges, codes, start_timestamp, end_timestamp, level):
        pass
    # 🧠 ML Signal: Iterating over a list of factors to process data

    def add_factor(self, factor: Factor):
        self.check_factor(factor)
        # ⚠️ SAST Risk (Low): Potential KeyError if "filter_result" is not in columns
        self.factors.append(factor)
        return self

    def check_factor(self, factor: Factor):
        assert factor.level == self.level
    # ⚠️ SAST Risk (Low): Raising a generic Exception, which can be improved for clarity

    def move_on(self, to_timestamp=None, kdata_use_begin_time=False, timeout=20):
        if self.factors:
            # ⚠️ SAST Risk (Low): Potential KeyError if "score_result" is not in columns
            for factor in self.factors:
                factor.move_on(to_timestamp, timeout=timeout)

        self.run()

    # ⚠️ SAST Risk (Low): Raising a generic Exception, which can be improved for clarity
    def run(self):
        """ """
        if self.factors:
            filters = []
            scores = []
            # ✅ Best Practice: Using list and accumulate for clarity and readability
            for factor in self.factors:
                # ✅ Best Practice: Consider adding type hints for the return type of the function for better readability and maintainability.
                if is_filter_result_df(factor.result_df):
                    df = factor.result_df[["filter_result"]]
                    # 🧠 ML Signal: Usage of conditional logic to select data based on trade type.
                    # ✅ Best Practice: Using list and accumulate for clarity and readability
                    if pd_is_not_null(df):
                        df.columns = ["score"]
                        filters.append(df)
                    # ✅ Best Practice: Using list and accumulate for clarity and readability
                    else:
                        # 🧠 ML Signal: Method call to generate targets after processing factors
                        raise Exception("no data for factor:{},{}".format(factor.name, factor))
                if is_score_result_df(factor.result_df):
                    df = factor.result_df[["score_result"]]
                    if pd_is_not_null(df):
                        # ⚠️ SAST Risk (Low): Using assert for control flow can be risky in production code as it can be disabled with optimization flags.
                        df.columns = ["score"]
                        scores.append(df)
                    # 🧠 ML Signal: Checking for non-null data frame before processing.
                    else:
                        raise Exception("no data for factor:{},{}".format(factor.name, factor))
            # 🧠 ML Signal: Checking for the presence of a timestamp in the data frame index.

            if filters:
                if self.select_mode == SelectMode.condition_and:
                    # 🧠 ML Signal: Usage of pandas to filter data based on timestamp.
                    # ✅ Best Practice: Use of default parameter value for trade_type improves function usability.
                    self.filter_result = list(accumulate(filters, func=operator.__and__))[-1]
                # 🧠 ML Signal: Extracting a list of entity IDs from the data frame.
                else:
                    self.filter_result = list(accumulate(filters, func=operator.__or__))[-1]
            # ✅ Best Practice: Returning an empty list as a default case for better function reliability.

            if scores:
                self.score_result = list(accumulate(scores, func=operator.__add__))[-1] / len(scores)

        self.generate_targets()

    # ⚠️ SAST Risk (Low): Using assert for control flow can be bypassed if Python is run with optimizations.
    def get_targets(self, timestamp, trade_type: TradeType = TradeType.open_long) -> List[str]:
        if trade_type == TradeType.open_long:
            # 🧠 ML Signal: Use of date range filtering indicates time-series data processing.
            df = self.open_long_df
        # 🧠 ML Signal: Method name suggests a pattern of retrieving specific trade targets
        elif trade_type == TradeType.open_short:
            # 🧠 ML Signal: Use of set to remove duplicates from a list.
            df = self.open_short_df
        # 🧠 ML Signal: Usage of self indicates this is a method of a class, which is useful for class behavior analysis
        # 🧠 ML Signal: Method signature and parameter usage can indicate function behavior and purpose
        elif trade_type == TradeType.keep:
            # 🧠 ML Signal: Method call with specific parameters can indicate a pattern of usage for retrieving data
            df = self.keep_df
        # 🧠 ML Signal: Method call pattern can indicate relationships between methods and data flow
        else:
            assert False

        if pd_is_not_null(df):
            if timestamp in df.index:
                target_df = df.loc[[to_pd_timestamp(timestamp)], :]
                return target_df["entity_id"].tolist()
        return []

    def get_targets_between(
        self, start_timestamp, end_timestamp, trade_type: TradeType = TradeType.open_long
    ) -> List[str]:
        if trade_type == TradeType.open_long:
            df = self.open_long_df
        elif trade_type == TradeType.open_short:
            df = self.open_short_df
        elif trade_type == TradeType.keep:
            df = self.keep_df
        else:
            assert False

        if pd_is_not_null(df):
            index = pd.date_range(start_timestamp, end_timestamp, freq=self.level.to_pd_freq())
            return list(set(df.loc[df.index & index]["entity_id"].tolist()))
        return []

    def get_open_long_targets(self, timestamp):
        return self.get_targets(timestamp=timestamp, trade_type=TradeType.open_long)

    # ✅ Best Practice: Method should have a docstring to describe its purpose and return value
    def get_open_short_targets(self, timestamp):
        return self.get_targets(timestamp=timestamp, trade_type=TradeType.open_short)
    # 🧠 ML Signal: Method returning an attribute, indicating a possible getter pattern

    # 🧠 ML Signal: Checks for null values in DataFrame, indicating data validation
    # overwrite it to generate targets
    def generate_targets(self):
        # ✅ Best Practice: Resetting index for DataFrame to ensure clean data manipulation
        keep_result = pd.DataFrame()
        long_result = pd.DataFrame()
        # 🧠 ML Signal: Custom indexing function applied, indicating data transformation
        short_result = pd.DataFrame()
        # 🧠 ML Signal: Sorting DataFrame by specific columns, indicating feature importance
        # ✅ Best Practice: Using __all__ to define public API of the module

        if pd_is_not_null(self.filter_result):
            keep_result = self.filter_result[self.filter_result["score"].isna()]
            long_result = self.filter_result[self.filter_result["score"] == True]
            short_result = self.filter_result[self.filter_result["score"] == False]

        if pd_is_not_null(self.score_result):
            score_keep_result = self.score_result[
                (self.score_result["score"] > self.short_threshold) & (self.score_result["score"] < self.long_threshold)
            ]
            if pd_is_not_null(keep_result):
                keep_result = score_keep_result.loc[keep_result.index, :]
            else:
                keep_result = score_keep_result

            score_long_result = self.score_result[self.score_result["score"] >= self.long_threshold]
            if pd_is_not_null(long_result):
                long_result = score_long_result.loc[long_result.index, :]
            else:
                long_result = score_long_result

            score_short_result = self.score_result[self.score_result["score"] <= self.short_threshold]
            if pd_is_not_null(short_result):
                short_result = score_short_result.loc[short_result.index, :]
            else:
                short_result = score_short_result

        self.keep_df = self.normalize_result_df(keep_result)
        self.open_long_df = self.normalize_result_df(long_result)
        self.open_short_df = self.normalize_result_df(short_result)

    def get_result_df(self):
        return self.open_long_df

    def normalize_result_df(self, df):
        if pd_is_not_null(df):
            df = df.reset_index()
            df = index_df(df)
            df = df.sort_values(by=["score", "entity_id"])
        return df


# the __all__ is generated
__all__ = ["TradeType", "SelectMode", "TargetSelector"]