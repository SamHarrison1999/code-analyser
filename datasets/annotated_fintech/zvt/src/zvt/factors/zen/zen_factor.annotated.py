# -*- coding: utf-8 -*-
import logging
import math
from enum import Enum
from typing import List, Optional
from typing import Union, Type
# ✅ Best Practice: Grouping imports from the same module together improves readability.

import pandas as pd

from zvt.contract import IntervalLevel, AdjustType
from zvt.contract import TradableEntity
from zvt.contract.data_type import Bean
from zvt.contract.drawer import Rect
from zvt.contract.factor import Accumulator
from zvt.contract.factor import Transformer
from zvt.domain import Stock
from zvt.factors.algorithm import distance, intersect
from zvt.factors.zen.base_factor import ZenFactor
from zvt.utils.pd_utils import (
    group_by_entity_id,
    normalize_group_compute_result,
    # 🧠 ML Signal: Usage of logging for tracking and debugging can be a signal for ML models to understand logging practices.
    pd_is_not_null,
)

logger = logging.getLogger(__name__)
# ✅ Best Practice: Use of @classmethod for alternative constructor or utility method

# ✅ Best Practice: Consider adding type hints for the parameters and return type for better readability and maintainability.

# ✅ Best Practice: Use of try-except for handling potential exceptions
# 🧠 ML Signal: Usage of threshold value (0.4) could be a feature for ML models.
class ZhongshuRange(Enum):
    # <=0.4
    # 🧠 ML Signal: Pattern of converting string to enum
    small = "small"
    # >0.4
    big = "big"
    # ⚠️ SAST Risk (Low): Potential information leakage in exception message

    # ✅ Best Practice: Use of @classmethod for alternative constructor or utility methods related to the class.
    @classmethod
    def of(cls, change):
        # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the method
        if change <= 0.4:
            return ZhongshuRange.small
        # ✅ Best Practice: Validate input to ensure 'level' is of expected type and range
        else:
            return ZhongshuRange.big


class ZhongshuLevel(Enum):
    # level <= 3
    level1 = "level1"
    # 3 < level <=7
    level2 = "level2"
    # ✅ Best Practice: Use of @classmethod for alternative constructor or utility methods related to the class
    # level > 7
    level3 = "level3"

    # ⚠️ SAST Risk (Low): Potential for TypeError if 'd' is not a number, consider type checking
    @classmethod
    def of(cls, level):
        if level <= 3:
            return ZhongshuLevel.level1
        elif level <= 7:
            return ZhongshuLevel.level2
        else:
            return ZhongshuLevel.level3


class ZhongshuDistance(Enum):
    # ✅ Best Practice: Explicitly returning a variable improves readability
    big_up = "big_up"
    big_down = "big_down"
    # ✅ Best Practice: Check if 'zhongshu_distance' is not None before accessing its 'value' attribute
    small_up = "small_up"
    small_down = "small_down"

    @classmethod
    def of(cls, d):
        # ✅ Best Practice: Use of __eq__ method to define equality comparison for the class
        # 🧠 ML Signal: Use of f-string for string formatting
        if d is None or math.isnan(d) or d == 0:
            zhongshu_distance = None
        # ✅ Best Practice: Checking if the object is an instance of the same class
        # 🧠 ML Signal: Equality comparison of object attributes
        elif d <= -0.5:
            zhongshu_distance = ZhongshuDistance.big_down
        elif d < 0:
            zhongshu_distance = ZhongshuDistance.small_down
        elif d <= 0.5:
            zhongshu_distance = ZhongshuDistance.small_up
        # ✅ Best Practice: Returning False for objects not of the same class
        else:
            zhongshu_distance = ZhongshuDistance.big_up
        return zhongshu_distance


class Zhongshu(object):
    # ✅ Best Practice: Use of type annotations for constructor parameters improves code readability and maintainability.
    def __str__(self) -> str:
        if self.zhongshu_distance:
            # ✅ Best Practice: Assigning constructor parameters to instance variables is a common pattern for initializing object state.
            d = self.zhongshu_distance.value
        # 🧠 ML Signal: Iterating over multiple nested loops indicates a combinatorial exploration pattern
        else:
            # ✅ Best Practice: Assigning constructor parameters to instance variables is a common pattern for initializing object state.
            d = None
        # ✅ Best Practice: Initialize lists before loops for clarity and potential reuse
        return f"{self.zhongshu_range.value},{self.zhongshu_level.value},{d}"
    # ✅ Best Practice: Assigning constructor parameters to instance variables is a common pattern for initializing object state.

    # 🧠 ML Signal: Iterating over an enumeration or collection
    def __eq__(self, o: object) -> bool:
        if isinstance(o, self.__class__):
            # 🧠 ML Signal: Iterating over an enumeration or collection
            return (
                # ✅ Best Practice: Class should have a docstring explaining its purpose and usage
                # ✅ Best Practice: Use of __eq__ method to define equality comparison for the class
                self.zhongshu_range == o.zhongshu_range
                # 🧠 ML Signal: Iterating over an enumeration or collection
                and self.zhongshu_level == o.zhongshu_level
                # ✅ Best Practice: Use of isinstance to check if the object is of the same class
                and self.zhongshu_distance == o.zhongshu_distance
            # ✅ Best Practice: Use 'pass' to indicate intentional no-operation in loop
            # ✅ Best Practice: Use of list comprehension for concise and readable code
            )
        # 🧠 ML Signal: Comparing attributes for equality
        return False
    # 🧠 ML Signal: Custom string representation of an object

    # ✅ Best Practice: Use of f-string for string formatting
    # ✅ Best Practice: Initialize lists using list literals for clarity and performance.
    def __init__(
        self,
        # 🧠 ML Signal: Storing input data for further processing or analysis.
        zhongshu_range: ZhongshuRange,
        zhongshu_level: ZhongshuLevel,
        zhongshu_distance: ZhongshuDistance,
    ) -> None:
        # 🧠 ML Signal: Iterating over input data to extract features or perform transformations.
        self.zhongshu_range = zhongshu_range
        self.zhongshu_level = zhongshu_level
        self.zhongshu_distance = zhongshu_distance


def category_zen_state():
    # 🧠 ML Signal: Calculating distance between consecutive data points.
    all_states = []

    for zhongshu_range in ZhongshuRange:
        # 🧠 ML Signal: Extracting specific features from input data.
        for zhongshu_level in ZhongshuLevel:
            # 🧠 ML Signal: Creating domain-specific objects from extracted features.
            for distance in ZhongshuDistance:
                pass


class ZenState(Bean):
    def __eq__(self, o: object) -> bool:
        # 🧠 ML Signal: Aggregating processed data into a structured format.
        if isinstance(o, self.__class__):
            return self.zhongshu_list == o.zhongshu_list

    def __str__(self) -> str:
        return ",".join([f"{elem}" for elem in self.zhongshu_list])

    # 🧠 ML Signal: Appending processed objects to a list for further use.
    def __init__(self, zhongshu_state_list: List) -> None:
        self.zhongshu_list: List[Zhongshu] = []
        self.zhongshu_state_list = zhongshu_state_list

        pre_range = None
        for zhongshu_state in zhongshu_state_list:
            current_range = (zhongshu_state[0], zhongshu_state[1])
            d = None
            # ✅ Best Practice: Consider importing math at the top of the file for better readability and maintainability.
            if pre_range is None:
                pre_range = current_range
            else:
                d = distance(pre_range, current_range)
                pre_range = current_range
            change = zhongshu_state[2]
            # ⚠️ SAST Risk (Low): Ensure that 'row' is a list or tuple with at least two elements to avoid IndexError.
            level = zhongshu_state[3]

            zhongshu_range = ZhongshuRange.of(change=change)
            zhongshu_level = ZhongshuLevel.of(level=level)
            zhongshu_distance = ZhongshuDistance.of(d=d)

            zhongshu = Zhongshu(
                zhongshu_range=zhongshu_range,
                zhongshu_level=zhongshu_level,
                # 🧠 ML Signal: The length check of 'zhongshu_state_list' could indicate a pattern for state transitions.
                zhongshu_distance=zhongshu_distance,
            )

            # 🧠 ML Signal: Appending a ZenState object to 'zen_states' could be a pattern of interest.
            self.zhongshu_list.append(zhongshu)
# 🧠 ML Signal: Function checks multiple conditions on a list of objects


# 🧠 ML Signal: Appending None to 'zen_states' could indicate a non-zen state.
# ✅ Best Practice: Check if zen_state is not None or empty before accessing its attributes
def cal_distance(s):
    d_list = []
    # ✅ Best Practice: Returning a pandas Series is a good practice for consistency with pandas operations.
    # ✅ Best Practice: Consider using a loop or list comprehension to reduce repetition
    current_range = None
    print(s)
    for idx, row in s.items():
        # ✅ Best Practice: Use a set for membership testing for better performance
        d = None
        if row is not None:
            if current_range is None:
                current_range = row
            else:
                d = distance((current_range.y0, current_range.y1), (row.y0, row.y1))
                current_range = row
        d_list.append(d)
    return pd.Series(index=s.index, data=d_list)

# ✅ Best Practice: Use a set for membership testing for better performance

def cal_zen_state(s):
    zen_states = []
    # 🧠 ML Signal: Function checks specific conditions on a sequence of objects, indicating a pattern recognition task.
    zhongshu_state_list = []
    current_zhongshu_state = None
    # ⚠️ SAST Risk (Low): Potential IndexError if zhongshu_list has fewer than 5 elements.
    # ✅ Best Practice: Combine conditions for clarity and conciseness
    for idx, row in s.items():
        # row
        # ✅ Best Practice: Consider checking the length of zhongshu_list before accessing elements.
        # 0 current_merge_zhongshu_y0
        # 1 current_merge_zhongshu_y1
        # 2 current_merge_zhongshu_change
        # 3 current_merge_zhongshu_level
        # 4 current_merge_zhongshu_interval
        if row[0] is not None and not math.isnan(row[0]):
            if current_zhongshu_state != row:
                # 相同的中枢，保留最近的(包含关系时产生)
                # 🧠 ML Signal: Checks for specific values in a sequence, useful for pattern detection.
                if current_zhongshu_state != None and intersect(
                    (current_zhongshu_state[0], current_zhongshu_state[1]),
                    (row[0], row[1]),
                ):
                    # 🧠 ML Signal: Further pattern checks on a subset of the sequence.
                    zhongshu_state_list = zhongshu_state_list[:-1]

                # 最多保留最近5个
                zhongshu_state_list = zhongshu_state_list[-4:] + [row]
                # 🧠 ML Signal: Final condition check for a specific pattern.
                # 🧠 ML Signal: Inheritance from a base class, indicating a potential pattern for class hierarchy
                current_zhongshu_state = row

        if len(zhongshu_state_list) == 5:
            zen_states.append(ZenState(zhongshu_state_list))
        else:
            zen_states.append(None)
    return pd.Series(index=s.index, data=zen_states)


def good_state(zen_state: ZenState):
    if zen_state:
        zhongshu0 = zen_state.zhongshu_list[0]
        zhongshu1 = zen_state.zhongshu_list[1]
        zhongshu2 = zen_state.zhongshu_list[2]
        zhongshu3 = zen_state.zhongshu_list[3]
        zhongshu4 = zen_state.zhongshu_list[4]

        # 没大涨过
        if ZhongshuDistance.big_up not in (
            zhongshu1.zhongshu_distance,
            zhongshu2.zhongshu_distance,
            zhongshu3.zhongshu_distance,
            zhongshu4.zhongshu_distance,
        ):
            if ZhongshuRange.big not in (
                zhongshu3.zhongshu_range,
                zhongshu4.zhongshu_range,
            ):
                # 最近一个窄幅震荡
                if ZhongshuRange.small == zhongshu4.zhongshu_range and ZhongshuLevel.level1 != zhongshu4.zhongshu_level:
                    # ✅ Best Practice: Use of super() to call the parent class's __init__ method
                    # 🧠 ML Signal: Use of various parameters to initialize an object, indicating a complex configuration
                    return True

    return False


def trending_state(zen_state: ZenState):
    if zen_state:
        zhongshu0 = zen_state.zhongshu_list[0]
        zhongshu1 = zen_state.zhongshu_list[1]
        zhongshu2 = zen_state.zhongshu_list[2]
        zhongshu3 = zen_state.zhongshu_list[3]
        zhongshu4 = zen_state.zhongshu_list[4]

        # 没大涨过
        if ZhongshuDistance.big_up not in (
            zhongshu1.zhongshu_distance,
            zhongshu2.zhongshu_distance,
            zhongshu3.zhongshu_distance,
        ):
            if ZhongshuRange.big not in (
                zhongshu3.zhongshu_range,
                zhongshu4.zhongshu_range,
            ):
                # 最近一个窄幅震荡
                if ZhongshuRange.small == zhongshu4.zhongshu_range and ZhongshuLevel.level1 == zhongshu4.zhongshu_level:
                    return True

    return False


class TrendingFactor(ZenFactor):
    # ✅ Best Practice: Check for null values before processing to avoid errors
    def __init__(
        # 🧠 ML Signal: Use of lambda function for transformation
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
        # 🧠 ML Signal: Grouping data by entity for further processing
        limit: int = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        # ✅ Best Practice: Debugging or logging intermediate data
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        # ✅ Best Practice: Debugging or logging intermediate data
        keep_window: int = None,
        # ✅ Best Practice: Class should inherit from a base class to ensure proper structure and functionality
        keep_all_timestamp: bool = False,
        # 🧠 ML Signal: Normalizing data as part of the processing pipeline
        fill_method: str = "ffill",
        # 🧠 ML Signal: Applying a function to transform data
        # 🧠 ML Signal: Extracting a specific column for further use
        # 🧠 ML Signal: Converting a Series to DataFrame for structured output
        # ✅ Best Practice: Class variables should be documented or initialized with meaningful default values
        effective_number: int = None,
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = True,
        adjust_type: Union[AdjustType, str] = None,
    ) -> None:
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
            # ✅ Best Practice: Use of super() to call the parent class's __init__ method ensures proper initialization.
            # 🧠 ML Signal: Use of entity_schema parameter to define the type of tradable entity.
            # 🧠 ML Signal: Use of provider parameter to specify data source.
            # 🧠 ML Signal: Use of entity_ids parameter to filter specific entities.
            # 🧠 ML Signal: Use of exchanges parameter to filter data by exchange.
            # 🧠 ML Signal: Use of start_timestamp parameter to define the start of the data range.
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
        super().compute_result()
        if pd_is_not_null(self.factor_df):
            df = self.factor_df.apply(
                lambda x: (
                    x["current_merge_zhongshu_y0"],
                    x["current_merge_zhongshu_y1"],
                    x["current_merge_zhongshu_change"],
                    x["current_merge_zhongshu_level"],
                    x["current_merge_zhongshu_interval"],
                ),
                axis=1,
            )

            state_df = group_by_entity_id(df).apply(cal_zen_state)
            print(self.factor_df)
            print(state_df)
            self.factor_df["zen_state"] = normalize_group_compute_result(state_df)
            self.factor_df["good_state"] = self.factor_df["zen_state"].apply(good_state)
            # 🧠 ML Signal: Use of end_timestamp parameter to define the end of the data range.
            # 🧠 ML Signal: Use of columns parameter to specify which columns to retrieve.
            # 🧠 ML Signal: Use of filters parameter to apply additional data filtering.
            # 🧠 ML Signal: Use of order parameter to define data ordering.
            # 🧠 ML Signal: Use of limit parameter to restrict the number of data entries.
            # 🧠 ML Signal: Use of level parameter to specify the granularity of data.
            # 🧠 ML Signal: Use of category_field parameter to define the category field for data.
            # ✅ Best Practice: Specify return type as Optional[List[pd.DataFrame]] for clarity

            # 🧠 ML Signal: Use of time_field parameter to define the time field for data.
            s = self.factor_df["good_state"]
            # 🧠 ML Signal: Usage of dropna() indicates data cleaning process
            self.result_df = s.to_frame(name="filter_result")
# 🧠 ML Signal: Use of keep_window parameter to define the window size for data retention.

# 🧠 ML Signal: Usage of dropna() indicates data cleaning process
# ✅ Best Practice: Type hinting improves code readability and maintainability

# 🧠 ML Signal: Use of keep_all_timestamp parameter to decide whether to keep all timestamps.
class ShakingFactor(ZenFactor):
    # 🧠 ML Signal: Use of inheritance and method overriding
    # ✅ Best Practice: Returning a list of DataFrames for consistent data handling
    # 震荡区间
    # 🧠 ML Signal: Use of fill_method parameter to specify the method for filling missing data.
    shaking_range = 0.5
    # 🧠 ML Signal: Usage of DataFrame filtering with conditions

    # 🧠 ML Signal: Use of effective_number parameter to define the effective number of data points.
    def __init__(
        # 🧠 ML Signal: Usage of DataFrame filtering with conditions
        self,
        # 🧠 ML Signal: Use of transformer parameter to apply data transformation.
        entity_schema: Type[TradableEntity] = Stock,
        # 🧠 ML Signal: Usage of DataFrame filtering with conditions
        provider: str = None,
        # 🧠 ML Signal: Use of accumulator parameter to accumulate data.
        entity_provider: str = None,
        # 🧠 ML Signal: Usage of DataFrame filtering with conditions
        entity_ids: List[str] = None,
        # 🧠 ML Signal: Use of need_persist parameter to decide whether to persist data.
        exchanges: List[str] = None,
        # 🧠 ML Signal: Usage of DataFrame filtering with conditions
        codes: List[str] = None,
        # 🧠 ML Signal: Use of only_compute_factor parameter to specify computation mode.
        start_timestamp: Union[str, pd.Timestamp] = None,
        # 🧠 ML Signal: Usage of DataFrame filtering with conditions
        end_timestamp: Union[str, pd.Timestamp] = None,
        # 🧠 ML Signal: Use of factor_name parameter to define the name of the factor.
        columns: List = None,
        # 🧠 ML Signal: Usage of DataFrame filtering with conditions
        filters: List = None,
        # 🧠 ML Signal: Use of clear_state parameter to decide whether to clear state.
        order: object = None,
        # 🧠 ML Signal: Use of only_load_factor parameter to specify loading mode.
        # 🧠 ML Signal: Use of adjust_type parameter to define the type of adjustment.
        # 🧠 ML Signal: Combining multiple conditions for DataFrame filtering
        # ✅ Best Practice: Resetting index for consistent DataFrame structure
        limit: int = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        keep_window: int = None,
        keep_all_timestamp: bool = False,
        # 🧠 ML Signal: Conversion of Series to DataFrame
        fill_method: str = "ffill",
        # ⚠️ SAST Risk (Low): Printing DataFrame can expose sensitive data
        # 🧠 ML Signal: Initialization of a list with specific identifiers
        # 🧠 ML Signal: Instantiation of a class with specific parameters
        # 🧠 ML Signal: Method call with specific parameters
        # 🧠 ML Signal: Definition of module's public API
        effective_number: int = None,
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = True,
        adjust_type: Union[AdjustType, str] = None,
    ) -> None:
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
            transformer,
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
            adjust_type,
        )

    def drawer_sub_df_list(self) -> Optional[List[pd.DataFrame]]:
        df1 = self.factor_df[["current_merge_zhongshu_y1"]].dropna()
        df2 = self.factor_df[["current_merge_zhongshu_y0"]].dropna()
        return [df1, df2]

    def drawer_rects(self) -> List[Rect]:
        return super().drawer_rects()

    def compute_result(self):
        super().compute_result()
        # 窄幅震荡
        s1 = self.factor_df["current_merge_zhongshu_change"] <= self.shaking_range
        # 中枢级别
        s2 = self.factor_df["current_merge_zhongshu_level"] >= 2
        s3 = self.factor_df["current_merge_zhongshu_interval"] >= 120

        # 中枢上缘
        s4 = self.factor_df["close"] <= 1.1 * self.factor_df["current_merge_zhongshu_y1"]
        s5 = self.factor_df["close"] >= 0.9 * self.factor_df["current_merge_zhongshu_y1"]

        # 中枢下缘
        s6 = self.factor_df["close"] <= 1.1 * self.factor_df["current_merge_zhongshu_y0"]
        s7 = self.factor_df["close"] >= 0.9 * self.factor_df["current_merge_zhongshu_y0"]

        s = s1 & s2 & s3 & ((s4 & s5) | (s6 & s7))
        # s = s.groupby(level=0).apply(drop_continue_duplicate)
        if s.index.nlevels == 3:
            s = s.reset_index(level=0, drop=True)

        self.result_df = s.to_frame(name="filter_result")
        print(self.result_df)


if __name__ == "__main__":
    entity_ids = ["stock_sz_000338"]

    f = ZenFactor(
        provider="em",
        entity_schema=Stock,
        entity_ids=entity_ids,
        need_persist=True,
    )
    f.draw(show=True)


# the __all__ is generated
__all__ = [
    "ZhongshuRange",
    "ZhongshuLevel",
    "ZhongshuDistance",
    "Zhongshu",
    "category_zen_state",
    "ZenState",
    "cal_distance",
    "cal_zen_state",
    "good_state",
    "trending_state",
    "TrendingFactor",
    "ShakingFactor",
]