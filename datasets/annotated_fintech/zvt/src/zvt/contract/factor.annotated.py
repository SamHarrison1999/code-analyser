# -*- coding: utf-8 -*-
import json
import logging
import time
from enum import Enum
from typing import List, Union, Optional, Type
# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies

import pandas as pd

from zvt.contract import IntervalLevel
from zvt.contract import zvt_context
from zvt.contract.api import get_data, df_to_db, del_data
from zvt.contract.base_service import EntityStateService
from zvt.contract.reader import DataReader, DataListener
from zvt.contract.schema import Mixin, TradableEntity
from zvt.contract.zvt_info import FactorState
# ✅ Best Practice: Use of Enum for defining a set of named constants improves code readability and maintainability.
from zvt.utils.pd_utils import pd_is_not_null, drop_continue_duplicate, is_filter_result_df, is_score_result_df
from zvt.utils.str_utils import to_snake_str
# ✅ Best Practice: Defining specific string values for each enum member enhances clarity and prevents errors.
from zvt.utils.time_utils import to_pd_timestamp


# ✅ Best Practice: Inheriting from 'object' is redundant in Python 3, as all classes are new-style by default.
# ✅ Best Practice: Use of __init__ method to initialize instance variables
class TargetType(Enum):
    positive = "positive"
    # ✅ Best Practice: Using a logger with the class name for better traceability
    negative = "negative"
    keep = "keep"
# 🧠 ML Signal: Initialization of an empty list, indicating potential dynamic data storage
# ✅ Best Practice: Class definition should include a docstring explaining its purpose and usage.
# ✅ Best Practice: Define an explicit constructor for the class


# ✅ Best Practice: Call the superclass constructor to ensure proper initialization
class Indicator(object):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.indicators = []


class Transformer(Indicator):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        input_df format::

                                      col1    col2    col3    ...
            entity_id    timestamp
                                      1.2     0.5     0.3     ...
                                      1.0     0.7     0.2     ...

        the return result would change the columns and  keep the format

        :param input_df:
        :return:
        """
        g = input_df.groupby(level=0)
        if len(g.groups) == 1:
            entity_id = input_df.index[0][0]

            df = input_df.reset_index(level=0, drop=True)
            ret_df = self.transform_one(entity_id=entity_id, df=df)
            ret_df["entity_id"] = entity_id

            return ret_df.set_index("entity_id", append=True).swaplevel(0, 1)
        # 🧠 ML Signal: Function signature with DataFrame parameter and return type
        # ✅ Best Practice: Class docstring is missing, consider adding one to describe the purpose and usage of the class.
        else:
            return g.apply(lambda x: self.transform_one(x.index[0][0], x.reset_index(level=0, drop=True)))
    # ✅ Best Practice: Use of type hints for function parameters and return type

    def transform_one(self, entity_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        df format::

                         col1    col2    col3    ...
            timestamp
                         1.2     0.5     0.3     ...
                         1.0     0.7     0.2     ...

        the return result would change the columns and  keep the format

        :param entity_id:
        :param df:
        :return:
        # 🧠 ML Signal: Extracts entity_id from the DataFrame index
        """
        return df
# ✅ Best Practice: Resetting index for easier manipulation of DataFrame


# ⚠️ SAST Risk (Low): Assumes acc_df has the same structure as input_df without validation
class Accumulator(Indicator):
    def __init__(self, acc_window: int = 1) -> None:
        """

        :param acc_window: the window size of acc for computing,default is 1
        """
        # 🧠 ML Signal: Calls a method to process a single entity's data
        super().__init__()
        # ⚠️ SAST Risk (Low): Assumes ret_df is not None without validation
        self.acc_window = acc_window

    def acc(self, input_df: pd.DataFrame, acc_df: pd.DataFrame, states: dict) -> (pd.DataFrame, dict):
        """

        :param input_df: new input
        :param acc_df: previous result
        :param states: current states of the entity
        :return: new result and states
        """
        # 🧠 ML Signal: Conditional logic based on group membership, indicating data filtering
        g = input_df.groupby(level=0)
        # 🧠 ML Signal: Initializes a dictionary to store states for multiple entities
        # 🧠 ML Signal: Accessing a specific group, common in data analysis
        if len(g.groups) == 1:
            entity_id = input_df.index[0][0]

            # 🧠 ML Signal: Checks for non-null DataFrame, indicating data validation pattern
            # ✅ Best Practice: Resetting index for cleaner DataFrame manipulation
            df = input_df.reset_index(level=0, drop=True)
            if pd_is_not_null(acc_df) and (entity_id == acc_df.index[0][0]):
                acc_one_df = acc_df.reset_index(level=0, drop=True)
            else:
                acc_one_df = None
            ret_df, state = self.acc_one(entity_id=entity_id, df=df, acc_df=acc_one_df, state=states.get(entity_id))
            # 🧠 ML Signal: Function call with multiple parameters, indicating complex data processing
            if pd_is_not_null(ret_df):
                ret_df["entity_id"] = entity_id
                ret_df = ret_df.set_index("entity_id", append=True).swaplevel(0, 1)
                ret_df["entity_id"] = entity_id
                return ret_df, {entity_id: state}
            # 🧠 ML Signal: Updating a dictionary with new state, indicating state management
            # 🧠 ML Signal: Applying a function over a group, common in data processing
            # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters
            return None, {entity_id: state}
        else:
            new_states = {}

            def cal_acc(x):
                entity_id = x.index[0][0]
                if pd_is_not_null(acc_df):
                    acc_g = acc_df.groupby(level=0)
                    acc_one_df = None
                    if entity_id in acc_g.groups:
                        acc_one_df = acc_g.get_group(entity_id)
                        if pd_is_not_null(acc_one_df):
                            acc_one_df = acc_one_df.reset_index(level=0, drop=True)
                # ✅ Best Practice: Returning multiple values for better function utility
                else:
                    # ✅ Best Practice: Return statement is clear and matches the function's documented return type
                    acc_one_df = None
                # ✅ Best Practice: Using a logger with the class name improves traceability and debugging.

                one_result, state = self.acc_one(
                    entity_id=entity_id,
                    # ✅ Best Practice: Include type annotations for function parameters and return type for better readability and maintainability.
                    df=x.reset_index(level=0, drop=True),
                    acc_df=acc_one_df,
                    state=states.get(x.index[0][0]),
                )

                new_states[entity_id] = state
                # 🧠 ML Signal: Directly returning the input DataFrame without processing may indicate a placeholder or a stub for future implementation.
                # 🧠 ML Signal: Checking class name against a list of specific names indicates a pattern for class registration.
                return one_result

            # ⚠️ SAST Risk (Low): Potential risk if `zvt_context` is not properly initialized or if `factor_cls_registry` is not a dictionary.
            ret_df = g.apply(lambda x: cal_acc(x))
            # ✅ Best Practice: Class docstring is missing, consider adding one for better documentation.
            # ✅ Best Practice: Use of __new__ method to customize class creation
            # 🧠 ML Signal: Dynamic registration of classes into a global context is a common pattern in plugin systems.
            return ret_df, new_states

    # ✅ Best Practice: Using type.__new__ to create a new class instance
    def acc_one(self, entity_id, df: pd.DataFrame, acc_df: pd.DataFrame, state: dict) -> (pd.DataFrame, dict):
        """
        df format::

                         col1    col2    col3    ...
            timestamp
                         1.2     0.5     0.3     ...
                         1.0     0.7     0.2     ...

        the new result and state

        :param df: current input df
        :param entity_id: current computing entity_id
        :param acc_df: current result of the entity_id
        :param state: current state of the entity_id
        :return: new result and state of the entity_id
        """
        return acc_df, state


class Scorer(object):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def score(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """

        :param input_df: current input df
        :return: df with normal score
        """
        return input_df


def _register_class(target_class):
    if target_class.__name__ not in ("Factor", "FilterFactor", "ScoreFactor", "StateFactor"):
        zvt_context.factor_cls_registry[target_class.__name__] = target_class

# ✅ Best Practice: Docstring is provided for the constructor parameters.

class FactorMeta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        _register_class(cls)
        return cls


class Factor(DataReader, EntityStateService, DataListener):
    #: Schema for storing states
    state_schema = FactorState
    #: define the schema for persist,its columns should be same as indicators in transformer or accumulator
    factor_schema: Type[Mixin] = None

    # ✅ Best Practice: Using a utility function to convert class name to snake case.
    #: transformer for this factor if not passed as __init__ argument
    transformer: Transformer = None
    #: accumulator for this factor if not passed as __init__ argument
    # 🧠 ML Signal: Usage of multiple inheritance and initialization of parent classes.
    accumulator: Accumulator = None

    def __init__(
        self,
        data_schema: Type[Mixin],
        entity_schema: Type[TradableEntity] = None,
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
        # 🧠 ML Signal: Usage of multiple inheritance and initialization of parent classes.
        effective_number: int = None,
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        clear_state: bool = False,
        only_load_factor: bool = False,
    ) -> None:
        """
        :param keep_all_timestamp:
        :param fill_method:
        :param effective_number:
        :param transformer:
        :param accumulator:
        :param need_persist: whether persist factor
        :param only_compute_factor: only compute factor nor result
        :param factor_name:
        :param clear_state:
        :param only_load_factor: only load factor and compute result
        """
        self.only_load_factor = only_load_factor
        # 🧠 ML Signal: Conditional logic based on class attributes.

        #: define unique name of your factor if you want to keep factor state
        #: the factor state is defined by factor_name and entity_id
        if not factor_name:
            self.name = to_snake_str(type(self).__name__)
        else:
            self.name = factor_name

        DataReader.__init__(
            # ⚠️ SAST Risk (Low): Potential risk if `get_data` is not properly validated or sanitized.
            self,
            data_schema,
            entity_schema,
            provider,
            entity_provider,
            entity_ids,
            exchanges,
            codes,
            start_timestamp,
            end_timestamp,
            # ✅ Best Practice: Check for conditions early to avoid unnecessary processing
            columns,
            filters,
            order,
            # ✅ Best Practice: Use of super() to call a method from the parent class
            limit,
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and functionality of the method.
            level,
            # 🧠 ML Signal: Registering a data listener, indicating event-driven architecture.
            category_field,
            # ✅ Best Practice: Ensure that `load_window_df` method handles exceptions or errors.
            time_field,
            keep_window,
        )

        # ✅ Best Practice: Ensure that `get_data` method handles exceptions or errors.
        EntityStateService.__init__(self, entity_ids=entity_ids)

        self.clear_state = clear_state

        self.keep_all_timestamp = keep_all_timestamp
        self.fill_method = fill_method
        self.effective_number = effective_number

        if transformer:
            self.transformer = transformer
        # ✅ Best Practice: Consider checking if 'df' is a DataFrame to ensure type safety.
        else:
            # 🧠 ML Signal: The method `decode_factor_df` is called on `self.factor_df`, indicating a transformation or decoding step.
            self.transformer = self.__class__.transformer
        # 🧠 ML Signal: Checking if a DataFrame is not null before processing is a common pattern.

        if accumulator:
            # 🧠 ML Signal: Iterating over DataFrame columns to apply transformations is a common pattern.
            self.accumulator = accumulator
        else:
            self.accumulator = self.__class__.accumulator
        # ⚠️ SAST Risk (Medium): Using json.loads with object_hook can lead to code execution if the input is not trusted.

        # 🧠 ML Signal: Use of lambda functions for inline data transformation.
        # ✅ Best Practice: Include a docstring to describe the method's purpose and return value
        self.need_persist = need_persist
        self.only_compute_factor = only_compute_factor

        #: 中间结果，不持久化
        # ✅ Best Practice: Return an empty dictionary as a default implementation
        #: data_df->pipe_df
        # ✅ Best Practice: Call to superclass method ensures base functionality is preserved.
        self.pipe_df: pd.DataFrame = None

        # ⚠️ SAST Risk (Medium): Potential SQL injection risk if `entity_id` is not properly sanitized.
        #: 计算因子的结果，可持久化,通过对pipe_df的计算得到
        #: pipe_df->factor_df
        self.factor_df: pd.DataFrame = None

        # 🧠 ML Signal: Conditional logic based on the presence of `entity_id` indicates different behavior paths.
        # ✅ Best Practice: Use of clear and descriptive variable names enhances readability
        #: result_df是用于选股的标准df,通过对factor_df的计算得到
        #: factor_df->result_df
        # 🧠 ML Signal: Logging usage pattern
        # ⚠️ SAST Risk (Low): Potential risk if pd_is_not_null is not properly defined or imported
        self.result_df: pd.DataFrame = None

        # 🧠 ML Signal: Assignment of one dataframe to another could indicate data transformation or preparation
        # 🧠 ML Signal: Logging usage pattern
        if self.clear_state:
            self.clear_state_data()
        # 🧠 ML Signal: Method call pattern
        elif self.need_persist or self.only_load_factor:
            self.load_factor()
            # 🧠 ML Signal: Logging usage pattern

            #: 根据已经计算的factor_df和computing_window来保留data_df
            # 🧠 ML Signal: Logging usage pattern
            #: 因为读取data_df的目的是为了计算factor_df,选股和回测只依赖factor_df
            #: 所以如果有持久化的factor_df,只需保留需要用于计算的data_df即可
            # 🧠 ML Signal: Method call pattern
            # 🧠 ML Signal: Checks if data is not null before processing
            if pd_is_not_null(self.data_df) and self.computing_window:
                # 🧠 ML Signal: Logging usage pattern
                # 🧠 ML Signal: Usage of transformer pattern for data processing
                dfs = []
                for entity_id, df in self.data_df.groupby(level=0):
                    latest_laved = get_data(
                        provider="zvt",
                        # 🧠 ML Signal: Checks if transformed data is not null before further processing
                        data_schema=self.factor_schema,
                        entity_id=entity_id,
                        order=self.factor_schema.timestamp.desc(),
                        # 🧠 ML Signal: Usage of accumulator pattern for data aggregation
                        limit=1,
                        # 🧠 ML Signal: Method checks for non-null DataFrame, indicating data validation pattern
                        index=[self.category_field, self.time_field],
                        return_type="domain",
                    )
                    # 🧠 ML Signal: Conditional checks for specific DataFrame types, indicating type-based logic
                    if latest_laved:
                        df1 = df[df.timestamp < latest_laved[0].timestamp].iloc[-self.computing_window :]
                        if pd_is_not_null(df1):
                            # 🧠 ML Signal: Conditional checks for specific DataFrame types, indicating type-based logic
                            df = df[df.timestamp >= df1.iloc[0].timestamp]
                    dfs.append(df)

                # ✅ Best Practice: Check if list is non-empty before using it
                # 🧠 ML Signal: Method with conditional logic based on instance attributes
                self.data_df = pd.concat(dfs)

        # 🧠 ML Signal: Assigning a subset of DataFrame columns, indicating feature selection pattern
        self.register_data_listener(self)
        # 🧠 ML Signal: Conditional logic based on instance attributes

        #: the compute logic is not triggered from load data
        # ✅ Best Practice: Method call without arguments, likely a utility function
        #: for the case:1)load factor from db 2)compute the result
        if self.only_load_factor:
            # 🧠 ML Signal: Conditional logic based on multiple instance attributes
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and functionality of the compute method.
            self.compute()

    # ✅ Best Practice: Method call without arguments, likely a utility function
    # 🧠 ML Signal: Logging usage pattern with self.logger.info
    def load_data(self):
        if self.only_load_factor:
            # 🧠 ML Signal: Logging usage pattern with self.logger.info
            return
        super().load_data()
    # 🧠 ML Signal: Capturing start time for performance measurement

    def load_factor(self):
        if self.only_compute_factor:
            # 🧠 ML Signal: Calculating elapsed time for performance measurement
            #: 如果只是为了计算因子，只需要读取acc_window的factor_df
            if self.accumulator is not None:
                # 🧠 ML Signal: Logging usage pattern with self.logger.info
                self.factor_df = self.load_window_df(
                    provider="zvt", data_schema=self.factor_schema, window=self.accumulator.acc_window
                # 🧠 ML Signal: Logging usage pattern with self.logger.info
                )
        # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
        else:
            # 🧠 ML Signal: Capturing start time for performance measurement
            self.factor_df = get_data(
                # 🧠 ML Signal: Conditional logic based on object attributes
                provider="zvt",
                data_schema=self.factor_schema,
                # 🧠 ML Signal: Return of different attributes based on condition
                # ✅ Best Practice: Check for None before using objects to avoid AttributeError
                # 🧠 ML Signal: Calculating elapsed time for performance measurement
                start_timestamp=self.start_timestamp,
                entity_ids=self.entity_ids,
                # 🧠 ML Signal: Logging usage pattern with self.logger.info
                end_timestamp=self.end_timestamp,
                # ✅ Best Practice: Use of None check to determine which object's indicators to use
                index=[self.category_field, self.time_field],
            # 🧠 ML Signal: Logging usage pattern with self.logger.info
            )

        self.decode_factor_df(self.factor_df)
    # ✅ Best Practice: Check if indicators is not empty before using it

    def decode_factor_df(self, df):
        col_map_object_hook = self.factor_col_map_object_hook()
        # 🧠 ML Signal: Usage of DataFrame slicing with dynamic column selection
        if pd_is_not_null(df) and col_map_object_hook:
            for col in col_map_object_hook:
                # ✅ Best Practice: Check for None before using objects to avoid AttributeError
                if col in df.columns:
                    # 🧠 ML Signal: Returning a DataFrame in a list for consistency
                    df[col] = df[col].apply(
                        # 🧠 ML Signal: Returns a list containing a single DataFrame if conditions are met
                        lambda x: json.loads(x, object_hook=col_map_object_hook.get(col)) if x else None
                    # ✅ Best Practice: Explicitly returning None for clarity
                    # ✅ Best Practice: Type hinting improves code readability and maintainability by specifying expected return type
                    )
    # 🧠 ML Signal: Returns None if conditions are not met

    # ✅ Best Practice: Check for None explicitly to handle null values.
    def factor_col_map_object_hook(self) -> dict:
        """

        :return:{col:object_hook}
        """
        return {}
    # ✅ Best Practice: Return a specific value for falsy order_type.

    # ✅ Best Practice: Use of descriptive color codes for different order types
    def clear_state_data(self, entity_id=None):
        super().clear_state_data(entity_id=entity_id)
        if entity_id:
            del_data(self.factor_schema, filters=[self.factor_schema.entity_id == entity_id], provider="zvt")
        else:
            del_data(self.factor_schema, provider="zvt")
    # 🧠 ML Signal: Checking if a DataFrame is a filter result

    def pre_compute(self):
        # ✅ Best Practice: Removing NaN values to ensure data integrity
        if not self.only_load_factor and not pd_is_not_null(self.pipe_df):
            self.pipe_df = self.data_df
    # 🧠 ML Signal: Dropping continuous duplicates in a DataFrame

    def do_compute(self):
        # 🧠 ML Signal: Mapping index to another DataFrame to extract 'close' values
        self.logger.info("compute factor start")
        self.compute_factor()
        # 🧠 ML Signal: Applying a function to transform 'filter_result' into flags
        # 🧠 ML Signal: Usage of pd.date_range to generate a range of dates
        self.logger.info("compute factor finish")

        self.logger.info("compute result start")
        # 🧠 ML Signal: Applying a function to transform 'filter_result' into colors
        self.compute_result()
        # 🧠 ML Signal: Creation of a MultiIndex for hierarchical indexing
        self.logger.info("compute result finish")

    # ⚠️ SAST Risk (Low): Potential data loss by removing duplicates without logging
    def compute_factor(self):
        # ✅ Best Practice: Check for both self.entity_ids and entity_ids to avoid unnecessary operations
        if self.only_load_factor:
            # 🧠 ML Signal: Reindexing DataFrame to align with a new index
            return
        #: 无状态的转换运算
        # 🧠 ML Signal: Usage of fillna with method and limit parameters
        if pd_is_not_null(self.data_df) and self.transformer:
            # ✅ Best Practice: Grouping by level 0 for hierarchical data processing
            self.pipe_df = self.transformer.transform(self.data_df)
        else:
            self.pipe_df = self.data_df
        # ✅ Best Practice: Use set operations to find new entity IDs efficiently

        #: 有状态的累加运算
        # ✅ Best Practice: Use set to avoid duplicate entity IDs
        if pd_is_not_null(self.pipe_df) and self.accumulator:
            self.factor_df, self.states = self.accumulator.acc(self.pipe_df, self.factor_df, self.states)
        # ⚠️ SAST Risk (Low): Potentially large data query without pagination
        else:
            self.factor_df = self.pipe_df

    def compute_result(self):
        if pd_is_not_null(self.factor_df):
            cols = []
            if is_filter_result_df(self.factor_df):
                cols.append("filter_result")
            if is_score_result_df(self.factor_df):
                cols.append("score_result")

            if cols:
                self.result_df = self.factor_df[cols]

    def after_compute(self):
        # ⚠️ SAST Risk (Low): Concatenating large DataFrames can lead to high memory usage
        # ✅ Best Practice: Sorting index after concatenation for consistent data order
        if self.only_load_factor:
            return
        if self.keep_all_timestamp:
            self.fill_gap()

        if self.need_persist and pd_is_not_null(self.factor_df):
            self.persist_factor()

    # ⚠️ SAST Risk (Low): External data fetching without validation
    def compute(self):
        self.pre_compute()

        # 🧠 ML Signal: Method signature and parameter types can be used to infer method usage patterns
        self.logger.info(f"[[[ ~~~~~~~~factor:{self.name} ~~~~~~~~]]]")
        self.logger.info("do_compute start")
        # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and parameters
        start_time = time.time()
        # 🧠 ML Signal: Decoding data frames could indicate data transformation patterns
        # ⚠️ SAST Risk (Low): Ensure that 'data' is validated before use to prevent potential data integrity issues
        # ✅ Best Practice: Add type hints for better code readability and maintainability
        self.do_compute()
        cost_time = time.time() - start_time
        self.logger.info("do_compute finished,cost_time:{}s".format(cost_time))

        # ⚠️ SAST Risk (Low): Concatenating large DataFrames can lead to high memory usage
        self.logger.info("after_compute start")
        # ✅ Best Practice: Sorting index after concatenation for consistent data order
        # ✅ Best Practice: Consider adding a docstring description for the 'data' parameter
        # ✅ Best Practice: Method docstring is provided, which improves code readability and maintainability
        start_time = time.time()
        # ✅ Best Practice: Docstring describes the purpose and parameters of the method
        # 🧠 ML Signal: Method call pattern that could be used to understand function usage
        self.after_compute()
        cost_time = time.time() - start_time
        self.logger.info("after_compute finished,cost_time:{}s".format(cost_time))
        self.logger.info(f"[[[ ^^^^^^^^factor:{self.name} ^^^^^^^^]]]")

    def drawer_main_df(self) -> Optional[pd.DataFrame]:
        if self.only_load_factor:
            # 🧠 ML Signal: Method with a clear purpose and parameters can be used to identify usage patterns
            return self.factor_df
        # ✅ Best Practice: Check if DataFrame is not null before proceeding
        return self.data_df

    # 🧠 ML Signal: Iterating over columns to apply transformations
    def drawer_factor_df_list(self) -> Optional[List[pd.DataFrame]]:
        if (self.transformer is not None or self.accumulator is not None) and pd_is_not_null(self.factor_df):
            indicators = None
            # 🧠 ML Signal: Using json.dumps with a custom encoder
            if self.transformer is not None:
                indicators = self.transformer.indicators
            elif self.accumulator is not None:
                # 🧠 ML Signal: Grouping DataFrame by level 0
                indicators = self.accumulator.indicators

            if indicators:
                return [self.factor_df[indicators]]
            # 🧠 ML Signal: Persisting state for each entity
            else:
                return [self.factor_df]
        return None

    def drawer_sub_df_list(self) -> Optional[List[pd.DataFrame]]:
        # ⚠️ SAST Risk (Medium): Potential SQL injection risk if df_to_db is not properly handling inputs
        if (self.transformer is not None or self.accumulator is not None) and pd_is_not_null(self.result_df):
            return [self.result_df]
        return None

    def drawer_annotation_df(self) -> Optional[pd.DataFrame]:
        # ✅ Best Practice: Logging errors with context
        # 🧠 ML Signal: Method checks a condition before accessing a DataFrame column
        def order_type_flag(order_type):
            if order_type is None:
                # 🧠 ML Signal: Accessing specific columns of a DataFrame
                # ✅ Best Practice: Logging exception details
                return None
            # 🧠 ML Signal: Method checks a condition before returning a specific DataFrame column
            if order_type:
                # 🧠 ML Signal: Clearing state data on exception
                return "B"
            # ✅ Best Practice: Explicitly specify the column name when returning a DataFrame slice
            if not order_type:
                return "S"
        # ⚠️ SAST Risk (Medium): Potential SQL injection risk if df_to_db is not properly handling inputs
        # ✅ Best Practice: Use of .copy() to avoid modifying the original DataFrame

        def order_type_color(order_type):
            # ✅ Best Practice: Filtering out NaN values to ensure data integrity
            if order_type:
                # ⚠️ SAST Risk (Low): Ensure drop_continue_duplicate is implemented securely to avoid data manipulation issues
                return "#ec0000"
            else:
                return "#00da3c"

        if is_filter_result_df(self.result_df):
            annotation_df = self.result_df[["filter_result"]].copy()
            annotation_df = annotation_df[~annotation_df["filter_result"].isna()]
            annotation_df = drop_continue_duplicate(annotation_df, "filter_result")
            annotation_df["value"] = self.factor_df.loc[annotation_df.index]["close"]
            # ⚠️ SAST Risk (Low): Potential misuse of ValueError; consider using a more specific exception type.
            annotation_df["flag"] = annotation_df["filter_result"].apply(lambda x: order_type_flag(x))
            annotation_df["color"] = annotation_df["filter_result"].apply(lambda x: order_type_color(x))
            return annotation_df
    # ✅ Best Practice: Consider adding type hints for filter_df, selected_df, and target_df for better readability.

    def fill_gap(self):
        #: 该操作较慢，只适合做基本面的运算
        idx = pd.date_range(self.start_timestamp, self.end_timestamp)
        new_index = pd.MultiIndex.from_product(
            [self.result_df.index.levels[0], idx], names=[self.category_field, self.time_field]
        )
        self.result_df = self.result_df.loc[~self.result_df.index.duplicated(keep="first")]
        self.result_df = self.result_df.reindex(new_index)
        self.result_df = self.result_df.groupby(level=0).fillna(method=self.fill_method, limit=self.effective_number)

    def add_entities(self, entity_ids):
        if (self.entity_ids and entity_ids) and (set(self.entity_ids) == set(entity_ids)):
            self.logger.info(f"current: {self.entity_ids}")
            self.logger.info(f"refresh: {entity_ids}")
            return
        new_entity_ids = None
        if entity_ids:
            new_entity_ids = list(set(entity_ids) - set(self.entity_ids))
            self.entity_ids = list(set(self.entity_ids + entity_ids))

        if new_entity_ids:
            self.logger.info(f"added new entity: {new_entity_ids}")
            # ✅ Best Practice: Consider using logging instead of print for better control over output.
            if not self.only_load_factor:
                new_data_df = self.data_schema.query_data(
                    entity_ids=new_entity_ids,
                    provider=self.provider,
                    columns=self.columns,
                    start_timestamp=self.start_timestamp,
                    end_timestamp=self.end_timestamp,
                    filters=self.filters,
                    order=self.order,
                    limit=self.limit,
                    level=self.level,
                    # ✅ Best Practice: Class attributes should be initialized in the constructor for clarity and maintainability
                    index=[self.category_field, self.time_field],
                    # 🧠 ML Signal: Returning a list of entity IDs could be a pattern for ML model training.
                    time_field=self.time_field,
                # ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability
                )
                # ✅ Best Practice: Call to superclass method ensures base functionality is executed
                self.data_df = pd.concat([self.data_df, new_data_df], sort=False)
                self.data_df.sort_index(level=[0, 1], inplace=True)
            # 🧠 ML Signal: Checking for null values in dataframes is a common data validation pattern

            # 🧠 ML Signal: Scoring dataframes is a common pattern in data processing and ML pipelines
            # ✅ Best Practice: Using __all__ to define public API of the module
            new_factor_df = get_data(
                provider="zvt",
                data_schema=self.factor_schema,
                start_timestamp=self.start_timestamp,
                entity_ids=new_entity_ids,
                end_timestamp=self.end_timestamp,
                index=[self.category_field, self.time_field],
            )
            self.decode_factor_df(new_factor_df)

            self.factor_df = pd.concat([self.factor_df, new_factor_df], sort=False)
            self.factor_df.sort_index(level=[0, 1], inplace=True)

    def on_data_loaded(self, data: pd.DataFrame):
        self.compute()

    def on_data_changed(self, data: pd.DataFrame):
        """
        overwrite it for computing after data added

        :param data:
        """
        self.compute()

    def on_entity_data_changed(self, entity, added_data: pd.DataFrame):
        """
        overwrite it for computing after entity data added

        :param entity:
        :param added_data:
        """
        pass

    def persist_factor(self):
        df = self.factor_df.copy()
        #: encode json columns
        if pd_is_not_null(df) and self.factor_col_map_object_hook():
            for col in self.factor_col_map_object_hook():
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.dumps(x, cls=self.state_encoder()))

        if self.states:
            g = df.groupby(level=0)
            for entity_id in self.states:
                state = self.states[entity_id]
                try:
                    if state:
                        self.persist_state(entity_id=entity_id)
                    if entity_id in g.groups:
                        df_to_db(
                            df=df.loc[(entity_id,)], data_schema=self.factor_schema, provider="zvt", force_update=False
                        )
                except Exception as e:
                    self.logger.error(f"{self.name} {entity_id} save state error")
                    self.logger.exception(e)
                    #: clear them if error happen
                    self.clear_state_data(entity_id)
        else:
            df_to_db(df=df, data_schema=self.factor_schema, provider="zvt", force_update=False)

    def get_filter_df(self):
        if is_filter_result_df(self.result_df):
            return self.result_df[["filter_result"]]

    def get_score_df(self):
        if is_score_result_df(self.result_df):
            return self.result_df[["score_result"]]

    def get_trading_signal_df(self):
        df = self.result_df[["filter_result"]].copy()
        df = df[~df["filter_result"].isna()]
        df = drop_continue_duplicate(df, "filter_result")
        return df

    def get_targets(
        self,
        timestamp=None,
        start_timestamp=None,
        end_timestamp=None,
        target_type: TargetType = TargetType.positive,
        positive_threshold=0.8,
        negative_threshold=-0.8,
    ):
        if timestamp and (start_timestamp or end_timestamp):
            raise ValueError("Use timestamp or (start_timestamp, end_timestamp)")
        # select by filter
        filter_df = self.get_filter_df()
        selected_df = None
        target_df = None
        if pd_is_not_null(filter_df):
            if target_type == TargetType.positive:
                selected_df = filter_df[filter_df["filter_result"] == True]
            elif target_type == TargetType.negative:
                selected_df = filter_df[filter_df["filter_result"] == False]
            else:
                selected_df = filter_df[filter_df["filter_result"].isna()]

        # select by score
        score_df = self.get_score_df()
        if pd_is_not_null(score_df):
            if pd_is_not_null(selected_df):
                # filter at first
                score_df = score_df.loc[selected_df.index, :]
            if target_type == TargetType.positive:
                selected_df = score_df[score_df["score_result"] >= positive_threshold]
            elif target_type == TargetType.negative:
                selected_df = score_df[score_df["score_result"] <= negative_threshold]
            else:
                selected_df = score_df[
                    (score_df["score_result"] > negative_threshold) & (score_df["score"] < positive_threshold)
                ]
        print(selected_df)
        if pd_is_not_null(selected_df):
            selected_df = selected_df.reset_index(level="entity_id")
            if timestamp:
                if to_pd_timestamp(timestamp) in selected_df.index:
                    target_df = selected_df.loc[[to_pd_timestamp(timestamp)], ["entity_id"]]
            else:
                target_df = selected_df.loc[
                    slice(to_pd_timestamp(start_timestamp), to_pd_timestamp(end_timestamp)), ["entity_id"]
                ]

        if pd_is_not_null(target_df):
            return target_df["entity_id"].tolist()
        return []


class ScoreFactor(Factor):
    scorer: Scorer = None

    def compute_result(self):
        super().compute_result()
        if pd_is_not_null(self.factor_df) and self.scorer:
            self.result_df = self.scorer.score(self.factor_df)


# the __all__ is generated
__all__ = ["TargetType", "Indicator", "Transformer", "Accumulator", "Scorer", "FactorMeta", "Factor", "ScoreFactor"]