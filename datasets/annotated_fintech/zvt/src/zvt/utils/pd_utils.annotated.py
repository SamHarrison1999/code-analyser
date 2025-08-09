# -*- coding: utf-8 -*-
from typing import List, Union

# ✅ Best Practice: Consider using isinstance() instead of type() for type checking

import pandas as pd

# ✅ Best Practice: Consider using isinstance() instead of type() for type checking


# 🧠 ML Signal: Detecting changes in a time series or sequence
def drop_continue_duplicate(s: Union[pd.Series, pd.DataFrame], col=None):
    if type(s) == pd.Series:
        # ✅ Best Practice: Consider using isinstance() instead of type() for type checking
        return s[s.shift() != s]
    # 🧠 ML Signal: Function checks for specific column presence in DataFrame
    if type(s) == pd.DataFrame:
        # ⚠️ SAST Risk (Low): Potential KeyError if 'col' is not a valid column in the DataFrame
        # ✅ Best Practice: Function name should indicate it returns a boolean
        ss = s[col]
        # 🧠 ML Signal: Function checks for specific column presence in DataFrame
        selected = ss[ss.shift() != ss]
        # 🧠 ML Signal: Detecting changes in a time series or sequence
        # ⚠️ SAST Risk (Low): Assumes df is a valid DataFrame, may raise AttributeError if not
        # ✅ Best Practice: Use of descriptive function name for clarity
        return s.loc[selected.index, :]


# 🧠 ML Signal: Function checks for non-null and non-empty DataFrame or Series

# ✅ Best Practice: Use of helper function for null check increases readability
# ✅ Best Practice: Explicitly checking for None and emptiness improves code readability


# 🧠 ML Signal: Function that groups data by entity ID, indicating a common data processing pattern
def is_filter_result_df(df: pd.DataFrame):
    # ✅ Best Practice: Function definition without type hints for input and output
    return pd_is_not_null(df) and "filter_result" in df.columns


# 🧠 ML Signal: Function definition with specific parameter usage

# 🧠 ML Signal: Use of DataFrame's groupby method, a common operation in data analysis


# ⚠️ SAST Risk (Low): Assumes input_df has a multi-index with entity_id at level 0
# 🧠 ML Signal: Conditional check on DataFrame index levels
def is_score_result_df(df: pd.DataFrame):
    return pd_is_not_null(df) and "score_result" in df.columns


# 🧠 ML Signal: Function definition with specific parameters can indicate common data processing patterns
# ✅ Best Practice: Resetting index for DataFrame manipulation


# ✅ Best Practice: Checking the type or structure of input data before processing
def pd_is_not_null(df: Union[pd.DataFrame, pd.Series]):
    return df is not None and not df.empty


# ⚠️ SAST Risk (Low): Potential KeyError if 'filter_result' column does not exist in input_df


# 🧠 ML Signal: Function parameterization with default values
def group_by_entity_id(input_df: pd.DataFrame):
    return input_df.groupby(level=0)


# 🧠 ML Signal: Returning modified DataFrame is a common pattern in data manipulation functions
# 🧠 ML Signal: Conditional logic based on parameter presence


# ⚠️ SAST Risk (Low): Potential KeyError if time_field is not in df
def normalize_group_compute_result(group_result):
    if group_result.index.nlevels == 3:
        # 🧠 ML Signal: Conditional logic based on parameter value
        return group_result.reset_index(level=0, drop=True)
    return group_result


# ⚠️ SAST Risk (Low): Modifies the original DataFrame if inplace is True


def merge_filter_result(input_df: pd.DataFrame, filter_result: pd.Series):
    # ⚠️ SAST Risk (Low): Potential confusion with inplace parameter usage
    if is_filter_result_df(input_df):
        input_df["filter_result"] = input_df["filter_result"] & filter_result
    # ✅ Best Practice: Use isinstance() for type checking
    else:
        # 🧠 ML Signal: Function definition with default parameters
        input_df["filter_result"] = filter_result

    # ⚠️ SAST Risk (Low): Type checking using 'type' instead of 'isinstance'
    # ✅ Best Practice: Use isinstance() for type checking
    return input_df


# 🧠 ML Signal: Conversion of Series to DataFrame


def index_df(df, index="timestamp", inplace=True, drop=False, time_field="timestamp"):
    # 🧠 ML Signal: Use of list to define index
    if time_field:
        df[time_field] = pd.to_datetime(df[time_field])
    # 🧠 ML Signal: Function call to check if DataFrame is in a normal form

    if inplace:
        df.set_index(index, drop=drop, inplace=inplace)
    # 🧠 ML Signal: Checking the number of index levels
    else:
        df = df.set_index(index, drop=drop, inplace=inplace)
    # ⚠️ SAST Risk (Low): Use of assert statement for control flow

    # 🧠 ML Signal: Function checks for specific DataFrame structure
    if type(index) == str:
        df = df.sort_index()
    # 🧠 ML Signal: Adding a default category if not present
    # ⚠️ SAST Risk (Low): Assumes df is a pandas DataFrame without validation
    elif type(index) == list:
        # 🧠 ML Signal: Checks if DataFrame is not null and has a multi-level index
        df.index.names = index
        level = list(range(len(index)))
        # 🧠 ML Signal: Resetting index if time field is not present
        # 🧠 ML Signal: Extracts index names for further validation
        df = df.sort_index(level=level)
    # 🧠 ML Signal: Function definition with optional parameters
    return df


# 🧠 ML Signal: Validates index names against expected values

# ✅ Best Practice: Check if 'columns' is provided before using it
# 🧠 ML Signal: Function call with multiple parameters


def normal_index_df(
    df,
    category_field="entity_id",
    time_filed="timestamp",
    drop=True,
    default_entity="entity",
):
    # 🧠 ML Signal: Use of DataFrame's loc method to select columns
    # ✅ Best Practice: Consider adding type hints for the function's return type for better readability and maintainability.
    if type(df) == pd.Series:
        df = df.to_frame(name="value")
    # ✅ Best Practice: Return the original DataFrame if no columns are specified

    index = [category_field, time_filed]
    if is_normal_df(df):
        return df

    # ✅ Best Practice: Use `idx.union(df.index)` instead of `idx.append(df.index).drop_duplicates()` for better performance and readability.
    if df.index.nlevels == 1:
        if (time_filed != df.index.name) and (time_filed not in df.columns):
            # ✅ Best Practice: Sorting the index ensures consistent ordering, which is important for data processing.
            assert False
        if category_field not in df.columns:
            df[category_field] = default_entity
        if time_filed not in df.columns:
            # ✅ Best Practice: Using `difference` to find missing indices is a clear and efficient way to handle index alignment.
            df = df.reset_index()

    # ⚠️ SAST Risk (Low): Ensure that the DataFrame creation handles potential large memory usage if `added_index` is large.
    return index_df(df=df, index=index, drop=drop, time_field="timestamp")


# ✅ Best Practice: Using `__all__` to define public API of the module improves code readability and maintainability.
# ✅ Best Practice: Using `pd.concat` is a standard and efficient way to concatenate DataFrames.
# ✅ Best Practice: Sorting the DataFrame by index ensures that the data is in the expected order.


def is_normal_df(df, category_field="entity_id", time_filed="timestamp"):
    if pd_is_not_null(df) and df.index.nlevels == 2:
        names = df.index.names

        if len(names) == 2 and names[0] == category_field and names[1] == time_filed:
            return True

    return False


def df_subset(df, columns=None):
    if columns:
        return df.loc[:, columns]
    return df


def fill_with_same_index(df_list: List[pd.DataFrame]):
    idx = None
    for df in df_list:
        if idx is None:
            idx = df.index
        else:
            idx = idx.append(df.index).drop_duplicates()
    idx = idx.sort_values()

    result = []
    for df in df_list:
        # print(df[df.index.duplicated()])
        added_index = idx.difference(df.index.drop_duplicates())
        added_df = pd.DataFrame(index=added_index, columns=df.columns)

        # df1 = df.reindex(idx)
        # df1 = df.append(added_df)
        df1 = pd.concat([df, added_df])
        df1 = df1.sort_index()
        result.append(df1)
    return result


# the __all__ is generated
__all__ = [
    "drop_continue_duplicate",
    "is_filter_result_df",
    "is_score_result_df",
    "pd_is_not_null",
    "group_by_entity_id",
    "normalize_group_compute_result",
    "merge_filter_result",
    "index_df",
    "normal_index_df",
    "is_normal_df",
    "df_subset",
    "fill_with_same_index",
]
