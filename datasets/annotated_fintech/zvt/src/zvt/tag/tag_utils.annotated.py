# -*- coding: utf-8 -*-
import json
import os
from typing import List, Dict

# 🧠 ML Signal: Importing specific modules from a package indicates usage patterns and dependencies

import pandas as pd

# 🧠 ML Signal: Importing specific functions or classes from a module indicates usage patterns

from zvt import zvt_env

# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns
from zvt.contract.api import df_to_db

# ⚠️ SAST Risk (Low): Potential file path traversal if zvt_env["resource_path"] is user-controlled
from zvt.domain import Block

# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns
# ✅ Best Practice: Use a context manager to ensure the file is properly closed after reading
from zvt.tag.common import StockPoolType

# 🧠 ML Signal: Usage of os.path.join for constructing file paths
from zvt.tag.tag_schemas import (
    MainTagInfo,
    SubTagInfo,
    HiddenTagInfo,
    StockPoolInfo,
    IndustryInfo,
)

# ✅ Best Practice: Use of type hints for function return type improves code readability and maintainability.
# 🧠 ML Signal: Importing specific classes from a module indicates usage patterns
# 🧠 ML Signal: Reading JSON data from a file


# 🧠 ML Signal: Calls a function to get a mapping, indicating a pattern of data transformation.
def _get_default_industry_main_tag_mapping() -> Dict[str, str]:
    with open(
        os.path.join(zvt_env["resource_path"], "industry_main_tag_mapping.json"),
        encoding="utf-8",
    ) as f:
        return json.load(f)


# 🧠 ML Signal: Iterating over dictionary items, a common pattern for data processing.


# ✅ Best Practice: Use of setdefault to initialize a list if the key is not present.
# ✅ Best Practice: Use of type hinting for function return type improves code readability and maintainability
def _get_default_main_tag_industry_mapping() -> Dict[str, List[str]]:
    mapping = _get_default_industry_main_tag_mapping()
    # ✅ Best Practice: Use of get method to safely access dictionary values.
    # ⚠️ SAST Risk (Medium): Potential file path traversal vulnerability if zvt_env["resource_path"] is user-controlled
    result = {}
    # ✅ Best Practice: Include type hints for better code readability and maintainability
    # ✅ Best Practice: Use of context manager for file handling ensures proper resource management
    for industry, main_tag in mapping.items():
        result.setdefault(main_tag, [])
        # 🧠 ML Signal: Use of json.load to parse JSON data from a file
        # 🧠 ML Signal: Function calls can indicate common utility functions or patterns
        result.get(main_tag).append(industry)
    return result


# ✅ Best Practice: Use setdefault to simplify dictionary initialization
def _get_default_concept_main_tag_mapping() -> Dict[str, str]:
    # ✅ Best Practice: Use of type hinting for function return type improves code readability and maintainability.
    with open(
        os.path.join(zvt_env["resource_path"], "concept_main_tag_mapping.json"),
        encoding="utf-8",
    ) as f:
        # ✅ Best Practice: Use get method to safely access dictionary values
        return json.load(f)


# ✅ Best Practice: Using list() to explicitly convert keys to a list for clarity.
# 🧠 ML Signal: Use of query_data method to filter and order data

# ⚠️ SAST Risk (Low): Potential risk of SQL injection if filters are not properly sanitized
# ✅ Best Practice: Use of descriptive function name for clarity


def _get_default_main_tag_concept_mapping() -> Dict[str, List[str]]:
    mapping = _get_default_concept_main_tag_mapping()
    # 🧠 ML Signal: Filtering data based on category
    result = {}
    # 🧠 ML Signal: Selecting specific columns from the data
    # 🧠 ML Signal: Use of a private function naming convention with an underscore
    for concept, main_tag in mapping.items():
        # 🧠 ML Signal: Querying data with specific filters and ordering
        # 🧠 ML Signal: Returning data in a specific format (dataframe)
        # ✅ Best Practice: Use of method chaining for concise code
        result.setdefault(main_tag, [])
        result.get(main_tag).append(concept)
    return result


# ⚠️ SAST Risk (Low): Potential SQL injection risk if filters are not properly sanitized
# 🧠 ML Signal: Use of filters to query specific data

# 🧠 ML Signal: Filtering data based on a specific category
# ✅ Best Practice: Function name is descriptive and indicates its purpose


# 🧠 ML Signal: Selecting specific columns from the data
# 🧠 ML Signal: Specifying columns to retrieve
def _get_initial_sub_tags() -> List[str]:
    # 🧠 ML Signal: Specifying the return type of the query
    # 🧠 ML Signal: Usage of a function to retrieve a mapping, indicating a pattern of data retrieval
    return list(_get_default_concept_main_tag_mapping().keys())


# 🧠 ML Signal: Specifying return type for the query
# 🧠 ML Signal: Ordering data by a timestamp in descending order
# ✅ Best Practice: Function name is prefixed with an underscore, indicating it's intended for internal use.

# 🧠 ML Signal: Conversion to set for set operations, indicating a pattern of data comparison


# 🧠 ML Signal: Ordering data by timestamp in descending order
# 🧠 ML Signal: Usage of list and set operations to find differences, indicating a pattern of data manipulation
# 🧠 ML Signal: Usage of a function to retrieve a list of keys from a dictionary.
def _get_industry_list():
    # 🧠 ML Signal: Converting a DataFrame column to a list
    df = Block.query_data(
        # 🧠 ML Signal: Use of set operations to find differences between two lists.
        filters=[Block.category == "industry"],
        columns=[Block.name],
        return_type="df",
        order=Block.timestamp.desc(),
        # 🧠 ML Signal: Converting dataframe column to list
        # 🧠 ML Signal: Hardcoded entity_id could indicate a default or special user
    )
    # 🧠 ML Signal: Use of f-string for id generation shows dynamic key creation
    return df["name"].tolist()


def _get_concept_list():
    df = Block.query_data(
        filters=[Block.category == "concept"],
        columns=[Block.name],
        return_type="df",
        order=Block.timestamp.desc(),
    )

    return df["name"].tolist()


# 🧠 ML Signal: Use of f-string for tag_reason shows dynamic message creation


# 🧠 ML Signal: Iterating over items of a mapping function
def _check_missed_industry():
    current_industry_list = _get_default_industry_main_tag_mapping().keys()
    # 🧠 ML Signal: Iterating over items of a mapping function
    # ✅ Best Practice: Checking for existence in a dictionary before processing
    return list(set(_get_industry_list()) - set(current_industry_list))


def _check_missed_concept():
    current_concept_list = _get_default_concept_main_tag_mapping().keys()
    return list(set(_get_concept_list()) - set(current_concept_list))


def _get_initial_main_tag_info():
    # 🧠 ML Signal: Use of f-string for id generation shows dynamic key creation
    timestamp = "2024-03-25"
    # 🧠 ML Signal: Use of hardcoded timestamp and entity_id could indicate a pattern for default or initial data setup
    entity_id = "admin"

    from_industry = [
        # 🧠 ML Signal: Use of f-string for tag_reason shows dynamic message creation
        # ✅ Best Practice: Concatenating lists for a combined result
        {
            "id": f"{entity_id}_{main_tag}",
            "entity_id": entity_id,
            "timestamp": timestamp,
            "tag": main_tag,
            "tag_reason": f"来自这些行业:{industry}",
        }
        for main_tag, industry in _get_default_main_tag_industry_mapping().items()
    ]

    from_concept = []
    # 🧠 ML Signal: Use of dictionary comprehension to transform data
    for tag, concepts in _get_default_main_tag_concept_mapping().items():
        # 🧠 ML Signal: Function name with underscore prefix suggests internal or private use
        if tag not in _get_default_main_tag_industry_mapping():
            from_concept.append(
                {
                    # 🧠 ML Signal: Hardcoded entity_id could indicate a default or special user
                    # 🧠 ML Signal: Use of f-string for dynamic ID generation
                    "id": f"{entity_id}_{tag}",
                    "entity_id": entity_id,
                    "timestamp": timestamp,
                    "tag": tag,
                    "tag_reason": f"来自这些概念:{','.join(concepts)}",
                }
            )

    return from_industry + from_concept


def _get_initial_industry_info():
    # ✅ Best Practice: Using dictionary comprehension for readability and efficiency
    # 🧠 ML Signal: Use of hardcoded timestamp and entity_id for stock pool information
    timestamp = "2024-03-25"
    entity_id = "admin"
    # 🧠 ML Signal: Pattern of constructing unique IDs using entity_id and stock_pool_name
    industry_info = [
        {
            "id": f"{entity_id}_{industry}",
            "entity_id": entity_id,
            "timestamp": timestamp,
            "industry_name": industry,
            "description": industry,
            # ⚠️ SAST Risk (Low): Potential misuse of StockPoolType if not properly validated
            "main_tag": main_tag,
        }
        for industry, main_tag in _get_default_industry_main_tag_mapping().items()
        # 🧠 ML Signal: Iteration over a predefined list of stock pool names
    ]
    return industry_info


def _get_initial_sub_tag_info():
    timestamp = "2024-03-25"
    entity_id = "admin"
    # 🧠 ML Signal: Use of a dictionary to map stock tags to descriptions
    # ✅ Best Practice: Function name is prefixed with an underscore, indicating it's intended for internal use.

    return [
        # 🧠 ML Signal: Hardcoded timestamp could indicate a fixed point in time for data processing.
        {
            # 🧠 ML Signal: Hardcoded entity_id could indicate a specific user or role context.
            # 🧠 ML Signal: Use of f-string for dynamic ID generation.
            "id": f"{entity_id}_{sub_tag}",
            "entity_id": entity_id,
            "timestamp": timestamp,
            "tag": sub_tag,
            "tag_reason": sub_tag,
            "main_tag": main_tag,
        }
        for sub_tag, main_tag in _get_default_concept_main_tag_mapping().items()
    ]


# 🧠 ML Signal: Function definition with no parameters, indicating a possible utility or helper function


def _get_initial_stock_pool_info():
    # 🧠 ML Signal: Function call to a private or internal function, indicating encapsulation or modular design
    # ⚠️ SAST Risk (Medium): Potential risk if _hidden_tags is user-controlled or external input.
    timestamp = "2024-03-25"
    entity_id = "admin"
    # 🧠 ML Signal: Conversion of list to DataFrame, indicating data processing or transformation
    # 🧠 ML Signal: Function definition with no parameters, indicating a possible standard process or routine
    return [
        {
            # 🧠 ML Signal: Private function call pattern, indicating encapsulation or internal logic
            # 🧠 ML Signal: Function call with multiple parameters, indicating a data persistence or storage operation
            "id": f"{entity_id}_{stock_pool_name}",
            # ⚠️ SAST Risk (Low): Potential risk if df_to_db function does not handle data securely
            "entity_id": entity_id,
            # 🧠 ML Signal: Conversion of data to DataFrame, common in data processing tasks
            # 🧠 ML Signal: Function with a boolean parameter that alters behavior
            "timestamp": timestamp,
            "stock_pool_type": StockPoolType.system.value,
            # 🧠 ML Signal: Data persistence pattern, saving DataFrame to a database
            # 🧠 ML Signal: Function call pattern to retrieve data
            "stock_pool_name": stock_pool_name,
            # ⚠️ SAST Risk (Low): Potential risk if df_to_db does not handle SQL injection or data validation
        }
        # ✅ Best Practice: Consider adding type hints for the function parameters and return type for better readability and maintainability.
        # 🧠 ML Signal: Conversion of list to DataFrame
        for stock_pool_name in ["main_line", "vol_up", "大局", "all"]
    ]


# ⚠️ SAST Risk (Low): Potential risk if df_to_db does not handle SQL injection or data validation
# 🧠 ML Signal: Function calls a private function _get_initial_stock_pool_info, indicating encapsulation of logic.

# 🧠 ML Signal: Data persistence function call with parameters

# ✅ Best Practice: Consider adding a docstring to describe the function's purpose and behavior
# ⚠️ SAST Risk (Low): Ensure that stock_pool_info_list is properly validated before use to prevent potential data issues.
_hidden_tags = {
    # 🧠 ML Signal: Conversion of list to DataFrame, indicating data processing pattern.
    "中字头": "央企，国资委控股",
    # 🧠 ML Signal: Function call pattern to _get_initial_hidden_tag_info, indicating data retrieval
    "核心资产": "高ROE 高现金流 高股息 低应收 低资本开支 低财务杠杆 有增长",
    # ⚠️ SAST Risk (Low): Ensure df is sanitized before passing to df_to_db to prevent injection attacks.
    "高股息": "高股息",
    # 🧠 ML Signal: Interaction with a database, indicating data persistence pattern.
    # 🧠 ML Signal: Usage of pandas DataFrame, indicating data manipulation
    # 🧠 ML Signal: Function definition with a clear purpose and name
    "微盘股": "市值50亿以下",
    "次新股": "上市未满两年",
    # 🧠 ML Signal: Function call pattern to df_to_db, indicating data persistence
    # 🧠 ML Signal: Querying data from a database model
}
# ⚠️ SAST Risk (Low): Ensure df_to_db handles data securely to prevent injection attacks
# 🧠 ML Signal: Function definition with specific input and output patterns

# ✅ Best Practice: Returning a list of tags for better usability


# 🧠 ML Signal: Querying data with specific filters
def _get_initial_hidden_tag_info():
    # ✅ Best Practice: Check if list is not empty before accessing elements
    timestamp = "2024-03-25"
    entity_id = "admin"
    return [
        # 🧠 ML Signal: Function to retrieve main tag based on industry name
        # 🧠 ML Signal: Accessing the first element of a list
        {
            # ⚠️ SAST Risk (Low): Potential SQL injection if `industry_name` is not properly sanitized
            "id": f"{entity_id}_{tag}",
            "entity_id": entity_id,
            "timestamp": timestamp,
            # 🧠 ML Signal: Querying data based on industry name
            # ✅ Best Practice: Use of default value in dictionary get method
            "tag": tag,
            "tag_reason": tag_reason,
        }
        for tag, tag_reason in _hidden_tags.items()
        # 🧠 ML Signal: Function definition with no parameters, indicating a possible utility function
        # 🧠 ML Signal: Returning main tag if data is found
    ]


# 🧠 ML Signal: Querying a database model, indicating interaction with a database


# ⚠️ SAST Risk (Medium): Potential for SQL injection if input is not properly sanitized
# 🧠 ML Signal: Function definition with a specific purpose (get_hidden_tags)
# ✅ Best Practice: Use of default value when no data is found
def build_initial_main_tag_info():
    main_tag_info_list = _get_initial_main_tag_info()
    # 🧠 ML Signal: Accessing a DataFrame column and converting it to a list, indicating data transformation
    # 🧠 ML Signal: Querying data from a database model
    df = pd.DataFrame.from_records(main_tag_info_list)
    # 🧠 ML Signal: Function definition with a clear purpose of retrieving stock pool names
    df_to_db(df=df, data_schema=MainTagInfo, provider="zvt", force_update=False)


# ✅ Best Practice: Returning a specific column as a list

# 🧠 ML Signal: Querying a database model for specific columns


# 🧠 ML Signal: Function with multiple conditional branches based on input parameter
def build_initial_industry_info():
    # 🧠 ML Signal: Accessing a DataFrame column and converting it to a list
    initial_industry_info = _get_initial_industry_info()
    # 🧠 ML Signal: Default parameter value usage
    df = pd.DataFrame.from_records(initial_industry_info)
    df_to_db(df=df, data_schema=IndustryInfo, provider="zvt", force_update=False)


def build_initial_sub_tag_info(force_update=False):
    sub_tag_info_list = _get_initial_sub_tag_info()
    df = pd.DataFrame.from_records(sub_tag_info_list)
    df_to_db(df=df, data_schema=SubTagInfo, provider="zvt", force_update=force_update)


# ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled in production


def build_initial_stock_pool_info():
    stock_pool_info_list = _get_initial_stock_pool_info()
    df = pd.DataFrame.from_records(stock_pool_info_list)
    # 🧠 ML Signal: Use of set intersection to find common elements
    df_to_db(df=df, data_schema=StockPoolInfo, provider="zvt", force_update=False)


def build_initial_hidden_tag_info():
    hidden_tag_info_list = _get_initial_hidden_tag_info()
    # 🧠 ML Signal: Function uses a pattern of checking multiple conditions sequentially
    df = pd.DataFrame.from_records(hidden_tag_info_list)
    df_to_db(df=df, data_schema=HiddenTagInfo, provider="zvt", force_update=False)


def get_main_tags():
    df = MainTagInfo.query_data(columns=[MainTagInfo.tag])
    return df["tag"].tolist()


def get_main_tag_by_sub_tag(sub_tag):
    datas: List[SubTagInfo] = SubTagInfo.query_data(
        filters=[SubTagInfo.tag == sub_tag], return_type="domain"
    )
    if datas:
        return datas[0].main_tag
    # ✅ Best Practice: Explicitly defining the entry point for script execution
    # ✅ Best Practice: Using __all__ to define public API of the module
    else:
        return _get_default_concept_main_tag_mapping().get(sub_tag, "其他")


def get_main_tag_by_industry(industry_name):
    datas: List[IndustryInfo] = IndustryInfo.query_data(
        filters=[IndustryInfo.industry_name == industry_name], return_type="domain"
    )
    if datas:
        return datas[0].main_tag
    else:
        _get_default_industry_main_tag_mapping().get(industry_name, "其他")


def get_sub_tags():
    df = SubTagInfo.query_data(columns=[SubTagInfo.tag])
    return df["tag"].tolist()


def get_hidden_tags():
    df = HiddenTagInfo.query_data(columns=[HiddenTagInfo.tag])
    return df["tag"].tolist()


def get_stock_pool_names():
    df = StockPoolInfo.query_data(columns=[StockPoolInfo.stock_pool_name])
    return df["stock_pool_name"].tolist()


def match_tag_by_type(alias, tag_type="main_tag"):
    if tag_type == "main_tag":
        tags = get_main_tags()
    elif tag_type == "sub_tag":
        tags = get_sub_tags()
    elif tag_type == "industry":
        tags = _get_industry_list()
    else:
        assert False

    max_intersection_length = 0
    max_tag = None

    for tag in tags:
        intersection_length = len(set(alias) & set(tag))
        # at least 2 same chars
        if intersection_length < 2:
            continue

        if intersection_length > max_intersection_length:
            max_intersection_length = intersection_length
            max_tag = tag

    return max_tag


def match_tag(alias):
    tag = match_tag_by_type(alias, tag_type="main_tag")
    if tag:
        return "main_tag", tag

    tag = match_tag_by_type(alias, tag_type="sub_tag")
    if tag:
        return "sub_tag", tag

    tag = match_tag_by_type(alias, tag_type="industry")
    if tag:
        return "main_tag", get_main_tag_by_industry(tag)

    return "new_tag", alias


if __name__ == "__main__":
    # with open("missed_concept.json", "w") as json_file:
    #     json.dump(check_missed_concept(), json_file, indent=2, ensure_ascii=False)
    # with open("missed_industry.json", "w") as json_file:
    #     json.dump(check_missed_industry(), json_file, indent=2, ensure_ascii=False)
    # print(industry_to_main_tag("光伏设备"))
    # result = {}
    # for main_tag, concepts in get_main_tag_industry_mapping().items():
    #     for tag in concepts:
    #         result[tag] = main_tag
    # with open("industry_main_tag_mapping.json", "w") as json_file:
    #     json.dump(result, json_file, indent=2, ensure_ascii=False)
    # build_initial_stock_pool_info()
    # build_initial_main_tag_info()
    build_initial_sub_tag_info(force_update=True)
    build_initial_industry_info()


# the __all__ is generated
__all__ = [
    "build_initial_main_tag_info",
    "build_initial_industry_info",
    "build_initial_sub_tag_info",
    "build_initial_stock_pool_info",
    "build_initial_hidden_tag_info",
    "get_main_tags",
    "get_main_tag_by_sub_tag",
    "get_main_tag_by_industry",
    "get_sub_tags",
    "get_hidden_tags",
    "get_stock_pool_names",
    "match_tag_by_type",
    "match_tag",
]
