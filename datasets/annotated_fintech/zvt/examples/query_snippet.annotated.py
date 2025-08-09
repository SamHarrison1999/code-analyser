# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from sqlalchemy import func

from zvt.api.selector import get_entity_ids_by_filter
from zvt.contract import Exchange
# ✅ Best Practice: Importing specific functions or classes instead of the entire module can improve code clarity and reduce memory usage.
from zvt.domain import Stock, BlockStock
# ⚠️ SAST Risk (Medium): Using json_extract with user-controlled input can lead to SQL injection if not properly sanitized.
from zvt.recorders.em import em_api
# ✅ Best Practice: Consider using parameterized queries to prevent SQL injection.
from zvt.tag.tag_schemas import StockTags


# 🧠 ML Signal: Usage of JSON path expressions in queries can indicate complex data structures.
def query_json():
    # 🧠 ML Signal: Function definition with no parameters, indicating a potential pattern for data retrieval functions

    # ✅ Best Practice: Use logging instead of print statements for better control over output and log levels.
    df = StockTags.query_data(
        # 🧠 ML Signal: Querying a database table with specific filters and columns
        filters=[func.json_extract(StockTags.sub_tags, '$."低空经济"') != None], columns=[StockTags.sub_tags]
    # ⚠️ SAST Risk (Low): Potential for SQL injection if filters or columns are not properly sanitized
    # 🧠 ML Signal: Function definition with no parameters, indicating a utility function
    )
    print(df)
# 🧠 ML Signal: Converting a DataFrame column to a list, indicating a common data transformation pattern
# 🧠 ML Signal: Function call with specific parameters, indicating usage pattern


# 🧠 ML Signal: Function definition and naming pattern could be used to identify utility functions in codebases
# 🧠 ML Signal: Function call to retrieve existing tags, indicating a filtering operation
def get_stocks_has_tag():
    df = StockTags.query_data(filters=[StockTags.latest.is_(True)], columns=[StockTags.entity_id])
    # ✅ Best Practice: Using set operations to find differences, which is efficient for this purpose
    # ✅ Best Practice: Initialize variables before use
    return df["entity_id"].tolist()

# 🧠 ML Signal: API call pattern to external service

def get_stocks_without_tag():
    # 🧠 ML Signal: Use of list concatenation pattern
    entity_ids = get_entity_ids_by_filter(provider="em", ignore_delist=True, ignore_st=True, ignore_new_stock=False)
    stocks_has_tag = get_stocks_has_tag()
    # 🧠 ML Signal: API call pattern to external service
    return list(set(entity_ids) - set(stocks_has_tag))
# 🧠 ML Signal: Function with default parameter value

# 🧠 ML Signal: Use of list concatenation pattern

# 🧠 ML Signal: Querying data from a database
def get_all_delist_stocks():
    # 🧠 ML Signal: API call pattern to external service
    # ⚠️ SAST Risk (Low): Potential SQL injection if filters are not properly sanitized
    stocks = []
    # 🧠 ML Signal: Default parameter usage can indicate common or expected use cases.
    # 🧠 ML Signal: Use of list concatenation pattern
    # 🧠 ML Signal: Converting DataFrame column to list
    df1 = em_api.get_tradable_list(entity_type="stock", exchange=Exchange.sh)
    stocks = stocks + df1["entity_id"].tolist()
    df2 = em_api.get_tradable_list(entity_type="stock", exchange=Exchange.sz)
    stocks = stocks + df2["entity_id"].tolist()
    df3 = em_api.get_tradable_list(entity_type="stock", exchange=Exchange.bj)
    # ⚠️ SAST Risk (Low): Potential SQL injection risk if `tag` is user-controlled and not properly sanitized.
    # ✅ Best Practice: Return statement at the end of the function
    stocks = stocks + df3["entity_id"].tolist()
    return stocks
# ✅ Best Practice: Explicitly returning a specific column as a list improves code readability and clarity.
# 🧠 ML Signal: Usage of `print` for output can indicate debugging or logging behavior.


def get_block_stocks(name="低空经济"):
    df = BlockStock.query_data(provider="em", filters=[BlockStock.name == name], columns=[BlockStock.stock_id])
    return df["stock_id"].tolist()


def get_sub_tag_stocks(tag="低空经济"):
    df = StockTags.query_data(
        provider="zvt",
        filters=[func.json_extract(StockTags.sub_tags, f'$."{tag}"') != None],
        columns=[StockTags.entity_id],
    )
    return df["entity_id"].tolist()


if __name__ == "__main__":
    # a = get_block_stocks()
    # b = get_sub_tag_stocks()
    # print(set(a) - set(b))
    print(Stock.query_data(provider="em", return_type="dict"))