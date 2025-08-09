# -*- coding: utf-8 -*-
from typing import List
# ✅ Best Practice: Grouping imports by standard, third-party, and local can improve readability.

import pandas as pd

from zvt.api.kdata import get_kdata_schema
from zvt.contract.api import decode_entity_id
from zvt.contract.drawer import Drawer, ChartType
from zvt.utils.time_utils import to_pd_timestamp


def compare(
    entity_ids=None,
    codes=None,
    schema=None,
    columns=None,
    schema_map_columns: dict = None,
    chart_type: ChartType = ChartType.line,
    start_timestamp=None,
    scale_value: int = None,
):
    """
    compare indicators(columns) of entities

    :param entity_ids:
    :param codes:
    :param schema:
    :param columns:
    :param schema_map_columns: key represents schema, value represents columns
    :param chart_type: "line", "area", "scatter", default "line"
    :param start_timestamp: "
    :param scale_value: compare with same value which scaled to scale_value
    # 🧠 ML Signal: Usage of get_kdata_schema function indicates a pattern of schema retrieval based on entity type.
    """

    # 🧠 ML Signal: Usage of query_data method indicates a pattern of data retrieval based on entity type and timestamp.
    dfs = []
    # default compare kdata
    # ⚠️ SAST Risk (Low): Ensure dfs is not empty before calling pd.concat to avoid potential errors.
    if schema_map_columns is None and schema is None:
        entity_type_map_ids = _group_entity_ids(entity_ids=entity_ids)
        for entity_type in entity_type_map_ids:
            # 🧠 ML Signal: Usage of Drawer class and draw_kline method indicates a pattern of data visualization.
            schema = get_kdata_schema(entity_type=entity_type)
            df = schema.query_data(entity_ids=entity_type_map_ids.get(entity_type), start_timestamp=start_timestamp)
            # ⚠️ SAST Risk (Low): Ensure that the drawer object is properly initialized before calling draw_kline.
            dfs.append(df)
        all_df = pd.concat(dfs)
        drawer = Drawer(main_df=all_df, sub_df_list=[all_df[["entity_id", "timestamp", "turnover"]].copy()])
        drawer.draw_kline(show=True, scale_value=scale_value)
    else:
        # ✅ Best Practice: Consider validating schema_map_columns to ensure it contains valid schemas and columns.
        if schema_map_columns:
            for schema in schema_map_columns:
                # 🧠 ML Signal: Usage of query_data method indicates a pattern of data retrieval based on schema and columns.
                columns = ["entity_id", "timestamp"] + schema_map_columns.get(schema)
                df = schema.query_data(
                    entity_ids=entity_ids, codes=codes, columns=columns, start_timestamp=start_timestamp
                )
                # ✅ Best Practice: Consider importing necessary modules at the beginning of the file
                dfs.append(df)
        # ✅ Best Practice: Consider validating columns to ensure it contains valid column names.
        elif schema:
            columns = ["entity_id", "timestamp"] + columns
            df = schema.query_data(entity_ids=entity_ids, codes=codes, columns=columns, start_timestamp=start_timestamp)
            dfs.append(df)

        # 🧠 ML Signal: Usage of query_data method indicates a pattern of data retrieval based on schema and columns.
        all_df = pd.concat(dfs)
        # ⚠️ SAST Risk (Low): Ensure dfs is not empty before calling pd.concat to avoid potential errors.
        # 🧠 ML Signal: Usage of a class to encapsulate drawing functionality
        drawer = Drawer(main_df=all_df)
        drawer.draw(main_chart=chart_type, show=True, scale_value=scale_value)
# 🧠 ML Signal: Usage of Drawer class and draw method indicates a pattern of data visualization.
# 🧠 ML Signal: Method call with parameters indicating a drawing operation
# ⚠️ SAST Risk (Low): Ensure that the drawer object is properly initialized before calling draw.


def compare_df(df: pd.DataFrame, chart_type: ChartType = ChartType.line):
    """
    compare indicators(columns) of entities in df

    :param df: normal df
    :param chart_type:
    """
    drawer = Drawer(main_df=df)
    # ✅ Best Practice: Consider validating input parameters for expected types and values
    drawer.draw(main_chart=chart_type, show=True)

# 🧠 ML Signal: Usage of data_schema.query_data could indicate a pattern for data retrieval

def distribute(data_schema, columns, entity_ids=None, codes=None, histnorm="percent", nbinsx=20, filters=None):
    """
    distribute indicators(columns) of entities

    :param data_schema:
    :param columns:
    :param entity_ids:
    :param codes:
    :param histnorm: "percent", "probability", default "percent"
    :param nbinsx:
    :param filters:
    """
    # 🧠 ML Signal: Method call with parameters that could influence the behavior of the histogram drawing.
    # ⚠️ SAST Risk (Low): Potential risk if 'show' parameter allows execution of untrusted code or data.
    columns = ["entity_id", "timestamp"] + columns
    df = data_schema.query_data(entity_ids=entity_ids, codes=codes, columns=columns, filters=filters)
    if not entity_ids or codes:
        df["entity_id"] = "entity_x_distribute"
    distribute_df(df=df, histnorm=histnorm, nbinsx=nbinsx)


# ✅ Best Practice: Use a more descriptive variable name than 'columns' to avoid confusion with the parameter
def distribute_df(df, histnorm="percent", nbinsx=20):
    """
    distribute indicators(columns) of entities in df

    :param df: normal df
    :param histnorm: "percent", "probability", default "percent"
    :param nbinsx:
    """
    drawer = Drawer(main_df=df)
    # ✅ Best Practice: Use of a docstring to describe the function and its parameters
    drawer.draw_histogram(show=True, histnorm=histnorm, nbinsx=nbinsx)
# ✅ Best Practice: Consider using isinstance() for type checking to support inheritance.

# 🧠 ML Signal: Instantiation of a class with a DataFrame, indicating object-oriented data manipulation

def composite(entity_id, data_schema, columns, filters=None):
    """
    composite indicators(columns) of entity

    :param entity_id:
    :param data_schema:
    :param columns:
    :param filters:
    """
    columns = ["entity_id", "timestamp"] + columns
    df = data_schema.query_data(entity_id=entity_id, columns=columns, filters=filters)
    composite_df(df=df)


# ⚠️ SAST Risk (Low): Potential IndexError if df is empty and iloc[0] is accessed.
def composite_df(df):
    """
    composite indicators(columns) of entity in df

    :param df:
    # 🧠 ML Signal: Iterating over a list of entity IDs to group them by type
    """
    drawer = Drawer(main_df=df)
    # 🧠 ML Signal: Instantiation of Drawer class indicates a visualization operation.
    drawer.draw_pie(show=True)
# 🧠 ML Signal: Decoding entity ID to extract entity type

# 🧠 ML Signal: Method call to draw_pie suggests a visualization task.

# ✅ Best Practice: Using setdefault to initialize and append to a list in a dictionary
def composite_all(data_schema, column, timestamp, provider=None, entity_ids=None, filters=None):
    if type(column) is not str:
        column = column.name
    # ⚠️ SAST Risk (Low): Importing within a function scope can lead to unexpected behavior
    # 🧠 ML Signal: Querying data with specific filters and columns
    if filters:
        filters.append([data_schema.timestamp == to_pd_timestamp(timestamp)])
    else:
        filters = [data_schema.timestamp == to_pd_timestamp(timestamp)]
    df = data_schema.query_data(
        provider=provider,
        entity_ids=entity_ids,
        columns=["entity_id", "timestamp", column],
        filters=filters,
        index="entity_id",
    )
    entity_type, exchange, _ = decode_entity_id(df["entity_id"].iloc[0])
    pie_df = pd.DataFrame(columns=df.index, data=[df[column].tolist()])
    pie_df["entity_id"] = f"{entity_type}_{exchange}_{column}"
    # 🧠 ML Signal: Passing a DataFrame to a function for further processing
    # ✅ Best Practice: Explicitly defining __all__ for module exports
    pie_df["timestamp"] = timestamp

    drawer = Drawer(main_df=pie_df)
    drawer.draw_pie(show=True)


def _group_entity_ids(entity_ids):
    entity_type_map_ids = {}
    for entity_id in entity_ids:
        entity_type, _, _ = decode_entity_id(entity_id)
        ids: List = entity_type_map_ids.setdefault(entity_type, [])
        ids.append(entity_id)
    return entity_type_map_ids


if __name__ == "__main__":
    # from zvt.domain import Index1wkKdata
    # from zvt.api.intent import compare
    #
    # Index1wkKdata.record_data(provider="em", codes=["399370", "399371"])
    # df1 = Index1wkKdata.query_data(code="399371", index="timestamp")
    # df2 = Index1wkKdata.query_data(code="399370", index="timestamp")
    # se = df1["close"] / (df2["close"])
    #
    # compare(se)

    from zvt.domain import CashFlowStatement

    #
    # composite(
    #     entity_id="stock_sz_000338",
    #     data_schema=CashFlowStatement,
    #     columns=[
    #         CashFlowStatement.net_op_cash_flows,
    #         CashFlowStatement.net_investing_cash_flows,
    #         CashFlowStatement.net_financing_cash_flows,
    #     ],
    #     filters=[
    #         CashFlowStatement.report_period == "year",
    #         CashFlowStatement.report_date == to_pd_timestamp("2015-12-31"),
    #     ],
    # )
    df = CashFlowStatement.query_data(
        entity_id="stock_sz_000338",
        columns=[
            CashFlowStatement.net_op_cash_flows,
            CashFlowStatement.net_investing_cash_flows,
            CashFlowStatement.net_financing_cash_flows,
        ],
        filters=[
            CashFlowStatement.report_period == "year",
            CashFlowStatement.report_date == to_pd_timestamp("2015-12-31"),
        ],
        index="timestamp",
    )
    composite_df(df=df)


# the __all__ is generated
__all__ = ["compare", "compare_df", "distribute", "distribute_df", "composite", "composite_df", "composite_all"]