# -*- coding: utf-8 -*-
# ✅ Best Practice: Group imports from the same module together for better readability.

from dash import dcc

# ✅ Best Practice: Group imports from the same module together for better readability.
from zvt.api.kdata import get_kdata_schema
from zvt.contract import zvt_context
from zvt.contract.api import decode_entity_id

# ✅ Best Practice: Group imports from the same module together for better readability.
from zvt.contract.drawer import Drawer

# 🧠 ML Signal: Function with conditional logic based on input value
from zvt.contract.reader import DataReader
from zvt.trader.trader_info_api import OrderReader, AccountStatsReader

# ✅ Best Practice: Group imports from the same module together for better readability.
# 🧠 ML Signal: Specific string values used to determine logic flow
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Returns specific color code based on condition


# ✅ Best Practice: Group imports from the same module together for better readability.
def order_type_color(order_type):
    # 🧠 ML Signal: Function uses conditional logic to map input to output, useful for learning decision boundaries
    if order_type == "order_long" or order_type == "order_close_short":
        # ✅ Best Practice: Group imports from the same module together for better readability.
        # 🧠 ML Signal: Default return value for unspecified conditions
        # 🧠 ML Signal: Specific string values are used to determine output, indicating categorical input
        return "#ec0000"
    else:
        return "#00da3c"


# ✅ Best Practice: Group imports from the same module together for better readability.
# 🧠 ML Signal: Default return value for non-matching conditions, useful for learning default behavior


def order_type_flag(order_type):
    # 🧠 ML Signal: Decoding entity_id to extract entity_type, which could be used to understand entity categorization patterns
    if order_type == "order_long" or order_type == "order_close_short":
        # ✅ Best Practice: Group imports from the same module together for better readability.
        return "B"
    # 🧠 ML Signal: Usage of get_kdata_schema to determine data schema based on entity type and other parameters
    else:
        return "S"


# ✅ Best Practice: Defaulting start_timestamp to order_reader's start_timestamp if not provided


# ✅ Best Practice: Defaulting end_timestamp to order_reader's end_timestamp if not provided
# 🧠 ML Signal: Instantiating DataReader with specific parameters, indicating data access patterns
def get_trading_signals_figure(
    order_reader: OrderReader,
    entity_id: str,
    start_timestamp=None,
    end_timestamp=None,
    adjust_type=None,
):
    entity_type, _, _ = decode_entity_id(entity_id)

    data_schema = get_kdata_schema(
        entity_type=entity_type, level=order_reader.level, adjust_type=adjust_type
    )
    if not start_timestamp:
        start_timestamp = order_reader.start_timestamp
    if not end_timestamp:
        end_timestamp = order_reader.end_timestamp
    kdata_reader = DataReader(
        data_schema=data_schema,
        entity_schema=zvt_context.tradable_schema_map.get(entity_type),
        # ⚠️ SAST Risk (Low): Potential for infinite loop if move_on does not handle timeout properly
        entity_ids=[entity_id],
        start_timestamp=start_timestamp,
        # 🧠 ML Signal: Copying data from order_reader, indicating data manipulation patterns
        end_timestamp=end_timestamp,
        level=order_reader.level,
        # 🧠 ML Signal: Filtering data based on entity_id, showing data selection patterns
    )
    # ✅ Best Practice: Consider adding type hints for the return type for better readability and maintainability.

    # generate the annotation df
    # ✅ Best Practice: Initialize variables before use to avoid potential reference errors.
    # 🧠 ML Signal: Creating new columns based on existing data, indicating feature engineering patterns
    order_reader.move_on(timeout=0)
    df = order_reader.data_df.copy()
    # ⚠️ SAST Risk (Low): Check if account_stats_reader is None to avoid potential AttributeError.
    # 🧠 ML Signal: Applying transformations to order_type to derive new features
    df = df[df.entity_id == entity_id].copy()
    if pd_is_not_null(df):
        # ✅ Best Practice: Printing the tail of the DataFrame for debugging or logging purposes
        # 🧠 ML Signal: Using Drawer to visualize data, indicating visualization patterns
        # 🧠 ML Signal: Drawing kline chart, showing visualization preferences
        # 🧠 ML Signal: Usage of a method from an object to generate a figure, indicating a pattern of data visualization.
        # 🧠 ML Signal: Iterating over a list of trader names to generate graphs, indicating a pattern of dynamic UI generation.
        # 🧠 ML Signal: Dynamic ID generation for UI components, useful for tracking user interactions.
        # ✅ Best Practice: Explicitly return the result to improve code clarity.
        df["value"] = df["order_price"]
        df["flag"] = df["order_type"].apply(lambda x: order_type_flag(x))
        df["color"] = df["order_type"].apply(lambda x: order_type_color(x))
    print(df.tail())

    drawer = Drawer(main_df=kdata_reader.data_df, annotation_df=df)
    return drawer.draw_kline(show=False, height=800)


def get_account_stats_figure(account_stats_reader: AccountStatsReader):
    graph_list = []

    # 账户统计曲线
    if account_stats_reader:
        fig = account_stats_reader.draw_line(show=False)

        for trader_name in account_stats_reader.trader_names:
            graph_list.append(
                dcc.Graph(id="{}-account".format(trader_name), figure=fig)
            )

    return graph_list
