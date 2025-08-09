# -*- coding: utf-8 -*-
from typing import List

import dash_daq as daq
from dash import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from zvt.contract import Mixin
from zvt.contract import zvt_context, IntervalLevel
from zvt.contract.api import get_entities, get_schema_by_name, get_schema_columns
from zvt.contract.drawer import StackedDrawer
from zvt.trader.trader_info_api import AccountStatsReader, OrderReader, get_order_securities
from zvt.trader.trader_info_api import get_trader_info
from zvt.trader.trader_schemas import TraderInfo
# ✅ Best Practice: Initialize lists to store account and order readers for later use
from zvt.ui import zvt_app
from zvt.ui.components.dcc_components import get_account_stats_figure
# ✅ Best Practice: Initialize lists to store account and order readers for later use
from zvt.utils.pd_utils import pd_is_not_null

# 🧠 ML Signal: Usage of type hinting for lists of custom objects
# 🧠 ML Signal: Function with conditional logic based on input values
account_readers = []
order_readers = []
# 🧠 ML Signal: Usage of type hinting for lists of strings
# ✅ Best Practice: Use of clear and descriptive variable names

# init the data
traders: List[TraderInfo] = []
# 🧠 ML Signal: Function uses conditional logic to determine return value based on input

trader_names: List[str] = []
# 🧠 ML Signal: Checks specific string values to determine behavior


# 🧠 ML Signal: Returns a specific color code based on condition
def order_type_flag(order_type):
    # ⚠️ SAST Risk (Medium): Use of global variables can lead to unexpected behavior and make the code harder to maintain.
    if order_type == "order_long" or order_type == "order_close_short":
        return "B"
    # 🧠 ML Signal: Returns a different color code for other conditions
    else:
        return "S"
# 🧠 ML Signal: Function call with specific parameter value, indicating a pattern of usage.


# ✅ Best Practice: Clearing lists before populating them to avoid stale data.
def order_type_color(order_type):
    if order_type == "order_long" or order_type == "order_close_short":
        return "#ec0000"
    else:
        return "#00da3c"
# ✅ Best Practice: Using list comprehension for readability and efficiency.


def load_traders():
    # ✅ Best Practice: Using list comprehension for readability and efficiency.
    # 🧠 ML Signal: Function call without parameters, indicating a pattern of usage.
    global traders
    global trader_names

    traders = get_trader_info(return_type="domain")
    account_readers.clear()
    order_readers.clear()
    for trader in traders:
        account_readers.append(AccountStatsReader(level=trader.level, trader_names=[trader.trader_name]))
        order_readers.append(
            OrderReader(start_timestamp=trader.start_timestamp, level=trader.level, trader_names=[trader.trader_name])
        )

    trader_names = [item.trader_name for item in traders]


load_traders()


def factor_layout():
    layout = html.Div(
        [
            # controls
            html.Div(
                className="three columns card",
                children=[
                    html.Div(
                        className="bg-white user-control",
                        children=[
                            html.Div(
                                className="padding-top-bot",
                                children=[
                                    html.H6("select trader:"),
                                    dcc.Dropdown(
                                        id="trader-selector",
                                        placeholder="select the trader",
                                        options=[{"label": item, "value": i} for i, item in enumerate(trader_names)],
                                    ),
                                ],
                            ),
                            # select entity type
                            html.Div(
                                className="padding-top-bot",
                                children=[
                                    html.H6("select entity type:"),
                                    dcc.Dropdown(
                                        id="entity-type-selector",
                                        placeholder="select entity type",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in zvt_context.tradable_schema_map.keys()
                                        ],
                                        value="stock",
                                        clearable=False,
                                    ),
                                ],
                            ),
                            # select entity provider
                            html.Div(
                                className="padding-top-bot",
                                children=[
                                    html.H6("select entity provider:"),
                                    dcc.Dropdown(id="entity-provider-selector", placeholder="select entity provider"),
                                ],
                            ),
                            # select entity
                            html.Div(
                                className="padding-top-bot",
                                children=[
                                    html.H6("select entity:"),
                                    dcc.Dropdown(id="entity-selector", placeholder="select entity"),
                                ],
                            ),
                            # select levels
                            html.Div(
                                className="padding-top-bot",
                                children=[
                                    html.H6("select levels:"),
                                    dcc.Dropdown(
                                        id="levels-selector",
                                        options=[
                                            {"label": level.name, "value": level.value}
                                            for level in (IntervalLevel.LEVEL_1WEEK, IntervalLevel.LEVEL_1DAY)
                                        ],
                                        value="1d",
                                        multi=True,
                                    ),
                                ],
                            ),
                            # select factor
                            html.Div(
                                className="padding-top-bot",
                                children=[
                                    html.H6("select factor:"),
                                    dcc.Dropdown(
                                        id="factor-selector",
                                        placeholder="select factor",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in zvt_context.factor_cls_registry.keys()
                                        ],
                                        value="TechnicalFactor",
                                    ),
                                ],
                            ),
                            # select data
                            html.Div(
                                children=[
                                    html.Div(
                                        [
                                            html.H6(
                                                "related/all data to show in sub graph",
                                                style={"display": "inline-block"},
                                            ),
                                            daq.BooleanSwitch(
                                                id="data-switch",
                                                on=True,
                                                style={
                                                    "display": "inline-block",
                                                    "float": "right",
                                                    "vertical-align": "middle",
                                                    "padding": "8px",
                                                },
                                            ),
                                        # ⚠️ SAST Risk (Low): Ensure that the callback function properly handles all input values to prevent potential security issues.
                                        ],
                                    ),
                                    dcc.Dropdown(id="data-selector", placeholder="schema"),
                                ],
                                style={"padding-top": "12px"},
                            ),
                            # select properties
                            html.Div(
                                children=[dcc.Dropdown(id="schema-column-selector", placeholder="properties")],
                                style={"padding-top": "6px"},
                            ),
                        ],
                    )
                ],
            ),
            # ✅ Best Practice: Check if trader_index is not None to avoid potential errors when accessing traders list
            # Graph
            html.Div(
                # 🧠 ML Signal: Accessing a list element by index
                className="nine columns card-left",
                children=[
                    # ✅ Best Practice: Check if entity_type is falsy and assign a default value
                    html.Div(
                        id="trader-details",
                        className="bg-white",
                    # 🧠 ML Signal: Creating a list of dictionaries for options
                    ),
                    html.Div(id="factor-details"),
                # 🧠 ML Signal: Function call with keyword argument
                # 🧠 ML Signal: Accessing a dictionary with a method call
                ],
            ),
        ]
    )

    return layout

# 🧠 ML Signal: List comprehension to create options
# 🧠 ML Signal: Function call with keyword argument

@zvt_app.callback(
    [
        Output("trader-details", "children"),
        Output("entity-type-selector", "options"),
        Output("entity-provider-selector", "options"),
        Output("entity-selector", "options"),
    ],
    [
        # 🧠 ML Signal: List comprehension with string formatting
        Input("trader-selector", "value"),
        Input("entity-type-selector", "value"),
        Input("entity-provider-selector", "value"),
    ],
)
def update_trader_details(trader_index, entity_type, entity_provider):
    # ✅ Best Practice: Initialize account_stats to None when trader_index is None
    # 🧠 ML Signal: List comprehension to create options
    if trader_index is not None:
        # change entity_type options
        entity_type = traders[trader_index].entity_type
        if not entity_type:
            # 🧠 ML Signal: Accessing a dictionary with a method call
            entity_type = "stock"
        # ✅ Best Practice: Check for None to avoid potential errors when accessing properties or methods.
        entity_type_options = [{"label": entity_type, "value": entity_type}]
        # 🧠 ML Signal: List comprehension to create options

        # ✅ Best Practice: Use of conditional logic to determine which schemas to use based on 'related'.
        # 🧠 ML Signal: Function call with multiple keyword arguments
        # account stats
        account_stats = get_account_stats_figure(account_stats_reader=account_readers[trader_index])
        # 🧠 ML Signal: Accessing a dictionary to retrieve schemas based on entity type.

        providers = zvt_context.tradable_schema_map.get(entity_type).providers
        # 🧠 ML Signal: List comprehension with string formatting
        # 🧠 ML Signal: Accessing a default set of schemas when 'related' is False.
        entity_provider_options = [{"label": name, "value": name} for name in providers]

        # entities
        # 🧠 ML Signal: List comprehension to transform schema objects into a list of dictionaries.
        # 🧠 ML Signal: Function checks for a non-empty schema_name, indicating conditional logic based on input
        entity_ids = get_order_securities(trader_name=trader_names[trader_index])
        df = get_entities(
            # ⚠️ SAST Risk (Low): Raising an exception to prevent update, ensure this is handled properly in the application.
            # 🧠 ML Signal: Usage of a function to retrieve schema by name, indicating a pattern of data retrieval
            provider=entity_provider,
            # 🧠 ML Signal: Decorator usage for callback registration
            entity_type=entity_type,
            # 🧠 ML Signal: Specifying output and input for a callback
            # 🧠 ML Signal: Use of decorators to define a callback function in a Dash application.
            # 🧠 ML Signal: Usage of a function to retrieve columns from a schema, indicating a pattern of data processing
            # 🧠 ML Signal: List comprehension used to transform data, indicating a pattern of data transformation
            # ⚠️ SAST Risk (Low): Raises an exception to prevent update, could be misused if not handled properly
            entity_ids=entity_ids,
            columns=["entity_id", "code", "name"],
            index="entity_id",
        )
        entity_options = [
            {"label": f'{entity_id}({entity["name"]})', "value": entity_id} for entity_id, entity in df.iterrows()
        ]

        return account_stats, entity_type_options, entity_provider_options, entity_options
    else:
        entity_type_options = [{"label": name, "value": name} for name in zvt_context.tradable_schema_map.keys()]
        account_stats = None
        # ✅ Best Practice: Decorator used to define a callback, improving code organization and readability
        # 🧠 ML Signal: Output definition for a callback, indicating a pattern of UI updates
        # 🧠 ML Signal: Multiple inputs for a callback, indicating a pattern of event-driven programming
        providers = zvt_context.tradable_schema_map.get(entity_type).providers
        entity_provider_options = [{"label": name, "value": name} for name in providers]
        df = get_entities(
            provider=entity_provider, entity_type=entity_type, columns=["entity_id", "code", "name"], index="entity_id"
        )
        # ✅ Best Practice: Convert single string to list for consistent processing
        entity_options = [
            {"label": f'{entity_id}({entity["name"]})', "value": entity_id} for entity_id, entity in df.iterrows()
        # 🧠 ML Signal: State used in a callback, indicating a pattern of maintaining state across events
        ]
        return account_stats, entity_type_options, entity_provider_options, entity_options
# 🧠 ML Signal: Querying data based on dynamic schema and columns


@zvt_app.callback(
    Output("data-selector", "options"), [Input("entity-type-selector", "value"), Input("data-switch", "on")]
)
# 🧠 ML Signal: Copying data for further processing
def update_entity_selector(entity_type, related):
    if entity_type is not None:
        # 🧠 ML Signal: Filtering data based on entity_id
        if related:
            schemas = zvt_context.entity_map_schemas.get(entity_type)
        else:
            # 🧠 ML Signal: Creating new columns based on existing data
            schemas = zvt_context.schemas
        return [{"label": schema.__name__, "value": schema.__name__} for schema in schemas]
    raise dash.PreventUpdate()

# ✅ Best Practice: Debugging or logging output

@zvt_app.callback(Output("schema-column-selector", "options"), [Input("data-selector", "value")])
def update_column_selector(schema_name):
    if schema_name:
        schema = get_schema_by_name(name=schema_name)
        # ✅ Best Practice: Sorting levels for consistent processing
        cols = get_schema_columns(schema=schema)
        # 🧠 ML Signal: Dynamic instantiation of classes based on registry

        return [{"label": col, "value": col} for col in cols]
    raise dash.PreventUpdate()


@zvt_app.callback(
    # 🧠 ML Signal: Using a composite pattern for drawing
    Output("factor-details", "children"),
    [
        Input("factor-selector", "value"),
        Input("entity-type-selector", "value"),
        Input("entity-selector", "value"),
        Input("levels-selector", "value"),
        Input("schema-column-selector", "value"),
    ],
    [State("trader-selector", "value"), State("data-selector", "value")],
# 🧠 ML Signal: Dynamic instantiation of classes based on registry
)
def update_factor_details(factor, entity_type, entity, levels, columns, trader_index, schema_name):
    # 🧠 ML Signal: Adding supplementary data to drawer
    # 🧠 ML Signal: Assigning processed data for annotation
    # ⚠️ SAST Risk (Low): Raising an exception without additional context
    if factor and entity_type and entity and levels:
        sub_df = None
        # add sub graph
        if columns:
            if type(columns) == str:
                columns = [columns]
            columns = columns + ["entity_id", "timestamp"]
            schema: Mixin = get_schema_by_name(name=schema_name)
            sub_df = schema.query_data(entity_id=entity, columns=columns)

        # add trading signals as annotation
        annotation_df = None
        if trader_index is not None:
            order_reader = order_readers[trader_index]
            annotation_df = order_reader.data_df.copy()
            annotation_df = annotation_df[annotation_df.entity_id == entity].copy()
            if pd_is_not_null(annotation_df):
                annotation_df["value"] = annotation_df["order_price"]
                annotation_df["flag"] = annotation_df["order_type"].apply(lambda x: order_type_flag(x))
                annotation_df["color"] = annotation_df["order_type"].apply(lambda x: order_type_color(x))
            print(annotation_df.tail())

        if type(levels) is list and len(levels) >= 2:
            levels.sort()
            drawers = []
            for level in levels:
                drawers.append(
                    zvt_context.factor_cls_registry[factor](
                        entity_schema=zvt_context.tradable_schema_map[entity_type], level=level, entity_ids=[entity]
                    ).drawer()
                )
            stacked = StackedDrawer(*drawers)

            return dcc.Graph(id=f"{factor}-{entity_type}-{entity}", figure=stacked.draw_kline(show=False, height=900))
        else:
            if type(levels) is list:
                level = levels[0]
            else:
                level = levels
            drawer = zvt_context.factor_cls_registry[factor](
                entity_schema=zvt_context.tradable_schema_map[entity_type],
                level=level,
                entity_ids=[entity],
                need_persist=False,
            ).drawer()
            if pd_is_not_null(sub_df):
                drawer.add_sub_df(sub_df)
            if pd_is_not_null(annotation_df):
                drawer.annotation_df = annotation_df

            return dcc.Graph(id=f"{factor}-{entity_type}-{entity}", figure=drawer.draw_kline(show=False, height=800))
    raise dash.PreventUpdate()