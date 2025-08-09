# -*- coding: utf-8 -*-
# ✅ Best Practice: Use of Enum for defining a set of named values
import logging
from enum import Enum

# ✅ Best Practice: Use of typing for type hinting improves code readability and maintainability
from typing import List, Optional

# ✅ Best Practice: Use of numpy for numerical operations is efficient and widely accepted
import numpy as np
import pandas as pd

# ✅ Best Practice: Use of pandas for data manipulation is efficient and widely accepted
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ✅ Best Practice: Use of plotly for interactive plots is a good choice for visualization

from zvt.contract.api import decode_entity_id

# ✅ Best Practice: Use of make_subplots for creating complex plot layouts
from zvt.contract.data_type import Bean
from zvt.contract.normal_data import NormalData

# ⚠️ SAST Risk (Low): Importing from external modules can introduce security risks if not properly managed
from zvt.utils.decorator import to_string

# ⚠️ SAST Risk (Low): Importing from external modules can introduce security risks if not properly managed
from zvt.utils.pd_utils import pd_is_not_null

logger = logging.getLogger(__name__)
# ✅ Best Practice: Enum members should be in uppercase to follow Python naming conventions.
# ⚠️ SAST Risk (Low): Importing from external modules can introduce security risks if not properly managed


# ✅ Best Practice: Enum members should be in uppercase to follow Python naming conventions.
# ⚠️ SAST Risk (Low): Importing from external modules can introduce security risks if not properly managed
class ChartType(Enum):
    """
    Chart type enum
    """

    # ✅ Best Practice: Enum members should be in uppercase to follow Python naming conventions.
    # ✅ Best Practice: Use of logging for tracking and debugging

    # ✅ Best Practice: Enum members should be in uppercase to follow Python naming conventions.
    #: candlestick chart
    kline = "kline"
    #: line chart
    # ✅ Best Practice: Enum members should be in uppercase to follow Python naming conventions.
    line = "line"
    #: area chart
    area = "area"
    # ✅ Best Practice: Enum members should be in uppercase to follow Python naming conventions.
    # ✅ Best Practice: Consider adding type hints for the class attributes for better readability and maintainability.
    # ✅ Best Practice: Use of default None values allows for optional parameters.
    #: scatter chart
    scatter = "scatter"
    # ✅ Best Practice: Initializing instance variables in the constructor.
    # ✅ Best Practice: Leading underscore in variable name suggests it's intended for internal use.
    #: histogram chart
    histogram = "histogram"
    #: pie chart
    # ⚠️ SAST Risk (Low): Ensure the `to_string` decorator is safe and does not expose sensitive information.
    pie = "pie"
    #: bar chart
    bar = "bar"


# ✅ Best Practice: Use of keyword arguments with default values improves function flexibility and readability

_zvt_chart_type_map_scatter_mode = {
    ChartType.line: "lines",
    ChartType.area: "none",
    ChartType.scatter: "markers",
}


@to_string
class Rect(Bean):
    """
    rect struct with left-bottom(x0, y0), right-top(x1, y1)
    """

    # ✅ Best Practice: Use of default parameter values for flexibility and readability
    # 🧠 ML Signal: Use of a method that wraps another method call, indicating a potential pattern for method delegation
    def __init__(self, x0=None, y0=None, x1=None, y1=None) -> None:
        #: left-bottom x0
        self.x0 = x0
        # ✅ Best Practice: Delegating functionality to another method promotes code reuse and separation of concerns
        # ✅ Best Practice: Use of None as default values to allow for optional parameters
        # 🧠 ML Signal: Use of enum or constant for chart type
        # 🧠 ML Signal: Method chaining pattern with self.draw
        #: left-bottom y0
        self.y0 = y0
        #: right-top x1
        self.x1 = x1
        #: right-top y1
        self.y1 = y1


class Draw(object):
    def draw_kline(
        self,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        return self.draw(
            # ✅ Best Practice: Using named parameters with default values improves code readability and maintainability.
            # ✅ Best Practice: Use of **kwargs for extensibility
            main_chart=ChartType.kline,
            width=width,
            height=height,
            title=title,
            keep_ui_state=keep_ui_state,
            show=show,
            scale_value=scale_value,
            **kwargs,
        )

    # ✅ Best Practice: Use of default parameter values for flexibility and readability
    # 🧠 ML Signal: The use of a method to draw an area chart indicates a pattern of data visualization.
    def draw_line(
        self,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        # ✅ Best Practice: Delegating functionality to another method (`self.draw`) promotes code reuse and separation of concerns.
        # 🧠 ML Signal: Method chaining pattern with self.draw
        # ✅ Best Practice: Use of named arguments for clarity
        return self.draw(
            main_chart=ChartType.line,
            width=width,
            height=height,
            title=title,
            keep_ui_state=keep_ui_state,
            show=show,
            scale_value=scale_value,
            **kwargs,
        )

    # 🧠 ML Signal: Method signature with multiple optional parameters indicates flexibility in usage patterns
    # ✅ Best Practice: Use of default parameter values for optional parameters improves function usability

    def draw_area(
        self,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
        # 🧠 ML Signal: Delegating functionality to another method (self.draw) shows a pattern of code reuse
        # ✅ Best Practice: Use of named arguments improves code readability and maintainability
    ):
        return self.draw(
            main_chart=ChartType.area,
            width=width,
            height=height,
            title=title,
            keep_ui_state=keep_ui_state,
            show=show,
            scale_value=scale_value,
            **kwargs,
            # ✅ Best Practice: Consider providing default values for parameters to improve function usability
        )

    # 🧠 ML Signal: Method chaining pattern with self.draw can indicate a fluent interface design
    # 🧠 ML Signal: Use of enum-like pattern with ChartType.bar can indicate a fixed set of options

    def draw_scatter(
        self,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        return self.draw(
            main_chart=ChartType.scatter,
            width=width,
            height=height,
            title=title,
            keep_ui_state=keep_ui_state,
            # ✅ Best Practice: Consider adding type hints for function parameters and return type for better readability and maintainability.
            show=show,
            # 🧠 ML Signal: Method chaining pattern with self.draw can be used to identify similar method calls.
            # ✅ Best Practice: Using named arguments improves code readability and maintainability.
            # 🧠 ML Signal: Usage of ChartType.pie indicates a specific chart type being used, which can be a feature for ML models.
            scale_value=scale_value,
            **kwargs,
        )

    def draw_histogram(
        self,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        return self.draw(
            ChartType.histogram,
            width=width,
            # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
            height=height,
            title=title,
            keep_ui_state=keep_ui_state,
            show=show,
            scale_value=scale_value,
            **kwargs,
        )

    def draw_bar(
        self,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        return self.draw(
            ChartType.bar,
            width=width,
            height=height,
            # ⚠️ SAST Risk (Low): Method is not implemented, which may lead to runtime errors if called
            title=title,
            # ✅ Best Practice: Use of default parameter values for flexibility and ease of use
            keep_ui_state=keep_ui_state,
            show=show,
            scale_value=scale_value,
            **kwargs,
        )

    def draw_pie(
        self,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        return self.draw(
            ChartType.pie,
            width=width,
            height=height,
            title=title,
            keep_ui_state=keep_ui_state,
            show=show,
            scale_value=scale_value,
            **kwargs,
        )

    def draw(
        self,
        main_chart=ChartType.kline,
        sub_chart="bar",
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):

        raise NotImplementedError()

    def default_layout(
        self,
        main_chart=None,
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        **layout_params,
    ):
        if keep_ui_state:
            uirevision = True
        else:
            uirevision = None

        if main_chart == ChartType.histogram:
            xaxis = None
        else:
            xaxis = dict(
                linecolor="#BCCCDC",
                showgrid=False,
                showspikes=True,  # Show spike line for X-axis
                # Format spike
                spikethickness=2,
                spikedash="dot",
                spikecolor="#999999",
                spikemode="across",
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            # 🧠 ML Signal: Method instantiation pattern with multiple parameters
                            dict(
                                count=3, label="3m", step="month", stepmode="backward"
                            ),
                            # ✅ Best Practice: Consider using keyword arguments for clarity
                            # 🧠 ML Signal: Object creation with multiple data sources
                            # ✅ Best Practice: Use descriptive variable names for readability
                            # 🧠 ML Signal: Method chaining pattern for data retrieval
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(
                    visible=True,
                ),
                type="date",
                # 🧠 ML Signal: Method chaining pattern for data retrieval
            )
        # 🧠 ML Signal: Method chaining pattern for data retrieval

        return dict(
            showlegend=True,
            plot_bgcolor="#FFF",
            hovermode="x",
            hoverdistance=100,  # Distance to show hover label of data point
            spikedistance=1000,  # Distance to show spike
            # 🧠 ML Signal: Method chaining pattern for data retrieval
            uirevision=uirevision,
            height=height,
            # 🧠 ML Signal: Method chaining pattern for data retrieval
            width=width,
            title=title,
            # ✅ Best Practice: Using a method to encapsulate drawing logic improves code organization and reusability.
            # 🧠 ML Signal: The use of default parameters can indicate common usage patterns and preferences.
            # ✅ Best Practice: Explicit return of the created object
            # 🧠 ML Signal: The use of a method chain suggests a fluent interface pattern.
            yaxis=dict(
                autorange=True,
                fixedrange=False,
                zeroline=False,
                linecolor="#BCCCDC",
                showgrid=False,
            ),
            xaxis=xaxis,
            legend_orientation="h",
            hoverlabel={"namelength": -1},
            # ✅ Best Practice: Include type hints for better code readability and maintainability
            **layout_params,
        )


# ✅ Best Practice: Explicitly returning None improves code clarity
# ✅ Best Practice: Type hinting improves code readability and maintainability


# ✅ Best Practice: Explicitly returning None can improve code clarity
# ✅ Best Practice: Type hinting improves code readability and maintainability
class Drawable(object):
    def drawer(self):
        # ✅ Best Practice: Explicitly returning None can improve code clarity
        # ✅ Best Practice: Use of type hinting improves code readability and maintainability
        drawer = Drawer(
            main_df=self.drawer_main_df(),
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            # ✅ Best Practice: Explicitly returning None clarifies the function's behavior
            main_data=self.drawer_main_data(),
            factor_df_list=self.drawer_factor_df_list(),
            # ✅ Best Practice: Explicitly returning None can improve code clarity
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            factor_data_list=self.drawer_factor_data_list(),
            sub_df_list=self.drawer_sub_df_list(),
            # ✅ Best Practice: Explicitly returning None can improve code clarity
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            sub_data_list=self.drawer_sub_data_list(),
            sub_col_chart=self.drawer_sub_col_chart(),
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            # ✅ Best Practice: Explicitly returning None clarifies the function's behavior
            annotation_df=self.drawer_annotation_df(),
            rects=self.drawer_rects(),
            # ✅ Best Practice: Explicitly returning None can improve code clarity
            # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
        )
        return drawer

    # ✅ Best Practice: Explicitly returning None can improve code clarity
    # ✅ Best Practice: Class should inherit from object for compatibility with Python 2 and 3

    def draw(
        self,
        # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled with optimization flags
        main_chart=ChartType.kline,
        width=None,
        # 🧠 ML Signal: Type hinting usage indicates a pattern for static type checking
        height=None,
        # ✅ Best Practice: Calculate 'part' once to avoid repeated computation.
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        # 🧠 ML Signal: Use of f-strings for dynamic variable naming.
        return self.drawer().draw(
            main_chart=main_chart,
            width=width,
            height=height,
            title=title,
            keep_ui_state=keep_ui_state,
            show=show,
            scale_value=scale_value,
            **kwargs,
        )

    def drawer_main_df(self) -> Optional[pd.DataFrame]:
        return None

    # 🧠 ML Signal: Use of list comprehension for domain calculation.

    def drawer_main_data(self) -> Optional[NormalData]:
        return None

    def drawer_factor_df_list(self) -> Optional[List[pd.DataFrame]]:
        return None

    def drawer_factor_data_list(self) -> Optional[List[NormalData]]:
        return None

    def drawer_sub_df_list(self) -> Optional[List[pd.DataFrame]]:
        return None

    def drawer_sub_data_list(self) -> Optional[List[NormalData]]:
        return None

    def drawer_annotation_df(self) -> Optional[pd.DataFrame]:
        return None

    def drawer_rects(self) -> Optional[List[Rect]]:
        return None

    def drawer_sub_col_chart(self) -> Optional[dict]:
        return None


class StackedDrawer(Draw):
    def __init__(self, *drawers) -> None:
        super().__init__()
        assert len(drawers) > 1
        self.drawers: List[Drawer] = drawers

    def make_y_layout(self, index, total, start_index=1, domain_range=(0, 1)):
        part = (domain_range[1] - domain_range[0]) / total

        if index == 1:
            yaxis = "yaxis"
            y = "y"
        else:
            yaxis = f"yaxis{index}"
            y = f"y{index}"

        return (
            yaxis,
            y,
            dict(
                anchor="x",
                autorange=True,
                fixedrange=False,
                zeroline=False,
                linecolor="#BCCCDC",
                showgrid=False,
                domain=[
                    domain_range[0] + part * (index - start_index),
                    domain_range[0] + part * (index - start_index + 1),
                ],
            ),
        )

    def draw(
        self,
        main_chart=ChartType.kline,
        sub_chart="bar",
        width=None,
        # ✅ Best Practice: Ensure the base class 'Draw' is defined or imported before use
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        stacked_fig = go.Figure()

        total = len(self.drawers)
        start = 1
        domain_range = (0, 1)
        for drawer in self.drawers:
            if drawer.has_sub_plot():
                # ✅ Best Practice: Use of type hints for function parameters and return type
                domain_range = (0.2, 1)
                start = 2
                break
        for index, drawer in enumerate(self.drawers, start=start):
            traces, sub_traces = drawer.make_traces(
                main_chart=main_chart,
                sub_chart=sub_chart,
                scale_value=scale_value,
                **kwargs,
            )

            # fix sub traces as the bottom
            if sub_traces:
                yaxis, y, layout = self.make_y_layout(
                    index=1, total=1, domain_range=(0, 0.2)
                )
                # ✅ Best Practice: Lazy initialization of main_data if not provided
                # update sub_traces with yaxis
                for trace in sub_traces:
                    trace.yaxis = y
                stacked_fig.add_traces(sub_traces)
                # ✅ Best Practice: Lazy initialization of factor_data_list if not provided
                stacked_fig.layout[yaxis] = layout

            # make y layouts
            # 🧠 ML Signal: Iterating over a list to transform data
            yaxis, y, layout = self.make_y_layout(
                index=index, total=total, start_index=start, domain_range=domain_range
            )

            # ✅ Best Practice: Lazy initialization of sub_data_list if not provided
            stacked_fig.layout[yaxis] = layout

            # update traces with yaxis
            # 🧠 ML Signal: Iterating over a list to transform data
            for trace in traces:
                # 🧠 ML Signal: Method usage pattern for adding factor data to an object
                trace.yaxis = y
            stacked_fig.add_traces(traces)
            # ✅ Best Practice: Explicitly specifying the type of 'df' improves code readability and maintainability

            # 🧠 ML Signal: Conversion of DataFrame to NormalData object
            # ✅ Best Practice: Check if the list is initialized before appending to it
            # update shapes with yaxis
            if drawer.rects:
                # ✅ Best Practice: Initialize the list if it is not already initialized
                for rect in drawer.rects:
                    # 🧠 ML Signal: Method that takes a DataFrame as input, indicating data processing or transformation
                    stacked_fig.add_shape(
                        # 🧠 ML Signal: Appending data to a list, common pattern for data collection
                        type="rect",
                        # 🧠 ML Signal: Usage of a method that wraps a DataFrame in a custom class
                        x0=rect.x0,
                        # ✅ Best Practice: Check if sub_data_list is initialized before appending
                        y0=rect.y0,
                        x1=rect.x1,
                        # ✅ Best Practice: Initialize sub_data_list as an empty list if not already initialized
                        y1=rect.y1,
                        # ✅ Best Practice: Method name should be descriptive and follow naming conventions
                        line=dict(color="RoyalBlue", width=1),
                        # 🧠 ML Signal: Appending data to a list, common pattern in data handling
                        # fillcolor="LightSkyBlue",
                        # 🧠 ML Signal: Checks for non-empty sub_data_list, indicating data validation logic
                        yref=y,
                        # ⚠️ SAST Risk (Low): Potential AttributeError if sub_data_list is not a list or does not have an empty method
                    )

            # annotations
            if pd_is_not_null(drawer.annotation_df):
                # ✅ Best Practice: Use copy to avoid modifying the original dataframe
                stacked_fig.layout["annotations"] = annotations(
                    drawer.annotation_df, yref=y
                )

        stacked_fig.update_layout(
            self.default_layout(
                main_chart=main_chart,
                width=width,
                height=height,
                title=title,
                keep_ui_state=keep_ui_state,
            )
        )

        if show:
            stacked_fig.show()
        else:
            return stacked_fig


class Drawer(Draw):
    def __init__(
        self,
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide unexpected errors
        main_df: pd.DataFrame = None,
        factor_df_list: List[pd.DataFrame] = None,
        sub_df_list: pd.DataFrame = None,
        main_data: NormalData = None,
        factor_data_list: List[NormalData] = None,
        # 🧠 ML Signal: Pattern of converting data to list for plotting
        sub_data_list: NormalData = None,
        sub_col_chart: Optional[dict] = None,
        rects: List[Rect] = None,
        annotation_df: pd.DataFrame = None,
        scale_value: int = None,
    ) -> None:
        """

        :param main_df: df for main chart
        :param factor_df_list: list of factor df on main chart
        :param sub_df_list: df for sub chart under main chart
        :param main_data: NormalData wrap main_df,use either
        :param factor_data_list: list of NormalData wrap factor_df,use either
        :param sub_data_list: NormalData wrap sub_df,use either
        :param annotation_df:
        """

        #: 主图数据
        if main_data is None:
            main_data = NormalData(main_df)
        # 🧠 ML Signal: Pattern of converting data to list for plotting
        self.main_data: NormalData = main_data

        #: 主图因子
        if not factor_data_list and factor_df_list:
            factor_data_list = []
            # 🧠 ML Signal: Pattern of converting data to list for plotting
            for df in factor_df_list:
                factor_data_list.append(NormalData(df))
        #: 每一个df可能有多个column, 代表多个指标，对于连续型的，可以放在一个df里面
        #: 对于离散型的，比如一些特定模式的连线，放在多个df里面较好，因为index不同
        self.factor_data_list: List[NormalData] = factor_data_list

        #: 副图数据
        if not sub_data_list and sub_df_list:
            sub_data_list = []
            for df in sub_df_list:
                sub_data_list.append(NormalData(df))
        #: 每一个df可能有多个column, 代表多个指标，对于连续型的，可以放在一个df里面
        #: 对于离散型的，比如一些特定模式的连线，放在多个df里面较好，因为index不同
        self.sub_data_list: List[NormalData] = sub_data_list

        #: 幅图col对应的图形，line or bar
        self.sub_col_chart = sub_col_chart

        #: 主图的标记数据
        self.annotation_df = annotation_df

        #: list of rect
        # ⚠️ SAST Risk (Low): Using assert for control flow can be disabled in optimized mode
        self.rects = rects

        self.scale_value = scale_value

    def add_factor_df(self, df: pd.DataFrame):
        self.add_factor_data(NormalData(df))

    def add_factor_data(self, data: NormalData):
        if not self.factor_data_list:
            self.factor_data_list = []
        # 🧠 ML Signal: Pattern of converting data to list for plotting
        self.factor_data_list.append(data)

    def add_sub_df(self, df: pd.DataFrame):
        self.add_sub_data(NormalData(df))

    def add_sub_data(self, data: NormalData):
        if not self.sub_data_list:
            self.sub_data_list = []
        # 🧠 ML Signal: Function that maps input to categorical output
        self.sub_data_list.append(data)

    def has_sub_plot(self):
        return self.sub_data_list is not None and not self.sub_data_list[0].empty()

    # 🧠 ML Signal: Pattern of converting data to list for plotting
    # 🧠 ML Signal: List comprehension used for data transformation
    def make_traces(
        self,
        main_chart=ChartType.kline,
        sub_chart="bar",
        yaxis="y",
        scale_value=None,
        **kwargs,
    ):
        traces = []
        sub_traces = []

        # 🧠 ML Signal: Conditional logic based on object attribute
        for entity_id, df in self.main_data.entity_map_df.items():
            df = df.select_dtypes(np.number)
            df = df.copy()
            if scale_value:
                for col in df.columns:
                    first = None
                    # 🧠 ML Signal: Conditional logic to determine chart type
                    for i in range(0, len(df)):
                        first = df[col][i]
                        if first != 0:
                            break
                    if first == 0:
                        continue
                    # 🧠 ML Signal: Method uses a default parameter value
                    scale = scale_value / first
                    # ⚠️ SAST Risk (Low): Potential risk if sub_traces is not initialized
                    df[col] = df[col] * scale
            # ⚠️ SAST Risk (Low): Potential risk if traces or sub_traces are not initialized
            # 🧠 ML Signal: Iterating over a list of objects
            code = entity_id
            try:
                _, _, code = decode_entity_id(entity_id)
            except Exception:
                pass

            # 构造主图
            if main_chart == ChartType.bar:
                for col in df.columns:
                    # ✅ Best Practice: Use of update_shapes to apply consistent properties to all shapes
                    trace_name = "{}_{}".format(code, col)
                    ydata = df[col].values.tolist()
                    traces.append(
                        go.Bar(
                            x=df.index, y=ydata, name=trace_name, yaxis=yaxis, **kwargs
                        )
                    )
            elif main_chart == ChartType.kline:
                trace_name = "{}_kdata".format(code)
                trace = go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    close=df["close"],
                    low=df["low"],
                    high=df["high"],
                    name=trace_name,
                    # ✅ Best Practice: Consider adding type hints for function parameters for better readability and maintainability.
                    yaxis=yaxis,
                    **kwargs,
                )
                traces.append(trace)
            elif main_chart in [ChartType.scatter, ChartType.line, ChartType.area]:
                mode = _zvt_chart_type_map_scatter_mode.get(main_chart)
                # ✅ Best Practice: Use descriptive variable names for better readability.
                for col in df.columns:
                    trace_name = "{}_{}".format(code, col)
                    ydata = df[col].values.tolist()
                    traces.append(
                        go.Scatter(
                            x=df.index,
                            y=ydata,
                            mode=mode,
                            name=trace_name,
                            yaxis=yaxis,
                            **kwargs,
                        )
                    )
            elif main_chart == ChartType.histogram:
                for col in df.columns:
                    trace_name = "{}_{}".format(code, col)
                    x = df[col].tolist()
                    trace = go.Histogram(x=x, name=trace_name, **kwargs)
                    traces.append(trace)
                    annotation = [
                        dict(
                            entity_id=entity_id,
                            timestamp=x[-1],
                            value=0,
                            flag=f"{trace_name}:{x[-1]}",
                        )
                    ]
                    annotation_df = pd.DataFrame.from_records(
                        annotation, index=["entity_id", "timestamp"]
                    )
                    # ⚠️ SAST Risk (Low): Using fig.show() can lead to potential security risks if the figure contains sensitive data.
                    if pd_is_not_null(self.annotation_df):
                        self.annotation_df = pd.concat(
                            [self.annotation_df, annotation_df]
                        )
                    # 🧠 ML Signal: Usage of DataFrame index and columns to generate table headers
                    else:
                        self.annotation_df = annotation_df
            # 🧠 ML Signal: Accessing DataFrame index level values
            elif main_chart == ChartType.pie:
                for _, row in df.iterrows():
                    # 🧠 ML Signal: Accessing DataFrame index level values
                    # 🧠 ML Signal: Iterating over DataFrame columns to extract values
                    # ✅ Best Practice: Use of Plotly's go.Table for structured data visualization
                    traces.append(
                        go.Pie(
                            name=entity_id,
                            labels=df.columns.tolist(),
                            values=row.tolist(),
                            **kwargs,
                        )
                    )
            else:
                assert False

            # 构造主图指标
            if self.factor_data_list:
                for factor_data in self.factor_data_list:
                    if not factor_data.empty():
                        factor_df = factor_data.entity_map_df.get(entity_id)
                        factor_df = factor_df.select_dtypes(np.number)
                        if pd_is_not_null(factor_df):
                            for col in factor_df.columns:
                                trace_name = "{}_{}".format(code, col)
                                ydata = factor_df[col].values.tolist()

                                # ✅ Best Practice: Creating a new Plotly Figure object
                                # ✅ Best Practice: Adding traces to the Plotly figure
                                # ✅ Best Practice: Updating layout with dynamic parameters
                                line = go.Scatter(
                                    x=factor_df.index,
                                    y=ydata,
                                    mode="lines",
                                    name=trace_name,
                                    yaxis=yaxis,
                                    **kwargs,
                                )
                                traces.append(line)

            # 构造幅图
            if self.has_sub_plot():
                for sub_data in self.sub_data_list:
                    # ✅ Best Practice: Displaying the figure using Plotly's show method
                    # ✅ Best Practice: Check if the DataFrame is not null before proceeding
                    sub_df = sub_data.entity_map_df.get(entity_id)
                    if pd_is_not_null(sub_df):
                        sub_df = sub_df.select_dtypes(np.number)
                        # 🧠 ML Signal: Iterating over grouped DataFrame by level 0 (entity_id)
                        for col in sub_df.columns:
                            trace_name = "{}_{}".format(code, col)
                            # ✅ Best Practice: Check if the DataFrame is not null before proceeding
                            ydata = sub_df[col].values.tolist()
                            # 🧠 ML Signal: Iterating over DataFrame rows

                            def color(i):
                                if i > 0:
                                    # ✅ Best Practice: Use of 'in' to check for key existence in dictionary
                                    return "red"
                                # ✅ Best Practice: Rounding value for consistency
                                # 🧠 ML Signal: Appending dictionary to list for annotations
                                else:
                                    return "green"

                            colors = [color(i) for i in ydata]

                            the_sub_chart = None
                            if self.sub_col_chart is not None:
                                the_sub_chart = self.sub_col_chart.get(col)
                            if not the_sub_chart:
                                the_sub_chart = sub_chart

                            if the_sub_chart == ChartType.line:
                                sub_trace = go.Scatter(
                                    x=sub_df.index,
                                    y=ydata,
                                    name=trace_name,
                                    yaxis="y2",
                                    marker=dict(color=colors),
                                )
                            else:
                                sub_trace = go.Bar(
                                    x=sub_df.index,
                                    y=ydata,
                                    name=trace_name,
                                    yaxis="y2",
                                    marker=dict(color=colors),
                                )
                            sub_traces.append(sub_trace)

        return traces, sub_traces

    # ✅ Best Practice: Use of __all__ to define public API of the module

    def add_rects(self, fig, yaxis="y"):
        if self.rects:
            for rect in self.rects:
                fig.add_shape(
                    type="rect",
                    x0=rect.x0,
                    y0=rect.y0,
                    x1=rect.x1,
                    y1=rect.y1,
                    line=dict(color="RoyalBlue", width=1),
                    # fillcolor="LightSkyBlue"
                )
            fig.update_shapes(dict(xref="x", yref=yaxis))

    def draw(
        self,
        main_chart=ChartType.kline,
        sub_chart="bar",
        width=None,
        height=None,
        title=None,
        keep_ui_state=True,
        show=False,
        scale_value=None,
        **kwargs,
    ):
        yaxis = "y"
        traces, sub_traces = self.make_traces(
            main_chart=main_chart,
            sub_chart=sub_chart,
            yaxis=yaxis,
            scale_value=scale_value,
            **kwargs,
        )

        if sub_traces:
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.8, 0.2],
                vertical_spacing=0.08,
                shared_xaxes=True,
            )
            fig.add_traces(traces, rows=[1] * len(traces), cols=[1] * len(traces))
            fig.add_traces(
                sub_traces, rows=[2] * len(sub_traces), cols=[1] * len(sub_traces)
            )
        else:
            fig = go.Figure()
            fig.add_traces(traces)

        # 绘制矩形
        self.add_rects(fig, yaxis=yaxis)

        fig.update_layout(
            self.default_layout(
                main_chart=main_chart,
                width=width,
                height=height,
                title=title,
                keep_ui_state=keep_ui_state,
            )
        )

        if sub_traces:
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.update_layout(
                xaxis2_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.1
            )
        # 绘制标志
        if pd_is_not_null(self.annotation_df):
            fig.layout["annotations"] = annotations(self.annotation_df, yref=yaxis)

        if show:
            fig.show()
        else:
            return fig

    def draw_table(
        self, width=None, height=None, title=None, keep_ui_state=True, **kwargs
    ):
        cols = (
            self.main_data.data_df.index.names + self.main_data.data_df.columns.tolist()
        )

        index1 = self.main_data.data_df.index.get_level_values(0).tolist()
        index2 = self.main_data.data_df.index.get_level_values(1).tolist()
        values = (
            [index1]
            + [index2]
            + [self.main_data.data_df[col] for col in self.main_data.data_df.columns]
        )

        data = go.Table(
            header=dict(
                values=cols,
                fill_color=["#000080", "#000080"]
                + ["#0066cc"] * len(self.main_data.data_df.columns),
                align="left",
                font=dict(color="white", size=13),
            ),
            cells=dict(values=values, fill=dict(color="#F5F8FF"), align="left"),
            **kwargs,
        )

        fig = go.Figure()
        fig.add_traces([data])
        fig.update_layout(
            self.default_layout(
                width=width, height=height, title=title, keep_ui_state=keep_ui_state
            )
        )

        fig.show()


def annotations(annotation_df: pd.DataFrame, yref="y"):
    """
    annotation_df format::

                                        value    flag    color
        entity_id    timestamp

    :param annotation_df:
    :param yref: specific yaxis e.g, y,y2,y3
    :return:
    """

    if pd_is_not_null(annotation_df):
        annotations = []
        for trace_name, df in annotation_df.groupby(level=0):
            if pd_is_not_null(df):
                for (_, timestamp), item in df.iterrows():
                    if "color" in item:
                        color = item["color"]
                    else:
                        color = "#ec0000"

                    value = round(item["value"], 2)
                    annotations.append(
                        dict(
                            x=timestamp,
                            y=value,
                            xref="x",
                            yref=yref,
                            text=item["flag"],
                            showarrow=True,
                            align="center",
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            # arrowcolor='#030813',
                            ax=-10,
                            ay=-30,
                            bordercolor="#c7c7c7",
                            borderwidth=1,
                            bgcolor=color,
                            opacity=0.8,
                        )
                    )
        return annotations
    return None


# the __all__ is generated
__all__ = [
    "ChartType",
    "Rect",
    "Draw",
    "Drawable",
    "StackedDrawer",
    "Drawer",
    "annotations",
]
