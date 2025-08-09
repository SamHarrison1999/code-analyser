# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ✅ Best Practice: Group standard library imports together at the top.

import math

# ✅ Best Practice: Group third-party library imports together.
import importlib
from typing import Iterable

import pandas as pd

# ✅ Best Practice: Class attributes should be documented to explain their purpose

import plotly.offline as py

# ✅ Best Practice: Use of a single underscore indicates intended private use
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot


class BaseGraph:
    _name = None

    def __init__(
        self,
        df: pd.DataFrame = None,
        layout: dict = None,
        graph_kwargs: dict = None,
        name_dict: dict = None,
        **kwargs,
    ):
        """

        :param df:
        :param layout:
        :param graph_kwargs:
        :param name_dict:
        :param kwargs:
            layout: dict
                go.Layout parameters
            graph_kwargs: dict
                Graph parameters, eg: go.Bar(**graph_kwargs)
        """
        self._df = df
        # 🧠 ML Signal: Method call indicating initialization or setup pattern.
        # ⚠️ SAST Risk (Low): The function assumes self._df is defined and has an 'empty' attribute, which may not be the case.

        self._layout = dict() if layout is None else layout
        self._graph_kwargs = dict() if graph_kwargs is None else graph_kwargs
        # 🧠 ML Signal: Method call pattern for data initialization
        self._name_dict = name_dict

        self.data = None

        # ✅ Best Practice: Use of self to access instance variables
        self._init_parameters(**kwargs)
        self._init_data()

    # ✅ Best Practice: Checking for None before initializing a dictionary

    def _init_data(self):
        """

        :return:
        """
        if self._df.empty:
            raise ValueError("df is empty.")
        # ⚠️ SAST Risk (Medium): No validation on 'graph_type' could lead to importing unintended modules or classes.

        self.data = self._get_data()

    # ⚠️ SAST Risk (Medium): Dynamic import using user-controlled input can lead to code execution vulnerabilities.

    def _init_parameters(self, **kwargs):
        """

        :param kwargs
        """
        # ✅ Best Practice: Specify the type of elements in the Iterable for better type hinting.

        # 🧠 ML Signal: Usage of dynamic class instantiation with kwargs.
        # ⚠️ SAST Risk (Medium): Using getattr with user-controlled input can lead to code execution vulnerabilities.
        # Instantiate graphics parameters
        self._graph_type = self._name.lower().capitalize()

        # Displayed column name
        if self._name_dict is None:
            # ✅ Best Practice: Initialize the notebook mode for Plotly to ensure compatibility with Jupyter notebooks.
            self._name_dict = {_item: _item for _item in self._df.columns}

    # ✅ Best Practice: Check if figure_list is not None to avoid TypeError.
    @staticmethod
    def get_instance_with_graph_parameters(graph_type: str = None, **kwargs):
        """

        :param graph_type:
        :param kwargs:
        :return:
        # ✅ Best Practice: Use of type hinting for the return type improves code readability and maintainability
        """
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide other issues; consider catching specific exceptions.
        try:
            _graph_module = importlib.import_module("plotly.graph_objs")
            _graph_class = getattr(_graph_module, graph_type)
        # ✅ Best Practice: Use the renderer parameter to specify the output environment for Plotly figures.
        except AttributeError:
            # 🧠 ML Signal: Use of a private method suggests encapsulation and internal API design
            _graph_module = importlib.import_module("qlib.contrib.report.graph")
            # ✅ Best Practice: Include a docstring that describes the return value and method purpose
            # 🧠 ML Signal: Returning a go.Layout object indicates usage of Plotly for visualization
            _graph_class = getattr(_graph_module, graph_type)
        return _graph_class(**kwargs)

    # 🧠 ML Signal: Use of list comprehension to transform data
    # 🧠 ML Signal: Method call with dynamic parameters
    @staticmethod
    def show_graph_in_notebook(figure_list: Iterable[go.Figure] = None):
        """

        :param figure_list:
        :return:
        """
        py.init_notebook_mode()
        for _fig in figure_list:
            # ✅ Best Practice: Add a descriptive docstring to explain the purpose and return value of the function
            # NOTE: displays figures: https://plotly.com/python/renderers/
            # default: plotly_mimetype+notebook
            # support renderers: import plotly.io as pio; print(pio.renderers)
            renderer = None
            # 🧠 ML Signal: Usage of Plotly's go.Figure, indicating data visualization
            try:
                # in notebook
                # ✅ Best Practice: Explicitly setting template to None for clarity
                _ipykernel = str(type(get_ipython()))
                # ✅ Best Practice: Use of a leading underscore in _name indicates it's intended for internal use.
                if "google.colab" in _ipykernel:
                    # ✅ Best Practice: Class should inherit from a base class to promote code reuse and maintainability
                    renderer = "colab"
            except NameError:
                # ✅ Best Practice: Use of a class attribute for a constant value
                pass
            # ✅ Best Practice: Class attribute _name is defined, which can be useful for identifying or categorizing instances.

            _fig.show(renderer=renderer)

    def _get_layout(self) -> go.Layout:
        """

        :return:
        # 🧠 ML Signal: Usage of dropna() indicates data cleaning, which is common in data preprocessing.
        """
        return go.Layout(**self._layout)

    # 🧠 ML Signal: Iterating over a dictionary to create a list of data columns is a common pattern in data processing.

    # ✅ Best Practice: Class should inherit from a base class to promote code reuse and maintainability
    def _get_data(self) -> list:
        """

        :return:
        """

        # ⚠️ SAST Risk (Low): Accessing dictionary keys without checking if they exist could lead to KeyError.
        # ✅ Best Practice: Consider providing a more detailed docstring explaining the return value and method purpose.
        # 🧠 ML Signal: Usage of instance method with specific parameters can indicate a pattern for ML models.
        # 🧠 ML Signal: Usage of class attributes to determine behavior.
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type,
                x=self._df.index,
                y=self._df[_col],
                name=_name,
                **self._graph_kwargs,
            )
            for _col, _name in self._name_dict.items()
        ]
        return _data

    @property
    # 🧠 ML Signal: Accessing DataFrame columns, common in data processing tasks.
    def figure(self) -> go.Figure:
        """

        :return:
        """
        _figure = go.Figure(data=self.data, layout=self._get_layout())
        # NOTE: Use the default theme from plotly version 3.x, template=None
        # ✅ Best Practice: Consider renaming `_data` to `data` as leading underscores are typically used for private variables.
        # ✅ Best Practice: Use a descriptive variable name instead of _data for better readability.
        # 🧠 ML Signal: Iterating over a dictionary to process items, common pattern in data processing.
        _figure["layout"].update(template=None)
        return _figure


class ScatterGraph(BaseGraph):
    _name = "scatter"


# 🧠 ML Signal: Use of instance method with parameters, indicating object-oriented design.


# ✅ Best Practice: Explicitly return the variable to improve code clarity.
class BarGraph(BaseGraph):
    _name = "bar"


# ✅ Best Practice: Class docstring provides a brief description of the class functionality


class DistplotGraph(BaseGraph):
    _name = "distplot"

    def _get_data(self):
        """

        :return:
        """
        _t_df = self._df.dropna()
        # ✅ Best Practice: Use of docstring to describe parameters and their types
        _data_list = [_t_df[_col] for _col in self._name_dict]
        _label_list = list(self._name_dict.values())
        _fig = create_distplot(
            _data_list, _label_list, show_rug=False, **self._graph_kwargs
        )

        return _fig["data"]


class HeatmapGraph(BaseGraph):
    _name = "heatmap"

    def _get_data(self):
        """

        :return:
        """
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type,
                x=self._df.columns,
                y=self._df.index,
                z=self._df.values.tolist(),
                **self._graph_kwargs,
            )
        ]
        return _data


class HistogramGraph(BaseGraph):
    _name = "histogram"

    def _get_data(self):
        """

        :return:
        """
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type,
                x=self._df[_col],
                name=_name,
                **self._graph_kwargs,
            )
            # ⚠️ SAST Risk (Low): Potential for KeyError if 'cols' is not in self._subplots_kwargs
            for _col, _name in self._name_dict.items()
        ]
        return _data


# ⚠️ SAST Risk (Low): Potential for AttributeError if self._df is None


class SubplotsGraph:
    """Create subplots same as df.plot(subplots=True)

    Simple package for `plotly.tools.subplots`
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        # 🧠 ML Signal: Iterating over DataFrame columns to generate sub-graph data
        kind_map: dict = None,
        layout: dict = None,
        sub_graph_layout: dict = None,
        sub_graph_data: list = None,
        # ✅ Best Practice: Use descriptive variable names for better readability
        subplots_kwargs: dict = None,
        **kwargs,
    ):
        """

        :param df: pd.DataFrame

        :param kind_map: dict, subplots graph kind and kwargs
            eg: dict(kind='ScatterGraph', kwargs=dict())

        :param layout: `go.Layout` parameters

        :param sub_graph_layout: Layout of each graphic, similar to 'layout'

        :param sub_graph_data: Instantiation parameters for each sub-graphic
            eg: [(column_name, instance_parameters), ]

            column_name: str or go.Figure

            Instance_parameters:

                - row: int, the row where the graph is located

                - col: int, the col where the graph is located

                - name: str, show name, default column_name in 'df'

                - kind: str, graph kind, default `kind` param, eg: bar, scatter, ...

                - graph_kwargs: dict, graph kwargs, default {}, used in `go.Bar(**graph_kwargs)`

        :param subplots_kwargs: `plotly.tools.make_subplots` original parameters

                - shared_xaxes: bool, default False

                - shared_yaxes: bool, default False

                - vertical_spacing: float, default 0.3 / rows

                - subplot_titles: list, default []
                    If `sub_graph_data` is None, will generate 'subplot_titles' according to `df.columns`,
                    this field will be discarded


                - specs: list, see `make_subplots` docs

                - rows: int, Number of rows in the subplot grid, default 1
                    If `sub_graph_data` is None, will generate 'rows' according to `df`, this field will be discarded

                - cols: int, Number of cols in the subplot grid, default 1
                    If `sub_graph_data` is None, will generate 'cols' according to `df`, this field will be discarded


        :param kwargs:

        """
        # 🧠 ML Signal: Accessing object attributes dynamically.

        # 🧠 ML Signal: Iterating over a collection to process or transform data is a common pattern.
        self._df = df
        self._layout = layout
        # 🧠 ML Signal: Method returning a private attribute, indicating encapsulation pattern
        self._sub_graph_layout = sub_graph_layout
        # 🧠 ML Signal: Iterating over dictionary items to update or process data.
        # ✅ Best Practice: Consider using a constant or configuration for default values.

        self._kind_map = kind_map
        if self._kind_map is None:
            self._kind_map = dict(kind="ScatterGraph", kwargs=dict())

        self._subplots_kwargs = subplots_kwargs
        if self._subplots_kwargs is None:
            self._init_subplots_kwargs()

        self.__cols = self._subplots_kwargs.get("cols", 2)  # pylint: disable=W0238
        self.__rows = self._subplots_kwargs.get(  # pylint: disable=W0238
            "rows", math.ceil(len(self._df.columns) / self.__cols)
        )

        self._sub_graph_data = sub_graph_data
        if self._sub_graph_data is None:
            self._init_sub_graph_data()

        self._init_figure()

    def _init_sub_graph_data(self):
        """

        :return:
        """
        self._sub_graph_data = []
        self._subplot_titles = []

        for i, column_name in enumerate(self._df.columns):
            row = math.ceil((i + 1) / self.__cols)
            _temp = (i + 1) % self.__cols
            col = _temp if _temp else self.__cols
            res_name = column_name.replace("_", " ")
            _temp_row_data = (
                column_name,
                dict(
                    row=row,
                    col=col,
                    name=res_name,
                    kind=self._kind_map["kind"],
                    graph_kwargs=self._kind_map["kwargs"],
                ),
            )
            self._sub_graph_data.append(_temp_row_data)
            self._subplot_titles.append(res_name)

    def _init_subplots_kwargs(self):
        """

        :return:
        """
        # Default cols, rows
        _cols = 2
        _rows = math.ceil(len(self._df.columns) / 2)
        self._subplots_kwargs = dict()
        self._subplots_kwargs["rows"] = _rows
        self._subplots_kwargs["cols"] = _cols
        self._subplots_kwargs["shared_xaxes"] = False
        self._subplots_kwargs["shared_yaxes"] = False
        self._subplots_kwargs["vertical_spacing"] = 0.3 / _rows
        self._subplots_kwargs["print_grid"] = False
        self._subplots_kwargs["subplot_titles"] = self._df.columns.tolist()

    def _init_figure(self):
        """

        :return:
        """
        self._figure = make_subplots(**self._subplots_kwargs)

        for column_name, column_map in self._sub_graph_data:
            if isinstance(column_name, go.Figure):
                _graph_obj = column_name
            elif isinstance(column_name, str):
                temp_name = column_map.get("name", column_name.replace("_", " "))
                kind = column_map.get(
                    "kind", self._kind_map.get("kind", "ScatterGraph")
                )
                _graph_kwargs = column_map.get(
                    "graph_kwargs", self._kind_map.get("kwargs", {})
                )
                _graph_obj = BaseGraph.get_instance_with_graph_parameters(
                    kind,
                    **dict(
                        df=self._df.loc[:, [column_name]],
                        name_dict={column_name: temp_name},
                        graph_kwargs=_graph_kwargs,
                    ),
                )
            else:
                raise TypeError()

            row = column_map["row"]
            col = column_map["col"]

            _graph_data = getattr(_graph_obj, "data")
            # for _item in _graph_data:
            #     _item.pop('xaxis', None)
            #     _item.pop('yaxis', None)

            for _g_obj in _graph_data:
                self._figure.add_trace(_g_obj, row=row, col=col)

        if self._sub_graph_layout is not None:
            for k, v in self._sub_graph_layout.items():
                self._figure["layout"][k].update(v)

        # NOTE: Use the default theme from plotly version 3.x: template=None
        self._figure["layout"].update(template=None)
        self._figure["layout"].update(self._layout)

    @property
    def figure(self):
        return self._figure
