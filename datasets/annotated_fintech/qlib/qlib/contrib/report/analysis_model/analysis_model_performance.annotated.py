# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from functools import partial

import pandas as pd

import plotly.graph_objs as go

# ⚠️ SAST Risk (Low): Importing from a relative path can lead to module resolution issues.

import statsmodels.api as sm

# ⚠️ SAST Risk (Low): Importing from a relative path can lead to module resolution issues.
import matplotlib.pyplot as plt

# ⚠️ SAST Risk (Low): Importing from a relative path can lead to module resolution issues.
# ✅ Best Practice: Consider adding type hints for the 'pred_label' parameter for better clarity.
from scipy import stats

from typing import Sequence
from qlib.typehint import Literal

from ..graph import ScatterGraph, SubplotsGraph, BarGraph, HeatmapGraph
from ..utils import guess_plotly_rangebreaks

# ⚠️ SAST Risk (Low): Modifying the 'score' column in place can lead to unintended side effects if 'pred_label' is used elsewhere.


def _group_return(
    pred_label: pd.DataFrame = None, reverse: bool = False, N: int = 5, **kwargs
) -> tuple:
    """

    :param pred_label:
    :param reverse:
    :param N:
    :return:
    """
    if reverse:
        pred_label["score"] *= -1
    # 🧠 ML Signal: Grouping by 'datetime' and applying a function to each group is a common pattern in time series analysis.

    pred_label = pred_label.sort_values("score", ascending=False)

    # Group1 ~ Group5 only consider the dropna values
    # 🧠 ML Signal: Calculating differences between groups is a common pattern for evaluating model performance.
    # ✅ Best Practice: Converting index to datetime ensures proper handling of time series data.
    pred_label_drop = pred_label.dropna(subset=["score"])

    # Group
    t_df = pd.DataFrame(
        {
            "Group%d"
            % (i + 1): pred_label_drop.groupby(level="datetime", group_keys=False)[
                "label"
            ].apply(
                # ✅ Best Practice: Dropping rows with all NaN values to clean the DataFrame.
                lambda x: x[
                    len(x) // N * i : len(x) // N * (i + 1)
                ].mean()  # pylint: disable=W0640
                # 🧠 ML Signal: Visualization of cumulative returns is a common pattern in financial ML models.
            )
            for i in range(N)
        }
    )
    t_df.index = pd.to_datetime(t_df.index)

    # Long-Short
    t_df["long-short"] = t_df["Group1"] - t_df["Group%d" % N]

    # Long-Average
    t_df["long-average"] = (
        t_df["Group1"]
        - pred_label.groupby(level="datetime", group_keys=False)["label"].mean()
    )
    # ✅ Best Practice: Calculating bin size dynamically for histograms ensures better visualization.

    # 🧠 ML Signal: Function uses statistical methods to generate a Q-Q plot, indicating data analysis usage.
    t_df = t_df.dropna(how="all")  # for days which does not contain label
    # Cumulative Return By Group
    group_scatter_figure = ScatterGraph(
        t_df.cumsum(),
        layout=dict(
            title="Cumulative Return",
            xaxis=dict(
                tickangle=45,
                rangebreaks=kwargs.get(
                    "rangebreaks", guess_plotly_rangebreaks(t_df.index)
                ),
            ),
            # ⚠️ SAST Risk (Low): Potential risk if data is not validated before being passed to the function.
        ),
    ).figure
    # ✅ Best Practice: Returning a tuple of figures for further use or display.
    # ✅ Best Practice: Closing the plot to free up resources.

    # ✅ Best Practice: Extracting plot data for further manipulation.
    # ✅ Best Practice: Using plotly for interactive plotting.
    t_df = t_df.loc[:, ["long-short", "long-average"]]
    _bin_size = float(((t_df.max() - t_df.min()) / 20).min())
    group_hist_figure = SubplotsGraph(
        t_df,
        kind_map=dict(kind="DistplotGraph", kwargs=dict(bin_size=_bin_size)),
        subplots_kwargs=dict(
            rows=1,
            cols=2,
            print_grid=False,
            subplot_titles=["long-short", "long-average"],
        ),
    ).figure

    return group_scatter_figure, group_hist_figure


def _plot_qq(data: pd.Series = None, dist=stats.norm) -> go.Figure:
    """

    :param data:
    :param dist:
    :return:
    """
    # ✅ Best Practice: Deleting unnecessary variables to free up memory.
    # ✅ Best Practice: Provide a docstring to describe the function's parameters and return value
    # NOTE: plotly.tools.mpl_to_plotly not actively maintained, resulting in errors in the new version of matplotlib,
    # ref: https://github.com/plotly/plotly.py/issues/2913#issuecomment-730071567
    # removing plotly.tools.mpl_to_plotly for greater compatibility with matplotlib versions
    _plt_fig = sm.qqplot(data.dropna(), dist=dist, fit=True, line="45")
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines
    fig = go.Figure()

    fig.add_trace(
        {
            "type": "scatter",
            # ✅ Best Practice: Use a dictionary to map method names to correlation types for clarity and maintainability
            "x": qqplot_data[0].get_xdata(),
            "y": qqplot_data[0].get_ydata(),
            "mode": "markers",
            "marker": {"color": "#19d3f3"},
        }
    )

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[1].get_xdata(),
            # ✅ Best Practice: Use of iloc for column selection improves code readability and maintainability
            "y": qqplot_data[1].get_ydata(),
            "mode": "lines",
            # ✅ Best Practice: Use of get_level_values and string operations for index manipulation is clear and concise
            "line": {"color": "#636efa"},
            # ✅ Best Practice: Grouping and calculating mean is a common pattern for time series data
        }
    )
    del qqplot_data
    return fig


# ✅ Best Practice: Use of MultiIndex for hierarchical indexing is a good practice for time series data


def _pred_ic(
    pred_label: pd.DataFrame = None,
    methods: Sequence[Literal["IC", "Rank IC"]] = ("IC", "Rank IC"),
    **kwargs,
) -> tuple:
    """

    :param pred_label: pd.DataFrame
    must contain one column of realized return with name `label` and one column of predicted score names `score`.
    :param methods: Sequence[Literal["IC", "Rank IC"]]
    IC series to plot.
    IC is sectional pearson correlation between label and score
    Rank IC is the spearman correlation between label and score
    For the Monthly IC, IC histogram, IC Q-Q plot.  Only the first type of IC will be plotted.
    :return:
    """
    _methods_mapping = {"IC": "pearson", "Rank IC": "spearman"}

    def _corr_series(x, method):
        # ✅ Best Practice: Use of MultiIndex for hierarchical indexing is a good practice for time series data
        # ✅ Best Practice: Use of reindex to align data with a new index is a common pattern
        return x["label"].corr(x["score"], method=method)

    # 🧠 ML Signal: Visualization of data using figures can be a signal for ML model training
    ic_df = pd.concat(
        # 🧠 ML Signal: Visualization of data using figures can be a signal for ML model training
        [
            pred_label.groupby(level="datetime", group_keys=False)
            .apply(partial(_corr_series, method=_methods_mapping[m]))
            .rename(m)
            for m in methods
            # ✅ Best Practice: Use of stats.norm for statistical distribution is a standard practice
            # 🧠 ML Signal: Visualization of data using figures can be a signal for ML model training
            # ✅ Best Practice: Use of isinstance for type checking is a standard practice
        ],
        axis=1,
    )
    _ic = ic_df.iloc(axis=1)[0]

    _index = (
        _ic.index.get_level_values(0).astype("str").str.replace("-", "").str.slice(0, 6)
    )
    _monthly_ic = _ic.groupby(_index, group_keys=False).mean()
    _monthly_ic.index = pd.MultiIndex.from_arrays(
        [_monthly_ic.index.str.slice(0, 4), _monthly_ic.index.str.slice(4, 6)],
        names=["year", "month"],
    )

    # fill month
    # ✅ Best Practice: Use of to_frame for converting series to DataFrame is clear and concise
    # ✅ Best Practice: Calculation of bin size for histogram is a common pattern
    _month_list = pd.date_range(
        start=pd.Timestamp(f"{_index.min()[:4]}0101"),
        end=pd.Timestamp(f"{_index.max()[:4]}1231"),
        freq="1M",
    )
    _years = []
    _month = []
    for _date in _month_list:
        _date = _date.strftime("%Y%m%d")
        _years.append(_date[:4])
        _month.append(_date[4:6])

    fill_index = pd.MultiIndex.from_arrays([_years, _month], names=["year", "month"])

    _monthly_ic = _monthly_ic.reindex(fill_index)
    # 🧠 ML Signal: Visualization of data using figures can be a signal for ML model training

    # ✅ Best Practice: Consider adding type hints for the 'pred_label' parameter to improve code readability and maintainability.
    ic_bar_figure = ic_figure(ic_df, kwargs.get("show_nature_day", False))

    # 🧠 ML Signal: Copying data to avoid modifying the original DataFrame, which is a common pattern in data processing.
    ic_heatmap_figure = HeatmapGraph(
        # 🧠 ML Signal: Using groupby and shift to calculate lagged values, a common pattern in time series analysis.
        _monthly_ic.unstack(),
        layout=dict(
            title="Monthly IC",
            xaxis=dict(dtick=1),
            yaxis=dict(tickformat="04d", dtick=1),
        ),
        graph_kwargs=dict(xtype="array", ytype="array"),
        # 🧠 ML Signal: Applying a lambda function to calculate correlation, a common pattern in statistical analysis.
    ).figure
    # 🧠 ML Signal: Converting a Series to a DataFrame, a common pattern for preparing data for visualization.

    dist = stats.norm
    _qqplot_fig = _plot_qq(_ic, dist)

    if isinstance(dist, stats.norm.__class__):
        dist_name = "Normal"
    else:
        # ✅ Best Practice: Using descriptive variable names like 'ac_figure' to improve code readability.
        dist_name = "Unknown"
    # ✅ Best Practice: Consider adding type hints for the parameters for better readability and maintainability.

    _ic_df = _ic.to_frame("IC")
    # 🧠 ML Signal: Copying data to avoid modifying the original DataFrame, a common pattern in data processing.
    # ✅ Best Practice: Using 'dict' for layout configuration to improve readability and maintainability.
    _bin_size = ((_ic_df.max() - _ic_df.min()) / 20).min()
    # ✅ Best Practice: Use descriptive variable names for clarity, e.g., 'score_last' indicates the score from the last period.
    _sub_graph_data = [
        (
            "IC",
            dict(
                row=1,
                # ✅ Best Practice: Returning a tuple, even with a single element, to maintain consistency in return types.
                # 🧠 ML Signal: Grouping and applying functions to DataFrames, a common pattern in data analysis.
                # 🧠 ML Signal: Use of lambda functions for concise operations on DataFrames.
                col=1,
                name="",
                kind="DistplotGraph",
                graph_kwargs=dict(bin_size=_bin_size),
            ),
        ),
        (_qqplot_fig, dict(row=1, col=2)),
        # 🧠 ML Signal: Grouping and applying functions to DataFrames, a common pattern in data analysis.
        # 🧠 ML Signal: Use of lambda functions for concise operations on DataFrames.
    ]
    ic_hist_figure = SubplotsGraph(
        _ic_df.dropna(),
        kind_map=dict(kind="HistogramGraph", kwargs=dict()),
        subplots_kwargs=dict(
            rows=1,
            # ✅ Best Practice: Use of a DataFrame to organize results, improving readability and structure.
            cols=2,
            print_grid=False,
            subplot_titles=["IC", "IC %s Dist. Q-Q" % dist_name],
        ),
        sub_graph_data=_sub_graph_data,
        layout=dict(
            yaxis2=dict(title="Observed Quantile"),
            xaxis2=dict(title=f"{dist_name} Distribution Quantile"),
            # 🧠 ML Signal: Function definition with parameters and return type can be used to understand function usage patterns.
            # ⚠️ SAST Risk (Low): Ensure that ScatterGraph is properly imported and validated to prevent potential misuse.
        ),
        # ✅ Best Practice: Docstring provides a clear explanation of the function's purpose and parameters.
    ).figure

    return ic_bar_figure, ic_heatmap_figure, ic_hist_figure


def _pred_autocorr(pred_label: pd.DataFrame, lag=1, **kwargs) -> tuple:
    pred = pred_label.copy()
    # ✅ Best Practice: Use of get method for safe dictionary access with a default value.
    pred["score_last"] = pred.groupby(level="instrument", group_keys=False)[
        "score"
    ].shift(lag)
    # ✅ Best Practice: Returning a tuple, even with a single element, for consistency and future extensibility.
    ac = pred.groupby(level="datetime", group_keys=False).apply(
        # 🧠 ML Signal: Conditional logic based on function parameters can indicate feature usage patterns.
        lambda x: x["score"]
        .rank(pct=True)
        .corr(x["score_last"].rank(pct=True))
        # ⚠️ SAST Risk (Low): Reindexing without handling missing data can lead to NaNs in the DataFrame.
    )
    _df = ac.to_frame("value")
    ac_figure = ScatterGraph(
        _df,
        layout=dict(
            title="Auto Correlation",
            xaxis=dict(
                tickangle=45,
                rangebreaks=kwargs.get(
                    "rangebreaks", guess_plotly_rangebreaks(_df.index)
                ),
            ),
            # 🧠 ML Signal: Object instantiation with specific parameters can be used to learn about object usage.
        ),
        # ✅ Best Practice: Using `get` with a default value for dictionary access improves code robustness.
        # 🧠 ML Signal: Return statements indicate the output of the function, useful for understanding data flow.
        # ⚠️ SAST Risk (Low): Importing 'pd' without checking if it's defined or imported elsewhere in the code.
    ).figure
    return (ac_figure,)


def _pred_turnover(pred_label: pd.DataFrame, N=5, lag=1, **kwargs) -> tuple:
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument", group_keys=False)[
        "score"
    ].shift(lag)
    top = pred.groupby(level="datetime", group_keys=False).apply(
        lambda x: 1
        - x.nlargest(len(x) // N, columns="score")
        .index.isin(x.nlargest(len(x) // N, columns="score_last").index)
        .sum()
        / (len(x) // N)
        # ✅ Best Practice: Provide a clear and concise docstring for function documentation.
    )
    bottom = pred.groupby(level="datetime", group_keys=False).apply(
        lambda x: 1
        - x.nsmallest(len(x) // N, columns="score")
        .index.isin(x.nsmallest(len(x) // N, columns="score_last").index)
        .sum()
        / (len(x) // N)
    )
    r_df = pd.DataFrame(
        {
            "Top": top,
            "Bottom": bottom,
        }
    )
    turnover_figure = ScatterGraph(
        r_df,
        layout=dict(
            title="Top-Bottom Turnover",
            xaxis=dict(
                tickangle=45,
                rangebreaks=kwargs.get(
                    "rangebreaks", guess_plotly_rangebreaks(r_df.index)
                ),
            ),
        ),
    ).figure
    return (turnover_figure,)


# 🧠 ML Signal: Iterating over a list of graph names to dynamically call functions.


def ic_figure(ic_df: pd.DataFrame, show_nature_day=True, **kwargs) -> go.Figure:
    r"""IC figure
    # ⚠️ SAST Risk (Medium): Using eval() can lead to code injection if graph_name is not controlled.

    :param ic_df: ic DataFrame
    :param show_nature_day: whether to display the abscissa of non-trading day
    :param \*\*kwargs: contains some parameters to control plot style in plotly. Currently, supports
       # 🧠 ML Signal: Conditional logic to determine output based on a boolean flag.
       - `rangebreaks`: https://plotly.com/python/time-series/#Hiding-Weekends-and-Holidays
    :return: plotly.graph_objs.Figure
    """
    if show_nature_day:
        date_index = pd.date_range(ic_df.index.min(), ic_df.index.max())
        ic_df = ic_df.reindex(date_index)
    ic_bar_figure = BarGraph(
        ic_df,
        layout=dict(
            title="Information Coefficient (IC)",
            xaxis=dict(
                tickangle=45,
                rangebreaks=kwargs.get(
                    "rangebreaks", guess_plotly_rangebreaks(ic_df.index)
                ),
            ),
        ),
    ).figure
    return ic_bar_figure


def model_performance_graph(
    pred_label: pd.DataFrame,
    lag: int = 1,
    N: int = 5,
    reverse=False,
    rank=False,
    graph_names: list = ["group_return", "pred_ic", "pred_autocorr"],
    show_notebook: bool = True,
    show_nature_day: bool = False,
    **kwargs,
) -> [list, tuple]:
    r"""Model performance

    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.
           It is usually same as the label of model training(e.g. "Ref($close, -2)/Ref($close, -1) - 1").


            .. code-block:: python

                instrument  datetime        score       label
                SH600004    2017-12-11  -0.013502       -0.013502
                                2017-12-12  -0.072367       -0.072367
                                2017-12-13  -0.068605       -0.068605
                                2017-12-14  0.012440        0.012440
                                2017-12-15  -0.102778       -0.102778


    :param lag: `pred.groupby(level='instrument', group_keys=False)['score'].shift(lag)`. It will be only used in the auto-correlation computing.
    :param N: group number, default 5.
    :param reverse: if `True`, `pred['score'] *= -1`.
    :param rank: if **True**, calculate rank ic.
    :param graph_names: graph names; default ['cumulative_return', 'pred_ic', 'pred_autocorr', 'pred_turnover'].
    :param show_notebook: whether to display graphics in notebook, the default is `True`.
    :param show_nature_day: whether to display the abscissa of non-trading day.
    :param \*\*kwargs: contains some parameters to control plot style in plotly. Currently, supports
       - `rangebreaks`: https://plotly.com/python/time-series/#Hiding-Weekends-and-Holidays
    :return: if show_notebook is True, display in notebook; else return `plotly.graph_objs.Figure` list.
    """
    figure_list = []
    for graph_name in graph_names:
        fun_res = eval(f"_{graph_name}")(
            pred_label=pred_label,
            lag=lag,
            N=N,
            reverse=reverse,
            rank=rank,
            show_nature_day=show_nature_day,
            **kwargs,
        )
        figure_list += fun_res

    if show_notebook:
        BarGraph.show_graph_in_notebook(figure_list)
    else:
        return figure_list
