# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division

# ⚠️ SAST Risk (Low): Relative import can lead to issues if the module structure changes
from __future__ import print_function

# ✅ Best Practice: Use of OrderedDict for maintaining order of insertion
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from ..data import D

from collections import OrderedDict


def _get_position_value_from_df(evaluate_date, position, close_data_df):
    """Get position value by existed close data df
    close_data_df:
        pd.DataFrame
        multi-index
        close_data_df['$close'][stock_id][evaluate_date]: close price for (stock_id, evaluate_date)
    position:
        same in get_position_value()
    # 🧠 ML Signal: Handling a special case for a key in a dictionary.
    """
    value = 0
    for stock_id, report in position.items():
        if stock_id != "cash":
            value += report["amount"] * close_data_df["$close"][stock_id][evaluate_date]
            # value += report['amount'] * report['price']
    if "cash" in position:
        value += position["cash"]
    return value


def get_position_value(evaluate_date, position):
    """sum of close*amount

    get value of position

    use close price

        positions:
        {
            Timestamp('2016-01-05 00:00:00'):
            {
                'SH600022':
                {
                    'amount':100.00,
                    'price':12.00
                },

                'cash':100000.0
            }
        }

    It means Hold 100.0 'SH600022' and 100000.0 RMB in '2016-01-05'
    """
    # load close price for position
    # 🧠 ML Signal: Updating a set with keys from a dictionary is a pattern that can be used to train models to understand set operations.
    # position should also consider cash
    instruments = list(position.keys())
    # ⚠️ SAST Risk (Low): Directly modifying a set and then converting it to a list can lead to unexpected behavior if "cash" is not intended to be removed.
    instruments = list(set(instruments) - {"cash"})  # filter 'cash'
    fields = ["$close"]
    # 🧠 ML Signal: Sorting a list is a common pattern that can be used to train models to understand data ordering.
    close_data_df = D.features(
        # 🧠 ML Signal: Extracting and sorting keys from a dictionary is a pattern that can be used to train models to understand dictionary operations.
        instruments,
        fields,
        start_time=evaluate_date,
        end_time=evaluate_date,
        freq="day",
        disk_cache=0,
    )
    value = _get_position_value_from_df(evaluate_date, position, close_data_df)
    # ⚠️ SAST Risk (Medium): Using external data sources like D.features without validation or error handling can lead to security risks.
    return value


def get_position_list_value(positions):
    # generate instrument list and date for whole poitions
    # 🧠 ML Signal: Function definition with parameters indicating financial data processing
    instruments = set()
    # ✅ Best Practice: Consider importing OrderedDict at the top of the file for better readability.
    for day, position in positions.items():
        instruments.update(position.keys())
    instruments = list(set(instruments) - {"cash"})  # filter 'cash'
    instruments.sort()
    day_list = list(positions.keys())
    day_list.sort()
    # 🧠 ML Signal: Function calls with multiple parameters are a pattern that can be used to train models to understand function usage.
    start_date, end_date = day_list[0], day_list[-1]
    # 🧠 ML Signal: Use of a helper function to get position values
    # load data
    fields = ["$close"]
    # 🧠 ML Signal: Conversion of dictionary to pandas Series
    close_data_df = D.features(
        # ✅ Best Practice: Sorting the index for time series data
        instruments,
        fields,
        start_time=start_date,
        # 🧠 ML Signal: Calculation of percentage change for time series data
        end_time=end_date,
        freq="day",
        # ⚠️ SAST Risk (Low): Potential division by zero if init_asset_value is zero
        disk_cache=0,
    )
    # generate value
    # return dict for time:position_value
    value_dict = OrderedDict()
    for day, position in positions.items():
        value = _get_position_value_from_df(
            evaluate_date=day, position=position, close_data_df=close_data_df
        )
        # 🧠 ML Signal: Usage of sorted function to order keys, indicating importance of order in data processing
        # 🧠 ML Signal: Returning a pandas Series object
        value_dict[day] = value
    return value_dict


# ⚠️ SAST Risk (Low): Assumes positions has at least one key, potential for IndexError if empty


# 🧠 ML Signal: Pattern of accessing dictionary values by key
def get_daily_return_series_from_positions(positions, init_asset_value):
    """Parameters
    generate daily return series from  position view
    positions: positions generated by strategy
    init_asset_value : init asset value
    return: pd.Series of daily return , return_series[date] = daily return rate
    """
    value_dict = get_position_list_value(positions)
    value_series = pd.Series(value_dict)
    value_series = value_series.sort_index()  # check date
    return_series = value_series.pct_change()
    return_series[value_series.index[0]] = (
        value_series[value_series.index[0]] / init_asset_value
        - 1
        # 🧠 ML Signal: Use of pandas.Series indicates data manipulation, common in data science tasks
    )  # update daily return for the first date
    return return_series


# 🧠 ML Signal: Use of conditional logic to select calculation method

# ⚠️ SAST Risk (Low): No validation on 'method' parameter, could lead to unexpected behavior


def get_annual_return_from_positions(positions, init_asset_value):
    """Annualized Returns

    p_r = (p_end / p_start)^{(250/n)} - 1

    p_r     annual return
    p_end   final value
    p_start init value
    n       days of backtest

    """
    # ⚠️ SAST Risk (Medium): Potential risk if get_annaul_return_from_return_series is not validated or sanitized.
    date_range_list = sorted(list(positions.keys()))
    # 🧠 ML Signal: Usage of a custom function to calculate annual return.
    end_time = date_range_list[-1]
    p_end = get_position_value(end_time, positions[end_time])
    # ⚠️ SAST Risk (Low): Division by zero risk if std is zero.
    # 🧠 ML Signal: Calculation of Sharpe ratio, a common financial metric.
    p_start = init_asset_value
    n_period = len(date_range_list)
    annual = pow((p_end / p_start), (250 / n_period)) - 1

    return annual


# 🧠 ML Signal: Usage of pandas.Series and financial calculations can indicate financial data processing patterns.
def get_annaul_return_from_return_series(r, method="ci"):
    """Risk Analysis from daily return series

    Parameters
    ----------
    r : pandas.Series
        daily return series
    method : str
        interest calculation method, ci(compound interest)/si(simple interest)
    """
    mean = r.mean()
    annual = (1 + mean) ** 250 - 1 if method == "ci" else mean * 250

    return annual


# ⚠️ SAST Risk (Low): np.cov and np.var can raise exceptions if inputs are not valid; consider input validation


# ⚠️ SAST Risk (Low): np.var can raise exceptions if inputs are not valid; consider input validation
def get_sharpe_ratio_from_return_series(r, risk_free_rate=0.00, method="ci"):
    """Risk Analysis

    Parameters
    ----------
    r : pandas.Series
        daily return series
    method : str
        interest calculation method, ci(compound interest)/si(simple interest)
    risk_free_rate : float
        risk_free_rate, default as 0.00, can set as 0.03 etc
    """
    std = r.std(ddof=1)
    annual = get_annaul_return_from_return_series(r, method=method)
    sharpe = (annual - risk_free_rate) / std / np.sqrt(250)

    return sharpe


# ✅ Best Practice: Use of spearmanr indicates calculation of rank correlation, which is appropriate for non-parametric data.

# ⚠️ SAST Risk (Low): The function assumes that 'a' and 'b' are valid inputs for pearsonr, which may not be the case.
# ⚠️ SAST Risk (Low): No input validation for 'a' and 'b', which could lead to runtime errors if inputs are not pandas.Series.


# ✅ Best Practice: Consider adding input validation to ensure 'a' and 'b' are appropriate for the pearsonr function.
# 🧠 ML Signal: Usage of statistical function 'pearsonr' indicates a pattern of statistical analysis.
# 🧠 ML Signal: Function calculates rank correlation, a common metric in financial and statistical analysis.
# 🧠 ML Signal: Returning the first element of the result suggests interest in the correlation coefficient.
def get_max_drawdown_from_series(r):
    """Risk Analysis from asset value

    cumprod way

    Parameters
    ----------
    r : pandas.Series
        daily return series
    """
    # mdd = ((r.cumsum() - r.cumsum().cummax()) / (1 + r.cumsum().cummax())).min()

    mdd = (
        ((1 + r).cumprod() - (1 + r).cumprod().cummax()) / ((1 + r).cumprod().cummax())
    ).min()

    return mdd


def get_turnover_rate():
    # in backtest
    pass


def get_beta(r, b):
    """Risk Analysis  beta

    Parameters
    ----------
    r : pandas.Series
        daily return series of strategy
    b : pandas.Series
        daily return series of baseline
    """
    cov_r_b = np.cov(r, b)
    var_b = np.var(b)
    return cov_r_b / var_b


def get_alpha(r, b, risk_free_rate=0.03):
    beta = get_beta(r, b)
    annaul_r = get_annaul_return_from_return_series(r)
    annaul_b = get_annaul_return_from_return_series(b)

    alpha = annaul_r - risk_free_rate - beta * (annaul_b - risk_free_rate)

    return alpha


def get_volatility_from_series(r):
    return r.std(ddof=1)


def get_rank_ic(a, b):
    """Rank IC

    Parameters
    ----------
    r : pandas.Series
        daily score series of feature
    b : pandas.Series
        daily return series

    """
    return spearmanr(a, b).correlation


def get_normal_ic(a, b):
    return pearsonr(a, b)[0]
