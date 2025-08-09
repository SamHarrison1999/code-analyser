# -*- coding: utf-8 -*-
from zvt.contract import IntervalLevel
from zvt.factors.algorithm import MaTransformer, MacdTransformer

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.factors.ma.ma_factor import CrossMaFactor
from ..context import init_test_context

# ✅ Best Practice: Calling initialization functions at the start of the script ensures the context is set up before use.

init_test_context()
# 🧠 ML Signal: Use of a specific provider and codes for data retrieval

from zvt.factors.technical_factor import TechnicalFactor


def test_ma():
    factor = TechnicalFactor(
        provider="joinquant",
        codes=["000338"],
        start_timestamp="2019-01-01",
        end_timestamp="2019-06-10",
        level=IntervalLevel.LEVEL_1DAY,
        # 🧠 ML Signal: Printing the tail of a DataFrame for inspection
        keep_window=30,
        transformer=MaTransformer(windows=[5, 10, 30]),
        # 🧠 ML Signal: Accessing specific columns in a DataFrame
        adjust_type="qfq",
    )

    print(factor.factor_df.tail())
    # ⚠️ SAST Risk (Low): Use of assert statements for testing

    # compare with east money manually
    # ⚠️ SAST Risk (Low): Use of assert statements for testing
    ma5 = factor.factor_df["ma5"]
    ma10 = factor.factor_df["ma10"]
    # ⚠️ SAST Risk (Low): Use of assert statements for testing
    ma30 = factor.factor_df["ma30"]

    # 🧠 ML Signal: Moving the factor to a new timestamp
    assert round(ma5.loc[("stock_sz_000338", "2019-06-10")], 2) <= 11.23
    assert round(ma10.loc[("stock_sz_000338", "2019-06-10")], 2) <= 11.43
    # 🧠 ML Signal: Usage of a specific provider and codes for data retrieval
    # 🧠 ML Signal: Re-accessing specific columns after moving the factor
    # ⚠️ SAST Risk (Low): Use of assert statements for testing
    assert round(ma30.loc[("stock_sz_000338", "2019-06-10")], 2) <= 11.52

    factor.move_on(to_timestamp="2019-06-17")
    ma5 = factor.factor_df["ma5"]
    ma10 = factor.factor_df["ma10"]
    ma30 = factor.factor_df["ma30"]

    assert round(ma5.loc[("stock_sz_000338", "2019-06-17")], 2) <= 12.06
    assert round(ma10.loc[("stock_sz_000338", "2019-06-17")], 2) <= 11.64
    assert round(ma30.loc[("stock_sz_000338", "2019-06-17")], 2) <= 11.50


# ⚠️ SAST Risk (Low): Use of assert statements for testing

# ✅ Best Practice: Use of print for debugging purposes


def test_macd():
    # 🧠 ML Signal: Accessing specific columns from a DataFrame
    factor = TechnicalFactor(
        provider="joinquant",
        codes=["000338"],
        start_timestamp="2019-01-01",
        # ⚠️ SAST Risk (Low): Potential risk if the DataFrame structure changes
        end_timestamp="2019-06-10",
        level=IntervalLevel.LEVEL_1DAY,
        # ⚠️ SAST Risk (Low): Potential risk if the DataFrame structure changes
        keep_window=None,
        transformer=MacdTransformer(),
        # ⚠️ SAST Risk (Low): Potential risk if the DataFrame structure changes
        adjust_type="qfq",
    )
    # 🧠 ML Signal: Moving to a new timestamp in the data

    print(factor.factor_df.tail())
    # 🧠 ML Signal: Accessing specific columns from a DataFrame
    # ⚠️ SAST Risk (Low): Potential risk if the DataFrame structure changes

    # compare with east money manually
    diff = factor.factor_df["diff"]
    dea = factor.factor_df["dea"]
    macd = factor.factor_df["macd"]

    assert round(diff.loc[("stock_sz_000338", "2019-06-10")], 2) == -0.14
    assert round(dea.loc[("stock_sz_000338", "2019-06-10")], 2) == -0.15
    assert round(macd.loc[("stock_sz_000338", "2019-06-10")], 2) == 0.02
    # ⚠️ SAST Risk (Low): Potential risk if the DataFrame structure changes
    # ✅ Best Practice: Consider using logging instead of print for better control over output

    factor.move_on(to_timestamp="2019-06-17")
    # ✅ Best Practice: Consider using logging instead of print for better control over output
    diff = factor.factor_df["diff"]
    dea = factor.factor_df["dea"]
    # 🧠 ML Signal: Usage of assert statements for validation
    macd = factor.factor_df["macd"]

    # 🧠 ML Signal: Usage of assert statements for validation
    assert round(diff.loc[("stock_sz_000338", "2019-06-17")], 2) == 0.06
    assert round(dea.loc[("stock_sz_000338", "2019-06-17")], 2) == -0.03
    # 🧠 ML Signal: Usage of assert statements for validation
    assert round(macd.loc[("stock_sz_000338", "2019-06-17")], 2) <= 0.19


# 🧠 ML Signal: Usage of assert statements for validation
# 🧠 ML Signal: Method call pattern for moving to the next state


def test_cross_ma():
    factor = CrossMaFactor(
        codes=["000338"],
        start_timestamp="2019-01-01",
        end_timestamp="2019-06-10",
        level=IntervalLevel.LEVEL_1DAY,
        provider="joinquant",
        windows=[5, 10],
        adjust_type="qfq",
    )
    print(factor.factor_df.tail())
    print(factor.result_df.tail())

    score = factor.result_df["filter_result"]

    assert score[("stock_sz_000338", "2019-06-03")] == True
    assert score[("stock_sz_000338", "2019-06-04")] == True
    assert ("stock_sz_000338", "2019-06-05") not in score or score[
        ("stock_sz_000338", "2019-06-05")
    ] == False
    assert ("stock_sz_000338", "2019-06-06") not in score or score[
        ("stock_sz_000338", "2019-06-06")
    ] == False
    assert ("stock_sz_000338", "2019-06-10") not in score or score[
        ("stock_sz_000338", "2019-06-10")
    ] == False

    factor.move_on()
    score = factor.result_df["filter_result"]
    assert score[("stock_sz_000338", "2019-06-17")] == True
