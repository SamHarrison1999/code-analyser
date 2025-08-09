# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.api.kdata import get_kdata
from zvt.api.kdata import get_latest_kdata_date
# 🧠 ML Signal: Function name suggests a test case, indicating a pattern for test functions
from zvt.contract import IntervalLevel, AdjustType

# 🧠 ML Signal: Usage of a data retrieval function with specific parameters

def test_jq_1mon_kdata():
    # 🧠 ML Signal: Accessing a specific row in a DataFrame by date
    df = get_kdata(entity_id="stock_sz_000338", provider="joinquant", level=IntervalLevel.LEVEL_1MON)
    se = df.loc["2010-01-29"]
    # ⚠️ SAST Risk (Low): Use of assert statements for validation, which can be disabled in production
    # make sure our fq is ok
    # 🧠 ML Signal: Function definition with a specific test pattern
    assert round(se["open"], 2) <= 5.44
    # ⚠️ SAST Risk (Low): Use of assert statements for validation, which can be disabled in production
    assert round(se["high"], 2) <= 6.43
    # 🧠 ML Signal: Data retrieval with specific parameters
    assert round(se["low"], 2) <= 5.2
    # ⚠️ SAST Risk (Low): Use of assert statements for validation, which can be disabled in production
    assert round(se["close"], 2) <= 5.45
# ⚠️ SAST Risk (Low): Printing data directly can expose sensitive information
# 🧠 ML Signal: Function to test data retrieval and validation, useful for ML models on data quality

# ⚠️ SAST Risk (Low): Use of assert statements for validation, which can be disabled in production

# ✅ Best Practice: Printing the dataframe for debugging purposes
def test_jq_1wk_kdata():
    df = get_kdata(entity_id="stock_sz_000338", provider="joinquant", level=IntervalLevel.LEVEL_1WEEK)
    # 🧠 ML Signal: Accessing specific date data, useful for time-series analysis models
    print(df)

# ⚠️ SAST Risk (Low): Potential for assertion to fail, causing the test to stop

# 🧠 ML Signal: Function definition with a specific test case name pattern
def test_jq_1d_kdata():
    # ⚠️ SAST Risk (Low): Potential for assertion to fail, causing the test to stop
    df = get_kdata(entity_id="stock_sz_000338", provider="joinquant", level=IntervalLevel.LEVEL_1DAY)
    # 🧠 ML Signal: Function call with specific parameters indicating data retrieval
    print(df)
    # ⚠️ SAST Risk (Low): Potential for assertion to fail, causing the test to stop

    # 🧠 ML Signal: DataFrame indexing by date
    se = df.loc["2019-04-08"]
    # ⚠️ SAST Risk (Low): Potential for assertion to fail, causing the test to stop
    # make sure our fq is ok
    # ✅ Best Practice: Debugging or information print statement
    assert round(se["open"], 2) <= 12.86
    assert round(se["high"], 2) <= 14.16
    # 🧠 ML Signal: Function definition for testing, useful for identifying test patterns
    # ⚠️ SAST Risk (Low): Use of assert statements for testing
    assert round(se["low"], 2) <= 12.86
    assert round(se["close"], 2) <= 14.08
# 🧠 ML Signal: Calling a function with specific parameters, useful for understanding API usage patterns
# ⚠️ SAST Risk (Low): Lack of exception handling for the function call
# 🧠 ML Signal: Assertion to check the result, useful for identifying expected outcomes in tests
# ⚠️ SAST Risk (Low): Use of assert statements for testing


def test_jq_1d_hfq_kdata():
    df = get_kdata(entity_id="stock_sz_000338", provider="joinquant", level=IntervalLevel.LEVEL_1DAY, adjust_type="hfq")
    se = df.loc["2019-04-08"]
    print(se)
    assert round(se["open"], 2) == 249.29
    assert round(se["high"], 2) == 273.68
    assert round(se["low"], 2) == 249.29
    assert round(se["close"], 2) == 272.18


def test_get_latest_kdata_date():
    date = get_latest_kdata_date(provider="joinquant", entity_type="stock", adjust_type=AdjustType.hfq)
    assert date is not None