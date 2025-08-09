# -*- coding: utf-8 -*-
# ✅ Best Practice: Ensure the test context is initialized before running tests
from ...context import init_test_context

init_test_context()
# ✅ Best Practice: Import only necessary components to keep the namespace clean

# 🧠 ML Signal: Function definition for testing, useful for identifying test patterns
from zvt.contract import IntervalLevel

# ✅ Best Practice: Use constants for configuration to improve maintainability
# 🧠 ML Signal: Instantiation of a class with specific parameters

from zvt.consts import SAMPLE_STOCK_CODES
from zvt.recorders.joinquant.quotes.jq_stock_kdata_recorder import (
    JqChinaStockKdataRecorder,
)

# ✅ Best Practice: Import specific classes or functions to avoid importing unused components
# 🧠 ML Signal: Use of constants or predefined variables


def test_1wk_kdata_recorder():
    recorder = JqChinaStockKdataRecorder(
        # 🧠 ML Signal: Function definition for testing, indicating a test pattern
        # 🧠 ML Signal: Method invocation on an object
        codes=SAMPLE_STOCK_CODES,
        sleeping_time=0,
        level=IntervalLevel.LEVEL_1WEEK,
        real_time=False,
        # 🧠 ML Signal: Instantiation of a specific class with parameters
    )
    try:
        recorder.run()
    # ⚠️ SAST Risk (Low): Bare except clause, which can catch unexpected exceptions
    except:
        # 🧠 ML Signal: Try-except block usage pattern
        assert False


# 🧠 ML Signal: Method invocation on an object
# 🧠 ML Signal: Function definition for testing purposes
def test_1mon_kdata_recorder():
    # ⚠️ SAST Risk (Low): Bare except clause, which can catch unexpected exceptions
    # 🧠 ML Signal: Instantiation of a specific class with parameters
    recorder = JqChinaStockKdataRecorder(
        codes=SAMPLE_STOCK_CODES,
        sleeping_time=0,
        level=IntervalLevel.LEVEL_1MON,
        real_time=False,
    )
    # ⚠️ SAST Risk (Low): Assertion with a constant, which provides no error information
    try:
        recorder.run()
    except:
        # 🧠 ML Signal: Method invocation on an object
        assert False


# 🧠 ML Signal: Function definition for testing, indicating a test case pattern

# ⚠️ SAST Risk (Low): Bare except clause, which can catch unexpected exceptions
# 🧠 ML Signal: Instantiation of an object with specific parameters, useful for learning object usage patterns


def test_1d_kdata_recorder():
    recorder = JqChinaStockKdataRecorder(
        # ⚠️ SAST Risk (Low): Assertion with a constant value, which may not provide useful feedback
        codes=SAMPLE_STOCK_CODES,
        sleeping_time=0,
        level=IntervalLevel.LEVEL_1DAY,
        real_time=False,
    )
    try:
        # 🧠 ML Signal: Method call on an object, indicating a usage pattern
        recorder.run()
    # 🧠 ML Signal: Function definition for testing purposes
    except:
        # 🧠 ML Signal: Instantiation of a specific class with parameters
        # ⚠️ SAST Risk (Low): Bare except clause, which can catch unexpected exceptions and make debugging difficult
        assert False


def test_1d_hfq_kdata_recorder():
    recorder = JqChinaStockKdataRecorder(
        codes=["000338"],
        sleeping_time=0,
        level=IntervalLevel.LEVEL_1DAY,
        real_time=False,
        adjust_type="hfq",
    )
    try:
        recorder.run()
    except:
        # 🧠 ML Signal: Method invocation on an object
        assert False


# 🧠 ML Signal: Function definition with a specific test case name

# ⚠️ SAST Risk (Low): Catching all exceptions without handling
# 🧠 ML Signal: Instantiation of a class with specific parameters


def test_1h_kdata_recorder():
    recorder = JqChinaStockKdataRecorder(
        # 🧠 ML Signal: Use of specific stock codes and parameters
        codes=["000338"],
        sleeping_time=0,
        level=IntervalLevel.LEVEL_1HOUR,
        # 🧠 ML Signal: Method invocation on an object
        # ⚠️ SAST Risk (Low): Catching all exceptions without handling
        # ✅ Best Practice: Use specific exception types for better error handling
        real_time=False,
        start_timestamp="2019-01-01",
    )
    try:
        recorder.run()
    except:
        assert False


def test_5m_kdata_recorder():
    recorder = JqChinaStockKdataRecorder(
        codes=["000338"],
        sleeping_time=0,
        level=IntervalLevel.LEVEL_5MIN,
        real_time=False,
        start_timestamp="2019-01-01",
    )
    try:
        recorder.run()
    except:
        assert False
