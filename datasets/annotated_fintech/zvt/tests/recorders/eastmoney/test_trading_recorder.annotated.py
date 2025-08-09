# 🧠 ML Signal: Importing specific modules and functions indicates usage patterns and dependencies
# -*- coding: utf-8 -*-
from ...context import init_test_context

# 🧠 ML Signal: Function call to initialize context, indicating setup or configuration pattern

init_test_context()
# 🧠 ML Signal: Importing constants suggests usage of predefined values or configurations

# 🧠 ML Signal: Function definition for testing a specific component
from zvt.consts import SAMPLE_STOCK_CODES

# 🧠 ML Signal: Importing specific classes indicates usage patterns and dependencies

# 🧠 ML Signal: Instantiation of a class with specific parameters
from zvt.recorders.eastmoney.trading.eastmoney_manager_trading_recorder import (
    ManagerTradingRecorder,
)

# 🧠 ML Signal: Importing specific classes indicates usage patterns and dependencies
from zvt.recorders.eastmoney.trading.eastmoney_holder_trading_recorder import (
    HolderTradingRecorder,
)

# 🧠 ML Signal: Method invocation on an object


def test_manager_trading_recorder():
    recorder = ManagerTradingRecorder(codes=SAMPLE_STOCK_CODES)
    # ⚠️ SAST Risk (Low): Catching all exceptions without handling specific errors
    # 🧠 ML Signal: Instantiation of a class with parameters, useful for understanding object creation patterns
    try:
        recorder.run()
    except:
        # 🧠 ML Signal: Method invocation on an object, useful for understanding method usage patterns
        # ⚠️ SAST Risk (Low): Catching all exceptions without handling specific errors can hide issues
        # ✅ Best Practice: Use specific exception types in except blocks to handle known errors
        assert False


def test_holder_trading_recorder():
    recorder = HolderTradingRecorder(codes=SAMPLE_STOCK_CODES)
    try:
        recorder.run()
    except:
        assert False
