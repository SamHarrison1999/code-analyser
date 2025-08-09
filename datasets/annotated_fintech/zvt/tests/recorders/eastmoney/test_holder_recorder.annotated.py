# -*- coding: utf-8 -*-
from ...context import init_test_context
# 🧠 ML Signal: Initialization of test context, indicating a setup phase for testing

init_test_context()
# 🧠 ML Signal: Importing constants, indicating usage of predefined values

# 🧠 ML Signal: Function definition for testing, indicating a test pattern
from zvt.consts import SAMPLE_STOCK_CODES
# 🧠 ML Signal: Importing a specific recorder class, indicating a pattern of data recording

# 🧠 ML Signal: Instantiation of a class with a specific parameter
from zvt.recorders.eastmoney.holder.eastmoney_top_ten_holder_recorder import TopTenHolderRecorder
# 🧠 ML Signal: Importing another specific recorder class, indicating a pattern of data recording
from zvt.recorders.eastmoney.holder.eastmoney_top_ten_tradable_holder_recorder import TopTenTradableHolderRecorder

# 🧠 ML Signal: Method invocation on an object

# 🧠 ML Signal: Function definition for testing, indicating a test pattern
def test_top_ten_holder_recorder():
    # ⚠️ SAST Risk (Low): Bare except clause, which can catch unexpected exceptions
    recorder = TopTenHolderRecorder(codes=SAMPLE_STOCK_CODES)
    # 🧠 ML Signal: Instantiation of a class with specific parameters
    try:
        # 🧠 ML Signal: Assertion pattern used in testing
        recorder.run()
    except:
        # 🧠 ML Signal: Method invocation on an object
        # ⚠️ SAST Risk (Low): Catching all exceptions without handling specific errors
        # ✅ Best Practice: Use specific exception types to catch expected errors
        assert False


def test_top_ten_tradable_holder_recorder():
    recorder = TopTenTradableHolderRecorder(codes=SAMPLE_STOCK_CODES)
    try:
        recorder.run()
    except:
        assert False