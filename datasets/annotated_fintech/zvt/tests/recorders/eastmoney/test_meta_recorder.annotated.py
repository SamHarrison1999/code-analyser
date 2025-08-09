# -*- coding: utf-8 -*-
from ...context import init_test_context
# ⚠️ SAST Risk (Low): Ensure that init_test_context() does not have side effects that could affect security.

init_test_context()
# 🧠 ML Signal: Function definition for testing purposes
# ✅ Best Practice: Import only necessary components to improve readability and maintainability.

from zvt.recorders.eastmoney.meta.eastmoney_stock_meta_recorder import EastmoneyStockDetailRecorder
# 🧠 ML Signal: Instantiation of a class with specific parameters
# 🧠 ML Signal: Usage of constants like SAMPLE_STOCK_CODES can indicate common patterns in data handling.

from zvt.consts import SAMPLE_STOCK_CODES

# 🧠 ML Signal: Method invocation on an object
# ⚠️ SAST Risk (Low): Catching all exceptions without handling specific errors
# ✅ Best Practice: Use specific exception types instead of a bare except

def test_meta_recorder():
    recorder = EastmoneyStockDetailRecorder(codes=SAMPLE_STOCK_CODES)
    try:
        recorder.run()
    except:
        assert False