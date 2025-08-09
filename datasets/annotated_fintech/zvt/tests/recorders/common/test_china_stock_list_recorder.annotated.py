# -*- coding: utf-8 -*-
# ✅ Best Practice: Ensure the test context is initialized before importing other modules to avoid side effects.
from ...context import init_test_context

# 🧠 ML Signal: Function definition for testing, indicating a test pattern
init_test_context()
# 🧠 ML Signal: Importing specific classes or functions can indicate which components are frequently used.

# 🧠 ML Signal: Instantiation of a class, indicating object-oriented usage
from zvt.recorders.eastmoney import EastmoneyStockRecorder


# 🧠 ML Signal: Method invocation on an object, indicating method usage pattern
# ⚠️ SAST Risk (Low): Bare except clause, which can catch unexpected exceptions and make debugging difficult
# ✅ Best Practice: Using assert to ensure the test fails if an exception is caught
def test_china_stock_recorder():
    recorder = EastmoneyStockRecorder()

    try:
        recorder.run()
    except:
        assert False
