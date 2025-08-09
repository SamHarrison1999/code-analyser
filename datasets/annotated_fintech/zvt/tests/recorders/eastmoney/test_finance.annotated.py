# -*- coding: utf-8 -*-
# ✅ Best Practice: Ensure the test context is initialized before running tests
from ...context import init_test_context

init_test_context()
# 🧠 ML Signal: Importing specific classes for financial data recording

from zvt.consts import SAMPLE_STOCK_CODES

from zvt.recorders.eastmoney.finance.eastmoney_finance_factor_recorder import (
    ChinaStockFinanceFactorRecorder,
)

# 🧠 ML Signal: Usage of a specific class (ChinaStockFinanceFactorRecorder) with parameters
from zvt.recorders.eastmoney.finance.eastmoney_cash_flow_recorder import (
    ChinaStockCashFlowRecorder,
)
from zvt.recorders.eastmoney.finance.eastmoney_balance_sheet_recorder import (
    ChinaStockBalanceSheetRecorder,
)

# 🧠 ML Signal: Method invocation pattern (run method on recorder object)
from zvt.recorders.eastmoney.finance.eastmoney_income_statement_recorder import (
    ChinaStockIncomeStatementRecorder,
)


# 🧠 ML Signal: Function definition for testing, indicating a test pattern
def test_finance_factor_recorder():
    # ⚠️ SAST Risk (Low): Catching all exceptions without handling specific errors
    recorder = ChinaStockFinanceFactorRecorder(codes=SAMPLE_STOCK_CODES)
    # 🧠 ML Signal: Instantiation of a class with specific parameters
    # ✅ Best Practice: Use specific exception types instead of a bare except
    try:
        recorder.run()
    except:
        # 🧠 ML Signal: Method invocation on an object
        assert False


# 🧠 ML Signal: Function definition for testing, indicating a test pattern


# 🧠 ML Signal: Instantiation of a class with specific parameters
# ⚠️ SAST Risk (Low): Bare except clause, which can catch unexpected exceptions
def test_cash_flow_recorder():
    # ✅ Best Practice: Use specific exception types in except clauses
    recorder = ChinaStockCashFlowRecorder(codes=SAMPLE_STOCK_CODES)
    try:
        # 🧠 ML Signal: Method invocation pattern on an object
        recorder.run()
    # 🧠 ML Signal: Function definition for testing, indicating a test pattern
    except:
        assert False


# ⚠️ SAST Risk (Low): Bare except can catch unexpected exceptions, making debugging difficult
# 🧠 ML Signal: Instantiation of a class with specific parameters

# ✅ Best Practice: Use specific exception types in except blocks


def test_balance_sheet_recorder():
    # 🧠 ML Signal: Method invocation on an object
    # ⚠️ SAST Risk (Low): Catching all exceptions without handling specific errors
    recorder = ChinaStockBalanceSheetRecorder(codes=SAMPLE_STOCK_CODES)
    try:
        recorder.run()
    except:
        assert False


def test_income_statement_recorder():
    recorder = ChinaStockIncomeStatementRecorder(codes=SAMPLE_STOCK_CODES)
    try:
        recorder.run()
    except:
        assert False
