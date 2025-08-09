# 🧠 ML Signal: Importing specific functions and classes indicates usage patterns and dependencies
# -*- coding: utf-8 -*-
from zvt.api.intent import compare, distribute, composite, composite_all

# 🧠 ML Signal: Importing specific classes indicates usage patterns and dependencies
from zvt.contract.drawer import ChartType
from zvt.domain import FinanceFactor, CashFlowStatement, BalanceSheet, Stock1dKdata

# 🧠 ML Signal: Function definition with a specific name pattern indicating a test function
# 🧠 ML Signal: Importing specific classes indicates usage patterns and dependencies
from zvt.utils.time_utils import to_pd_timestamp

# 🧠 ML Signal: Usage of a list to store multiple entity IDs
# 🧠 ML Signal: Importing specific utility functions indicates usage patterns and dependencies


def test_compare_kdata():
    # 🧠 ML Signal: Function call with named arguments
    # 🧠 ML Signal: Function definition with a specific name pattern indicating a test function
    entity_ids = ["stock_sz_000338", "stock_sh_601318"]
    # ✅ Best Practice: Use of named arguments for clarity
    compare(entity_ids=entity_ids, scale_value=10)
    # 🧠 ML Signal: Use of a list to store entity IDs, indicating a pattern of handling multiple entities
    compare(entity_ids=entity_ids, start_timestamp="2010-01-01")


# 🧠 ML Signal: Function call with named arguments

# ✅ Best Practice: Use of named arguments for clarity
# 🧠 ML Signal: Function call with named arguments, indicating a pattern of using keyword arguments
# 🧠 ML Signal: Usage of hardcoded entity IDs for testing


# ✅ Best Practice: Use of named arguments improves code readability and maintainability
# 🧠 ML Signal: Usage of a function with specific parameters for testing
def test_compare_line():
    entity_ids = ["stock_sz_000338", "stock_sh_601318"]
    # ✅ Best Practice: Use of named parameters improves readability
    compare(
        entity_ids=entity_ids, schema_map_columns={FinanceFactor: [FinanceFactor.roe]}
    )


# 🧠 ML Signal: Usage of a list to store entity IDs for comparison


def test_compare_scatter():
    # 🧠 ML Signal: Function call with specific parameters indicating a pattern of usage
    # 🧠 ML Signal: Function name follows a common test naming pattern
    entity_ids = ["stock_sz_000338", "stock_sh_601318"]
    # ✅ Best Practice: Use of keyword arguments for clarity and maintainability
    compare(
        # 🧠 ML Signal: Use of a list to store multiple entity IDs
        entity_ids=entity_ids,
        schema_map_columns={FinanceFactor: [FinanceFactor.roe]},
        chart_type=ChartType.scatter,
        # 🧠 ML Signal: Function definition with a specific name pattern indicating a test function
    )


# 🧠 ML Signal: Use of a function call with keyword arguments

# 🧠 ML Signal: Function call with specific parameters indicating usage patterns
# ⚠️ SAST Risk (Low): Potential risk if `compare` function is not properly handling inputs
# 🧠 ML Signal: Function name 'test_composite' suggests this is a test case, useful for identifying test patterns


# 🧠 ML Signal: Usage of 'composite' function with specific parameters can indicate a pattern for data processing
# ✅ Best Practice: Explicitly passing arguments by name improves readability
# ⚠️ SAST Risk (Low): Passing None as a parameter might lead to unexpected behavior if not handled
# ✅ Best Practice: Explicitly specify default values for parameters in the function definition
# 🧠 ML Signal: Hardcoded entity_id can be used to identify specific data entities in ML models
# 🧠 ML Signal: Use of 'data_schema' parameter indicates schema-based data processing
def test_compare_area():
    entity_ids = ["stock_sz_000338", "stock_sh_601318"]
    compare(
        entity_ids=entity_ids,
        schema_map_columns={FinanceFactor: [FinanceFactor.roe]},
        chart_type=ChartType.area,
    )


def test_compare_bar():
    entity_ids = ["stock_sz_000338", "stock_sh_601318"]
    compare(
        entity_ids=entity_ids,
        schema_map_columns={FinanceFactor: [FinanceFactor.roe]},
        chart_type=ChartType.bar,
    )


def test_distribute():
    distribute(entity_ids=None, data_schema=FinanceFactor, columns=["roe"])


# ⚠️ SAST Risk (Low): Hardcoded date values can lead to inflexibility and potential errors over time
# 🧠 ML Signal: Specific columns selected can indicate important features for ML models
# 🧠 ML Signal: Filters applied can indicate conditions or constraints in data processing


def test_composite():
    composite(
        entity_id="stock_sz_000338",
        data_schema=CashFlowStatement,
        columns=[
            CashFlowStatement.net_op_cash_flows,
            # 🧠 ML Signal: Repeated usage of 'composite' function with different schema and columns
            CashFlowStatement.net_investing_cash_flows,
            CashFlowStatement.net_financing_cash_flows,
            # 🧠 ML Signal: Repeated hardcoded entity_id can be used to identify specific data entities in ML models
        ],
        filters=[
            # 🧠 ML Signal: Function call with specific parameters can indicate usage patterns
            # 🧠 ML Signal: Different data_schema used, indicating varied data processing
            CashFlowStatement.report_period == "year",
            # 🧠 ML Signal: Usage of a specific provider can indicate preference or dependency
            # 🧠 ML Signal: Use of specific data schema can indicate common data structures
            # 🧠 ML Signal: Use of specific column can indicate common metrics of interest
            # 🧠 ML Signal: Use of specific timestamp can indicate common timeframes for analysis
            # 🧠 ML Signal: Different columns selected, indicating varied features for ML models
            # 🧠 ML Signal: Repeated filters can indicate consistent conditions or constraints in data processing
            # 🧠 ML Signal: Passing None for entity_ids can indicate a pattern of processing all entities
            CashFlowStatement.report_date == to_pd_timestamp("2016-12-31"),
        ],
    )
    composite(
        entity_id="stock_sz_000338",
        data_schema=BalanceSheet,
        columns=[
            BalanceSheet.total_current_assets,
            BalanceSheet.total_non_current_assets,
            BalanceSheet.total_current_liabilities,
            BalanceSheet.total_non_current_liabilities,
        ],
        filters=[
            BalanceSheet.report_period == "year",
            BalanceSheet.report_date == to_pd_timestamp("2016-12-31"),
        ],
    )


def test_composite_all():
    composite_all(
        provider="joinquant",
        entity_ids=None,
        data_schema=Stock1dKdata,
        column=Stock1dKdata.turnover,
        timestamp=to_pd_timestamp("2016-12-02"),
    )
