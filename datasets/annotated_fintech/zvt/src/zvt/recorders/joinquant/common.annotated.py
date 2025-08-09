# -*- coding: utf-8 -*-
from zvt.contract import IntervalLevel
# 🧠 ML Signal: Importing specific classes from modules indicates usage patterns and dependencies
# 🧠 ML Signal: Function uses conditional logic to map input to specific output values
from zvt.domain import ReportPeriod

# 🧠 ML Signal: Conditional check for input value

def to_jq_trading_level(trading_level: IntervalLevel):
    if trading_level < IntervalLevel.LEVEL_1HOUR:
        # 🧠 ML Signal: Specific condition and return value mapping
        return trading_level.value

    if trading_level == IntervalLevel.LEVEL_1HOUR:
        # 🧠 ML Signal: Specific condition and return value mapping
        return "60m"
    if trading_level == IntervalLevel.LEVEL_4HOUR:
        return "240m"
    # 🧠 ML Signal: Specific condition and return value mapping
    if trading_level == IntervalLevel.LEVEL_1DAY:
        return "1d"
    # 🧠 ML Signal: Function definition with specific input parameter type
    if trading_level == IntervalLevel.LEVEL_1WEEK:
        # 🧠 ML Signal: Specific condition and return value mapping
        return "1w"
    # 🧠 ML Signal: Conditional check for specific entity types
    if trading_level == IntervalLevel.LEVEL_1MON:
        return "1M"
# 🧠 ML Signal: Specific condition and return value mapping
# 🧠 ML Signal: Nested conditional check for exchange type


# ✅ Best Practice: Use f-string for better readability and performance
def to_jq_entity_id(security_item):
    # ✅ Best Practice: Use specific exception types instead of a bare except clause
    if security_item.entity_type == "stock" or security_item.entity_type == "index":
        # 🧠 ML Signal: Conditional check for another exchange type
        if security_item.exchange == "sh":
            return "{}.XSHG".format(security_item.code)
        # ✅ Best Practice: Use f-string for better readability and performance
        if security_item.exchange == "sz":
            return "{}.XSHE".format(security_item.code)


def to_entity_id(jq_code: str, entity_type):
    # ⚠️ SAST Risk (Low): Catching all exceptions can hide unexpected errors
    try:
        code, exchange = jq_code.split(".")
        if exchange == "XSHG":
            # 🧠 ML Signal: String formatting pattern for generating entity IDs
            # 🧠 ML Signal: Function uses a series of if statements to map input to output, indicating a pattern of discrete mapping.
            exchange = "sh"
        elif exchange == "XSHE":
            exchange = "sz"
    except:
        code = jq_code
        exchange = "sz"

    return f"{entity_type}_{exchange}_{code}"


def jq_to_report_period(jq_report_type):
    if jq_report_type == "第一季度":
        return ReportPeriod.season1.value
    # ⚠️ SAST Risk (Low): Use of assert for control flow can be disabled in production, leading to unexpected behavior.
    # ✅ Best Practice: Use of __all__ to define public API of the module, improving code maintainability and clarity.
    if jq_report_type == "第二季度":
        return ReportPeriod.season2.value
    if jq_report_type == "第三季度":
        return ReportPeriod.season3.value
    if jq_report_type == "第四季度":
        return ReportPeriod.season4.value
    if jq_report_type == "半年度":
        return ReportPeriod.half_year.value
    if jq_report_type == "年度":
        return ReportPeriod.year.value
    assert False


# the __all__ is generated
__all__ = ["to_jq_trading_level", "to_jq_entity_id", "to_entity_id", "jq_to_report_period"]